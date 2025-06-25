import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import mediapipe as mp

# --- Инициализация MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- Определение класса генератора ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return F.relu(out)

class AdaptiveLayerInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaptiveLayerInstanceNorm, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        instance_mean = x.mean([2, 3], keepdim=True)
        instance_var = x.var([2, 3], keepdim=True, unbiased=False)
        layer_mean = x.mean([0, 2, 3], keepdim=True)
        layer_var = x.var([0, 2, 3], keepdim=True, unbiased=False)
        mean = self.rho * instance_mean + (1 - self.rho) * layer_mean
        var = self.rho * instance_var + (1 - self.rho) * layer_var
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class AttentionModule(nn.Module):
    def __init__(self, in_channels, window_size=8):
        super(AttentionModule, self).__init__()
        self.window_size = window_size
        self.conv_f = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_g, f_l):
        B, C, H, W = f_g.size()
        window_size, stride = self.window_size, self.window_size // 2
        output = torch.zeros_like(f_l)
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                h_start, h_end = i, min(i + window_size, H)
                w_start, w_end = j, min(j + window_size, W)
                if h_end - h_start < 2 or w_end - w_start < 2:
                    continue
                f_patch = self.conv_f(f_g[:, :, h_start:h_end, w_start:w_end])
                g_patch = self.conv_g(f_l[:, :, h_start:h_end, w_start:w_end])
                h_patch = self.conv_h(f_l[:, :, h_start:h_end, w_start:w_end])
                B, C_f, H_patch, W_patch = f_patch.size()
                f, g, h = [x.view(B, -1, H_patch * W_patch) for x in (f_patch, g_patch, h_patch)]
                attention = self.softmax(torch.bmm(f.permute(0, 2, 1), g))
                output[:, :, h_start:h_end, w_start:w_end] = torch.bmm(h, attention.permute(0, 2, 1)).view(B, C, h_end - h_start, w_end - w_start)
        return output

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_residual_blocks=4):
        super(Generator, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=3), AdaptiveLayerInstanceNorm(ngf), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), AdaptiveLayerInstanceNorm(ngf * 2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1), AdaptiveLayerInstanceNorm(ngf * 4), nn.ReLU(True))
        self.res_blocks = nn.ModuleList([ResidualBlock(ngf * 4) for _ in range(n_residual_blocks)])
        self.attention = AttentionModule(ngf * 4)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1), AdaptiveLayerInstanceNorm(ngf * 2), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1), AdaptiveLayerInstanceNorm(ngf), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.Conv2d(ngf, out_channels, kernel_size=7, stride=1, padding=3), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        out = e3
        for block in self.res_blocks:
            out = block(out)
        attention_map = self.attention(out, out)
        return self.dec3(self.dec2(self.dec1(out))), attention_map

# --- Сегментация лица с помощью MediaPipe ---
def segment_face_with_mediapipe(image, background_color=(255, 255, 255), expand_mask=True, expand_factor=1.2):
    try:
        height, width = image.shape[:2]
        results = face_mesh.process(image)

        if not results.multi_face_landmarks:
            print("Лицо не обнаружено на изображении")
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for idx in mp_face_mesh.FACEMESH_FACE_OVAL:
                pt = face_landmarks.landmark[idx[0]]
                x, y = int(pt.x * width), int(pt.y * height)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            if expand_mask:
                x, y, w, h = cv2.boundingRect(hull)
                new_w = int(w * expand_factor)
                new_h = int(h * expand_factor)
                new_x = max(0, x - (new_w - w) // 2)
                new_y = max(0, y - (new_h - h) // 2)
                new_x2 = min(width, new_x + new_w)
                new_y2 = min(height, new_y + new_h)

                expanded_mask = np.zeros_like(mask)
                cv2.ellipse(expanded_mask,
                           (new_x + (new_x2 - new_x) // 2, new_y + (new_y2 - new_y) // 2),
                           ((new_x2 - new_x) // 2, (new_y2 - new_y) // 2),
                           0, 0, 360, 255, -1)
                mask = cv2.bitwise_or(mask, expanded_mask)

        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = (mask > 128).astype(np.uint8) * 255

        face_mask_bool = mask > 0
        background = np.full(image.shape, background_color, dtype=np.uint8)
        segmented_face = np.where(np.expand_dims(face_mask_bool, axis=-1), image, background).astype(np.uint8)
        return segmented_face

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

# --- Предобработка изображения ---
def preprocess_image(image, image_size=128):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# --- Постобработка результата ---
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu() * 0.5 + 0.5
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

# --- Основная функция ---
def main():
    # Параметры
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 128
    background_color = (255, 255, 255)
    weights_path = "generator_AtoB_1.pth"

    # Инициализация модели
    generator = Generator().to(device)
    try:
        generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        generator.eval()
    except FileNotFoundError:
        print(f"Ошибка: Файл весов {weights_path} не найден")
        return
    except RuntimeError as e:
        print(f"Ошибка загрузки весов: {e}")
        return

    # Захват видео с веб-камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось захватить кадр")
            break

        # Преобразование кадра OpenCV (BGR) в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Сегментация лица с веб-камеры
        segmented_frame = segment_face_with_mediapipe(frame_rgb, background_color=background_color)
        if segmented_frame is None:
            segmented_frame = frame_rgb

        # Преобразование сегментированного кадра в PIL Image
        pil_image = Image.fromarray(segmented_frame)

        # Предобработка изображения с веб-камеры
        input_tensor = preprocess_image(pil_image, image_size).to(device)

        # Инференс
        with torch.no_grad():
            fake_B, _ = generator(input_tensor)
        output_image = postprocess_image(fake_B)

        # Добавление текста с информацией
        output_with_text = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.putText(output_with_text, "Anime Style", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Отображение результатов
        cv2.imshow("Generated Anime Face", output_with_text)
        cv2.imshow("Webcam Original", frame)
        cv2.imshow("Webcam Segmented Face", cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR))

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()