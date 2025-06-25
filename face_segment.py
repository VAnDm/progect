import cv2
import numpy as np
import mediapipe as mp
import os

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Для обработки статических изображений
    max_num_faces=1,        # Обрабатываем только одно лицо
    refine_landmarks=True,  # Улучшенные ключевые точки (например, для глаз)
    min_detection_confidence=0.5
)

def segment_face_with_mediapipe(image_path, background_color=(0, 0, 0), expand_mask=True, expand_factor=1.2):
    """
    Отделяет лицо от заднего фона на изображении с помощью MediaPipe Face Mesh.

    Args:
        image_path (str): Путь к изображению.
        background_color (tuple): Цвет для замены заднего фона (RGB).
                                 Если None, задний фон будет прозрачным.
        expand_mask (bool): Если True, расширяет маску для захвата волос и шеи.
        expand_factor (float): Коэффициент расширения маски (1.0 - без изменений, >1.0 - расширение).

    Returns:
        numpy.ndarray: Изображение с отделенным лицом и измененным задним фоном.
                       Вернет None, если не удалось обработать изображение.
    """
    try:
        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Обработка изображения с помощью MediaPipe
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print(f"Лицо не обнаружено на изображении: {image_path}")
            return None

        # Создание маски лица
        mask = np.zeros((height, width), dtype=np.uint8)

        # Получение ключевых точек лица
        for face_landmarks in results.multi_face_landmarks:
            # Используем контур лица (MediaPipe предоставляет индексы для FACEMESH_FACE_OVAL)
            points = []
            for idx in mp_face_mesh.FACEMESH_FACE_OVAL:
                pt = face_landmarks.landmark[idx[0]]
                x, y = int(pt.x * width), int(pt.y * height)
                points.append([x, y])

            # Преобразуем точки в numpy массив
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)

            # Заполняем маску лица
            cv2.fillConvexPoly(mask, hull, 255)

            # Расширение маски (для захвата волос и шеи)
            if expand_mask:
                # Находим ограничивающий прямоугольник
                x, y, w, h = cv2.boundingRect(hull)
                # Расширяем прямоугольник
                new_w = int(w * expand_factor)
                new_h = int(h * expand_factor)
                new_x = max(0, x - (new_w - w) // 2)
                new_y = max(0, y - (new_h - h) // 2)
                new_x2 = min(width, new_x + new_w)
                new_y2 = min(height, new_y + new_h)

                # Создаем новую маску с расширенной областью
                expanded_mask = np.zeros_like(mask)
                cv2.ellipse(expanded_mask,
                           (new_x + (new_x2 - new_x) // 2, new_y + (new_y2 - new_y) // 2),
                           ((new_x2 - new_x) // 2, (new_y2 - new_y) // 2),
                           0, 0, 360, 255, -1)
                mask = cv2.bitwise_or(mask, expanded_mask)

        # Сглаживание маски
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = (mask > 128).astype(np.uint8) * 255

        # Применение маски
        face_mask_bool = mask > 0
        original_image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if background_color is None:
            # Создание изображения с прозрачным фоном (RGBA)
            segmented_image = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2RGBA)
            alpha_channel = np.where(face_mask_bool, 255, 0).astype(np.uint8)
            segmented_image[:, :, 3] = alpha_channel
            return segmented_image
        else:
            # Создание изображения с заданным цветом фона
            background = np.full(original_image_np.shape, background_color, dtype=np.uint8)
            segmented_face = np.where(np.expand_dims(face_mask_bool, axis=-1), original_image_np, background).astype(np.uint8)
            return segmented_face

    except Exception as e:
        print(f"Произошла ошибка при обработке {image_path}: {e}")
        return None

if __name__ == '__main__':
    # Укажите путь к вашей директории с изображениями
    directory = '../datasets/img_align_celeba/img_align_celeba/'
    # Измените background_color на (255, 255, 255) для белого фона
    background_color = (255, 255, 255)  # Белый фон

    # Обработка всех изображений в директории
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            segmented_image = segment_face_with_mediapipe(image_path, background_color=background_color, expand_mask=True, expand_factor=1.2)
            if segmented_image is not None:
                # Сохранение с учетом типа фона
                if background_color is None:
                    # Для прозрачного фона сохраняем как PNG
                    output_path = image_path.rsplit('.', 1)[0] + '.png'
                    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA))
                else:
                    # Для цветного фона сохраняем в исходном формате
                    cv2.imwrite(image_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
                print(f"Обработано и заменено изображение: {filename}")