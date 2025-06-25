import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# --- 1. Generator (UGATIT Generator) ---
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
                output[:, :, h_start:h_end, w_start:w_end] = torch.bmm(h, attention.permute(0, 2, 1)).view(B, C, H_patch, W_patch)
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

# --- 2. Discriminator (Multi-Scale Discriminator) ---
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            model += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(nf * 2, 512)
        model += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True), nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_discriminators=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_discriminators = num_discriminators
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
        self.discriminators = nn.ModuleList([NLayerDiscriminator(input_nc, ndf, n_layers) for _ in range(num_discriminators)])

    def forward(self, input):
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(input))
            if i != self.num_discriminators - 1:
                input = self.downsample(input)
        return outputs

# --- 3. Loss Functions ---
def generator_loss(fake_output, real_output, fake_cycle_output, real_cycle_output, real_attention, fake_attention, real_cycle_attention, fake_cycle_attention, lambda_cycle=10.0, lambda_identity=5.0, lambda_attention=1.0):
    adversarial_loss = sum(torch.mean(F.relu(1.0 - output)) for output in fake_output) / len(fake_output)
    cycle_loss = F.l1_loss(fake_cycle_output, real_cycle_output)
    identity_loss = F.l1_loss(fake_output[0], real_output[0])  # Simplified for speed
    attention_loss = F.l1_loss(real_attention, fake_attention) + F.l1_loss(real_cycle_attention, fake_cycle_attention)
    total_loss = adversarial_loss + lambda_cycle * cycle_loss + lambda_identity * identity_loss + lambda_attention * attention_loss
    return total_loss, identity_loss, adversarial_loss, cycle_loss, attention_loss

def discriminator_loss(real_output, fake_output):
    d_loss_real = sum(torch.mean(F.relu(1.0 - output)) for output in real_output) / len(real_output)
    d_loss_fake = sum(torch.mean(F.relu(1.0 + output)) for output in fake_output) / len(fake_output)
    return d_loss_real, d_loss_fake, d_loss_real + d_loss_fake

# --- 4. Data Loaders ---
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(image) if self.transform else image

class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(image) if self.transform else image

# --- 5. Training Loop ---
if __name__ == '__main__':
    # Parameters
    celeba_root = 'datasets/img_align_celeba (2)/img_align_celeba'
    anime_root = 'datasets/images2'
    image_size, batch_size, num_epochs, save_interval, num_visualizations = 128, 8, 100, 10, 4

    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Datasets and Loaders
    dataset_A, dataset_B = CelebADataset(celeba_root, transform), AnimeFaceDataset(anime_root, transform)
    data_loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    data_loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # Models and Optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_AtoB, generator_BtoA = Generator().to(device), Generator().to(device)
    discriminator_A, discriminator_B = MultiScaleDiscriminator(input_nc=3).to(device), MultiScaleDiscriminator(input_nc=3).to(device)

    # Multi-GPU Support
    if torch.cuda.device_count() > 1:
        generator_AtoB, generator_BtoA = nn.DataParallel(generator_AtoB), nn.DataParallel(generator_BtoA)
        discriminator_A, discriminator_B = nn.DataParallel(discriminator_A), nn.DataParallel(discriminator_B)

    optimizer_G = torch.optim.Adam(list(generator_AtoB.parameters()) + list(generator_BtoA.parameters()), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D_A, optimizer_D_B = torch.optim.Adam(discriminator_A.parameters(), lr=0.0001, betas=(0.5, 0.999)), torch.optim.Adam(discriminator_B.parameters(), lr=0.0001, betas=(0.5, 0.999))

    scaler = GradScaler('cuda')
    os.makedirs('generated_images', exist_ok=True)

    # Training Loop
    for epoch in range(num_epochs):
        data_iter = tqdm(zip(data_loader_A, data_loader_B), total=min(len(data_loader_A), len(data_loader_B)), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, (real_A_batch, real_B_batch) in enumerate(data_iter):
            real_A_batch, real_B_batch = real_A_batch.to(device), real_B_batch.to(device)

            # Train Discriminators
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()
            with autocast('cuda'):
                fake_B, fake_attention_B_map = generator_AtoB(real_A_batch)
                fake_A, fake_attention_A_map = generator_BtoA(real_B_batch)
                d_loss_A = discriminator_loss(discriminator_A(real_A_batch), discriminator_A(fake_A.detach()))
                d_loss_B = discriminator_loss(discriminator_B(real_B_batch), discriminator_B(fake_B.detach()))
                d_loss = (d_loss_A[2] + d_loss_B[2]) * 0.5
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D_A)
            scaler.step(optimizer_D_B)
            scaler.update()

            # Train Generators
            optimizer_G.zero_grad()
            with autocast('cuda'):
                fake_B, fake_attention_B_map = generator_AtoB(real_A_batch)
                fake_A, fake_attention_A_map = generator_BtoA(real_B_batch)
                cycled_A, cycled_attention_A_map = generator_BtoA(fake_B)
                cycled_B, cycled_attention_B_map = generator_AtoB(fake_A)
                identity_A, identity_B = generator_BtoA(real_A_batch)[0], generator_AtoB(real_B_batch)[0]
                real_attention_A_map, real_attention_B_map = generator_AtoB(real_A_batch)[1], generator_BtoA(real_B_batch)[1]
                real_cycle_attention_A_map, fake_cycle_attention_B_map = generator_BtoA(fake_B)[1], generator_AtoB(fake_A)[1]
                g_loss_AtoB = generator_loss(discriminator_B(fake_B), discriminator_B(real_B_batch), cycled_B, real_B_batch, real_attention_A_map, fake_attention_B_map, real_cycle_attention_A_map, fake_cycle_attention_B_map)
                g_loss_BtoA = generator_loss(discriminator_A(fake_A), discriminator_A(real_A_batch), cycled_A, real_A_batch, real_attention_B_map, fake_attention_A_map, cycled_attention_B_map, real_cycle_attention_A_map)
                identity_loss = (F.l1_loss(identity_A, real_A_batch) + F.l1_loss(identity_B, real_A_batch)) * 5.0
                total_g_loss = g_loss_AtoB[0] + g_loss_BtoA[0] + identity_loss

            scaler.scale(total_g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Visualization
            if (i + 1) % save_interval == 0:
                with torch.no_grad():
                    real_A_sample, real_B_sample = real_A_batch[:num_visualizations], real_B_batch[:num_visualizations]
                    fake_B_sample, fake_A_sample = generator_AtoB(real_A_sample)[0], generator_BtoA(real_B_sample)[0]
                    save_image(torch.cat((real_A_sample * 0.5 + 0.5, fake_B_sample * 0.5 + 0.5), dim=3), f'generated_images/AtoB_epoch_{epoch + 1}_batch_{i + 1}.png', nrow=num_visualizations)
                    save_image(torch.cat((real_B_sample * 0.5 + 0.5, fake_A_sample * 0.5 + 0.5), dim=3), f'generated_images/BtoA_epoch_{epoch + 1}_batch_{i + 1}.png', nrow=num_visualizations)

            # Logging
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader_A)}], D_loss: {d_loss.item():.4f}, G_loss: {total_g_loss.item():.4f}, GPU Memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")

        # Save Models
        if (epoch + 1) % 1 == 0:
            torch.save(generator_AtoB.state_dict(), f'generator_AtoB_{epoch + 1}.pth')
            torch.save(generator_BtoA.state_dict(), f'generator_BtoA_{epoch + 1}.pth')
            torch.save(discriminator_A.state_dict(), f'discriminator_A_{epoch + 1}.pth')
            torch.save(discriminator_B.state_dict(), f'discriminator_B_{epoch + 1}.pth')

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                save_image(generator_AtoB(real_A_batch[:1])[0] * 0.5 + 0.5, f'fake_B_epoch_{epoch + 1}.png')
                save_image(generator_BtoA(real_B_batch[:1])[0] * 0.5 + 0.5, f'fake_A_epoch_{epoch + 1}.png')
