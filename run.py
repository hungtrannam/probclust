import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import piq  # pip install piq

# ===== Thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Hàm tính loss histogram màu

# ===== 1. Hàm cơ sở Gauss học được
class LearnableBasis(nn.Module):
    def __init__(self, num_basis, grid):
        super().__init__()
        self.grid = torch.tensor(grid, dtype=torch.float32).to(device)  # [M]
        self.mus = nn.Parameter(torch.linspace(grid[0], grid[-1], num_basis))  # [K]
        self.log_sigmas = nn.Parameter(torch.zeros(num_basis))  # [K]

    def forward(self):
        grid = self.grid.view(1, -1)  # [1, M]
        mus = self.mus.view(-1, 1)    # [K, 1]
        sigmas = torch.exp(self.log_sigmas).view(-1, 1)  # [K, 1]
        phi = torch.exp(-0.5 * ((grid - mus) / sigmas) ** 2)
        phi = phi / torch.trapezoid(phi, self.grid, dim=1).unsqueeze(1)
        return phi  # [K, M]

# ===== 2. Encoder: ảnh RGB → alpha
class EncoderToAlpha(nn.Module):
    def __init__(self, num_basis):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_basis),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.encoder(x)

# ===== 3. Decoder: PDF → ảnh RGB
class DecoderFromPDF(nn.Module):
    def __init__(self, pdf_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(pdf_dim, 128 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return self.deconv(x)

# ===== 4. Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ===== 5. Khởi tạo mô hình
grid_dim = 2000
grid = np.linspace(-50, 50, grid_dim)
num_basis =100

encoder = EncoderToAlpha(num_basis).to(device)
basis_model = LearnableBasis(num_basis, grid).to(device)
decoder = DecoderFromPDF(grid_dim).to(device)

params = list(encoder.parameters()) + list(basis_model.parameters()) + list(decoder.parameters())
optimizer = torch.optim.RMSprop(params, lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
loss_fn = nn.MSELoss()

# ===== 6. Huấn luyện
print("🔁 Training...")
for epoch in range(100):
    total_loss = 0
    for i, (img, _) in enumerate(loader):
        if i >= 100: break
        img = img.to(device)
        alpha = encoder(img)             # [B, K]
        basis = basis_model()            # [K, M]
        pdf = alpha @ basis              # [B, M]
        recon = decoder(pdf)             # [B, 3, 32, 32]

        ssim_loss = 1 - piq.ssim(recon, img, data_range=1.)
        loss = ssim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1:02d}: loss = {total_loss/100:.4f}")

# ===== 7. Vẽ kết quả
os.makedirs("output/pdf_basis", exist_ok=True)
for i, (img, _) in enumerate(loader):
    if i >= 10: break
    img = img.to(device)
    with torch.no_grad():
        alpha = encoder(img)
        basis = basis_model()
        pdf = (alpha @ basis).squeeze().cpu().numpy()
        recon = decoder(alpha @ basis).squeeze().permute(1, 2, 0).cpu().numpy()
        original = img.squeeze().permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(9, 2.5))
    axs[0].imshow(original)
    axs[0].set_title("Ảnh gốc"); axs[0].axis("off")

    axs[1].imshow(recon)
    axs[1].set_title("Tái dựng"); axs[1].axis("off")

    axs[2].plot(grid, pdf)
    axs[2].fill_between(grid, pdf, alpha=0.3)
    axs[2].set_title("PDF từ cơ sở học được")
    axs[2].set_ylim(0, np.max(pdf)*1.1); axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"output/pdf_basis/sample_{i}.png")
    plt.close()


