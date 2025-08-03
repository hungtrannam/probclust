import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from piq import ssim

# ===== Thiết lập =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 28
latent_dim = 128
batch_size = 64
dataset_path = 'dataset/trainingSample'

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
train_dataset = ImageFolder(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ===== Basis Gauss phân bố đều =====
class GaussianBasis2D(nn.Module):
    def __init__(self, K, M):
        super().__init__()
        self.K = K
        self.M = M
        coords = torch.linspace(0, 1, M)
        X, Y = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

        # Khởi tạo μ phủ đều ảnh
        n = int(np.ceil(np.sqrt(K)))
        mu_grid = torch.linspace(0.1, 0.9, n)
        mx, my = torch.meshgrid(mu_grid, mu_grid, indexing='ij')
        mx = mx.flatten()[:K]
        my = my.flatten()[:K]
        self.mu_x = nn.Parameter(mx)
        self.mu_y = nn.Parameter(my)
        self.log_sigma = nn.Parameter(torch.zeros(K))

    def forward(self):
        basis_list = []
        for k in range(self.K):
            gx = self.X - self.mu_x[k]
            gy = self.Y - self.mu_y[k]
            sigma = torch.exp(self.log_sigma[k])
            g = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))
            basis_list.append(g)
        return torch.stack(basis_list, dim=0)  # [K, M, M]

# ===== Mô hình chính =====
class VAR(nn.Module):
    def __init__(self, latent_dim=128, image_size=28):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )
        self.basis_model = GaussianBasis2D(latent_dim, image_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
            nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        alpha = self.encoder(x)                          # [B, K]
        alpha = torch.softmax(alpha, dim=1)              # convex weights
        basis = self.basis_model()                       # [K, M, M]
        pdf = torch.einsum('bk,kij->bij', alpha, basis)  # [B, M, M]
        # Normalize PDF to [0,1]
        pdf_min = pdf.view(pdf.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        pdf_max = pdf.view(pdf.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        pdf = (pdf - pdf_min) / (pdf_max - pdf_min + 1e-6)
        recon = self.decoder(pdf.unsqueeze(1))           # [B, 1, M, M]
        return recon, pdf, alpha

# ===== Loss kết hợp =====
def entropy_loss(alpha):
    return -torch.sum(alpha * torch.log(alpha + 1e-8), dim=1).mean()

def compute_loss(x, recon, alpha, λ=0.01):
    mse = F.mse_loss(recon, x)
    ssim_val = ssim(recon, x, data_range=1.0)
    ent = entropy_loss(alpha)
    return mse + ssim_val + λ * ent

# ===== Huấn luyện =====
model = VAR(latent_dim, image_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        recon, pdf, alpha = model(x)
        loss = compute_loss(x, recon, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, α max={alpha.max():.3f}, α min={alpha.min():.3f}, pdf min={pdf.min():.3f}, max={pdf.max():.3f}")



model.eval()
x, _ = next(iter(train_loader))
x = x.to(device)

with torch.no_grad():
    recon, pdf, alpha = model(x)

alpha_np = alpha.cpu().numpy()
pdf_np = pdf.cpu().numpy()
x_np = x.cpu().numpy()
recon_np = recon.cpu().numpy()

fig, axes = plt.subplots(3, 3, figsize=(10, 9))
titles = ["Ảnh gốc", "PDF", "Ảnh tái tạo"]
for i in range(3):
    axes[i, 0].imshow(x_np[i, 0], cmap='gray'); axes[i, 0].set_title(titles[0]); axes[i, 0].axis('off')
    axes[i, 1].imshow(pdf_np[i], cmap='viridis'); axes[i, 1].set_title(titles[1]); axes[i, 1].axis('off')
    axes[i, 2].imshow(recon_np[i, 0], cmap='gray'); axes[i, 2].set_title(titles[2]); axes[i, 2].axis('off')
plt.tight_layout()
plt.show()

# Vẽ vector α
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    sns.barplot(x=np.arange(latent_dim), y=alpha_np[i], ax=axes[i], color='blue')
    axes[i].set_title(f'α vector #{i}')
plt.tight_layout()
plt.show()
