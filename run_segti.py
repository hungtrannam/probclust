import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import cv2

# ==== 1. CNN Feature Extractor ====
class CNNFeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Lấy feature map sau layer3 để giữ nhiều thông tin không gian
        self.model = nn.Sequential(*list(backbone.children())[:-3])
        self.model.eval().to(device)
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_pil):
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat_map = self.model(img_tensor)  # [1, C, H, W]
        return feat_map.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]

# ==== 2. Trộn thêm thông tin màu + vị trí ====
def build_hyper_feature(image_path):
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np = np.array(img) / 255.0
    H, W, _ = img_np.shape

    # CNN feature
    extractor = CNNFeatureExtractor()
    cnn_feat = extractor.extract(img)  # [H, W, C]

    # Tạo lưới toạ độ
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    coords = np.stack([xx, yy], axis=-1)  # [H, W, 2]

    # Nối (RGB + coords + CNN_feature)
    hyper_feat = np.concatenate([img_np, coords, cnn_feat], axis=-1)
    return hyper_feat.reshape(-1, hyper_feat.shape[-1])

# ==== 3. Chiếu về 1D và tính PDF ====
def compute_1d_pdf(features, bins=256, bandwidth=0.3):
    pca = PCA(n_components=1)
    proj = pca.fit_transform(features).ravel()

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(proj[:, None])

    x = np.linspace(proj.min(), proj.max(), bins)[:, None]
    pdf = np.exp(kde.score_samples(x))
    pdf /= pdf.sum()
    return x.ravel(), pdf, proj

# ==== 4. Chạy ví dụ ====
if __name__ == "__main__":
    img_path = "image.jpg"
    print("[INFO] Trích xuất hyper-feature từ CNN + màu...")
    features = build_hyper_feature(img_path)
    print(f"[INFO] Số lượng vector đặc trưng: {features.shape}")

    print("[INFO] Chiếu về 1D và tính PDF...")
    x, pdf, proj = compute_1d_pdf(features)

    plt.figure(figsize=(8, 4))
    plt.plot(x, pdf, color='black')
    plt.title("CNN-based Hyper-Feature PDF (1D)")
    plt.xlabel("Trục đặc trưng")
    plt.ylabel("Mật độ")
    plt.show()
