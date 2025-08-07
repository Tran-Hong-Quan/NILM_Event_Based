import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib

# --- CẤU HÌNH ---
image_path = "training_images/tulanh_segment_0012.png"
p_mean_value = 75.0  # ví dụ

# --- THÔNG SỐ ---
image_size = 32
img_input_size = image_size * image_size
p_input_size = 1

# --- TẢI LABEL ENCODER ---
label_encoder = joblib.load("label_encoder.pkl")
num_classes = len(label_encoder.classes_)

# --- ĐỊNH NGHĨA MÔ HÌNH ---
class MLPEnhancedDualBranch(nn.Module):
    def __init__(self, img_input_size, p_input_size=1, num_classes=3):
        super().__init__()
        self.img_branch = nn.Sequential(
            nn.Linear(img_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.p_branch = nn.Sequential(
            nn.Linear(p_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, p_mean):
        img_feat = self.img_branch(img)
        p_feat = self.p_branch(p_mean)
        combined = torch.cat((img_feat, p_feat), dim=1)
        out = self.classifier(combined)
        return out

# --- KHỞI TẠO MÔ HÌNH VÀ LOAD TRỌNG SỐ ---
model = MLPEnhancedDualBranch(img_input_size, p_input_size, num_classes)
model.load_state_dict(torch.load("mlp_dual_branch.pth", map_location="cpu"))
model.eval()

# --- XỬ LÝ ẢNH ---
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
img = Image.open(image_path).convert("L")
img_tensor = transform(img).view(-1)  # flatten
p_mean_tensor = torch.tensor([p_mean_value], dtype=torch.float32)

# --- DỰ ĐOÁN ---
with torch.no_grad():
    output = model(img_tensor.unsqueeze(0), p_mean_tensor.unsqueeze(0))  # thêm batch dim
    predicted_class = output.argmax(dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

print(f"✅ Ảnh: {image_path}")
print(f"⚡ P_mean = {p_mean_value}")
print(f"🔍 Dự đoán: {predicted_label}")
