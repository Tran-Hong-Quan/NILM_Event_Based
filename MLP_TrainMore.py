import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import joblib
from MLP import MLP

# --- Config ---
csv_path = "MLP_data.csv"  # CSV có thể chứa dữ liệu cũ + mới
image_size = 32
batch_size = 32
num_epochs = 5  # train thêm 5 epoch
learning_rate = 0.001
checkpoint_path = "MLP_checkpoint.pth"
label_encoder_path = "MLP_label_encoder.pkl"

# --- Dataset ---
class DualBranchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert("L")
        if self.transform:
            img = self.transform(img)
        img = img.view(-1)
        p_mean = torch.tensor([row['P_mean']], dtype=torch.float32)
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        return img, p_mean, label

# --- Load dữ liệu ---
df = pd.read_csv(csv_path)

# Load label encoder cũ (nếu cùng nhãn)
label_encoder = joblib.load(label_encoder_path)
df['label_encoded'] = label_encoder.transform(df['label'])
num_classes = len(label_encoder.classes_)

# --- Chia train/val ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_dataset = DualBranchDataset(train_df, transform)
val_dataset = DualBranchDataset(val_df, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(img_input_size=image_size*image_size, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# --- Load checkpoint cũ ---
start_epoch = 0
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0)  # Nếu không có thì mặc định 0
    print(f"✅ Đã load checkpoint, tiếp tục huấn luyện từ epoch {start_epoch+1}")
except FileNotFoundError:
    print("⚠️ Không tìm thấy checkpoint, huấn luyện từ đầu.")

# --- Training loop ---
for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    total_loss = 0
    for img, p_mean, label in train_loader:
        img, p_mean, label = img.to(device), p_mean.to(device), label.to(device)

        output = model(img, p_mean)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, p_mean, label in val_loader:
            img, p_mean, label = img.to(device), p_mean.to(device), label.to(device)
            output = model(img, p_mean)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dataset):.4f} | Val Acc: {acc:.2f}%")

    # --- Save checkpoint mỗi epoch ---
    torch.save({
        "epoch": epoch + 1,  # lưu epoch tiếp theo
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, checkpoint_path)

# --- Save model state riêng ---
torch.save(model.state_dict(), "MLP.pth")
print("💾 Mô hình đã được lưu và có thể tiếp tục huấn luyện lần sau.")
