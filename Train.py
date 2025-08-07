import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import joblib

# --- Config ---
csv_path = "output_data.csv"
image_size = 32
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# --- Load CSV ---
df = pd.read_csv(csv_path)
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

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

# --- Split dataset ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_dataset = DualBranchDataset(train_df, transform)
val_dataset = DualBranchDataset(val_df, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- MLP Dual-Branch Model ---
class MLPDualBranch(nn.Module):
    def __init__(self, img_input_size, p_input_size=1, num_classes=3):
        super().__init__()

        # Nhánh ảnh (xử lý nhiều tầng hơn)
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

        # Nhánh P_mean (xử lý nhẹ nhưng giữ đặc trưng)
        self.p_branch = nn.Sequential(
            nn.Linear(p_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Nhánh kết hợp
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


# --- Training setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPDualBranch(img_input_size=image_size*image_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training loop ---
for epoch in range(num_epochs):
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
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_dataset):.4f} | Val Acc: {acc:.2f}%")

# --- Save model and label encoder ---
torch.save(model.state_dict(), "mlp_dual_branch.pth")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Mô hình đã được lưu.")
