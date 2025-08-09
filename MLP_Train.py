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
from MLP import MLP

# --- Config ---
csv_path = "MLP_data.csv"
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

# --- Training setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(img_input_size=image_size*image_size, num_classes=num_classes).to(device)
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
torch.save(model.state_dict(), "MLMP.pth")
joblib.dump(label_encoder, "MLP_label_encoder.pkl")
print("✅ Mô hình đã được lưu.")
