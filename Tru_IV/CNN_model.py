import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CNN_TwoBranch_1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_TwoBranch_1, self).__init__()

        # --- Nhánh 1: Ảnh 33x33 ---
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8, 128),
            nn.ReLU(),
        )

        # --- Nhánh 2: Float ---
        self.float_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # --- Kết hợp 2 nhánh ---
        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_float, x_img):
        img_feat = self.cnn_fc(self.cnn_branch(x_img))
        float_feat = self.float_branch(x_float)
        combined = torch.cat((img_feat, float_feat), dim=1)
        out = self.combined_fc(combined)
        return out