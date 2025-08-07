import torch.nn as nn
import torch

class MLP(nn.Module):
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