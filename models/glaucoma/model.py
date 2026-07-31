import torch.nn as nn


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return {"logits": logits}


def build_model(cfg):
    mcfg = cfg["model"]
    return Simple3DCNN(
        in_channels=mcfg.get("input_channels", 1),
        num_classes=mcfg.get("num_classes", 2),
        dropout=mcfg.get("dropout", 0.3),
    )
