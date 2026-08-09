import torch
import torch.nn as nn


def _norm3d(channels: int, kind: str = "group", groups: int = 8) -> nn.Module:
    """GroupNorm is batch-size independent (fixes BatchNorm with bs<8).

    'group' (default) computes stats per-sample -> stable with batch_size 2-4.
    'batch' keeps the old behaviour when explicitly requested.
    """
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    g = min(groups, channels)
    while channels % g != 0:  # GroupNorm requires channels % groups == 0
        g -= 1
    return nn.GroupNorm(g, channels)


def _pool3d(kind: str, kernel_size) -> nn.Module:
    if kind == "avg":
        return nn.AvgPool3d(kernel_size, stride=kernel_size)
    return nn.MaxPool3d(kernel_size, stride=kernel_size)


class Simple3DCNN(nn.Module):
    """3D CNN for OCT glaucoma classification.

    Fixes vs the original architecture:
      1. GroupNorm instead of BatchNorm3d  -> trains stably at batch_size 2-4.
      2. Anisotropic pooling (2,2,1)       -> keeps full B-scan depth resolution,
                                             so the thin RNFL signal survives
                                             to the final AdaptiveAvgPool.
      3. Optional residual connections     -> smoother gradients for deeper nets.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        dropout=0.3,
        hidden=(16, 32, 64, 128),
        norm="group",
        norm_groups=8,
        pool_strides=((2, 2, 1), (2, 2, 1), (2, 2, 1)),
        pool_type="max",
        residual=True,
    ):
        super().__init__()
        hidden = tuple(hidden)
        strides = [tuple(s) for s in pool_strides]
        if len(strides) != len(hidden):  # one stride per conv block
            strides = [(2, 2, 1)] * len(hidden)

        self.features = nn.Sequential()
        cin = in_channels
        for i, cout in enumerate(hidden):
            block = [
                nn.Conv3d(cin, cout, kernel_size=3, padding=1),
                _norm3d(cout, norm, norm_groups),
                nn.ReLU(inplace=True),
            ]
            if residual and i > 0:
                block.append(SimpleResidual3d(cout, norm, norm_groups))
            self.features.append(nn.Sequential(*block))
            self.features.append(_pool3d(pool_type, strides[i]))
            cin = cout

        self.features.append(nn.AdaptiveAvgPool3d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return {"logits": logits}


class SimpleResidual3d(nn.Module):
    """Conv(3)->Norm->ReLU + skip connection. Keeps spatial dims, refines channels."""

    def __init__(self, channels: int, norm: str = "group", groups: int = 8):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = _norm3d(channels, norm, groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm(self.conv(x)))


def build_model(cfg):
    mcfg = cfg["model"]
    return Simple3DCNN(
        in_channels=mcfg.get("input_channels", 1),
        num_classes=mcfg.get("num_classes", 2),
        dropout=mcfg.get("dropout", 0.3),
        hidden=tuple(mcfg.get("hidden", (16, 32, 64, 128))),
        norm=mcfg.get("norm", "group"),
        norm_groups=mcfg.get("norm_groups", 8),
        pool_strides=tuple(mcfg.get("pool_strides", ((2, 2, 1), (2, 2, 1), (2, 2, 1)))),
        pool_type=mcfg.get("pool_type", "max"),
        residual=mcfg.get("residual", True),
    )


def probe_model_shapes():
    """Quick shape sanity: run a random 200³ volume through the net."""
    m = Simple3DCNN().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 1, 96, 96, 96))
    print("features out:", m.features(torch.randn(2, 1, 96, 96, 96)).shape)
    print("logits:", out["logits"].shape)


if __name__ == "__main__":
    probe_model_shapes()
