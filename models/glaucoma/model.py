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


class SwinUNETRClassifier(nn.Module):
    """SwinUNETR encoder turned into a classifier (decoder removed).

    MONAI's SwinUNETR is a segmentation net; this wrapper keeps only the Swin
    transformer encoder (``swinViT``), drops every decoder/encoder conv block,
    and attaches a global pooling + MLP head. Input spatial dims must be
    divisible by 32 (the ``patch_size ** 5`` constraint of SwinUNETR, e.g.
    96/128).
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        dropout=0.3,
        feature_size=48,
        use_checkpoint=False,
        pretrained_ckpt="",
        freeze_encoder=False,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )
        for name in (
            "encoder1",
            "encoder2",
            "encoder3",
            "encoder4",
            "encoder10",
            "decoder5",
            "decoder4",
            "decoder3",
            "decoder2",
            "decoder1",
            "out",
        ):
            if hasattr(self.backbone, name):
                delattr(self.backbone, name)

        self.pool = nn.AdaptiveAvgPool3d(1)
        feat = feature_size * 16
        hidden_dim = max(64, feat // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if pretrained_ckpt:
            self._load_pretrained(pretrained_ckpt)
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _load_pretrained(self, path):
        """Load the official MONAI SwinUNETR SSL checkpoint (``model_swinvit.pt``).

        Its keys live under ``module.<name>`` and cover only the Swin encoder;
        they are matched onto ``backbone.swinViT.*``. Conv blocks and the head
        stay randomly initialized.
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()}
        module = self.backbone.swinViT
        current = module.state_dict()
        matched = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        if not matched:
            raise ValueError(f"No Swin encoder weights in checkpoint matched: {path}")
        result = module.load_state_dict(matched, strict=False)
        print(
            f"[swinunetr] loaded {len(matched)} tensors from {path}; "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
        )

    def forward(self, x):
        hidden = self.backbone.swinViT(x, self.backbone.normalize)
        deep = self.pool(hidden[4])
        logits = self.classifier(deep)
        return {"logits": logits}


def build_model(cfg):
    mcfg = cfg["model"]
    architecture = mcfg.get("architecture", "simple3dcnn")
    if architecture == "simple3dcnn":
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
    if architecture == "swinunetr":
        shape = cfg.get("data", {}).get("model_input_shape")
        if shape is None or shape % 32 != 0:
            raise ValueError(
                "architecture 'swinunetr' requires data.model_input_shape divisible by 32 "
                f"(e.g. 96 or 128); got {shape!r}"
            )
        return SwinUNETRClassifier(
            in_channels=mcfg.get("input_channels", 1),
            num_classes=mcfg.get("num_classes", 2),
            dropout=mcfg.get("dropout", 0.3),
            feature_size=mcfg.get("feature_size", 48),
            use_checkpoint=mcfg.get("use_checkpoint", False),
            pretrained_ckpt=mcfg.get("pretrained_ckpt", ""),
            freeze_encoder=mcfg.get("freeze_encoder", False),
        )
    raise ValueError(f"Unknown glaucoma architecture: {architecture}")


def probe_model_shapes():
    """Quick shape sanity: run a random 200³ volume through the net."""
    m = Simple3DCNN().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 1, 96, 96, 96))
    print("features out:", m.features(torch.randn(2, 1, 96, 96, 96)).shape)
    print("logits:", out["logits"].shape)


if __name__ == "__main__":
    probe_model_shapes()
