import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Pyramid3DCNN(nn.Module):
    """Multi-scale (FPN-style) 3D CNN for OCT glaucoma classification.

    Same resolution/optimization fixes as ``Simple3DCNN`` (GroupNorm,
    anisotropic (2,2,1) pooling so the thin RNFL layer survives, residual
    blocks), plus a feature pyramid:

      * every stage produces a level at half the previous H/W, full depth;
      * the coarsest ``fuse_levels`` levels get a 1x1 lateral projection and
        are merged top-down (trilinear upsample + add + 3x3 smooth), so coarse
        globe shape semantics are injected into the finer levels;
      * the classifier sees every level after global pooling (unfused levels
        raw, fused levels at ``fusion_dim``), i.e. both fine RNFL detail and
        global geometry.

    ``fuse_levels=0`` disables the top-down path and leaves a plain
    multi-scale pooled head (ablation).
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        dropout=0.3,
        hidden=(16, 32, 64, 128),
        norm="group",
        norm_groups=8,
        pool_strides=((2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1)),
        pool_type="max",
        residual=True,
        fusion_dim=64,
        fuse_levels=3,
    ):
        super().__init__()
        hidden = tuple(hidden)
        if len(hidden) < 2:
            raise ValueError("Pyramid3DCNN needs at least 2 pyramid levels")
        strides = [tuple(s) for s in pool_strides]
        if len(strides) != len(hidden):
            strides = [(2, 2, 1)] * len(hidden)
        self.fuse_levels = max(0, min(int(fuse_levels), len(hidden)))
        self.fuse_start = len(hidden) - self.fuse_levels

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden[0], kernel_size=3, padding=1),
            _norm3d(hidden[0], norm, norm_groups),
            nn.ReLU(inplace=True),
        )

        stages = []
        cin = hidden[0]
        for i, cout in enumerate(hidden):
            block = []
            if i > 0:
                block += [
                    nn.Conv3d(cin, cout, kernel_size=3, padding=1),
                    _norm3d(cout, norm, norm_groups),
                    nn.ReLU(inplace=True),
                ]
            if residual:
                block.append(SimpleResidual3d(cout, norm, norm_groups))
            stages.append(nn.Sequential(*block))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.pools = nn.ModuleList([_pool3d(pool_type, s) for s in strides])

        self.laterals = nn.ModuleList()
        self.smooths = nn.ModuleList()
        for i in range(self.fuse_start, len(hidden)):
            self.laterals.append(nn.Conv3d(hidden[i], fusion_dim, kernel_size=1))
        for _ in range(max(0, self.fuse_levels - 1)):
            self.smooths.append(
                nn.Sequential(
                    nn.Conv3d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
                    _norm3d(fusion_dim, norm, norm_groups),
                    nn.ReLU(inplace=True),
                )
            )

        head_in = sum(hidden[: self.fuse_start]) + fusion_dim * self.fuse_levels
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for stage, pool in zip(self.stages, self.pools):
            x = pool(stage(x))
            feats.append(x)

        parts = []
        if self.fuse_levels:
            fused = [lat(f) for lat, f in zip(self.laterals, feats[self.fuse_start :])]
            for j in range(len(fused) - 2, -1, -1):
                up = F.interpolate(fused[j + 1], size=fused[j].shape[-3:], mode="trilinear", align_corners=False)
                fused[j] = self.smooths[j](fused[j] + up)
            parts += [self.pool(f) for f in feats[: self.fuse_start]]
            parts += [self.pool(f) for f in fused]
        else:
            parts = [self.pool(f) for f in feats]

        logits = self.classifier(torch.cat([p.flatten(1) for p in parts], dim=1))
        return {"logits": logits}


class _ResNeXtBlock3d(nn.Module):
    """Grouped bottleneck residual block (mirrors ``scripts/final_model.ResXBlock3D``).

    Keeps GroupNorm so batch 2-4 stays stable, and accepts an arbitrary stride so the
    first stage can use (2,2,1) and keep depth resolution.
    """

    def __init__(self, cin, cout, stride=(1, 1, 1), groups=8, base_width=8, norm_groups=8):
        super().__init__()
        width = int(cout * base_width / 64.0) * groups
        self.c1 = nn.Conv3d(cin, width, kernel_size=1, bias=False)
        self.g1 = _norm3d(width, "group", norm_groups)
        self.c2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.g2 = _norm3d(width, "group", norm_groups)
        self.c3 = nn.Conv3d(width, cout, kernel_size=1, bias=False)
        self.downsample = stride != (1, 1, 1) or cin != cout
        if self.downsample:
            self.sc = nn.Conv3d(cin, cout, kernel_size=1, stride=stride, bias=False)
            self.sg = _norm3d(cout, "group", norm_groups)

    def forward(self, x):
        out = self.c3(F.relu(self.g2(self.c2(F.relu(self.g1(self.c1(x)))))))
        if self.downsample:
            x = self.sg(self.sc(x))
        return F.relu(x + out)


class PyramidResNeXt3D(nn.Module):
    """ResNeXt3D backbone (grouped bottlenecks, GroupNorm) + FPN pyramid head.

    Same stages/widths as ``scripts/final_model.Enc3DResNeXt`` (the winning
    ``single3d-resxt3d`` encoder: features (32,64,128,192), strides
    (2,2,1) then (2,2,2) x3), but every stage is exposed and merged top-down:

      * level 0 keeps depth at the input resolution (stride (2,2,1));
      * the last ``fuse_levels`` levels get 1x1 laterals, trilinear top-down
        merge and a 3x3 smooth;
      * the classifier concatenates global-pooled features from every level
        (unfused raw, fused at ``fusion_dim``).

    ``fuse_levels=0`` reduces it to the ResNeXt backbone plus a multi-scale head.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        dropout=0.3,
        features=(32, 64, 128, 192),
        groups=8,
        base_width=8,
        fusion_dim=128,
        fuse_levels=3,
        norm_groups=8,
    ):
        super().__init__()
        features = tuple(features)
        if len(features) < 2:
            raise ValueError("PyramidResNeXt3D needs at least 2 levels")
        self.fuse_levels = max(0, min(int(fuse_levels), len(features)))
        self.fuse_start = len(features) - self.fuse_levels

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            _norm3d(features[0], "group", norm_groups),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        cin = features[0]
        for i, cout in enumerate(features):
            stride = (2, 2, 1) if i == 0 else (2, 2, 2)
            self.stages.append(
                nn.ModuleList(
                    [
                        _ResNeXtBlock3d(cin, cout, stride, groups, base_width, norm_groups),
                        _ResNeXtBlock3d(cout, cout, (1, 1, 1), groups, base_width, norm_groups),
                    ]
                )
            )
            cin = cout

        self.laterals = nn.ModuleList()
        self.smooths = nn.ModuleList()
        for i in range(self.fuse_start, len(features)):
            self.laterals.append(nn.Conv3d(features[i], fusion_dim, kernel_size=1))
        for _ in range(max(0, self.fuse_levels - 1)):
            self.smooths.append(
                nn.Sequential(
                    nn.Conv3d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
                    _norm3d(fusion_dim, "group", norm_groups),
                    nn.ReLU(inplace=True),
                )
            )

        head_in = sum(features[: self.fuse_start]) + fusion_dim * self.fuse_levels
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for downsample, refine in self.stages:
            feats.append(refine(downsample(x)))
            x = feats[-1]

        parts = []
        if self.fuse_levels:
            fused = [lat(f) for lat, f in zip(self.laterals, feats[self.fuse_start :])]
            for j in range(len(fused) - 2, -1, -1):
                up = F.interpolate(fused[j + 1], size=fused[j].shape[-3:], mode="trilinear", align_corners=False)
                fused[j] = self.smooths[j](fused[j] + up)
            parts += [self.pool(f) for f in feats[: self.fuse_start]]
            parts += [self.pool(f) for f in fused]
        else:
            parts = [self.pool(f) for f in feats]

        logits = self.classifier(torch.cat([p.flatten(1) for p in parts], dim=1))
        return {"logits": logits}


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
    if architecture == "pyramid3dcnn":
        return Pyramid3DCNN(
            in_channels=mcfg.get("input_channels", 1),
            num_classes=mcfg.get("num_classes", 2),
            dropout=mcfg.get("dropout", 0.3),
            hidden=tuple(mcfg.get("hidden", (16, 32, 64, 128))),
            norm=mcfg.get("norm", "group"),
            norm_groups=mcfg.get("norm_groups", 8),
            pool_strides=tuple(mcfg.get("pool_strides", ((2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1)))),
            pool_type=mcfg.get("pool_type", "max"),
            residual=mcfg.get("residual", True),
            fusion_dim=mcfg.get("fusion_dim", 64),
            fuse_levels=mcfg.get("fuse_levels", 3),
        )
    if architecture == "pyramid_resnext3d":
        return PyramidResNeXt3D(
            in_channels=mcfg.get("input_channels", 1),
            num_classes=mcfg.get("num_classes", 2),
            dropout=mcfg.get("dropout", 0.3),
            groups=mcfg.get("groups", 8),
            base_width=mcfg.get("base_width", 8),
            fusion_dim=mcfg.get("fusion_dim", 128),
            fuse_levels=mcfg.get("fuse_levels", 3),
            norm_groups=mcfg.get("norm_groups", 8),
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
