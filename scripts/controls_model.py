import torch
import torch.nn as nn

from scripts.final_model import VIEWS, CrossGate, Enc3DResNeXt, Proj, Timm2D


class FixedCrossGate(CrossGate):
    def __init__(self, D, n_2d, heads=None):
        super().__init__(D, n_2d, heads=heads)
        del self.gate

    def forward(self, c3, toks2d):
        q = c3.unsqueeze(1)
        h = self.norm(toks2d)
        out, w = self.attn(q, h, h, need_weights=True)
        self.last_w = w[..., 0, :].detach()
        self.last_gate = 1.0
        return c3 + out.squeeze(1)


class ControlsModel(nn.Module):
    def __init__(self, n_2d=2, D=256, num_classes=2, enc2d="maxvit_tiny_rw_224",
                 enc2d_pretrained=True, enc3d_features=(32, 64, 128, 192),
                 view_indices=None, fusion="crossgate", use_3d=True, gate_fixed=False):
        super().__init__()
        if fusion not in ("crossgate", "concat"):
            raise ValueError("fusion must be 'crossgate' or 'concat'")
        self.fusion_mode = fusion
        self.use_3d = bool(use_3d)
        self.gate_fixed = bool(gate_fixed)
        self.view_indices = tuple(range(n_2d)) if view_indices is None else tuple(int(i) for i in view_indices)
        self.n_2d = len(self.view_indices)
        if self.n_2d < 1 or any(index < 0 or index >= len(VIEWS) for index in self.view_indices):
            raise ValueError("view_indices out of range")
        if self.fusion_mode == "crossgate" and not self.use_3d:
            raise ValueError("CrossGate requires the 3D branch; use fusion='concat' for 2D-only controls")
        self.enc3d = Enc3DResNeXt(features=tuple(enc3d_features)) if self.use_3d else None
        self.enc2ds = nn.ModuleList([Timm2D(enc2d, pretrained=enc2d_pretrained) for _ in range(self.n_2d)])
        dims = ([self.enc3d.out_dim] if self.use_3d else []) + [e.out_dim for e in self.enc2ds]
        self.projs = nn.ModuleList([Proj(d, D) for d in dims])
        if self.fusion_mode == "concat":
            self.fusion = None
            self.head = nn.Linear(D * len(dims), num_classes)
        else:
            gate_cls = FixedCrossGate if self.gate_fixed else CrossGate
            self.fusion = gate_cls(D, self.n_2d)
            self.head = nn.Linear(D, num_classes)

    def embed(self, x3d, views):
        es = [self.enc3d(x3d)] if self.use_3d else []
        for slot, view_index in enumerate(self.view_indices):
            es.append(self.enc2ds[slot](views[:, view_index]))
        return es

    def fuse(self, x3d, views):
        es = self.embed(x3d, views)
        toks = torch.stack([p(e) for p, e in zip(self.projs, es)], dim=1)
        if self.fusion_mode == "concat":
            return toks.flatten(1), toks
        z = self.fusion(toks[:, 0], toks[:, 1:])
        return z, toks

    def forward(self, x3d, views):
        z, _ = self.fuse(x3d, views)
        return self.head(z)

    def gcam3d_module(self):
        if not self.use_3d:
            raise RuntimeError("This variant has no 3D branch")
        return self.enc3d.gcam_module()

    def gcam2d_module(self, i):
        return self.enc2ds[i].gcam_module()
