import numpy as np
import torch
import torch.nn as nn


class _Conv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.down0 = _Conv(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.down1 = _Conv(base, base * 2)
        self.down2 = _Conv(base * 2, base * 4)
        self.bot = _Conv(base * 4, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.conv2 = _Conv(base * 8, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.conv1 = _Conv(base * 4, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.conv0 = _Conv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        s0 = self.down0(x)
        h = self.pool(s0)
        s1 = self.down1(h)
        h = self.pool(s1)
        s2 = self.down2(h)
        h = self.pool(s2)
        h = self.bot(h)
        h = self.conv2(torch.cat([self.up2(h), s2], 1))
        h = self.conv1(torch.cat([self.up1(h), s1], 1))
        h = self.conv0(torch.cat([self.up0(h), s0], 1))
        return self.head(h)


def mask_input(x, n_pix):
    b, c, h, w = x.shape
    xin = x.clone()
    mask = torch.zeros(b, 1, h, w, dtype=torch.bool, device=x.device)
    total = h * w
    for bi in range(b):
        n = min(n_pix, total - 1)
        if n <= 0:
            continue
        flat = torch.randperm(total, device=x.device)[:n]
        ys = flat // w
        xs = flat % w
        oy = torch.randint(3, (n,), device=x.device) - 1
        ox = torch.randint(3, (n,), device=x.device) - 1
        ok = (oy != 0) | (ox != 0)
        while not ok.all():
            bad = ~ok
            nb = int(bad.sum())
            oy[bad] = torch.randint(3, (nb,), device=x.device) - 1
            ox[bad] = torch.randint(3, (nb,), device=x.device) - 1
            ok = (oy != 0) | (ox != 0)
        ny = (ys + oy).clamp(0, h - 1)
        nx = (xs + ox).clamp(0, w - 1)
        xin[bi, 0, ys, xs] = x[bi, 0, ny, nx]
        mask[bi, 0, ys, xs] = True
    return xin, mask


def n2v_loss(pred, target, mask):
    return nn.functional.mse_loss(pred[mask], target[mask])


@torch.no_grad()
def denoise_volume(net, vol, device, batch=16):
    net.eval()
    n = vol.shape[0]
    out = np.empty_like(vol)
    for i in range(0, n, batch):
        b = vol[i : i + batch]
        x = torch.from_numpy(b.astype(np.float32) / 255.0)[:, None].to(device)
        y = net(x).clamp_(0.0, 1.0)[:, 0].cpu().numpy()
        out[i : i + batch] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    net.train()
    return out
