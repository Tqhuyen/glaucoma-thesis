import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resolution_study as rs

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_OK = True
except Exception:
    torch = None
    TORCH_OK = False


def find_dir(root, size):
    d = os.path.join(root, f"glaucoma_all_{size}")
    if os.path.isfile(os.path.join(d, "Training_volumes.npy")):
        return d
    return None


class VolumeDataset(Dataset):
    def __init__(self, vols, labels, n_max=0, seed=0):
        self.vols = vols
        self.labels = np.asarray(labels)
        idx = np.arange(len(self.labels))
        if n_max and n_max < len(idx):
            idx = np.random.default_rng(seed).choice(idx, size=n_max, replace=False)
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        v = np.asarray(self.vols[j])
        if v.ndim == 4:
            v = v[0]
        return torch.from_numpy(v.astype(np.float32) / 255.0)[None], int(self.labels[j])


class Block3D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.dn = None
        if stride != 1 or cin != cout:
            self.dn = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        idt = x if self.dn is None else self.dn(x)
        h = F.relu(self.n1(self.c1(x)), inplace=True)
        h = self.n2(self.c2(h))
        return F.relu(h + idt, inplace=True)


class Proxy3DCNN(nn.Module):
    def __init__(self, width=24, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, width, 3, 2, 1, bias=False), nn.GroupNorm(8, width),
                                  nn.ReLU(inplace=True), nn.MaxPool3d(2))
        self.a = Block3D(width, width * 2, 2)
        self.b = Block3D(width * 2, width * 4, 2)
        self.c = Block3D(width * 4, width * 8, 2)
        self.norm = nn.GroupNorm(8, width * 8)
        self.head = nn.Linear(width * 8, num_classes)

    def embed(self, x):
        h = self.c(self.b(self.a(self.stem(x))))
        return F.relu(self.norm(h), inplace=True).mean((2, 3, 4))

    def forward(self, x):
        return self.head(self.embed(x))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys, fs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        fs.append(model.embed(x).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(fs)


def train_proxy(vols, labels, val_vols, val_labels, bs, epochs, device, seed, lr=1e-3, width=24,
                n_train=0, n_val=0):
    tr = VolumeDataset(vols, labels, n_max=n_train, seed=seed)
    va = VolumeDataset(val_vols, val_labels, n_max=n_val, seed=seed + 1)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0)
    torch.manual_seed(seed)
    model = Proxy3DCNN(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(tl)))
    ce = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = ce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"    epoch {ep+1}/{epochs} ({time.time()-t0:.0f}s)", flush=True)
    probs, ys, feats = predict(model, vl, device)
    met = rs.classification_metrics(probs, ys)
    met["train_min"] = round((time.time() - t0) / 60.0, 2)
    return model, met, feats, ys


def pca_np(X, k=16):
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--sizes", type=int, nargs="+", default=[96, 128])
    ap.add_argument("--split", default="Training")
    ap.add_argument("--val-split", default="Validation")
    ap.add_argument("--train-n", type=int, default=600)
    ap.add_argument("--val-n", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--bs-large", type=int, default=6)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-exp1", type=int, default=40)
    ap.add_argument("--out", default=os.path.join("figures", "resolution"))
    ap.add_argument("--push-to-hf", action="store_true")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-private", type=int, default=1)
    args = ap.parse_args()

    if not TORCH_OK:
        raise RuntimeError("torch not importable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[cfg] device={device} sizes={args.sizes} train_n={args.train_n} val_n={args.val_n} "
          f"epochs={args.epochs} bs={args.bs} width={args.width}", flush=True)

    data = {}
    for s in args.sizes:
        d = find_dir(args.data_root, s)
        if d is None:
            raise RuntimeError(f"missing data/{os.path.basename(d) if d else 'glaucoma_all_'+str(s)}")
        data[s] = {
            "train_v": np.load(os.path.join(d, f"{args.split}_volumes.npy"), mmap_mode="r"),
            "train_l": np.load(os.path.join(d, f"{args.split}_labels.npy")),
            "val_v": np.load(os.path.join(d, f"{args.val_split}_volumes.npy"), mmap_mode="r"),
            "val_l": np.load(os.path.join(d, f"{args.val_split}_labels.npy")),
        }
        print(f"[data] {s}^3 {data[s]['train_v'].shape} val {data[s]['val_v'].shape}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    report = {"sizes": args.sizes, "device": str(device), "train_n": args.train_n,
              "val_n": args.val_n, "epochs": args.epochs}

    if args.n_exp1 > 0 and 96 in data:
        ref = 128 if 128 in data else max(args.sizes)
        rv = data[ref]["val_v"]
        idx = np.random.default_rng(0).choice(len(rv), size=min(args.n_exp1, len(rv)), replace=False)
        ps, ss = [], []
        for j in idx:
            x = np.asarray(rv[j])
            x = (x[0] if x.ndim == 4 else x).astype(np.float32)
            d96 = rs.downsample_volume(x, (96,) * 3, "gaussian_trilinear")
            up = rs.resize_volume(d96, (ref,) * 3)
            ps.append(rs.psnr(x, up))
            ss.append(rs.ssim3d(x, up))
        report["exp1"] = {"ref": ref, "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
        print(f"[exp1] 96 vs {ref}: PSNR {np.mean(ps):.2f} dB SSIM {np.mean(ss):.4f}", flush=True)

    models, metrics, feats = {}, {}, {}
    for s in args.sizes:
        print(f"[exp2] training {s}^3 ...", flush=True)
        m, met, f, yv = train_proxy(data[s]["train_v"], data[s]["train_l"], data[s]["val_v"],
                                    data[s]["val_l"], args.bs, args.epochs, device, args.seed,
                                    width=args.width, n_train=args.train_n, n_val=args.val_n)
        models[s], metrics[s], feats[s] = m, met, f
        print(f"[exp2] {s}^3 AUC={met['auc_roc']:.4f} AP={met['auc_pr']:.4f} F1={met['f1']:.4f} "
              f"bAcc={met['balanced_acc']:.4f} ECE={met['ece']:.4f} ({met['train_min']} min)", flush=True)
    report["exp2"] = metrics
    VAL_Y = yv

    if 96 in feats and 128 in feats:
        cka = rs.linear_cka(feats[128], feats[96])
        sil = {s: rs.silhouette_np(pca_np(feats[s], 16), VAL_Y) for s in (96, 128)}
        drop = (sil[128] - sil[96]) / sil[128] if sil[128] else float("nan")
        report["exp3"] = {"cka": float(cka), "sil_128": float(sil[128]), "sil_96": float(sil[96]),
                          "sil_drop": float(drop)}
        print(f"[exp3] CKA={cka:.4f} sil128={sil[128]:.4f} sil96={sil[96]:.4f} drop={drop*100:.2f}%", flush=True)

    if args.bs_large and 96 in data and 128 in data:
        print(f"[exp4] 128^3 bs={args.bs} vs 96^3 bs={args.bs_large} ...", flush=True)
        _, m128s, _, _ = train_proxy(data[128]["train_v"], data[128]["train_l"], data[128]["val_v"],
                                     data[128]["val_l"], args.bs, args.epochs, device, args.seed,
                                     width=args.width, n_train=args.train_n, n_val=args.val_n)
        _, m96l, _, _ = train_proxy(data[96]["train_v"], data[96]["train_l"], data[96]["val_v"],
                                    data[96]["val_l"], args.bs_large, args.epochs, device, args.seed,
                                    width=args.width, n_train=args.train_n, n_val=args.val_n)
        report["exp4"] = {"128_bs%d" % args.bs: m128s, "96_bs%d" % args.bs_large: m96l}
        print(f"[exp4] AUC 128 bs{args.bs}={m128s['auc_roc']:.4f} | 96 bs{args.bs_large}={m96l['auc_roc']:.4f}", flush=True)

    checks = {}
    if "exp1" in report:
        checks["exp1_ssim>0.85"] = bool(report["exp1"]["ssim"] > 0.85)
        checks["exp1_psnr>30"] = bool(report["exp1"]["psnr"] > 30.0)
    if 96 in metrics and 128 in metrics:
        checks["exp2_auc_drop<1.5%"] = bool((metrics[128]["auc_roc"] - metrics[96]["auc_roc"]) < 0.015)
    if "exp3" in report:
        checks["exp3_cka>0.85"] = bool(report["exp3"]["cka"] > 0.85)
        checks["exp3_sil_drop<10%"] = bool(report["exp3"]["sil_drop"] < 0.10)
    if "exp4" in report:
        k96 = "96_bs%d" % args.bs_large
        k128 = "128_bs%d" % args.bs
        checks["exp4_96largebs>=128smallbs"] = bool(
            report["exp4"][k96]["auc_roc"] >= report["exp4"][k128]["auc_roc"] - 0.005)
    passed = sum(1 for v in checks.values() if v)
    best = 96 if (checks.get("exp4_96largebs>=128smallbs", False) and passed >= max(3, len(checks) - 1)) else 128
    report["checks"] = checks
    report["passed"] = f"{passed}/{len(checks)}"
    report["recommendation"] = f"CHON {best}^3"
    report_path = os.path.join(args.out, "resolution_compare_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("\n===== KET LUAN =====")
    for k, v in checks.items():
        print(("  PASS " if v else "  FAIL ") + k)
    print(f"=> KHUYEN NGHI: {best}^3 ({report['passed']}) | report: {report_path}", flush=True)

    if args.push_to_hf:
        from huggingface_hub import HfApi
        try:
            from pipeline.utils import load_env_file
            load_env_file()
        except Exception:
            pass
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN missing")
        api = HfApi(token=token)
        user = api.whoami()["name"]
        repo = args.hf_repo or f"{user}/harvard-oct-glaucoma-{best}"
        folder = os.path.join(args.data_root, f"glaucoma_all_{best}")
        api.create_repo(repo_id=repo, repo_type="dataset", private=bool(args.hf_private), exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder,
                          commit_message=f"Harvard-GF {best}^3 resized dataset ({report['passed']} checks)")
        print("uploaded:", "https://huggingface.co/datasets/" + repo, flush=True)


if __name__ == "__main__":
    main()
