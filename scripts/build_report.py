"""Build the self-contained HTML experiment report (Vietnamese) from archived results.

Inputs (download with huggingface_hub.snapshot_download, see the report appendix):
  artifacts/vast/training_runs/<raw200 run>/      metrics.jsonl, test_results.json, config_used.json
  artifacts/<kaggle run>-results-v0/               same for bilateral-96 / aniso runs
  artifacts/report_src/glaucoma/post/report-assets  per-sample probabilities, case images, HSI images
  artifacts/report_src/glaucoma/runs-modal          W&B trace of the Modal depth-attention run
  artifacts/report_src/hsi/run/outputs              MP-HSIR pilot evaluation
  artifacts/vast/dl_lab_run/outputs                 Urban100 run

    python scripts/build_report.py --out reports/experiments_report.html
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ART = Path("artifacts")
ASSETS = ART / "report_src/glaucoma/post/report-assets"
HSI_OUT = ART / "report_src/hsi/run/outputs"
STEPS_PER_EPOCH = 262.5

MODELS = {
    "raw200-simple-s42": ("Simple3DCNN", "raw 200³", 8, 42),
    "raw200-pyramid-s42": ("Pyramid3DCNN", "raw 200³", 8, 42),
    "raw200-pyramid-e15-s42": ("Pyramid3DCNN", "raw 200³", 15, 42),
    "raw200-resnext-s42": ("PyramidResNeXt3D", "raw 200³", 8, 42),
    "raw200-resnext-s43": ("PyramidResNeXt3D", "raw 200³", 8, 43),
    "simple3dcnn-bilateral96-s42": ("Simple3DCNN", "bilateral 96³", 15, 42),
    "pyramid-bilateral96-s42": ("Pyramid3DCNN", "bilateral 96³", 15, 42),
    "pyramid-resnext-bilateral96-s42": ("PyramidResNeXt3D", "bilateral 96³", 15, 42),
}
SHORT = {
    "raw200-simple-s42": "Simple·raw·8ep",
    "raw200-pyramid-s42": "Pyramid·raw·8ep",
    "raw200-pyramid-e15-s42": "Pyramid·raw·15ep",
    "raw200-resnext-s42": "ResNeXt·raw·s42",
    "raw200-resnext-s43": "ResNeXt·raw·s43",
    "simple3dcnn-bilateral96-s42": "Simple·bil96",
    "pyramid-bilateral96-s42": "Pyramid·bil96",
    "pyramid-resnext-bilateral96-s42": "ResNeXt·bil96",
}
COLORS = dict(zip(MODELS, plt.cm.tab10.colors))
WHY_VI = {
    "all models correct (glaucoma)": "cả 5 model raw đúng (glaucoma)",
    "all models correct (normal)": "cả 5 model raw đúng (bình thường)",
    "only Pyramid-e15 correct": "chỉ Pyramid raw 15 ep đúng",
    "Pyramid-e15 correct, Simple wrong": "Pyramid raw 15 ep đúng, Simple raw sai",
    "Pyramid-e15 wrong, Simple correct": "Pyramid raw 15 ep sai, Simple raw đúng",
    "all models wrong": "cả 5 model raw sai",
}


def img_tag(path_or_fig, width=None, jpeg=False, alt=""):
    buf = io.BytesIO()
    if isinstance(path_or_fig, (str, Path)):
        im = Image.open(path_or_fig).convert("RGB")
        im.save(buf, "JPEG" if jpeg else "PNG", quality=88)
        mime = "jpeg" if jpeg else "png"
    else:
        path_or_fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(path_or_fig)
        mime = "png"
    w = f' style="width:{width}"' if width else ""
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/{mime};base64,{data}" alt="{html.escape(alt)}"{w}>'


def table(df, fmt=None, highlight=None):
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            s = fmt[c](v) if c in fmt and pd.notna(v) else ("—" if pd.isna(v) else html.escape(str(v)))
            cls = ' class="best"' if highlight and highlight(c, v, df[c]) else ""
            cells.append(f"<td{cls}>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def pct(v):
    return f"{100 * v:.2f}%"


def f3(v):
    return f"{v:.3f}"


def f2(v):
    return f"{v:.2f}"


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def metrics(p, y, rng):
    pred = (p > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec_pos = tp / max(tp + fp, 1)
    prec_neg = tn / max(tn + fn, 1)
    f1_pos = 2 * prec_pos * sens / max(prec_pos + sens, 1e-9)
    f1_neg = 2 * prec_neg * spec / max(prec_neg + spec, 1e-9)
    mcc = (tp * tn - fp * fn) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    idx = rng.integers(0, len(y), (2000, len(y)))
    accs = (pred[idx] == y[idx]).mean(1)
    aucs = np.array([roc_auc_score(y[i], p[i]) for i in idx[:500]])
    return {
        "acc": (pred == y).mean(),
        "acc_lo": np.percentile(accs, 2.5),
        "acc_hi": np.percentile(accs, 97.5),
        "bal": (sens + spec) / 2,
        "sens": sens,
        "spec": spec,
        "f1_pos": f1_pos,
        "f1_macro": (f1_pos + f1_neg) / 2,
        "mcc": mcc,
        "auc": roc_auc_score(y, p),
        "auc_lo": np.percentile(aucs, 2.5),
        "auc_hi": np.percentile(aucs, 97.5),
        "ap": average_precision_score(y, p),
        "brier": float(((p - y) ** 2).mean()),
        "ece": ece(p, y),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def mcnemar(a_right, b_right):
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def load_probs():
    raw = np.load(ASSETS / "glaucoma/test_probs.npz")
    bil = np.load(ASSETS / "glaucoma/test_probs_bilateral96.npz")
    assert (raw["labels"] == bil["labels"]).all()
    probs = {k: raw[k] for k in raw.files if k != "labels"}
    probs.update({k: bil[k] for k in bil.files if k != "labels"})
    return raw["labels"].astype(int), {k: probs[k] for k in MODELS}


def run_dir(name):
    d = ART / "vast/training_runs" / name
    return d if d.exists() else ART / f"{name}-results-v0"


def history(name):
    rows = [json.loads(line) for line in open(run_dir(name) / "metrics.jsonl")]
    val = pd.DataFrame(
        [{"step": r["step"], "loss": r["val/loss"], "acc": r["val/acc"]} for r in rows if "val/acc" in r]
    )
    return val.drop_duplicates("step", keep="last")


def reported(name):
    t = json.loads((run_dir(name) / "test_results.json").read_text())
    return t["test_acc"]


def glaucoma_section(rng):
    y, probs = load_probs()
    rows, m_all = [], {}
    for k, (arch, data, ep, seed) in MODELS.items():
        m = metrics(probs[k], y, rng)
        m_all[k] = m
        rows.append(
            {
                "Model": arch,
                "Dữ liệu": data,
                "Epoch": ep,
                "Seed": seed,
                "Acc": m["acc"],
                "CI 95% acc": f"{100 * m['acc_lo']:.1f}–{100 * m['acc_hi']:.1f}",
                "Bal. acc": m["bal"],
                "Sens.": m["sens"],
                "Spec.": m["spec"],
                "F1 (+)": m["f1_pos"],
                "MCC": m["mcc"],
                "AUC": m["auc"],
                "CI 95% AUC": f"{m['auc_lo']:.3f}–{m['auc_hi']:.3f}",
                "AP": m["ap"],
                "Brier": m["brier"],
                "ECE": m["ece"],
                "Acc báo cáo lúc train": reported(k),
            }
        )
    ens_p = np.mean([probs[k] for k in MODELS], 0)
    ens = metrics(ens_p, y, rng)
    rows.append(
        {
            "Model": "Ensemble trung bình 8 model",
            "Dữ liệu": "raw + bil96",
            "Epoch": None,
            "Seed": None,
            "Acc": ens["acc"],
            "CI 95% acc": f"{100 * ens['acc_lo']:.1f}–{100 * ens['acc_hi']:.1f}",
            "Bal. acc": ens["bal"],
            "Sens.": ens["sens"],
            "Spec.": ens["spec"],
            "F1 (+)": ens["f1_pos"],
            "MCC": ens["mcc"],
            "AUC": ens["auc"],
            "CI 95% AUC": f"{ens['auc_lo']:.3f}–{ens['auc_hi']:.3f}",
            "AP": ens["ap"],
            "Brier": ens["brier"],
            "ECE": ens["ece"],
            "Acc báo cáo lúc train": None,
        }
    )
    main = pd.DataFrame(rows)

    def best(col, v, series):
        if col in ("Acc", "Bal. acc", "MCC", "AUC", "AP", "F1 (+)"):
            return pd.notna(v) and v == series.iloc[:-1].max()
        if col in ("Brier", "ECE"):
            return pd.notna(v) and v == series.iloc[:-1].min()
        return False

    fmt = {c: pct for c in ("Acc", "Bal. acc", "Sens.", "Spec.", "Acc báo cáo lúc train")}
    fmt.update({c: f3 for c in ("F1 (+)", "MCC", "AUC", "AP", "Brier", "ECE")})
    fmt.update({"Epoch": lambda v: str(int(v)), "Seed": lambda v: str(int(v))})
    main_table = table(main, fmt, best)

    thin = json.loads(
        (ART / "report_src/glaucoma/runs-modal/pyramid-thinhead-bilateral96-s42/summary.json").read_text()
    )
    aniso = json.loads((run_dir("pyramid-aniso96x200-s42") / "test_results.json").read_text())
    agg = pd.DataFrame(
        [
            {
                "Model": "ResNeXt3D đơn nhánh (single3d-resxt3d) — SOTA luận văn",
                "Dữ liệu": "raw → 96³",
                "Epoch": 15,
                "Acc": 0.7711,
                "Bal. acc": 0.7719,
                "F1 (+)": 0.7836,
                "MCC": 0.542,
                "AUC": 0.8538,
                "Nguồn": "docs/3d-backbone-sweep.md",
            },
            {
                "Model": "ResNeXt3D + 2×MaxViT + CrossGate (mô hình chính)",
                "Dữ liệu": "raw 200³ + 2D",
                "Epoch": 5,
                "Acc": 0.7667,
                "Bal. acc": 0.7692,
                "F1 (+)": 0.7752,
                "MCC": 0.536,
                "AUC": 0.8476,
                "Nguồn": "docs/final-model-raw-results.md (ngưỡng 0,43 tối ưu trên val)",
            },
            {
                "Model": "Pyramid3DCNN + depth-attention head (Modal)",
                "Dữ liệu": "bilateral 96³",
                "Epoch": 15,
                "Acc": thin.get("test/acc", 0.7467),
                "Bal. acc": 0.7539,
                "F1 (+)": None,
                "MCC": 0.510,
                "AUC": None,
                "Nguồn": "W&B trace (weights mất cùng workspace Modal)",
            },
            {
                "Model": "Pyramid3DCNN anisotropic 96×96×200",
                "Dữ liệu": "bilateral aniso",
                "Epoch": 10,
                "Acc": aniso["test_acc"],
                "Bal. acc": 0.7355,
                "F1 (+)": aniso["test_f1_pos"],
                "MCC": 0.470,
                "AUC": None,
                "Nguồn": "Kaggle run (tiền xử lý không ghi trong config)",
            },
        ]
    )
    agg_table = table(
        agg, {"Acc": pct, "Bal. acc": pct, "F1 (+)": f3, "MCC": f3, "AUC": f3, "Epoch": lambda v: str(int(v))}
    )

    figs = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    order = list(MODELS)
    xs = np.arange(len(order))
    for i, k in enumerate(order):
        m = m_all[k]
        ax[0].bar(i, 100 * m["acc"], color=COLORS[k])
        ax[0].errorbar(
            i,
            100 * m["acc"],
            yerr=[[100 * (m["acc"] - m["acc_lo"])], [100 * (m["acc_hi"] - m["acc"])]],
            color="black",
            capsize=3,
        )
        ax[1].bar(i, m["auc"], color=COLORS[k])
        ax[1].errorbar(i, m["auc"], yerr=[[m["auc"] - m["auc_lo"]], [m["auc_hi"] - m["auc"]]], color="black", capsize=3)
    ax[0].axhline(77.11, ls="--", color="crimson", label="SOTA luận văn 77,11%")
    ax[1].axhline(0.8538, ls="--", color="crimson", label="SOTA luận văn AUC 0,854")
    for a, t in zip(ax, ("Test accuracy (%) ± CI 95%", "ROC-AUC ± CI 95%")):
        a.set_xticks(xs, [SHORT[k] for k in order], rotation=35, ha="right")
        a.set_title(t)
        a.legend(loc="lower right", fontsize=8)
    ax[0].set_ylim(55, 82)
    ax[1].set_ylim(0.6, 0.9)
    figs["bars"] = img_tag(fig, "100%")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for k in order:
        fpr, tpr, _ = roc_curve(y, probs[k])
        ax.plot(fpr, tpr, color=COLORS[k], label=f"{SHORT[k]} (AUC {m_all[k]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=7.5)
    ax.set_title("ROC trên 900 ảnh test")
    figs["roc"] = img_tag(fig, "520px")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for k in order:
        h = history(k)
        ax[0].plot(h.step / STEPS_PER_EPOCH, 100 * h.acc, color=COLORS[k], label=SHORT[k], lw=1.2)
        ax[1].plot(h.step / STEPS_PER_EPOCH, h.loss, color=COLORS[k], label=SHORT[k], lw=1.2)
    ax[0].set_title("Validation accuracy (%) theo epoch")
    ax[1].set_title("Validation loss theo epoch")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=7.5, ncol=2)
    figs["curves"] = img_tag(fig, "100%")

    right = {k: (probs[k] > 0.5) == y for k in order}
    n = len(order)
    dis = np.corrcoef(np.stack([probs[k] for k in order]))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dis, cmap="viridis", vmin=0.5, vmax=1)
    ax.set_xticks(range(n), [SHORT[k] for k in order], rotation=45, ha="right")
    ax.set_yticks(range(n), [SHORT[k] for k in order])
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{dis[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dis[i, j] < 0.75 else "black",
            )
    fig.colorbar(im, ax=ax, label="tương quan Pearson của P(glaucoma)")
    ax.set_title("Hai model ra quyết định giống nhau đến mức nào")
    figs["disagree"] = img_tag(fig, "560px")

    n_right = np.sum([right[k] for k in order], 0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for lab, name, col in ((1, "glaucoma", "#d62728"), (0, "bình thường", "#1f77b4")):
        counts = np.bincount(n_right[y == lab], minlength=n + 1)
        ax.bar(np.arange(n + 1) + (0.2 if lab else -0.2), counts, width=0.4, color=col, label=name)
    ax.set_xlabel("Số model (trên 8) phân loại đúng ca đó")
    ax.set_ylabel("Số ca")
    ax.legend()
    ax.set_title("Độ khó của từng ca test")
    figs["difficulty"] = img_tag(fig, "620px")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    pairs = [("raw200-pyramid-e15-s42", "pyramid-bilateral96-s42"), ("raw200-pyramid-e15-s42", "raw200-resnext-s42")]
    for a, (k1, k2) in zip(ax, pairs):
        a.scatter(probs[k2][y == 0], probs[k1][y == 0], s=7, alpha=0.5, c="#1f77b4", label="bình thường")
        a.scatter(probs[k2][y == 1], probs[k1][y == 1], s=7, alpha=0.5, c="#d62728", label="glaucoma")
        a.axhline(0.5, color="gray", lw=0.8)
        a.axvline(0.5, color="gray", lw=0.8)
        a.set_xlabel(f"P(glaucoma) — {SHORT[k2]}")
        a.set_ylabel(f"P(glaucoma) — {SHORT[k1]}")
        r = np.corrcoef(probs[k1], probs[k2])[0, 1]
        a.set_title(f"r = {r:.2f}")
        a.legend(fontsize=8)
    figs["scatter"] = img_tag(fig, "100%")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for a, k in zip(axes.ravel(), order):
        m = m_all[k]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                )
        a.set_xticks([0, 1], ["dự đoán −", "dự đoán +"])
        a.set_yticks([0, 1], ["thật −", "thật +"])
        a.set_title(SHORT[k], fontsize=9)
    fig.tight_layout()
    figs["cm"] = img_tag(fig, "100%")

    tests = [
        ("raw200-pyramid-e15-s42", "raw200-pyramid-s42", "15 epoch so với 8 epoch (cùng Pyramid, raw)"),
        ("raw200-pyramid-e15-s42", "raw200-simple-s42", "Pyramid so với Simple (raw)"),
        ("raw200-pyramid-s42", "raw200-resnext-s42", "Pyramid so với ResNeXt (raw, 8 ep)"),
        ("raw200-resnext-s42", "raw200-resnext-s43", "Seed 42 so với 43 (ResNeXt raw)"),
        ("pyramid-bilateral96-s42", "raw200-pyramid-e15-s42", "Bilateral 96³ so với raw 200³ (Pyramid, 15 ep)"),
        ("pyramid-resnext-bilateral96-s42", "pyramid-bilateral96-s42", "ResNeXt so với Pyramid (bil96)"),
        ("pyramid-resnext-bilateral96-s42", "simple3dcnn-bilateral96-s42", "ResNeXt so với Simple (bil96)"),
    ]
    mc_rows = []
    for a, b, what in tests:
        bb, cc, p = mcnemar(right[a], right[b])
        mc_rows.append(
            {
                "So sánh": what,
                "A": SHORT[a],
                "B": SHORT[b],
                "Δ acc (A−B)": m_all[a]["acc"] - m_all[b]["acc"],
                "A đúng, B sai": bb,
                "A sai, B đúng": cc,
                "p (McNemar exact)": p,
                "Kết luận": "khác biệt có ý nghĩa (p<0,05)" if p < 0.05 else "chưa đủ bằng chứng",
            }
        )
    mc_table = table(
        pd.DataFrame(mc_rows),
        {"Δ acc (A−B)": lambda v: f"{100 * v:+.2f} điểm", "p (McNemar exact)": lambda v: f"{v:.4f}"},
    )

    cases = json.loads((ASSETS / "glaucoma/cases.json").read_text())
    cards = []
    for c in cases:
        i = c["index"]
        prob_rows = "".join(
            f'<tr><td>{SHORT[k]}</td><td class="{"ok" if (probs[k][i] > 0.5) == c["label"] else "bad"}">'
            f"{probs[k][i]:.2f}</td></tr>"
            for k in order
        )
        sal = "".join(
            f"<figure>{img_tag(ASSETS / f'glaucoma/case{i}_saliency_{r}.png', '190px', True)}"
            f"<figcaption>saliency {SHORT[r]}</figcaption></figure>"
            for r in ("raw200-pyramid-e15-s42", "raw200-simple-s42")
        )
        cards.append(f"""<div class="case"><h4>Ca #{i} — nhãn thật: {"glaucoma" if c["label"] else "bình thường"}
<span class="tag">{html.escape(WHY_VI.get(c["why"], c["why"]))}</span></h4><div class="row">
<figure>{img_tag(ASSETS / f"glaucoma/case{i}_bscan.png", "190px", True)}<figcaption>B-scan giữa khối</figcaption></figure>
<figure>{img_tag(ASSETS / f"glaucoma/case{i}_enface.png", "190px", True)}<figcaption>en-face (trung bình theo độ sâu)</figcaption></figure>
{sal}<table class="mini"><tr><th>Model</th><th>P(glaucoma)</th></tr>{prob_rows}</table></div></div>""")

    stats = {
        "best": max(order, key=lambda k: m_all[k]["acc"]),
        "best_auc": max(order, key=lambda k: m_all[k]["auc"]),
        "m": m_all,
        "ens": ens,
        "all_wrong": int((n_right == 0).sum()),
        "all_right": int((n_right == n).sum()),
    }
    return main_table, agg_table, mc_table, figs, "".join(cards), stats


DENOISE = ART / "case_denoise"
DN_METHODS = ["original", "bilateral", "wavelet", "tv", "nlm", "gaussian", "median"]


def denoise_cases_section():
    mj = json.loads((DENOISE / "metrics.json").read_text())
    y, probs = load_probs()
    raw_strong = ["raw200-pyramid-s42", "raw200-pyramid-e15-s42"]
    bil = ["simple3dcnn-bilateral96-s42", "pyramid-bilateral96-s42", "pyramid-resnext-bilateral96-s42"]
    rows, cards = [], []
    for c in mj["cases"]:
        i, m = c["index"], c["methods"]
        lab = int(y[i])
        cand = {k: v for k, v in m.items() if k != "original"}
        best = max(cand, key=lambda k: cand[k]["cnr_bg"])
        raw_ok = sum((probs[k][i] > 0.5) == lab for k in raw_strong)
        bil_ok = sum((probs[k][i] > 0.5) == lab for k in bil)
        rows.append(
            {
                "Ca": f"#{i}",
                "Nhãn": "glaucoma" if lab else "bình thường",
                "CNR_bg gốc": m["original"]["cnr_bg"],
                "CNR_bg Bilateral": m["bilateral"]["cnr_bg"],
                "CNR_bg Wavelet": m["wavelet"]["cnr_bg"],
                "CNR_bg TV": m["tv"]["cnr_bg"],
                "CNR_bg NLM": m["nlm"]["cnr_bg"],
                "β Bilateral": m["bilateral"]["beta"],
                "β Wavelet": m["wavelet"]["beta"],
                "σ nền gốc → Bilateral": f"{m['original']['bg_sigma']:.1f} → {m['bilateral']['bg_sigma']:.1f}",
                "CNR_bg cao nhất": best,
                "Model raw (Pyramid) đúng": f"{raw_ok}/2",
                "Model bilateral 96³ đúng": f"{bil_ok}/3",
            }
        )
        cells = "".join(
            f"<figure>{img_tag(DENOISE / f'case{i}_{k}.png', '150px', True)}<figcaption>{k}"
            + (f" · CNR_bg {m[k]['cnr_bg']:.2f} · β {m[k]['beta']:.2f}" if k != "original" else "")
            + "</figcaption></figure>"
            for k in DN_METHODS
        )
        cards.append(
            f'<div class="case"><h4>Ca #{i} — {"glaucoma" if lab else "bình thường"} · raw đúng {raw_ok}/2 · '
            f'bilateral đúng {bil_ok}/3</h4><div class="row">{cells}</div></div>'
        )
    df = pd.DataFrame(rows)
    fmt = {c: f3 for c in df.columns if c.startswith(("CNR_bg ", "β "))}
    fmt["CNR_bg cao nhất"] = str
    tab = table(df, fmt)
    r = (probs["raw200-pyramid-e15-s42"] > 0.5) == y
    b = (probs["pyramid-bilateral96-s42"] > 0.5) == y
    stats = {
        "raw_only": int((r & ~b).sum()),
        "bil_only": int((~r & b).sum()),
        "spec_raw": r[y == 0].mean(),
        "spec_bil": b[y == 0].mean(),
        "sens_raw": r[y == 1].mean(),
        "sens_bil": b[y == 1].mean(),
    }
    return tab, "".join(cards), stats, mj["params"]


HSI_COND_VI = {
    "gauss30": "Gaussian σ30",
    "gauss50": "Gaussian σ50",
    "gauss70": "Gaussian σ70",
    "case1_noniid": "Case 1 non-iid",
    "case2_stripe": "Case 2 +stripe",
    "case3_deadline": "Case 3 +deadline",
    "case4_impulse": "Case 4 impulse",
    "case4_mixture": "Case 4 hỗn hợp",
    "physics": "Nhiễu vật lý (D1)",
    "physics_heavy": "Nhiễu vật lý ×2",
}


def hsi_section():
    t = pd.read_csv(HSI_OUT / "eval.csv")
    cond = t.groupby(["condition", "checkpoint"]).psnr.mean().unstack()[["official", "control", "physics"]]
    cond_ex = (
        t[t.set != "Xiongan"]
        .groupby(["condition", "checkpoint"])
        .psnr.mean()
        .unstack()[["official", "control", "physics"]]
    )
    cond = cond.loc[list(HSI_COND_VI)]
    cond_ex = cond_ex.loc[list(HSI_COND_VI)]
    piv = t.pivot_table(index=["set", "image", "condition"], columns="checkpoint", values="psnr")
    win = (piv.physics > piv.control).groupby(level="condition").mean().loc[list(HSI_COND_VI)]
    rows = []
    for c in HSI_COND_VI:
        rows.append(
            {
                "Điều kiện": HSI_COND_VI[c],
                "Official": cond.loc[c, "official"],
                "Control": cond.loc[c, "control"],
                "D1 physics": cond.loc[c, "physics"],
                "D1 − control": cond.loc[c, "physics"] - cond.loc[c, "control"],
                "D1 thắng control (% ảnh)": win.loc[c],
                "Official (bỏ Xiong'an)": cond_ex.loc[c, "official"],
                "Control (bỏ X)": cond_ex.loc[c, "control"],
                "D1 (bỏ X)": cond_ex.loc[c, "physics"],
            }
        )
    df = pd.DataFrame(rows)
    fmt = {c: f2 for c in df.columns if c not in ("Điều kiện", "D1 thắng control (% ảnh)", "D1 − control")}
    fmt["D1 − control"] = lambda v: f"{v:+.2f}"
    fmt["D1 thắng control (% ảnh)"] = lambda v: f"{100 * v:.0f}%"
    cond_table = table(df, fmt)

    by_set = t.groupby(["set", "checkpoint"])[["psnr", "sam"]].mean().unstack()
    srows = []
    for s in by_set.index:
        srows.append(
            {
                "Bộ test": s,
                "Trong train?": "không (held-out scene)" if s == "Xiongan" else "cùng scene, khác vùng",
                "PSNR official": by_set.loc[s, ("psnr", "official")],
                "PSNR control": by_set.loc[s, ("psnr", "control")],
                "PSNR D1": by_set.loc[s, ("psnr", "physics")],
                "SAM official": by_set.loc[s, ("sam", "official")],
                "SAM control": by_set.loc[s, ("sam", "control")],
                "SAM D1": by_set.loc[s, ("sam", "physics")],
            }
        )
    sdf = pd.DataFrame(srows)
    set_table = table(sdf, {c: f2 for c in sdf.columns[2:]})

    gate = json.loads((HSI_OUT / "stage0_gate.json").read_text())
    gate_table = table(
        pd.DataFrame(gate).rename(
            columns={
                "set": "Bộ",
                "condition": "Điều kiện",
                "paper": "Paper (dB)",
                "ours": "Đo lại (dB)",
                "delta": "Lệch",
            }
        ),
        {"Lệch": lambda v: f"{v:+.2f}"},
    )

    real = pd.DataFrame([json.loads(line) for line in open(HSI_OUT / "real_noref.jsonl")])
    real = real.drop_duplicates(["checkpoint", "scene"], keep="last")
    rp = real.pivot_table(index="checkpoint", values=["stripe_index", "noise_sigma", "sam_to_input"], aggfunc="mean")
    rp = rp.loc[["raw_input", "official", "control", "physics"]].reset_index()
    rp.columns = [
        "Checkpoint",
        "Noise σ ước lượng (thấp = mượt hơn)",
        "SAM so với input (°)",
        "Stripe index (thấp = ít sọc)",
    ]
    real_table = table(rp, {c: (lambda v: f"{v:.4f}") for c in rp.columns[1:]})

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(HSI_COND_VI))
    for i, (tag, col) in enumerate((("official", "#7f7f7f"), ("control", "#1f77b4"), ("physics", "#d62728"))):
        ax[0].bar(x + (i - 1) * 0.27, cond[tag], 0.27, color=col, label=tag)
        ax[1].bar(x + (i - 1) * 0.27, cond_ex[tag], 0.27, color=col, label=tag)
    for a, ttl in zip(ax, ("PSNR trung bình 6 bộ test", "PSNR trung bình, bỏ Xiong'an (scene chưa thấy)")):
        a.set_xticks(x, [HSI_COND_VI[c] for c in HSI_COND_VI], rotation=35, ha="right")
        a.set_ylim(28, 43)
        a.set_title(ttl)
        a.legend(fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig_cond = img_tag(fig, "100%")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for arm, col in (("control", "#1f77b4"), ("physics", "#d62728")):
        log = pd.read_json(HSI_OUT / f"train_{arm}_log.jsonl", lines=True)
        ax.plot(log.minutes, log.loss.rolling(20, min_periods=1).mean(), color=col, label=arm)
    ax.set_xlabel("phút")
    ax.set_ylabel("L1 loss (trung bình trượt)")
    ax.set_title("Loss khi fine-tune (cùng thời lượng 292 phút)")
    ax.legend()
    fig_loss = img_tag(fig, "560px")

    records = json.loads((ASSETS / "hsi/records.json").read_text())
    gallery = []
    for set_name in ("Xiongan", "WDC", "Chikusei", "PaviaU", "Houston"):
        for condition in ("case4_mixture", "physics_heavy", "gauss50", "case2_stripe", "physics"):
            rec = next((r for r in records if r["set"] == set_name and r["condition"] == condition), None)
            if rec is None:
                continue
            cells = [
                f"<figure>{img_tag(ASSETS / f'hsi/{set_name}_clean.png', '150px', True)}<figcaption>sạch</figcaption></figure>",
                f"<figure>{img_tag(ASSETS / f'hsi/{set_name}_{condition}_noisy.png', '150px', True)}"
                f"<figcaption>nhiễu ({rec['psnr_noisy']:.1f} dB)</figcaption></figure>",
            ]
            for tag in ("official", "control", "physics"):
                cells.append(
                    f"<figure>{img_tag(ASSETS / f'hsi/{set_name}_{condition}_{tag}.png', '150px', True)}"
                    f"<figcaption>{tag}: {rec[tag]['psnr']:.2f} dB · SAM {rec[tag]['sam']:.1f}°</figcaption></figure>"
                )
            errs = "".join(
                f"<figure>{img_tag(ASSETS / f'hsi/{set_name}_{condition}_{tag}_err.png', '150px', True)}"
                f"<figcaption>|sai số| {tag}</figcaption></figure>"
                for tag in ("official", "control", "physics")
            )
            gallery.append(
                f'<div class="case"><h4>{set_name} — {HSI_COND_VI[condition]}</h4>'
                f'<div class="row">{"".join(cells)}</div><div class="row">{errs}</div></div>'
            )
    real_gal = []
    for scene in ("Dioni", "Loukia", "Erato"):
        cells = "".join(
            f"<figure>{img_tag(ASSETS / f'hsi/real_{scene}_{tag}.png', '170px', True)}<figcaption>{tag} (RGB)</figcaption></figure>"
            for tag in ("input", "official", "control", "physics")
        )
        bands = "".join(
            f"<figure>{img_tag(ASSETS / f'hsi/real_{scene}_{tag}_band412.png', '170px', True)}<figcaption>{tag} band ~412 nm</figcaption></figure>"
            for tag in ("input", "official", "control", "physics")
        )
        real_gal.append(
            f'<div class="case"><h4>HyRank {scene} (Hyperion thật, không có ảnh sạch)</h4>'
            f'<div class="row">{cells}</div><div class="row">{bands}</div></div>'
        )
    return (
        cond_table,
        set_table,
        gate_table,
        real_table,
        fig_cond,
        fig_loss,
        "".join(gallery),
        "".join(real_gal),
        cond,
        cond_ex,
    )


def urban_section():
    out = ART / "vast/dl_lab_run/outputs"
    s = pd.read_csv(out / "scores.csv")
    summ = json.loads((out / "summary.json").read_text())["tests"][0]
    means = s.groupby(["adapted", "flip"]).psnr.mean()
    flip = min(s.flip.unique(), key=lambda f: abs(means.loc[(False, f)] - summ["unadapted"]))
    per = s[s.flip == flip].pivot_table(index="image", columns="adapted", values="psnr")
    gain = per[True] - per[False]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(gain, bins=30, color="#2ca02c")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("PSNR sau thích nghi − trước (dB), từng ảnh")
    ax.set_ylabel("số ảnh")
    ax.set_title(f"Urban100 σ25: {len(gain)} ảnh, trung bình +{gain.mean():.3f} dB")
    return summ, img_tag(fig, "560px"), gain


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1d2330;background:#f6f7fb;line-height:1.55}
header{background:linear-gradient(120deg,#1f3a68,#3c6db0);color:#fff;padding:28px 40px}
header h1{margin:0 0 6px 0;font-size:26px} header p{margin:2px 0;opacity:.9}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #dde2ec;padding:8px 40px;z-index:5;font-size:14px}
nav a{margin-right:18px;color:#1f3a68;text-decoration:none;font-weight:600}
main{max-width:1250px;margin:0 auto;padding:10px 32px 60px}
section{scroll-margin-top:48px;background:#fff;border:1px solid #e3e7ef;border-radius:10px;padding:18px 26px;margin:22px 0}
h2{color:#1f3a68;border-bottom:2px solid #e3e7ef;padding-bottom:6px} h3{color:#2b4f86;margin-top:26px}
.tw{overflow-x:auto} table{border-collapse:collapse;font-size:13px;margin:10px 0;width:100%}
th,td{border:1px solid #dfe3ea;padding:5px 8px;text-align:center} th{background:#eef2f9}
td:first-child,th:first-child{text-align:left} td.best{background:#dff3e3;font-weight:700}
.kpi{display:flex;gap:14px;flex-wrap:wrap}.kpi div{flex:1;min-width:210px;background:#f2f6fd;border:1px solid #d9e3f3;border-radius:8px;padding:10px 14px}
.kpi b{display:block;font-size:22px;color:#1f3a68}
.case{border:1px solid #e3e7ef;border-radius:8px;padding:8px 12px;margin:12px 0;background:#fbfcfe}
.case h4{margin:4px 0 8px 0}.row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
figure{margin:0;text-align:center;font-size:11.5px;color:#4a5568} img{border-radius:4px;max-width:100%}
.tag{background:#fff3cd;border:1px solid #f0d78c;border-radius:10px;padding:1px 8px;font-size:12px;margin-left:8px;font-weight:500}
table.mini{width:auto;font-size:12px} td.ok{background:#dff3e3} td.bad{background:#fde2e1}
.note{background:#fff8e6;border-left:4px solid #f0b429;padding:8px 12px;margin:10px 0;font-size:14px}
.good{background:#eaf7ee;border-left:4px solid #2f9e44;padding:8px 12px;margin:10px 0;font-size:14px}
.bad2{background:#fdecec;border-left:4px solid #d9480f;padding:8px 12px;margin:10px 0;font-size:14px}
code{background:#eef1f6;padding:1px 5px;border-radius:4px;font-size:12.5px}
"""


def build(out):
    rng = np.random.default_rng(0)
    main_table, agg_table, mc_table, gfig, cases_html, gs = glaucoma_section(rng)
    cond_table, set_table, gate_table, real_table, fig_cond, fig_loss, gallery, real_gal, cond, cond_ex = hsi_section()
    usumm, ufig, ugain = urban_section()
    dn_table, dn_cards, dn, dn_params = denoise_cases_section()
    m = gs["m"]
    best = gs["best"]
    e15, p8, simple = m["raw200-pyramid-e15-s42"], m["raw200-pyramid-s42"], m["raw200-simple-s42"]
    bil_rx = m["pyramid-resnext-bilateral96-s42"]
    ens = gs["ens"]
    d1c = cond.loc["physics", "physics"] - cond.loc["physics", "control"]
    d1h = cond.loc["physics_heavy", "physics"] - cond.loc["physics_heavy", "control"]

    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Báo cáo thí nghiệm — Glaucoma 3D CNN, MP-HSIR, Urban100</title>
<style>{CSS}</style></head><body>
<header><h1>Báo cáo thí nghiệm: Glaucoma 3D CNN · MP-HSIR khử nhiễu ảnh siêu phổ · Urban100</h1>
<p>Đợt chạy 07–11/11/2026 · Kaggle (T4), Modal, vast.ai (RTX 3090) · Toàn bộ model, checkpoint, log lưu trên Hugging Face (private)</p>
<p>Tập test glaucoma: Harvard-GF, 900 khối OCT (489 glaucoma / 411 bình thường), cùng thứ tự mẫu cho mọi model</p></header>
<nav><a href="#tomtat">Tóm tắt</a><a href="#glaucoma">Glaucoma</a><a href="#khacbiet">Model khác nhau ở đâu</a><a href="#ca">Ca minh hoạ</a><a href="#khunhieu">Khử nhiễu cho các ca</a>
<a href="#hsi">MP-HSIR</a><a href="#urban">Urban100</a><a href="#hatang">Chi phí & hạ tầng</a><a href="#gioihan">Giới hạn</a><a href="#phuluc">Phụ lục</a></nav>
<main>
<section id="tomtat"><h2>1. Tóm tắt kết quả</h2>
<div class="kpi">
<div>Model 3D tốt nhất đợt này (acc)<b>{SHORT[best]} — {pct(m[best]["acc"])}</b>AUC {m[best]["auc"]:.3f} · MCC {m[best]["mcc"]:.3f}</div>
<div>SOTA hiện tại của luận văn<b>ResNeXt3D đơn nhánh — 77,11%</b>AUC 0,854 · MCC 0,542 (chưa bị vượt)</div>
<div>Pyramid raw 200³: 15 so với 8 epoch<b>{pct(e15["acc"])} vs {pct(p8["acc"])}</b>AUC {e15["auc"]:.3f} vs {p8["auc"]:.3f}</div>
<div>MP-HSIR: D1 vật lý − control<b>+{d1c:.2f} dB · +{d1h:.2f} dB</b>nhiễu vật lý / nhiễu vật lý ×2</div>
<div>Urban100 σ25 thích nghi<b>+{usumm["gain"]:.3f} dB</b>{usumm["n"]} ảnh, không ảnh nào tụt &gt;0,05 dB</div>
</div>
<ul>
<li><b>Glaucoma:</b> không model nào đợt này vượt SOTA 77,11%. Model mạnh nhất về accuracy là <b>{SHORT[best]}</b>; trên AUC (không phụ thuộc ngưỡng) mạnh nhất là <b>{SHORT[gs["best_auc"]]}</b> ({m[gs["best_auc"]]["auc"]:.3f}).
Ensemble trung bình 8 model đạt {pct(ens["acc"])} / AUC {ens["auc"]:.3f} — tham khảo, không dùng để chọn model.</li>
<li><b>Kiến trúc:</b> với raw 200³, Pyramid3DCNN tốt hơn rõ Simple3DCNN và PyramidResNeXt3D (AUC 0,815 so với 0,720 / 0,710); với bilateral 96³ cả ba sát nhau (AUC 0,806–0,827), ResNeXt nhỉnh nhất. Các model tốt hội tụ về gần cùng một hàm quyết định (r 0,90–0,97) dù khác dữ liệu đầu vào (mục 3.2).</li>
<li><b>Train lâu hơn giúp:</b> Pyramid raw 15 epoch tăng {100 * (e15["acc"] - p8["acc"]):+.2f} điểm acc so với 8 epoch (McNemar ở mục 3).</li>
<li><b>MP-HSIR:</b> Stage 0 tái lập đúng paper (Gaussian ±0,02 dB, Case 1–3 ±0,17 dB). Fine-tune với nhiễu vật lý (D1) thắng control cùng ngân sách ở mọi điều kiện khi tính trung bình 6 bộ test (bỏ Xiong'an thì ngang nhau ở nhiễu chuẩn, hơn rõ ở nhiễu vệ tinh); nhưng <b>cả hai bản fine-tune đều sụp trên scene chưa thấy (Xiong'an)</b> do lệch phổ/cường độ — phát hiện quan trọng nhất của pilot.</li>
</ul></section>

<section id="glaucoma"><h2>2. Glaucoma 3D CNN — so sánh toàn bộ model</h2>
<h3>2.1 Các kiến trúc</h3>
<table><tr><th>Model</th><th>Ý tưởng</th><th>Tham số</th></tr>
<tr><td>Simple3DCNN</td><td>4 khối conv 3D + GroupNorm + residual, pool (2,2,1) giữ trục độ sâu B-scan, GAP cuối</td><td>0,88M</td></tr>
<tr><td>Pyramid3DCNN</td><td>Backbone như trên + FPN 3D: lateral 1×1×1 từ 3 tầng sâu, top-down cộng dồn, GAP từng tầng rồi ghép; giữ chi tiết lớp RNFL mỏng ở tầng nông</td><td>1,13M</td></tr>
<tr><td>PyramidResNeXt3D</td><td>Khối ResNeXt (grouped conv, cardinality 8) + cùng FPN, fusion 128</td><td>1,61M</td></tr>
<tr><td>+ depth-attention head</td><td>Thay GAP bằng attention theo trục độ sâu (chạy trên Modal)</td><td>—</td></tr></table>
<div class="note">Hai dữ liệu đầu vào: <b>raw 200³</b> (khối OCT gốc) và <b>bilateral 96³</b> (khử nhiễu bilateral rồi resize 96³). Recipe chung: AdamW lr 1e-4, wd 0,01, warmup 5% + cosine, AMP, chọn checkpoint theo val loss, ngưỡng 0,5. Raw: batch 1×8 accum; bil96: 4×2. Recipe này <b>khác</b> recipe của SOTA (lr 4e-4, batch 16, chọn theo val acc) — xem mục 7.</div>
<h3>2.2 Kết quả trên 900 ảnh test (tính lại từ dự đoán từng mẫu)</h3>
{main_table}
<p style="font-size:13px">Ô xanh = tốt nhất cột (không tính ensemble). CI 95% bằng bootstrap 2000 lần (AUC 500 lần). Cột cuối là accuracy log lúc train (AMP); chênh vài mẫu với bản tính lại fp32 ở 2 run bil96.</p>
<h3>2.3 Các mốc chỉ có số tổng hợp</h3>{agg_table}
<h3>2.4 Biểu đồ</h3>{gfig["bars"]}<div class="row">{gfig["roc"]}{gfig["difficulty"]}</div>
<h3>2.5 Quá trình huấn luyện</h3>{gfig["curves"]}
<p>Val loss thấp nhất: ResNeXt bil96 (0,505, epoch 6,5). Trên raw 200³: Pyramid 15 epoch (0,519) tiếp tục giảm tới epoch 11 rồi dừng sớm theo patience; Pyramid 8 epoch đạt đáy đúng epoch cuối → run 8 epoch chưa bão hoà. Simple và ResNeXt raw chững ở 0,60–0,62 từ epoch ~7.</p>
<h3>2.6 Ma trận nhầm lẫn</h3>{gfig["cm"]}
</section>

<section id="khacbiet"><h2>3. Model khác nhau ở đâu, tốt hơn nhau ở đâu</h2>
<h3>3.1 Kiểm định từng cặp trên cùng 900 ca (McNemar exact)</h3>{mc_table}
<h3>3.2 Hai cụm hành vi</h3><div class="row">{gfig["disagree"]}<div style="max-width:560px">
<p>Tương quan xác suất dự đoán chia 8 model thành <b>hai cụm</b>:</p>
<ul><li><b>Cụm mạnh</b> (acc 72–75%, AUC 0,79–0,83): Pyramid raw 8/15 ep và cả 3 model bilateral 96³ — tương quan 0,90–0,97 với nhau dù <b>khác dữ liệu đầu vào</b>. Pyramid raw 15 ep và Pyramid bil96 chỉ bất đồng đúng/sai ở 8,2% ca.</li>
<li><b>Cụm yếu</b> (acc 64–67%, AUC 0,71–0,72): Simple raw và ResNeXt raw (2 seed tương quan 0,98 → ổn định giữa seed, nhưng ổn định ở một nghiệm kém). Tương quan với cụm mạnh chỉ 0,59–0,71.</li></ul>
<p>Hệ quả: model tốt hội tụ về gần cùng một hàm quyết định → ghép các model trong cụm mạnh khó tăng nhiều (ensemble 8 model ở bảng 2.2). Trên raw 200³, chỉ kiến trúc giữ được trục độ sâu tới tầng FPN (Pyramid) mới vào được cụm mạnh.</p>
<p>Có <b>{gs["all_right"]}</b> ca cả 8 model đều đúng và <b>{gs["all_wrong"]}</b> ca cả 8 đều sai (ca "khó tuyệt đối" — đáng kiểm tra nhãn hoặc chất lượng ảnh).</p>
<p><b>Điểm mạnh từng model:</b> Simple3DCNN raw có sensitivity {pct(simple["sens"])} nhưng specificity {pct(simple["spec"])}; Pyramid raw 15 ep cân bằng hơn (sens {pct(e15["sens"])} / spec {pct(e15["spec"])}); ResNeXt bil96 có AUC {bil_rx["auc"]:.3f}. Xem cột Sens./Spec. ở bảng 2.2.</p></div></div>
<h3>3.3 Xác suất dự đoán của hai model trên cùng ca</h3>{gfig["scatter"]}
<p>Trái: Pyramid raw 200³ (15 ep) và Pyramid bilateral 96³ — gần như trên đường chéo: cùng đúng, cùng sai trên cùng ca. Phải: Pyramid raw 15 ep và ResNeXt raw — cùng dữ liệu nhưng phân tán mạnh; ResNeXt raw dồn xác suất về vùng 0,4–0,7 (ít tự tin, specificity chỉ 52,8%).</p>
</section>

<section id="ca"><h2>4. Ca minh hoạ: ảnh đầu vào, saliency và dự đoán của từng model</h2>
<p>Saliency = |∂ logit glaucoma / ∂ input| lấy trung bình theo độ sâu (vùng sáng = voxel ảnh hưởng nhiều nhất), thang màu chuẩn hoá riêng từng ảnh. Ô xanh/đỏ = model đúng/sai ở ngưỡng 0,5. Các ca được chọn ngẫu nhiên (seed 0) trong từng nhóm, nhóm xác định theo <b>5 model raw 200³</b>; bảng xác suất hiện thêm 3 model bil96 để đối chiếu.</p>
{cases_html}</section>

<section id="khunhieu"><h2>4b. Nên dùng kiểu khử nhiễu nào cho các ca minh hoạ?</h2>
<div class="note">Cùng quy trình khảo sát khử nhiễu của luận văn (<code>docs/denoise-top3-thesis-review.md</code>): 6 bộ lọc 2D theo từng B-scan,
tham số giữ nguyên (Bilateral σ<sub>color</sub>={dn_params["bilateral"]["sigma_color"]}, σ<sub>spatial</sub>={dn_params["bilateral"]["sigma_spatial"]};
Wavelet db4 BayesShrink soft; TV λ={dn_params["tv"]["weight"]}; NLM h={dn_params["nlm"]["h"]}), chỉ số không tham chiếu tính trên toàn khối:
CNR_bg = (μ<sub>tín hiệu</sub>−μ<sub>nền</sub>)/σ<sub>nền</sub> (tương phản mô/nền), σ nền (mức speckle nền), β = tương quan gradient với ảnh gốc ở 15% vị trí biên mạnh nhất (mức giữ biên).
Khối OCT lấy trực tiếp từ <code>tqhuyen/harvard-oct-glaucoma-200</code>. Script: <code>scripts/case_denoise_review.py</code>.</div>
{dn_table}
<h3>Kết luận cho 10 ca</h3><ul>
<li><b>Chỉ Bilateral và Wavelet cải thiện ảnh</b>: CNR_bg tăng ở cả 10/10 ca (Bilateral +6,6% đến +19,6%, Wavelet +7,8% đến +14,6%).
TV, Gaussian, Median <b>giảm</b> CNR_bg dưới ảnh gốc ở cả 10 ca; NLM giảm ở 9/10 ca (ngoại lệ #813, chỉ +0,04). β của 4 bộ lọc này chỉ 0,44–0,64 — ảnh mờ ranh giới lớp võng mạc (xem hình). <b>Không dùng 4 bộ lọc này</b> cho các ca này.</li>
<li><b>Bilateral</b> thắng CNR_bg ở cả 4 ca glaucoma và giảm σ nền mạnh nhất (−27,5% so với −17,0% của Wavelet), nhưng làm phẳng kết cấu bên trong lớp mô.
<b>Wavelet</b> thắng CNR_bg ở 5/6 ca bình thường và giữ biên tốt nhất (β 0,84–0,92, so với 0,83–0,88 của Bilateral), đổi lại còn nhiều speckle.</li>
<li><b>Bằng chứng phân loại</b> (chỉ có cho Bilateral, vì chưa train model trên Wavelet): trên toàn tập test, pipeline Bilateral tăng specificity
({pct(dn["spec_raw"])} → {pct(dn["spec_bil"])}, Pyramid raw 15 ep → Pyramid bil96) nhưng giảm sensitivity ({pct(dn["sens_raw"])} → {pct(dn["sens_bil"])}).
Đúng như vậy trên các ca: ca bình thường #813 chỉ model Bilateral nhận đúng; hai ca glaucoma khó #371 và #774 thì cả 3 model Bilateral bỏ sót.
Hai pipeline sai ở những ca khác nhau: {dn["raw_only"]} ca raw đúng mà Bilateral sai, {dn["bil_only"]} ca ngược lại.</li></ul>
<div class="good"><b>Khuyến nghị:</b> (1) <b>Bilateral</b> là lựa chọn chính hiện có bằng chứng (CNR_bg và giảm nhiễu nền tốt nhất; model bil96 có AUC cao nhất 0,827) — hợp khi ưu tiên giảm dương tính giả.
(2) Với mục tiêu <b>không bỏ sót glaucoma sớm</b> (RNFL mỏng như #371, #774), nên giữ thêm nhánh <b>raw</b> hoặc thử <b>Wavelet</b> (giữ biên tốt nhất, can thiệp nhẹ) — đây là giả thuyết, cần thí nghiệm phân loại có kiểm soát với Wavelet trước khi kết luận.
(3) Loại TV, NLM, Gaussian, Median.</div>
<div class="note">Giới hạn: 10 ca không phải mẫu đại diện (chọn theo hành vi model); CNR_bg/β là chỉ số gián tiếp, mask Otsu không phải phân đoạn RNFL; ảnh hiển thị là vùng crop B-scan giữa khối, cùng thang xám với ảnh gốc.</div>
{dn_cards}</section>

<section id="hsi"><h2>5. MP-HSIR — khử nhiễu ảnh siêu phổ vệ tinh (pilot D1)</h2>
<div class="note">Thiết kế: checkpoint chính thức <code>Remote_sensing.ckpt</code> (ICCV 2025) → fine-tune <b>292 phút</b> mỗi nhánh trên RTX 3090, batch 16, lr 5e-5, cùng seed và lịch LR.
<b>Control</b> = nhiễu chuẩn MP-HSIR; <b>D1</b> = 60% nhiễu vật lý pushbroom (Poisson-Gaussian theo tín hiệu, sọc gain/offset theo cột detector, hồ sơ SNR theo bước sóng, pixel chết/bão hoà) + 40% nhiễu chuẩn.
Dữ liệu train: 2832 patch từ WDC, PaviaC, PaviaU, Houston, Chikusei (vùng ngoài crop test; kiểm chứng tương quan crop = 1,0000). Xiong'an không có trong train.</div>
<h3>5.1 Stage 0 — tái lập paper</h3>{gate_table}
<p>Gaussian và Case 1–3 khớp paper. Case 4 lệch +1,4 dB dù dùng impulse đơn hay hỗn hợp cộng dồn → định nghĩa Case 4 của paper chưa tái lập được từ code công bố.</p>
<h3>5.2 PSNR theo điều kiện nhiễu</h3>{cond_table}{fig_cond}
<h3>5.3 Theo bộ test — phát hiện quan trọng</h3>{set_table}
<div class="bad2"><b>Xiong'an (scene không có trong train) sụp mạnh sau fine-tune.</b> Ảnh output (mục 5.5) cho thấy control làm ảnh tối và lệch màu: mạng đã học thống kê phổ/cường độ của 5 scene train và "kéo" scene mới về đó. Official 37,71 dB → control 28,82 dB (−8,89) và D1 30,84 dB (−6,87): D1 sụp ít hơn 2 dB, nhưng vẫn kém official → fine-tune trên ít scene làm mất khả năng tổng quát.</div>
<div class="good">Trên các scene đã thấy (khác vùng), D1 ngang control ở nhiễu chuẩn (chênh trong khoảng ±0,2 dB) và vượt xa ở nhiễu vật lý (+{cond_ex.loc["physics", "physics"] - cond_ex.loc["physics", "control"]:.2f} dB) và nhiễu vật lý ×2 (+{cond_ex.loc["physics_heavy", "physics"] - cond_ex.loc["physics_heavy", "control"]:.2f} dB) → thêm nhiễu vật lý gần như không phải trả giá trên nhiễu chuẩn mà tăng mạnh độ bền với nhiễu kiểu vệ tinh.</div>
{fig_loss}
<h3>5.4 Ảnh vệ tinh thật (HyRank / Hyperion) — đo không tham chiếu</h3>{real_table}
<p>D1 cho stripe index thấp nhất (ít sọc nhất) và SAM so với input thấp hơn control (ít méo phổ hơn); control mượt nhất nhưng méo phổ nhiều nhất.</p>{real_gal}
<h3>5.5 Ảnh output (giả màu R≈650 / G≈550 / B≈470 nm, crop 256×256) và bản đồ sai số</h3><div class="note">Bản đồ |sai số| (trung bình theo band) được chuẩn hoá thang màu <b>riêng từng ảnh</b>: dùng để so <b>cấu trúc</b> sai số (ví dụ control sai theo cả vùng do lệch cường độ, official sai ngẫu nhiên kiểu nhiễu), không so độ lớn giữa các model — độ lớn xem PSNR/SAM dưới mỗi ảnh. Cùng một ảnh nhiễu cho cả 3 model.</div>{gallery}
</section>

<section id="urban"><h2>6. Urban100 σ25 — thích nghi diffusion (deep-learning-laboratory)</h2>
<table><tr><th>Ảnh</th><th>Trước</th><th>Sau</th><th>Tăng</th><th>Ảnh tụt &gt;0,05 dB</th><th>p (Holm)</th><th>Tốt nhất công bố</th></tr>
<tr><td>{usumm["n"]}</td><td>{usumm["unadapted"]:.3f}</td><td>{usumm["adapted"]:.3f}</td><td>+{usumm["gain"]:.3f}</td><td>{100 * usumm["worse_005"]:.0f}%</td><td>{usumm["p_holm"]:.3g}</td><td>{usumm["published_best"]}</td></tr></table>
{ufig}<p>Thích nghi giúp {int((ugain > 0).sum())}/{len(ugain)} ảnh; vẫn kém mức tốt nhất đã công bố {usumm["published_best"] - usumm["adapted"]:.2f} dB.</p></section>

<section id="hatang"><h2>7. Chi phí, hạ tầng và sự cố</h2>
<table><tr><th>Nền tảng</th><th>Dùng cho</th><th>Ghi chú</th></tr>
<tr><td>Kaggle T4</td><td>4 run bilateral 96³ / aniso</td><td>Hết quota GPU tuần</td></tr>
<tr><td>Modal</td><td>depth-attention head</td><td>Workspace bị khoá → mất weights, chỉ còn trace W&B</td></tr>
<tr><td>vast.ai RTX 3090 (Ba Lan, $0,211/h)</td><td>6 run raw 200³, Urban100, MP-HSIR</td><td>~21 giờ, ≈ $4,4–4,6 trên ngân sách $5; còn $0,42</td></tr></table>
<div class="note">Sự cố đã xử lý: (1) W&B chặn IP máy vast (403) → replay metrics từ máy local; (2) <code>pkill -f</code> qua SSH tự khớp lệnh của chính nó; (3) <b>CLI vastai mới hỏi xác nhận khi destroy</b> → controller và watchdog đều "Aborted"; máy chạy thêm ~50 phút (~$0,18), phần lớn thời gian đó được dùng để tạo tài liệu cho report và lưu trữ toàn bộ lên HF trước khi huỷ thủ công — đã sửa script bằng <code>-y</code> và coi "aborted" là lỗi.</div>
<p><b>Khác biệt recipe với SOTA</b> (lý do chính khi so sánh): SOTA dùng lr 4e-4, batch 16, 15 epoch, chọn theo val acc, 96³ raw; các run đợt này dùng lr 1e-4, batch 8, chọn theo val loss. Không có augmentation ở cả hai.</p></section>

<section id="gioihan"><h2>8. Giới hạn và bước tiếp theo</h2><ul>
<li>Mọi so sánh glaucoma là <b>1 seed</b> (trừ ResNeXt raw có 2 seed); khác biệt &lt;~3 điểm acc thường chưa có ý nghĩa thống kê (mục 3.1).</li>
<li>Không tối ưu ngưỡng (không có xác suất trên val trong đợt này) — AUC là chỉ số công bằng nhất giữa các model.</li>
<li>Saliency dạng gradient chỉ mang tính định hướng, không phải bằng chứng giải thích.</li>
<li>MP-HSIR: tham số nhiễu vật lý lấy từ tài liệu, chưa fit từ ảnh L1 thật; Stage 1–3 của kế hoạch (pretrain, EnMAP/PRISMA) ngoài ngân sách.</li>
<li>Đề xuất: (1) chuẩn hoá một protocol (15 ep, batch 16, augmentation, chọn theo val AUC, 3 seed) cho Pyramid/ResNeXt; (2) kết hợp raw + bilateral (bất đồng cao → bổ sung); (3) MP-HSIR: fine-tune có regularizer giữ gần checkpoint gốc hoặc thêm nhiều scene để tránh sụp trên scene mới.</li></ul></section>

<section id="phuluc"><h2>9. Phụ lục — nơi lưu toàn bộ dữ liệu</h2>
<table><tr><th>Repo HF (private)</th><th>Nội dung</th></tr>
<tr><td><code>tqhuyen/glaucoma-raw200-runs</code></td><td><code>runs/</code> 5 run raw 200³ (best_model, checkpoints, TensorBoard, metrics, config) · <code>runs-kaggle/</code> 4 run bil96/aniso · <code>runs-modal/</code> trace W&B 5 run Modal · <code>post/urban100-s25-seed2</code> · <code>post/report-assets</code> (dự đoán từng mẫu, ảnh ca, saliency, ảnh HSI) · <code>logs/</code> mọi log và script đã chạy</td></tr>
<tr><td><code>tqhuyen/hsi-mphsir-pilot</code></td><td><code>run/control</code>, <code>run/physics</code> (checkpoint, log), <code>run/outputs</code> (eval.csv, Stage 0, HyRank) · <code>data/</code> patch train, crop test, cảnh HyRank, checkpoint gốc đã dùng</td></tr>
<tr><td><code>tqhuyen/hsi-mphsir-smoke</code></td><td>Lần chạy smoke 2 phút/nhánh</td></tr>
<tr><td>W&B <code>glaucoma-thesis</code></td><td>Mọi run glaucoma (raw 200³ replay từ local)</td></tr></table>
<p>Code: <code>glaucoma-thesis</code> (models/glaucoma, scripts/vast_*, scripts/report_assets.py, scripts/build_report.py) · <code>deep-learning-laboratory/hsi_mphsir</code>.</p></section>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/experiments_report.html"))
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
