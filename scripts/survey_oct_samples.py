import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    cache = Path(tempfile.gettempdir()) / "gf_vol_cache"
    paths = sorted(cache.glob("raw_*.npy"))[:20]
    output = Path("figures/oct_sample_survey")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for page in range(0, len(paths), 7):
        batch = paths[page : page + 7]
        fig, axes = plt.subplots(len(batch), 3, figsize=(12, 2.7 * len(batch)), squeeze=False, layout="constrained")
        for row, path in enumerate(batch):
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            if raw.shape != (200, 200, 200) or raw.dtype != np.uint8:
                raise ValueError(f"Unexpected sample: {path}")
            data = raw.transpose(0, 2, 1)
            left = data[-1, :, :].T
            right = data[:, 0, :].T
            np.testing.assert_array_equal(left[:, 0], right[:, -1])
            unfolded = np.concatenate([left[:, :32][:, ::-1], right[:, -32:-1][:, ::-1]], axis=1)
            for ax, pixels, title in zip(
                axes[row],
                [raw[100], raw[:, :, 100].T, unfolded],
                ["X=100 section", "Y=100 section", "Unfolded visible edge"],
            ):
                ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
                ax.set_title(f"{path.stem}: {title}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([0, 100, 199])
            axes[row, 2].axvline(31, color="#c58112", linewidth=0.6, linestyle="--")
            records.append(
                {"sample": path.stem, "source": str(path), "shape": list(raw.shape), "shared_edge_equal": True}
            )
        fig.savefig(output / f"survey_{page // 7 + 1}.png", dpi=150)
        plt.close(fig)
    (output / "survey.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Compared {len(records)} volumes: {output.resolve()}")


if __name__ == "__main__":
    main()
