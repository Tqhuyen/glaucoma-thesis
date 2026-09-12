"""Controls-only extensions; leave the reviewed final-training bundle unchanged."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from scripts import final_training as ft


class ControlsDataset(ft.FinalDataset):
    def __init__(self, *args, use_3d=True, **kwargs):
        self.use_3d = use_3d
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.use_3d:
            return super().__getitem__(index)
        from scripts import final_model as fm

        views = np.asarray(self.views[index])
        if self.train:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
            _, views = fm.aug_pair(np.zeros((1, 1, 1), dtype=np.uint8), views, rng)
        return (
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            torch.from_numpy(views.astype(np.float32) / 255)[:, None],
            torch.tensor(int(self.labels[index])),
        )


class Trainer(ft.Trainer):
    def commit(self, *, best=False):
        super().commit(best=best)
        hook = getattr(self, "_epoch_hook", None)
        if (
            hook is not None
            and self.epoch > self._last_notified_epoch
            and self.cursor == 0
            and self.history
            and self.history[-1]["epoch"] == self.epoch
        ):
            self._last_notified_epoch = self.epoch
            hook(self)

    def fit(self, evaluate, *, boundary_hook=None, test_evaluate=None, epoch_hook=None):
        """Call epoch_hook(self) once per newly completed, checkpointed epoch.

        History and train/val/optional test logging are complete, and checkpoint
        saving/synchronization is included in timing measured by the hook.
        No timing is added to history; resumed completed epochs are not replayed.
        """
        self._epoch_hook = epoch_hook
        self._last_notified_epoch = self.epoch
        try:
            return super().fit(evaluate, boundary_hook=boundary_hook, test_evaluate=test_evaluate)
        finally:
            self._epoch_hook = None


def calibrated_report(model, val, test, batch_size, *, smoke=False, train=None, num_workers=0, amp_dtype="float16"):
    """Mirror the frozen report protocol while passing AMP precision to every split."""
    from scripts import final_model as fm

    _, vy, vl = ft.predict(model, val, batch_size, num_workers=num_workers, amp_dtype=amp_dtype)
    temperature = fm.temperature_scale(vl, vy, max_iter=10 if smoke else 200)
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Validation temperature fit did not produce a finite positive temperature")
    vp = torch.softmax(torch.tensor(vl) / temperature, 1)[:, 1].numpy()
    threshold = fm.tune_threshold(vp, vy)
    _, ty, tl = ft.predict(model, test, batch_size, num_workers=num_workers, amp_dtype=amp_dtype)
    tp = torch.softmax(torch.tensor(tl) / temperature, 1)[:, 1].numpy()
    n_boot = 20 if smoke else 1000
    result = {
        "temperature": temperature,
        "threshold": threshold,
        "val": {
            **fm.full_metrics(vp, vy, threshold),
            "loss": F.cross_entropy(torch.tensor(vl) / temperature, torch.tensor(vy)).item(),
        },
        "test": {
            **fm.full_metrics(tp, ty, threshold),
            "loss": F.cross_entropy(torch.tensor(tl) / temperature, torch.tensor(ty)).item(),
        },
        "val_ci": fm.bootstrap_ci(vp, vy, n_boot=n_boot, threshold=threshold),
        "test_ci": fm.bootstrap_ci(tp, ty, n_boot=n_boot, threshold=threshold),
    }
    if train is not None:
        _, tr_labels, trl = ft.predict(model, train, batch_size, num_workers=num_workers, amp_dtype=amp_dtype)
        trp = torch.softmax(torch.tensor(trl) / temperature, 1)[:, 1].numpy()
        result["train"] = {
            **fm.full_metrics(trp, tr_labels, threshold),
            "loss": F.cross_entropy(torch.tensor(trl) / temperature, torch.tensor(tr_labels)).item(),
        }
        result["train_ci"] = fm.bootstrap_ci(trp, tr_labels, n_boot=n_boot, threshold=threshold)
    return result, tp, ty
