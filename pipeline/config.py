"""Config schema validation: every run fails in <1 second on a malformed config,
never 20 minutes in. Checks presence AND type AND value ranges."""
from __future__ import annotations

from typing import Any

_BASE_SCHEMA: list[tuple[str, tuple, Any]] = [
    ("run_name", (str,), lambda v: len(v) > 0),
    ("seed", (int,), None),
    ("model.type", (str,), lambda v: len(v) > 0),
    ("hf.use_auth", (bool,), None),
    ("hf.fast_transfer", (bool,), None),
    ("train.epochs", (int,), lambda v: v >= 1),
    ("train.batch_size", (int,), lambda v: v >= 1),
    ("train.grad_accum_steps", (int,), lambda v: v >= 1),
    ("train.lr", (float,), lambda v: 0 < v < 1),
    ("train.weight_decay", (float, int), lambda v: v >= 0),
    ("train.warmup_ratio", (float, int), lambda v: 0 <= v < 1),
    ("train.max_grad_norm", (float, int), lambda v: v > 0),
    ("train.amp", (bool,), None),
    ("train.compile", (bool,), None),
    ("train.log_every_steps", (int,), lambda v: v >= 1),
    ("train.eval_every_steps", (int,), lambda v: v >= 1),
    ("train.save_every_steps", (int,), lambda v: v >= 1),
    ("train.keep_last_checkpoints", (int,), lambda v: v >= 1),
    ("train.early_stop_patience", (int,), lambda v: v >= 1),
    ("logging.tensorboard", (bool,), None),
    ("logging.jsonl", (bool,), None),
    ("logging.wandb", (bool,), None),
    ("output_dir", (str,), None),
]


_MODEL_SCHEMAS = {
    "nlp": [
        ("model.name", (str,), lambda v: len(v) > 0),
        ("model.num_labels", (int,), lambda v: v >= 2),
        ("data.dataset", (str,), None),
        ("data.text_column", (str,), None),
        ("data.label_column", (str,), None),
        ("data.max_length", (int,), lambda v: 8 <= v <= 8192),
        ("data.val_fraction", (float,), lambda v: 0 < v < 0.5),
        ("data.test_split", (str,), None),
    ],
    "glaucoma": [
        ("model.architecture", (str,), None),
        ("model.input_channels", (int,), lambda v: v >= 1),
        ("model.num_classes", (int,), lambda v: v >= 2),
    ],
}


def _get(cfg: dict, dotted: str):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(dotted)
        node = node[k]
    return node


def validate_config(cfg: dict) -> None:
    errors = []

    model_type = cfg.get("model", {}).get("type", "unknown")
    schema = list(_BASE_SCHEMA) + _MODEL_SCHEMAS.get(model_type, [])

    for dotted, types, check in schema:
        try:
            val = _get(cfg, dotted)
        except KeyError:
            errors.append(f"missing key: {dotted}")
            continue
        if bool not in types and isinstance(val, bool):
            errors.append(f"{dotted}: expected {types}, got bool ({val!r})")
            continue
        if not isinstance(val, types):
            errors.append(
                f"{dotted}: expected {[t.__name__ for t in types]}, "
                f"got {type(val).__name__} ({val!r})"
            )
            continue
        if check is not None and not check(val):
            errors.append(f"{dotted}: value {val!r} out of allowed range")

    if errors:
        raise ValueError("CONFIG VALIDATION FAILED:\n  - " + "\n  - ".join(errors))
