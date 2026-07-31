from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from pipeline.distributed import is_distributed, main_process_first
from pipeline.utils import optimal_num_proc, optimal_num_workers, seed_worker, setup_hf_env


def build_dataloaders(cfg, smoke=False):
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    token = setup_hf_env(cfg.get("hf", {}))

    num_proc = optimal_num_proc(dcfg.get("num_proc", "auto"))
    num_workers = optimal_num_workers(dcfg.get("num_workers", "auto"))
    print(f"[cpu] preprocessing num_proc={num_proc} | dataloader num_workers={num_workers}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], token=token)

    with main_process_first():
        raw = load_dataset(dcfg["dataset"], token=token, num_proc=num_proc)

    split = raw["train"].train_test_split(test_size=dcfg["val_fraction"], seed=cfg["seed"])
    train_ds, val_ds = split["train"], split["test"]

    test_ds = None
    test_split_name = dcfg.get("test_split")
    if test_split_name and test_split_name in raw:
        test_ds = raw[test_split_name]

    if smoke:
        train_ds = train_ds.select(range(min(len(train_ds), cfg["smoke"]["max_train_samples"])))
        val_ds = val_ds.select(range(min(len(val_ds), cfg["smoke"]["max_val_samples"])))
        if test_ds is not None:
            test_ds = test_ds.select(range(min(len(test_ds), cfg["smoke"]["max_val_samples"])))

    validate_dataset(train_ds, dcfg, cfg["model"]["num_labels"])

    def tokenize(batch):
        return tokenizer(
            batch[dcfg["text_column"]],
            truncation=True,
            max_length=dcfg["max_length"],
            padding=False,
        )

    cols_to_remove = [c for c in train_ds.column_names if c != dcfg["label_column"]]
    map_kwargs = dict(batched=True, num_proc=num_proc, remove_columns=cols_to_remove)

    train_ds = train_ds.map(tokenize, desc="tokenize train", **map_kwargs)
    val_ds = val_ds.map(tokenize, desc="tokenize val", **map_kwargs)
    if test_ds is not None:
        test_ds = test_ds.map(tokenize, desc="tokenize test", **map_kwargs)

    train_ds = train_ds.rename_column(dcfg["label_column"], "labels")
    val_ds = val_ds.rename_column(dcfg["label_column"], "labels")
    if test_ds is not None:
        test_ds = test_ds.rename_column(dcfg["label_column"], "labels")

    def collate(features):
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        batch = tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = labels
        return batch

    batch_size = cfg["smoke"]["batch_size"] if smoke else tcfg["batch_size"]
    common = dict(
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        common["prefetch_factor"] = dcfg.get("prefetch_factor", 4)

    if is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, sampler=train_sampler, drop_last=True, **common)
        val_loader = DataLoader(val_ds, sampler=val_sampler, **common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        val_loader = DataLoader(val_ds, shuffle=False, **common)

    test_loader = DataLoader(test_ds, shuffle=False, **common) if test_ds is not None else None
    return train_loader, val_loader, test_loader


def validate_dataset(ds, dcfg, num_labels):
    problems = []

    if dcfg["text_column"] not in ds.column_names:
        problems.append(f"text_column '{dcfg['text_column']}' not found. Columns: {ds.column_names}")
    if dcfg["label_column"] not in ds.column_names:
        problems.append(f"label_column '{dcfg['label_column']}' not found. Columns: {ds.column_names}")

    if not problems:
        sample = ds.select(range(min(1000, len(ds))))
        texts = sample[dcfg["text_column"]]
        labels = sample[dcfg["label_column"]]

        n_empty = sum(1 for t in texts if not isinstance(t, str) or not t.strip())
        if n_empty:
            problems.append(f"{n_empty}/1000 sampled rows have empty/non-string text.")

        bad = [lab for lab in labels if not (0 <= int(lab) < num_labels)]
        if bad:
            problems.append(
                f"Labels outside [0, {num_labels}): e.g. {bad[:5]}. "
                f"Fix model.num_labels or remap labels."
            )

    if problems:
        raise ValueError("DATASET VALIDATION FAILED:\n  - " + "\n  - ".join(problems))
    print(f"[data] validation passed ({len(ds):,} rows)")
