import pytest

from pipeline.config import validate_config


def test_valid_glaucoma_config():
    cfg = {
        "run_name": "test",
        "seed": 42,
        "model": {"type": "glaucoma", "architecture": "simple3dcnn", "input_channels": 1, "num_classes": 2},
        "hf": {"use_auth": False, "fast_transfer": False},
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "lr": 0.001,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "max_grad_norm": 1.0,
            "amp": True,
            "compile": False,
            "log_every_steps": 1,
            "eval_every_steps": 4,
            "save_every_steps": 4,
            "keep_last_checkpoints": 2,
            "early_stop_patience": 3,
        },
        "logging": {"tensorboard": False, "jsonl": False, "wandb": False},
        "output_dir": "/tmp/test",
    }
    validate_config(cfg)


def test_missing_model_type():
    cfg = {"run_name": "test", "seed": 42, "train": {}, "logging": {}, "output_dir": "/tmp", "hf": {}}
    with pytest.raises(ValueError, match="missing key"):
        validate_config(cfg)


def test_invalid_lr():
    cfg = {
        "run_name": "test",
        "seed": 42,
        "model": {"type": "glaucoma", "architecture": "simple3dcnn", "input_channels": 1, "num_classes": 2},
        "hf": {"use_auth": False, "fast_transfer": False},
        "train": {"epochs": 1, "batch_size": 2, "grad_accum_steps": 1, "lr": -0.1, "weight_decay": 0, "warmup_ratio": 0, "max_grad_norm": 1, "amp": False, "compile": False, "log_every_steps": 1, "eval_every_steps": 1, "save_every_steps": 1, "keep_last_checkpoints": 1, "early_stop_patience": 1},
        "logging": {"tensorboard": False, "jsonl": False, "wandb": False},
        "output_dir": "/tmp",
    }
    with pytest.raises(ValueError, match="out of allowed range"):
        validate_config(cfg)


def test_bool_as_int_guarded():
    cfg = {
        "run_name": "test",
        "seed": 42,
        "model": {"type": "glaucoma", "architecture": "simple3dcnn", "input_channels": 1, "num_classes": 2},
        "hf": {"use_auth": True, "fast_transfer": False},
        "train": {"epochs": 1, "batch_size": 2, "grad_accum_steps": 1, "lr": 0.001, "weight_decay": 0, "warmup_ratio": 0, "max_grad_norm": 1, "amp": True, "compile": False, "log_every_steps": 1, "eval_every_steps": 1, "save_every_steps": 1, "keep_last_checkpoints": 1, "early_stop_patience": 1},
        "logging": {"tensorboard": True, "jsonl": False, "wandb": False},
        "output_dir": "/tmp",
    }
    validate_config(cfg)


def test_valid_nlp_config():
    cfg = {
        "run_name": "test-nlp",
        "seed": 42,
        "model": {"type": "nlp", "name": "distilbert-base-uncased", "num_labels": 4},
        "hf": {"use_auth": False, "fast_transfer": False},
        "data": {
            "dataset": "ag_news",
            "text_column": "text",
            "label_column": "label",
            "max_length": 256,
            "val_fraction": 0.05,
            "test_split": "test",
        },
        "train": {
            "epochs": 1,
            "batch_size": 8,
            "grad_accum_steps": 1,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "max_grad_norm": 1.0,
            "amp": False,
            "compile": False,
            "log_every_steps": 1,
            "eval_every_steps": 4,
            "save_every_steps": 4,
            "keep_last_checkpoints": 2,
            "early_stop_patience": 3,
        },
        "logging": {"tensorboard": False, "jsonl": False, "wandb": False},
        "output_dir": "/tmp",
    }
    validate_config(cfg)
