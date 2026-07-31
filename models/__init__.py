from importlib import import_module
from typing import Callable

_MODEL_REGISTRY = {
    "glaucoma": "models.glaucoma",
    "nlp": "models.nlp",
}


def get_model_builder(model_type: str) -> Callable:
    module_name = _MODEL_REGISTRY.get(model_type)
    if module_name is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    mod = import_module(module_name)
    return mod.build_model


def get_dataloader_builder(model_type: str) -> Callable:
    module_name = _MODEL_REGISTRY.get(model_type)
    if module_name is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    mod = import_module(module_name)
    return mod.build_dataloaders
