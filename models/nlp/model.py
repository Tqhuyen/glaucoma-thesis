from transformers import AutoModelForSequenceClassification


def build_model(cfg):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
        token=cfg.get("_hf_token"),
    )
