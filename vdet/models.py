"""VideoMAE violence classifier + checkpoint I/O."""
import torch
import torch.nn as nn

MODEL_ZOO = {
    "base": "MCG-NJU/videomae-base-finetuned-kinetics",    # ViT-B, 87M params
    "small": "MCG-NJU/videomae-small-finetuned-kinetics",  # ViT-S, 22M params
}


class VideoClassifier(nn.Module):
    """Kinetics-pretrained VideoMAE with a single violence logit.

    Input: (B, T, 3, H, W) ImageNet-normalised frames. Output: (B,) logits.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    @property
    def num_frames(self):
        return self.net.config.num_frames

    def forward(self, x):
        return self.net(pixel_values=x).logits.squeeze(-1)


def _set_sdpa(config):
    try:
        config._attn_implementation = "sdpa"
    except Exception:
        pass


def build_model(arch="base", config_dict=None):
    """Pretrained Kinetics weights when config_dict is None, else an empty shell."""
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    if config_dict is None:
        name = MODEL_ZOO.get(arch, arch)
        try:
            net = VideoMAEForVideoClassification.from_pretrained(
                name, num_labels=1, ignore_mismatched_sizes=True, attn_implementation="sdpa")
        except (TypeError, ValueError):
            net = VideoMAEForVideoClassification.from_pretrained(
                name, num_labels=1, ignore_mismatched_sizes=True)
        net.config.id2label = {0: "violence"}
        net.config.label2id = {"violence": 0}
    else:
        config = VideoMAEConfig.from_dict(config_dict)
        _set_sdpa(config)
        net = VideoMAEForVideoClassification(config)
    return VideoClassifier(net)


def save_checkpoint(model, path, meta=None):
    sd = {k: v.detach().half().cpu() if v.is_floating_point() else v.cpu()
          for k, v in model.state_dict().items()}
    torch.save({"config": model.net.config.to_dict(), "state_dict": sd, "meta": meta or {}}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config_dict=ckpt["config"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), ckpt.get("meta", {})


def layer_decay_param_groups(model, weight_decay=0.05, layer_decay=0.75):
    """AdamW groups with layer-wise LR decay: the head learns fastest, the
    patch embedding slowest. Each group carries an `lr_scale`."""
    n = model.net.config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".embeddings." in name:
            lid = 0
        elif ".encoder.layer." in name:
            lid = int(name.split(".encoder.layer.")[1].split(".")[0]) + 1
        else:
            lid = n + 1
        no_wd = p.ndim == 1 or name.endswith(".bias")
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr_scale": layer_decay ** (n + 1 - lid),
                           "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())
