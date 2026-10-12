"""Spatio-temporal Grad-CAM for VideoMAE: *where* in the clip the model sees violence."""
import numpy as np
import torch


class VideoGradCAM:
    """Grad x activation on the last transformer block's tubelet tokens.

    Returns a (T/tubelet, H/patch, W/patch) map in [0, 1] — 8 x 14 x 14 for the
    default 16-frame, 224 px VideoMAE.
    """

    def __init__(self, model):
        self.model = model
        cfg = model.net.config
        self.grid = (cfg.num_frames // cfg.tubelet_size, cfg.image_size // cfg.patch_size,
                     cfg.image_size // cfg.patch_size)
        self._act = None
        self._handle = model.net.videomae.encoder.layer[-1].register_forward_hook(self._hook)

    def remove(self):
        self._handle.remove()

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.requires_grad:
            h.retain_grad()
        self._act = h

    def __call__(self, x):
        """x: (1, T, 3, H, W) normalised clip on the model's device/dtype."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            logit = self.model(x)
            logit.sum().backward()
        act, grad = self._act, self._act.grad
        cam = (act * grad).sum(-1).relu()[0].float()
        cam = cam.view(*self.grid).detach().cpu().numpy()
        cam /= cam.max() + 1e-8
        return cam, float(torch.sigmoid(logit.float()).item())


def cams_for_timeline(detector, timeline, max_windows=120):
    """Grad-CAM for the windows the scorer kept (the high-scoring ones)."""
    if not timeline.kept:
        return {}
    gc = VideoGradCAM(detector.model)
    sc = detector.scorer()
    times = sorted(timeline.kept)
    if len(times) > max_windows:
        times = [times[i] for i in np.linspace(0, len(times) - 1, max_windows).round().astype(int)]
    out = {}
    try:
        for t in times:
            out[t] = gc(sc.to_tensor(timeline.kept[t][None]))[0]
    finally:
        gc.remove()
    return out
