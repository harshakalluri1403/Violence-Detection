"""Annotated video rendering and timeline plots."""
import os
from functools import lru_cache

import cv2
import numpy as np

# Palette (light surface). Series colours carry identity; status colours carry
# meaning and are always paired with a text label.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL, GOOD = "#d03b3b", "#0ca30c"


def _bgr(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def apply_style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "legend.frameon": False,
    })


def plot_timeline(timeline, events, threshold=0.6, gt=None, title=None, path=None, ax=None):
    """Violence probability over time with alarm events (and optional ground truth)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_style(plt)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10, 2.8), dpi=130)
    if gt:
        for i, (s, e) in enumerate(gt):
            ax.axvspan(s, e, ymin=0, ymax=1, color=INK2, alpha=0.10, lw=0,
                       label="Ground-truth violence" if i == 0 else None)
    for i, ev in enumerate(events):  # alarm bar sits above the 0-1 plot area
        ax.axvspan(ev.start, ev.end, ymin=1.03 / 1.12, ymax=1.09 / 1.12, color=CRITICAL, lw=0,
                   label="Alarm" if i == 0 else None)
    ax.plot(timeline.t_center, timeline.prob, color=SERIES[0], lw=1, alpha=0.35)
    ax.plot(timeline.t_center, timeline.smoothed(), color=SERIES[0], lw=2, label="Violence probability")
    ax.axhline(threshold, color=MUTED, lw=1)
    ax.text(timeline.duration, threshold, f" threshold {threshold:.2f}", va="center", ha="left",
            color=MUTED, fontsize=8, clip_on=False)
    ax.set_xlim(0, max(timeline.duration, 1e-3))
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(violence)")
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=3, fontsize=8)
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        return fig
    return ax


@lru_cache(maxsize=16)
def _font(size, bold=False, mono=False):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager as fm
        family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
        return ImageFont.truetype(fm.findfont(fm.FontProperties(family=family,
                                                                weight="bold" if bold else "normal")), size)
    except Exception:
        return ImageFont.load_default()


def _rgba(hex_color, a=255):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


def _timeline_strip(timeline, events, width, height=40, threshold=0.6):
    """Whole-video score curve with incident bands, drawn once per video (BGR)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (width, height), (20, 20, 19, 255))
    d = ImageDraw.Draw(img, "RGBA")
    dur = max(timeline.duration, 1e-3)
    y0, y1 = 6, height - 4
    X = lambda t: t / dur * (width - 1)  # noqa: E731
    Y = lambda p: y1 - p * (y1 - y0)  # noqa: E731
    for ev in events:
        d.rectangle([X(ev.start), 0, X(ev.end), height], fill=_rgba(CRITICAL, 95))
    d.line([(0, Y(threshold)), (width, Y(threshold))], fill=(137, 135, 129, 150), width=1)
    if len(timeline.prob) > 1:
        pts = [(X(t), Y(p)) for t, p in zip(timeline.t_center, timeline.smoothed())]
        d.polygon([(pts[0][0], y1)] + pts + [(pts[-1][0], y1)], fill=(57, 135, 229, 70))
        d.line(pts, fill=(57, 135, 229, 255), width=2, joint="curve")
    return np.asarray(img.convert("RGB"))[..., ::-1].copy()


def _hud(frame, t, score, alarm):
    """Status bar (state, timestamp, probability meter) + alarm border, PIL-rendered."""
    from PIL import Image, ImageDraw
    h, w = frame.shape[:2]
    img = Image.fromarray(np.ascontiguousarray(frame[..., ::-1])).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bh = max(30, round(h * 0.09))
    cy, pad, fs = bh // 2, round(bh * 0.42), round(bh * 0.46)
    white = (255, 255, 255, 255)
    d.rectangle([0, 0, w, bh], fill=_rgba(CRITICAL, 235) if alarm else (12, 12, 12, 150))
    r = max(4, round(bh * 0.15))
    d.ellipse([pad - r, cy - r, pad + r, cy + r], fill=white if alarm else _rgba(GOOD))
    d.text((pad + 2 * r + 4, cy), "VIOLENCE DETECTED" if alarm else "No violence",
           font=_font(fs, bold=True), fill=white, anchor="lm")
    pf = _font(fs, bold=True)
    mw, mh = round(w * 0.2), max(6, round(bh * 0.22))
    mx1 = w - pad - d.textlength("100%", font=pf) - 8
    mx0 = mx1 - mw
    d.rounded_rectangle([mx0, cy - mh / 2, mx1, cy + mh / 2], radius=mh / 2, fill=(255, 255, 255, 70))
    if score > 0.02:
        d.rounded_rectangle([mx0, cy - mh / 2, mx0 + max(mh, mw * score), cy + mh / 2], radius=mh / 2,
                            fill=white if alarm else _rgba(GOOD))
    d.text((w - pad, cy), f"{min(score, 0.99):.0%}", font=pf, fill=white, anchor="rm")  # never show certainty
    if w >= 520:
        d.text((mx0 - 14, cy), f"{int(t // 60):02d}:{t % 60:04.1f}", font=_font(round(fs * 0.85), mono=True),
               fill=(255, 255, 255, 210), anchor="rm")
    if alarm:
        d.rectangle([0, 0, w - 1, h - 1], outline=_rgba(CRITICAL), width=max(4, w // 140))
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))[..., ::-1]


def _cam_heat(cam2d, w, h):
    """Upsampled, blurred Grad-CAM slice in [0, 1] at frame resolution."""
    cam = cv2.resize(cam2d.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(np.clip(cam, 0, 1), (0, 0), sigmaX=max(w / 60, 1))
    return cam / (cam.max() + 1e-6)


def _overlay_cam(frame, cam, max_alpha=0.6):
    """Blend an Inferno heatmap whose opacity follows the heat (no hard edges)."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
    a = (np.clip((cam - 0.25) / 0.6, 0, 1) * max_alpha)[..., None]
    return (frame * (1 - a) + heat * a).astype(np.uint8)


def _cam_for_time(cams, t, window_sec, num_slices):
    """Pick the Grad-CAM slice of the window that best covers time t."""
    best, best_d = None, 1e9
    for t_end, cam in cams.items():
        start = t_end - window_sec
        if start <= t <= t_end:
            d = abs(t - (start + window_sec / 2))
            if d < best_d:
                best_d = d
                k = int(np.clip((t - start) / window_sec * num_slices, 0, num_slices - 1))
                best = cam[k]
    return best


def _writer(path, size, fps):
    try:
        import imageio_ffmpeg
        w = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", quality=7,
                                        pix_fmt_in="rgb24", macro_block_size=1,
                                        output_params=["-movflags", "+faststart"])
        w.send(None)
        return lambda f: w.send(np.ascontiguousarray(f[..., ::-1])), w.close
    except Exception:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        return vw.write, vw.release


def render_annotated(src, out_path, timeline, events, threshold=0.6, cams=None, max_width=960):
    """Write a copy of `src` with a status bar, alarm border, a whole-video
    timeline strip with a playhead, and (optionally) Grad-CAM during alarms."""
    cap = cv2.VideoCapture(src)
    fps = timeline.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / max(W, 1))
    w, h = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    strip_h = max(32, h // 9) // 2 * 2
    write, close = _writer(out_path, (w, h + strip_h), fps)
    strip = _timeline_strip(timeline, events, w, strip_h, threshold)
    n_slices = next(iter(cams.values())).shape[0] if cams else 0
    cam_s, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        score = timeline.score_at(t)
        alarm = any(e.start <= t <= e.end for e in events)
        if alarm and cams:
            cam = _cam_for_time(cams, t, timeline.window_sec, n_slices)
            if cam is not None:
                heat = _cam_heat(cam, w, h)
                cam_s = heat if cam_s is None else 0.75 * cam_s + 0.25 * heat  # no flicker
        elif cam_s is not None:
            cam_s = cam_s * 0.8 if cam_s.max() > 0.3 else None  # fade out after an alarm
        if cam_s is not None:
            frame = _overlay_cam(frame, cam_s)
        frame = _hud(frame, t, score, alarm)
        s = strip.copy()
        cx = int(np.clip(t / max(timeline.duration, 1e-3) * (w - 1), 0, w - 1))
        s[:, cx + 1:] = (s[:, cx + 1:] * 0.5).astype(np.uint8)
        cv2.line(s, (cx, 0), (cx, strip_h - 1), (255, 255, 255), 2)
        write(np.vstack([frame, s]))
        i += 1
    cap.release()
    close()
    return out_path


def export_clips(src, events, out_dir, pad=2.0, stem="incident"):
    """Cut one evidence clip per incident (raw source frames, +/- pad seconds)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    spans = [(max(0.0, e.start - pad), e.end + pad) for e in events]
    paths = [os.path.join(out_dir, f"{stem}_{k + 1:02d}_{s:.0f}s-{e:.0f}s.mp4") for k, (s, e) in enumerate(spans)]
    writers = [None] * len(spans)
    i = 0
    while spans:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        for k, (s, e) in enumerate(spans):
            if s <= t <= e:
                if writers[k] is None:
                    writers[k] = _writer(paths[k], (W, H), fps)
                writers[k][0](frame[:H, :W])
        if t > spans[-1][1]:
            break
        i += 1
    cap.release()
    for w in writers:
        if w is not None:
            w[1]()
    return [p for p, w in zip(paths, writers) if w is not None]


def video_to_gif(segments, gif_path, fps=10, width=560, colors=192):
    """Cut (mp4, start_s, duration_s) segments, join them and write a GIF with
    ffmpeg's two-pass palette (far cleaner than a fixed 256-colour palette)."""
    import subprocess
    import imageio_ffmpeg
    inputs, chains = [], []
    for k, (path, t0, dur) in enumerate(segments):
        inputs += ["-ss", f"{t0:.2f}", "-t", f"{dur:.2f}", "-i", path]
        chains.append(f"[{k}:v]fps={fps},scale={width}:-2:flags=lanczos,setsar=1[v{k}]")
    joined = "".join(f"[v{k}]" for k in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[cat]"
    graph = ";".join(chains + [joined, "[cat]split[a][b]", f"[a]palettegen=max_colors={colors}:stats_mode=diff[p]",
                               "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"])
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", graph, "-loop", "0", gif_path], check=True)
    return os.path.getsize(gif_path)


def save_gif(frames_bgr, path, fps=10, width=480):
    from PIL import Image
    imgs = []
    for f in frames_bgr:
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        imgs.append(Image.fromarray(f[..., ::-1]))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0,
                 optimize=True)
    return os.path.getsize(path)
