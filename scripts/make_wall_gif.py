#!/usr/bin/env python
"""The README hero: a "control-room" camera wall.

Four held-out test videos play side by side as cameras. Each is analysed the
way `detect.py` analyses a recorded file (centred smoothing, hysteresis
incidents, Grad-CAM), and the wall shows four camera tiles, an incident log
that fills as incidents begin and end, and probability traces. Every score and
heatmap is real model output; the footage was never seen in training.

    python scripts/make_wall_gif.py --checkpoint runs/vmae_b_mixed/model.pt \
        --bench data/longbench_preview --out assets/demo.gif
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_longbench import gt_events  # noqa: E402
from vdet.viz import CRITICAL, GOOD, _cam_for_time, _cam_heat, _font, _overlay_cam, video_to_gif  # noqa: E402

# Dark "control room" palette (dark-surface steps of the project palette).
PAGE, PANEL, LINE = (13, 13, 13), (26, 26, 25), (44, 44, 42)
INK, INK2, MUTED = (240, 239, 236), (195, 194, 183), (137, 135, 129)
BLUE = (57, 135, 229)
RED = tuple(int(CRITICAL[i:i + 2], 16) for i in (1, 3, 5))
GREEN = tuple(int(GOOD[i:i + 2], 16) for i in (1, 3, 5))

W, H = 960, 540
TILE_W, TILE_H, GAP = 316, 178, 8
GRID_X, GRID_Y = 12, 48
SIDE_X, SIDE_W = GRID_X + 2 * TILE_W + GAP + 12, 960 - (GRID_X + 2 * TILE_W + GAP + 12) - 12
LANE_Y, LANE_H = GRID_Y + 2 * TILE_H + GAP + 12, 24


class Cam:
    """One tile: a source video, where the clip starts, and its analysed timeline."""

    def __init__(self, name, kind, video, offset, tl, events, cams):
        self.name, self.kind, self.video, self.offset = name, kind, video, offset
        self.t, self.score = tl.t_center, tl.smoothed()
        self.events = [(e.start, e.end) for e in events]
        self.cams, self.window = cams, tl.window_sec
        self.cap = cv2.VideoCapture(video)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.src_idx = int(round(offset * self.src_fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.src_idx)
        self.frame, self.heat = None, None

    def frame_at(self, t):
        """Sequentially read the source up to time t (absolute, seconds)."""
        target = int(t * self.src_fps)
        while self.src_idx <= target or self.frame is None:
            ok, f = self.cap.read()
            if not ok:
                break
            self.frame, self.src_idx = f, self.src_idx + 1
        return self.frame

    def state_at(self, t):
        score = float(np.interp(t, self.t, self.score)) if len(self.t) else 0.0
        return score, any(s <= t <= e for s, e in self.events)


def _pill(d, xy, text, font, fg, bg, pad=(7, 3), anchor_right=False):
    x, y = xy
    tw = d.textlength(text, font=font)
    h = font.size + 2 * pad[1]
    x0 = x - tw - 2 * pad[0] if anchor_right else x
    d.rounded_rectangle([x0, y, x0 + tw + 2 * pad[0], y + h], radius=h / 2, fill=bg)
    d.text((x0 + pad[0], y + h / 2), text, font=font, fill=fg, anchor="lm")


def render_tile(cam, t, size):
    frame = cam.frame_at(cam.offset + t)
    tile = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    score, alarm = cam.state_at(cam.offset + t)
    cam_slice = _cam_for_time(cam.cams, cam.offset + t, cam.window, 8) if alarm and cam.cams else None
    if cam_slice is not None:
        heat = _cam_heat(cam_slice, size[0], size[1]) ** 1.5  # focus on the hot spots on small tiles
        cam.heat = heat if cam.heat is None else 0.7 * cam.heat + 0.3 * heat
    elif cam.heat is not None:
        cam.heat = cam.heat * 0.8 if cam.heat.max() > 0.3 else None
    if cam.heat is not None:
        tile = _overlay_cam(tile, cam.heat, max_alpha=0.7)
    return Image.fromarray(tile[..., ::-1]), score, alarm


def draw_frame(cams, tau, duration, log, thr):
    img = Image.new("RGB", (W, H), PAGE)
    d = ImageDraw.Draw(img)
    f_title, f_sub = _font(17, bold=True), _font(12)
    f_label, f_small, f_mono = _font(11, bold=True), _font(11), _font(13, mono=True)

    # Header
    d.text((GRID_X, 24), "VIOLENCE DETECTION", font=f_title, fill=INK, anchor="lm")
    d.text((GRID_X + d.textlength("VIOLENCE DETECTION", font=f_title) + 10, 25),
           "camera wall  ·  held-out test footage", font=f_sub, fill=MUTED, anchor="lm")
    clock = f"{int(tau // 60):02d}:{tau % 60:04.1f}"
    d.text((W - 12, 24), clock, font=f_mono, fill=INK2, anchor="rm")
    px = W - 12 - d.textlength(clock, font=f_mono) - 12
    d.polygon([(px - 9, 18), (px - 9, 30), (px, 24)], fill=INK2)

    # Camera tiles
    for k, cam in enumerate(cams):
        x = GRID_X + (k % 2) * (TILE_W + GAP)
        y = GRID_Y + (k // 2) * (TILE_H + GAP)
        tile, score, alarm = render_tile(cam, tau, (TILE_W, TILE_H))
        img.paste(tile, (x, y))
        d.rectangle([x - 1, y - 1, x + TILE_W, y + TILE_H], outline=RED if alarm else LINE, width=3 if alarm else 1)
        _pill(d, (x + 8, y + 8), f"{cam.name}  ·  {cam.kind}", f_label, INK, (0, 0, 0))
        if alarm:
            _pill(d, (x + TILE_W - 8, y + 8), f"VIOLENCE {min(score, 0.99):.0%}", f_label, (255, 255, 255), RED,
                  anchor_right=True)
        else:
            _pill(d, (x + TILE_W - 8, y + 8), "● OK", f_label, GREEN, (0, 0, 0), anchor_right=True)
        d.rectangle([x, y + TILE_H - 4, x + TILE_W - 1, y + TILE_H - 1], fill=(0, 0, 0))
        d.rectangle([x, y + TILE_H - 4, x + int((TILE_W - 1) * score), y + TILE_H - 1], fill=RED if alarm else GREEN)
        # incident log: an entry appears when its incident begins, and closes when it ends
        now = cam.offset + tau
        for (s, e) in cam.events:
            if not 0 <= s - cam.offset <= tau:
                continue
            m = (cam.t >= s) & (cam.t <= min(e, now))
            log[(cam.name, s)] = {"cam": cam.name, "t": s - cam.offset,
                                  "peak": float(cam.score[m].max()) if m.any() else score,
                                  "end": e - cam.offset if e <= now else None}

    # Sidebar: incident log + system facts
    sy0, sy1 = GRID_Y - 1, GRID_Y + 2 * TILE_H + GAP
    d.rounded_rectangle([SIDE_X, sy0, SIDE_X + SIDE_W, sy1], radius=8, fill=PANEL)
    d.text((SIDE_X + 14, sy0 + 20), "INCIDENTS", font=f_label, fill=INK2, anchor="lm")
    n_open = sum(1 for e in log.values() if e["end"] is None)
    _pill(d, (SIDE_X + SIDE_W - 12, sy0 + 11), f"{n_open} active" if n_open else "all clear", f_label,
          (255, 255, 255) if n_open else GREEN, RED if n_open else (20, 20, 19), anchor_right=True)
    entries = sorted(log.values(), key=lambda e: -e["t"])[:5]
    yy = sy0 + 44
    if not entries:
        d.text((SIDE_X + 14, yy + 10), "No incidents", font=f_small, fill=MUTED, anchor="lm")
    for e in entries:
        active = e["end"] is None
        d.ellipse([SIDE_X + 14, yy + 6, SIDE_X + 22, yy + 14], fill=RED if active else MUTED)
        d.text((SIDE_X + 30, yy + 10), e["cam"], font=f_label, fill=INK, anchor="lm")
        d.text((SIDE_X + SIDE_W - 14, yy + 10), f"{int(e['t'] // 60):02d}:{e['t'] % 60:04.1f}", font=f_small,
               fill=INK2, anchor="rm")
        peak = min(e["peak"], 0.99)  # never display certainty
        detail = (f"violence  ·  peak {peak:.0%}" if active
                  else f"cleared after {e['end'] - e['t']:.1f} s  ·  peak {peak:.0%}")
        d.text((SIDE_X + 30, yy + 27), detail, font=f_small, fill=INK2 if active else MUTED, anchor="lm")
        d.line([SIDE_X + 14, yy + 42, SIDE_X + SIDE_W - 14, yy + 42], fill=LINE)
        yy += 50
    facts = ["VideoMAE-B  ·  86M parameters", "3.2 s window  ·  a decision every 0.4 s",
             "alarm at 0.6  ·  clears below 0.4"]
    for i, line in enumerate(facts):
        d.text((SIDE_X + 14, sy1 - 58 + 18 * i), line, font=f_small, fill=MUTED, anchor="lm")

    # Live probability lanes
    lx0, lx1 = GRID_X + 64, W - 12
    d.rounded_rectangle([GRID_X, LANE_Y - 6, W - 12, LANE_Y + 4 * (LANE_H + 4) + 2], radius=8, fill=PANEL)
    for k, cam in enumerate(cams):
        y0 = LANE_Y + k * (LANE_H + 4)
        y1 = y0 + LANE_H
        d.text((GRID_X + 12, (y0 + y1) / 2), cam.name, font=f_label, fill=INK2, anchor="lm")
        X = lambda t: lx0 + (t / duration) * (lx1 - lx0)  # noqa: E731
        Y = lambda p: y1 - 2 - p * (LANE_H - 4)  # noqa: E731
        d.line([lx0, Y(thr), lx1, Y(thr)], fill=LINE)
        for (s, e) in cam.events:
            a, b = max(s - cam.offset, 0), min(e - cam.offset, tau)
            if b > a:
                d.rectangle([X(a), y0, X(b), y1], fill=(70, 24, 24))
        m = (cam.t >= cam.offset) & (cam.t <= cam.offset + tau)
        ts, ss = cam.t[m] - cam.offset, cam.score[m]
        if len(ts) > 1:
            pts = [(X(a), Y(b)) for a, b in zip(ts, ss)]
            d.polygon([(pts[0][0], y1)] + pts + [(pts[-1][0], y1)], fill=(27, 52, 84))
            d.line(pts, fill=BLUE, width=2)
        d.line([X(tau), y0, X(tau), y1], fill=INK2)
    return img


def choose_cams(entries, duration, lead=(2.5, 7.0, 11.5), kinds=("CCTV", "STREET", "CCTV")):
    """Pick 3 cameras with a correctly detected fight (alarms staggered, a mix
    of CCTV and street footage) and 1 CCTV camera that stays quiet throughout."""
    fights, quiet = [], None
    for e in entries:
        for (s, en) in e["gts"]:
            hit = any(a <= en and b >= s for a, b in e["events"])
            if hit and s >= 4:
                fights.append((e, s))
        if quiet is None or e["kind"] == "CCTV":
            spans = list(e["events"]) + e["gts"]
            for o in np.arange(4.0, e["duration"] - duration - 1, 1.0):
                if all(b < o - 1 or a > o + duration + 1 for a, b in spans):
                    if quiet is None or (quiet[0]["kind"] != "CCTV" and e["kind"] == "CCTV"):
                        quiet = (e, float(o))
                    break
    chosen, used = [], set()
    for want, kind in zip(lead, kinds):
        pool = [f for f in fights if f[0]["kind"] == kind] + [f for f in fights if f[0]["kind"] != kind]
        for e, s in pool:
            if e["name"] in used or (quiet and e["name"] == quiet[0]["name"]):
                continue
            o = s - want
            if o >= 0 and o + duration <= e["duration"]:
                chosen.append((e, o))
                used.add(e["name"])
                break
    if quiet:
        chosen.insert(1, quiet)
    return chosen[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/vmae_b_mixed/model.pt")
    ap.add_argument("--bench", default="data/longbench_preview")
    ap.add_argument("--out", default="assets/demo.gif")
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gif-width", type=int, default=840)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    from vdet.explain import cams_for_timeline
    from vdet.stream import ViolenceDetector
    det = ViolenceDetector(args.checkpoint)
    entries = []
    for path in sorted(glob.glob(os.path.join(args.bench, "*.mp4"))):
        if path.endswith(".tmp.mp4"):
            continue
        meta = json.load(open(path.replace(".mp4", ".json")))
        tl = det.analyze(path, batch=32, keep_above=args.threshold)
        events = tl.events(on=args.threshold, off=args.threshold - 0.2)
        name = os.path.basename(path)[:-4]
        entries.append({"name": name, "path": path, "kind": "CCTV" if name.startswith("rwf") else "STREET",
                        "tl": tl, "ev": events, "events": [(ev.start, ev.end) for ev in events],
                        "gts": gt_events(meta["segments"]), "duration": tl.duration})
        print(name, "fights", [(round(a), round(b)) for a, b in entries[-1]["gts"]],
              "incidents", [(round(a, 1), round(b, 1)) for a, b in entries[-1]["events"]])
    chosen = choose_cams(entries, args.duration)
    cams = []
    for k, (e, o) in enumerate(chosen):
        cm = cams_for_timeline(det, e["tl"])
        cams.append(Cam(f"CAM {k + 1:02d}", e["kind"], e["path"], o, e["tl"], e["ev"], cm))
        print(f"CAM {k + 1:02d}: {e['name']} from {o:.1f}s")

    import imageio_ffmpeg
    mp4 = args.out.replace(".gif", ".mp4")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    w = imageio_ffmpeg.write_frames(mp4, (W, H), fps=args.fps, codec="libx264", quality=9, macro_block_size=1)
    w.send(None)
    log = {}
    for n in range(int(args.duration * args.fps)):
        w.send(np.asarray(draw_frame(cams, n / args.fps, args.duration, log, args.threshold)))
    w.close()
    size = video_to_gif([(mp4, 0, args.duration)], args.out, fps=args.fps, width=args.gif_width, colors=176)
    print(f"{args.out}: {size / 1e6:.1f} MB ({len(log)} incidents logged)")


if __name__ == "__main__":
    main()
