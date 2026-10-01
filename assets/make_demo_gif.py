"""Regenerate assets/demo.gif: two Hockey Fight clips (one fight, one not),
each labelled with its ground-truth class, as a looping hero image.

    python assets/make_demo_gif.py

Requires: opencv-python, pillow, numpy, matplotlib (for the bundled font).
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(HERE, "demo.gif")

bold_path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
reg_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
f_bold = ImageFont.truetype(bold_path, 26)
f_cap = ImageFont.truetype(reg_path, 14)
f_tag = ImageFont.truetype(bold_path, 13)

W, BANNER_H, CAP_H, N = 440, 48, 30, 16


def read_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    grab = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in grab:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    while len(out) < n and out:
        out.append(out[-1])
    return out


def center(d, text, font, cx, y, fill):
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text((cx - (r - l) / 2, y), text, font=font, fill=fill)


def compose(frame, label, is_fight):
    img = Image.fromarray(frame)
    h = int(img.height * (W / img.width))
    img = img.resize((W, h), Image.LANCZOS)
    H = BANNER_H + h + CAP_H
    canvas = Image.new("RGB", (W, H), (13, 17, 23))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, BANNER_H], fill=(229, 72, 77) if is_fight else (46, 164, 94))
    center(d, label, f_bold, W / 2, (BANNER_H - 30) / 2, (255, 255, 255))
    canvas.paste(img, (0, BANNER_H))
    d.rectangle([8, BANNER_H + 8, 94, BANNER_H + 28], fill=(0, 0, 0))
    d.text((14, BANNER_H + 10), "ground truth", font=f_tag, fill=(200, 200, 200))
    center(d, "Hockey Fight dataset  •  VGG16 + LSTM classifier",
           f_cap, W / 2, BANNER_H + h + 7, (170, 178, 189))
    return canvas


def main():
    clips = [(os.path.join(DATA, "fi100_xvid.avi"), "FIGHT", True),
             (os.path.join(DATA, "no100_xvid.avi"), "NO FIGHT", False)]
    frames = []
    for path, label, is_fight in clips:
        comp = [compose(f, label, is_fight) for f in read_frames(path, N)]
        frames += comp + [comp[-1]] * 4
    maxw = max(f.width for f in frames)
    maxh = max(f.height for f in frames)
    norm = []
    for f in frames:
        if f.size != (maxw, maxh):
            bg = Image.new("RGB", (maxw, maxh), (13, 17, 23))
            bg.paste(f, (0, (maxh - f.height) // 2))
            f = bg
        norm.append(f)
    pal = norm[0].quantize(colors=128, method=Image.FASTOCTREE)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in norm]
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=90, loop=0,
              optimize=True, disposal=2)
    print("wrote", OUT, round(os.path.getsize(OUT) / 1024, 1), "KB")


if __name__ == "__main__":
    main()
