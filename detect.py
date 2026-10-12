#!/usr/bin/env python
"""Detect violence in a video file, webcam or RTSP/HTTP stream.

  File  -> annotated video + JSON report + timeline plot
      python detect.py clip.mp4 --out out/clip_annotated.mp4 --explain
      python detect.py clip.mp4 --clips              # + one evidence clip per incident
  Live  -> on-screen overlay, alarms printed (and appended to a JSONL log)
      python detect.py --live 0                       # webcam 0
      python detect.py --live rtsp://user:pass@cam/stream --log alarms.jsonl --headless
      python detect.py --live 0 --webhook https://hooks.slack.com/services/...
"""
import argparse
import json
import os
import queue
import threading
import time

import urllib.request

import cv2
import numpy as np

from vdet.stream import ViolenceDetector
from vdet.viz import CRITICAL, GOOD, _bgr, export_clips, plot_timeline, render_annotated


def post_webhook(url, payload):
    """Fire-and-forget JSON POST (Slack/Discord/Teams-compatible `text` field)."""
    def send():
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).close()
        except Exception as e:  # never let alerting crash detection
            print(f"[webhook] failed: {e}")
    threading.Thread(target=send, daemon=True).start()


def analyze_file(det, args):
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.source))[0]
    t0 = time.time()
    tl = det.analyze(args.source, batch=args.batch,
                     keep_above=args.threshold if args.explain else None)
    elapsed = time.time() - t0
    events = tl.events(on=args.threshold, off=max(args.threshold - 0.2, 0.05))
    cams = None
    if args.explain and events:
        from vdet.explain import cams_for_timeline
        cams = cams_for_timeline(det, tl)
    report = {
        "source": args.source,
        "duration_s": round(tl.duration, 2),
        "violent": bool(events),
        "max_probability": round(float(tl.smoothed().max()) if len(tl.prob) else 0.0, 4),
        "events": [e.to_dict() for e in events],
        "threshold": args.threshold,
        "window_s": det.window_sec, "stride_s": det.stride_sec,
        "processing_s": round(elapsed, 2),
        "timeline": {"t": np.round(tl.t_center, 2).tolist(), "p": np.round(tl.prob, 4).tolist()},
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1)
    plot_timeline(tl, events, args.threshold, title=os.path.basename(args.source),
                  path=os.path.join(args.out_dir, f"{stem}_timeline.png"))
    if not args.no_video:
        render_annotated(args.source, os.path.join(args.out_dir, f"{stem}_annotated.mp4"),
                         tl, events, args.threshold, cams)
    if args.clips and events:
        clips = export_clips(args.source, events, os.path.join(args.out_dir, f"{stem}_incidents"), stem=stem)
        report["incident_clips"] = clips
        with open(json_path, "w") as f:
            json.dump(report, f, indent=1)
    verdict = "VIOLENCE" if events else "no violence"
    print(f"{args.source}: {verdict} | {len(events)} event(s) | max p={report['max_probability']:.2f} | "
          f"{tl.duration:.1f}s video in {elapsed:.1f}s")
    for e in events:
        print(f"   {e.start:7.1f}s -> {e.end:7.1f}s   peak {e.peak:.2f}")
    print(f"outputs in {args.out_dir}/")


def run_live(det, args):
    """Reader thread keeps the newest frames flowing; inference runs in the
    main loop at the window stride, so a slow model never freezes the video."""
    src = int(args.live) if str(args.live).isdigit() else args.live
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.live}")
    frames = queue.Queue(maxsize=256)
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            ok, f = cap.read()
            if not ok:
                break
            try:
                frames.put((time.time(), f), timeout=1)
            except queue.Full:
                pass
        stop.set()

    threading.Thread(target=reader, daemon=True).start()
    sc = det.scorer(batch=1)
    t_start, hist_t, hist_p = None, [], []
    in_alarm, alarm_start = False, 0.0
    log = open(args.log, "a") if args.log else None
    on, off = args.threshold, max(args.threshold - 0.2, 0.05)
    while not stop.is_set() or not frames.empty():
        try:
            ts, frame = frames.get(timeout=1)
        except queue.Empty:
            continue
        t_start = t_start or ts
        t = ts - t_start
        for t_end, p in sc.push(frame, t):
            hist_t.append(t_end)
            hist_p.append(p)
        score = float(np.mean(hist_p[-3:])) if hist_p else 0.0
        if not in_alarm and score >= on:
            in_alarm, alarm_start = True, t
            print(f"[ALARM] violence at t={t:.1f}s (p={score:.2f})", flush=True)
            if args.webhook:
                post_webhook(args.webhook, {"text": f"Violence detected on {args.live} at {time.ctime(ts)} "
                                                    f"(p={score:.2f})", "event": "start",
                                            "source": str(args.live), "time": ts, "probability": score})
            if log:
                log.write(json.dumps({"event": "start", "t": round(t, 2), "wall": time.ctime(ts), "p": score}) + "\n")
                log.flush()
        elif in_alarm and score < off:
            in_alarm = False
            print(f"[clear] t={t:.1f}s (lasted {t - alarm_start:.1f}s)", flush=True)
            if args.webhook:
                post_webhook(args.webhook, {"text": f"Incident on {args.live} ended after {t - alarm_start:.0f}s",
                                            "event": "end", "source": str(args.live), "time": ts,
                                            "duration_s": t - alarm_start})
            if log:
                log.write(json.dumps({"event": "end", "t": round(t, 2), "wall": time.ctime(ts)}) + "\n")
                log.flush()
        if not args.headless:
            view = frame.copy()
            h, w = view.shape[:2]
            color = _bgr(CRITICAL) if in_alarm else _bgr(GOOD)
            cv2.rectangle(view, (0, 0), (w, 36), color if in_alarm else (30, 30, 30), -1)
            cv2.putText(view, f"{'VIOLENCE' if in_alarm else 'no violence'}  {score:.0%}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if in_alarm:
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), color, 6)
            cv2.imshow("violence detection (q to quit)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    stop.set()
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="video file")
    ap.add_argument("--live", help="webcam index or stream URL")
    ap.add_argument("--checkpoint", default="checkpoints/vdet_base.pt")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--window", type=float, default=3.2, help="seconds of video per decision")
    ap.add_argument("--stride", type=float, default=0.4, help="seconds between decisions")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--explain", action="store_true", help="Grad-CAM heatmaps during alarms")
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    ap.add_argument("--clips", action="store_true", help="save one evidence clip per incident")
    ap.add_argument("--device", default=None)
    ap.add_argument("--int8", action="store_true", help="CPU only: int8 dynamic quantisation, ~1.4x faster")
    ap.add_argument("--log", help="live mode: append alarm events to this JSONL file")
    ap.add_argument("--headless", action="store_true", help="live mode: no display window")
    ap.add_argument("--webhook", help="live mode: POST JSON alerts to this URL (Slack/Discord/Teams/n8n)")
    args = ap.parse_args()
    if not args.source and args.live is None:
        ap.error("give a video file or --live SOURCE")
    if args.int8 and args.explain:
        print("--int8 models cannot run Grad-CAM; ignoring --explain")
        args.explain = False
    det = ViolenceDetector(args.checkpoint, device=args.device, window_sec=args.window, stride_sec=args.stride,
                           int8=args.int8)
    if args.live is not None:
        run_live(det, args)
    else:
        analyze_file(det, args)


if __name__ == "__main__":
    main()
