#!/usr/bin/env python3
import argparse, csv, math, os
from pathlib import Path
import cv2
import numpy as np

def load_times(csv_path):
    times = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        if "time_s" not in r.fieldnames:
            raise SystemExit("CSV must have a 'time_s' column.")
        for row in r:
            try:
                t = float(row["time_s"])
                times.append(t)
            except (KeyError, ValueError):
                continue
    times.sort()
    return times

def times_to_flash_frames(times, fps, n_frames, flash_dur_s):
    """Return a boolean mask over frames where the dot should be visible."""
    flash = np.zeros(n_frames, dtype=bool)
    half = int(round((flash_dur_s * fps) / 2.0))
    for t in times:
        c = int(round(t * fps))          # center frame for the flash
        lo = max(0, c - half)
        hi = min(n_frames - 1, c + half)
        flash[lo:hi+1] = True
    return flash

def main():
    ap = argparse.ArgumentParser(description="Flash a white dot bottom-left at times from a CSV over a video.")
    ap.add_argument("--video", default="emily_mov_1_short.mov", help="Input video path (default: emily_mov_1_short.mov)")
    ap.add_argument("--csv", default="emily_mov_1_short.footstrikes.csv", help="CSV with a 'time_s' column")
    ap.add_argument("--out", default=None, help="Output video path (default: <video_basename>.flash.mp4)")
    ap.add_argument("--flash_ms", type=float, default=100.0, help="Flash duration in milliseconds (default: 100)")
    ap.add_argument("--radius", type=int, default=12, help="Dot radius in pixels (default: 12)")
    ap.add_argument("--margin", type=int, default=30, help="Margin from bottom-left edges in px (default: 30)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    vid_path = Path(args.video)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not vid_path.exists():
        raise SystemExit(f"Video not found: {vid_path}")

    times = load_times(csv_path)
    if not times:
        raise SystemExit("No valid times found in CSV.")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flash_mask = times_to_flash_frames(times, fps, n_frames, args.flash_ms / 1000.0)

    # Output path / codec
    out_path = Path(args.out) if args.out else vid_path.with_suffix("").with_name(vid_path.stem + ".flash.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible; switch to "avc1" if you have H.264
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open writer for: {out_path}")

    # Dot position (bottom-left)
    x = args.margin + args.radius
    y = height - args.margin - args.radius
    color = (255, 255, 255)  # white (BGR)
    thickness = -1           # filled circle

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if 0 <= i < n_frames and flash_mask[i]:
            cv2.circle(frame, (x, y), args.radius, color, thickness)
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()

    print(f"✅ Wrote: {out_path}")
    print(f"   Frames: {n_frames}  FPS: {fps:.3f}")
    print(f"   Flashes: {sum(flash_mask)} frames (~{(sum(flash_mask)/fps):.2f} s total on-screen time)")

if __name__ == "__main__":
    main()
