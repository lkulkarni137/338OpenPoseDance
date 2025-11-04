#!/usr/bin/env python
import argparse, json, math, re
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter, find_peaks

# ---------- Keypoint maps ----------
BODY25 = {
    "R_ANKLE":  10, "L_ANKLE":  13,
    "R_HEEL":   24, "L_HEEL":   21,
    "R_BIGTOE": 22, "L_BIGTOE": 19,
    "R_SMALLTOE": 23, "L_SMALLTOE": 20,
}
COCO = {"R_ANKLE": 10, "L_ANKLE": 13}

FRAME_RE = re.compile(r"_(\d+)_keypoints\.json$")

def parse_frame_num(p: Path) -> int:
    m = FRAME_RE.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse frame number from filename: {p.name}")
    return int(m.group(1))

def load_openpose_frame(json_path):
    with open(json_path, "r") as f:
        js = json.load(f)
    people = js.get("people", [])
    if not people:
        return None
    p = people[0]
    if "pose_keypoints_2d" in p:
        pts = np.array(p["pose_keypoints_2d"], dtype=float).reshape(-1, 3)
    elif "pose_keypoints" in p:
        pts = np.array(p["pose_keypoints"], dtype=float).reshape(-1, 3)
    else:
        return None
    return pts  # (K,3): x,y,c

def choose_model_map(model: str):
    m = model.lower()
    if m in ("body_25","body25","25","body-25"):
        return BODY25, "body25"
    if m in ("coco","coco_18","18"):
        return COCO, "coco"
    return BODY25, "body25"

def extract_series(pts_per_frame, idx_list, conf_min=0.2):
    y = []
    for pts in pts_per_frame:
        if pts is None:
            y.append(np.nan); continue
        best, best_c = np.nan, -1.0
        for idx in idx_list:
            if idx is None or idx >= len(pts): continue
            _, yy, c = pts[idx]
            if c >= conf_min and c > best_c:
                best, best_c = yy, c
        y.append(best if best_c >= conf_min else np.nan)
    return np.array(y, dtype=float)

def interpolate_nans(a):
    a = a.copy()
    isn = np.isnan(a)
    if not isn.any(): return a
    idx = np.arange(len(a))
    good = ~isn
    if good.sum() >= 2:
        a[isn] = np.interp(idx[isn], idx[good], a[good])
    return a

def detect_footstrikes_from_y(y, fps, frame_nums, smooth_win=21, smooth_poly=3,
                              prominence_px=3.0, distance_frames=None):
    """
    Strike = local MINIMUM of y (low point). We detect minima as peaks of (-y_s).
    Times are computed from TRUE frame numbers: t = frame_num / fps.
    """
    y = interpolate_nans(y)
    if np.all(np.isnan(y)):
        return np.array([], dtype=int), np.array([], dtype=float), np.array([])

    win = min(smooth_win if smooth_win % 2 else smooth_win+1, max(3, (len(y)//2)*2+1))
    y_s = savgol_filter(y, window_length=win, polyorder=min(smooth_poly, win-1), mode="interp")

    if distance_frames is None:
        distance_frames = max(1, int(round(0.25 * fps)))

    # local minima of y -> peaks of -y
    peaks, _ = find_peaks(-y_s, prominence=prominence_px, distance=distance_frames)

    # map peak indices -> real frame numbers -> seconds
    peak_frames = np.array(frame_nums)[peaks]
    t = peak_frames / float(fps)
    return peaks, t, y_s

def main():
    ap = argparse.ArgumentParser(description="Detect foot-strike times (local minima of y) from OpenPose JSONs; use real frame numbers from filenames.")
    ap.add_argument("--json_dir", required=True, help="Directory with OpenPose *_keypoints.json")
    ap.add_argument("--fps", type=float, required=True, help="Video FPS")
    ap.add_argument("--model", default="body25", help="body25 (default) or coco")
    ap.add_argument("--conf_min", type=float, default=0.2)
    ap.add_argument("--side", choices=["left","right","both"], default="both")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--prominence_px", type=float, default=3.0)
    ap.add_argument("--min_interval_s", type=float, default=0.25)
    args = ap.parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    kpmap, model_name = choose_model_map(args.model)
    if model_name == "body25":
        right_pref = [kpmap.get("R_HEEL"), kpmap.get("R_ANKLE"), kpmap.get("R_BIGTOE")]
        left_pref  = [kpmap.get("L_HEEL"), kpmap.get("L_ANKLE"), kpmap.get("L_BIGTOE")]
    else:
        right_pref = [kpmap.get("R_ANKLE")]
        left_pref  = [kpmap.get("L_ANKLE")]

    # Collect and sort JSONs by TRUE frame number
    files = list(json_dir.glob("*_keypoints.json"))
    if not files:
        raise SystemExit(f"[error] no JSON files in {json_dir}")
    files.sort(key=parse_frame_num)
    frame_nums = [parse_frame_num(p) for p in files]

    pts_per_frame = [load_openpose_frame(p) for p in files]

    ys = {}
    if args.side in ("right","both"):
        ys["right"] = extract_series(pts_per_frame, right_pref, conf_min=args.conf_min)
    if args.side in ("left","both"):
        ys["left"]  = extract_series(pts_per_frame, left_pref,  conf_min=args.conf_min)

    strikes = []
    for side, y in ys.items():
        peaks_idx, peaks_t, y_s = detect_footstrikes_from_y(
            y, fps=args.fps, frame_nums=frame_nums,
            prominence_px=args.prominence_px,
            distance_frames=max(1, int(round(args.min_interval_s * args.fps))),
        )
        for pi, t in zip(peaks_idx, peaks_t):
            strikes.append((t, side, int(frame_nums[pi]), float(y_s[pi])))

    strikes.sort(key=lambda r: r[0])

    # Write CSV
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "side", "frame_num", "y_smoothed"])
        for t, side, fnum, ypix in strikes:
            w.writerow([f"{t:.6f}", side, fnum, f"{ypix:.2f}"])

    print(f"✅ wrote {len(strikes)} strikes → {out_csv}")
    last_f = frame_nums[-1]
    est_dur = last_f / float(args.fps)
    print(f"   JSONs cover frames 0…{last_f} ⇒ ~{est_dur:.2f}s at {args.fps:.3f} fps")
    if est_dur < 0.75 * 22:  # rough sanity if your clip is ~22s
        print("   ⚠️ Looks short. If your video is ~22s, OpenPose likely skipped frames.")
        print("   Re-run OpenPose with: --process_real_time 0 --frame_step 1 --display 0 --render_pose 0")

if __name__ == "__main__":
    main()
