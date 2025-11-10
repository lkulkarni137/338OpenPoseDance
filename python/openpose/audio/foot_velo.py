#!/usr/bin/env python
import argparse, json, re, csv
from pathlib import Path
import numpy as np
import cv2
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

def extract_series_single(pts_per_frame, idx, conf_min=0.2):
    """Return y-series for exactly one keypoint index (np.nan when low confidence)."""
    y = []
    for pts in pts_per_frame:
        if pts is None or idx is None:
            y.append(np.nan); continue
        if idx >= len(pts):
            y.append(np.nan); continue
        _, yy, c = pts[idx]
        y.append(yy if (c >= conf_min) else np.nan)
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

# ---------- Peak candidates (stance plateaus) ----------
def detect_candidates_maxima(
    y, fps, frame_nums,
    smooth_win=21, smooth_poly=3,
    base_prominence_px=4.0,
    distance_frames=None,
    height_percentile=70,
    min_width_s=0.04,
):
    """
    Find candidate foot-down plateaus as local MAXIMA of smoothed y.
    Returns: peaks_idx, y_s, y_s_d1(px/s), y_s_d2(px/s^2)
    """
    y = interpolate_nans(y)
    if np.all(np.isnan(y)):
        return np.array([], dtype=int), np.array([]), np.array([]), np.array([])

    # smoothing window (odd)
    win = smooth_win if smooth_win % 2 else smooth_win + 1
    win = min(win, max(3, (len(y)//2)*2 + 1))
    y_s = savgol_filter(y, window_length=win, polyorder=min(smooth_poly, win-1), mode="interp")

    # First and second derivatives (SavGol)
    p_for_deriv = max(3, min(smooth_poly, win-1))
    y_s_d1 = savgol_filter(y, window_length=win, polyorder=p_for_deriv, deriv=1, delta=1.0, mode="interp") * fps
    y_s_d2 = savgol_filter(y, window_length=win, polyorder=p_for_deriv, deriv=2, delta=1.0, mode="interp") * (fps**2)

    # thresholds
    q25, q75 = np.nanpercentile(y_s, [25, 75])
    iqr = max(1.0, q75 - q25)
    prom = max(base_prominence_px, 0.4 * iqr)
    hfloor = np.nanpercentile(y_s, height_percentile)

    if distance_frames is None:
        distance_frames = max(1, int(round(0.30 * fps)))  # candidate spacing

    min_width_frames = max(1, int(round(min_width_s * fps)))

    peaks, _ = find_peaks(
        y_s,
        prominence=prom,
        distance=distance_frames,
        height=hfloor,
        width=min_width_frames
    )
    return peaks, y_s, y_s_d1, y_s_d2

# ---------- Contact onset refinement ----------
def refine_contact_from_peak(i_peak, y_s, y_s_d1, y_s_d2, fps,
                             search_back_s=0.12,
                             within_px=8.0,
                             v_min_px_s=50.0,
                             v_quiet_px_s=18.0,
                             use_accel=False):
    """
    Given a plateau peak index, search backwards up to `search_back_s`
    to find the **contact onset** frame (earliest frame near-peak & quiet).
    """
    n = len(y_s)
    win = max(1, int(round(search_back_s * fps)))
    j0 = max(0, i_peak - win)
    y_peak = y_s[i_peak]

    candidate = i_peak
    for j in range(i_peak, j0-1, -1):
        near_ground = (y_peak - y_s[j]) <= within_px
        quiet = abs(y_s_d1[j]) <= v_quiet_px_s
        if j-1 >= 0:
            was_moving_down = (y_s_d1[j-1] > v_min_px_s)
        else:
            was_moving_down = False

        ok = near_ground and quiet and was_moving_down
        if use_accel:
            ok = ok and (y_s_d2[j] <= 0.0 or y_s_d2[j] < np.nanpercentile(y_s_d2[j0:i_peak+1], 30))
        if ok:
            candidate = j
    return candidate

# ---------- Merge toe & heel candidates (one step = earliest of the two) ----------
def merge_toe_heel_contacts(idx_toe, idx_heel, frame_nums, fps_time, merge_window_s=0.10):
    """
    Given contact indices for toe and heel (same side), merge any pair whose
    times are within merge_window_s into a single event at the EARLIER time/index.
    Returns merged sorted unique indices.
    """
    events = []
    for i in idx_toe:
        t = frame_nums[i] / float(fps_time)
        events.append((t, i))
    for i in idx_heel:
        t = frame_nums[i] / float(fps_time)
        events.append((t, i))
    if not events:
        return np.array([], dtype=int)

    events.sort(key=lambda x: x[0])
    merged = []
    for t, i in events:
        if not merged:
            merged.append((t, i))
            continue
        t_prev, i_prev = merged[-1]
        if (t - t_prev) <= merge_window_s:
            # collapse into one: keep earlier time/index
            if t < t_prev:
                merged[-1] = (t, i)
            # else keep previous
        else:
            merged.append((t, i))
    return np.array([i for (_, i) in merged], dtype=int)

# ---------- Lift-off hysteresis (prevent duplicates during long stance) ----------
def filter_strikes_by_lift(indices, y_s_combined, frame_nums, fps,
                           min_interval_s=0.25,
                           lift_drop_px=14.0,
                           lift_drop_iqr_frac=0.30):
    """
    Hysteresis on a combined foot trace (max of toe & heel y).
    Accept a new strike only after sufficient 'lift' (drop in y) since last.
    """
    if len(indices) == 0:
        return indices

    q25, q75 = np.nanpercentile(y_s_combined, [25, 75])
    iqr = max(1.0, q75 - q25)
    drop_thr = max(lift_drop_px, float(lift_drop_iqr_frac) * iqr)

    accepted = []
    last_i = None
    last_t = None
    last_val = None
    idx2t = {i: frame_nums[i] / float(fps) for i in indices}

    for i in sorted(indices):
        t = idx2t[i]
        val = y_s_combined[i]
        if last_i is None:
            accepted.append(i)
            last_i, last_t, last_val = i, t, val
            continue
        if (t - last_t) < min_interval_s:
            continue
        seg = y_s_combined[last_i:i+1] if i > last_i else np.array([np.nan])
        min_between = np.nanmin(seg) if seg.size > 0 else np.nan
        lifted = (np.isfinite(min_between) and (last_val - min_between) >= drop_thr)
        if lifted:
            accepted.append(i)
            last_i, last_t, last_val = i, t, val
    return np.array(accepted, dtype=int)

def main():
    ap = argparse.ArgumentParser(
        description="Detect foot-strike **contact onset** using toe & heel (merged), with lift-off hysteresis."
    )
    ap.add_argument("--json_dir", required=True, help="Directory with OpenPose *_keypoints.json")
    ap.add_argument("--fps", type=float, required=True, help="JSON sampling FPS used for detection")
    ap.add_argument("--time_from_video", default=None,
                    help="Optional path to the .mov/.mp4 to stamp times using the video's FPS (prevents drift).")
    ap.add_argument("--model", default="body25", help="body25 (default) or coco")
    ap.add_argument("--conf_min", type=float, default=0.2)
    ap.add_argument("--side", choices=["left","right","both"], default="both")
    ap.add_argument("--out", required=True, help="Output CSV")

    # Peak selection controls (candidates)
    ap.add_argument("--prominence_px", type=float, default=4.0,
                    help="Baseline prominence (auto-raised ~0.4*IQR).")
    ap.add_argument("--height_pct", type=int, default=70,
                    help="Only accept peaks above this percentile of smoothed y.")
    ap.add_argument("--min_width_s", type=float, default=0.04,
                    help="Minimum peak width in seconds.")
    ap.add_argument("--min_interval_s", type=float, default=0.25,
                    help="Minimum time between same-side strikes (seconds).")
    ap.add_argument("--smooth_win", type=int, default=21,
                    help="Savitzky–Golay window length (odd).")
    ap.add_argument("--smooth_poly", type=int, default=3,
                    help="Savitzky–Golay polyorder (>=3 recommended).")

    # Contact refinement
    ap.add_argument("--contact_search_back_s", type=float, default=0.12,
                    help="Search window back from plateau peak to find contact onset (s).")
    ap.add_argument("--contact_within_px", type=float, default=8.0,
                    help="Within this many px of peak y to consider 'near ground'.")
    ap.add_argument("--v_min_px_s", type=float, default=50.0,
                    help="Velocity threshold for 'moving down' before contact (px/s).")
    ap.add_argument("--v_quiet_px_s", type=float, default=18.0,
                    help="Velocity threshold for 'near still' at contact (px/s).")
    ap.add_argument("--use_accel", action="store_true",
                    help="Also require small negative acceleration at contact.")

    # Lift-off hysteresis
    ap.add_argument("--lift_drop_px", type=float, default=14.0,
                    help="Required drop (px) from last strike peak before next strike allowed.")
    ap.add_argument("--lift_drop_iqr_frac", type=float, default=0.30,
                    help="Required drop as fraction of IQR(y). Effective drop is max(px, frac*IQR).")

    # Toe/heel merge window
    ap.add_argument("--toe_heel_merge_s", type=float, default=0.10,
                    help="Merge toe & heel contacts within this time window into one (earliest kept).")

    args = ap.parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # FPS used for timestamps (prevents drift if different from --fps)
    fps_time = float(args.fps)
    if args.time_from_video:
        cap = cv2.VideoCapture(args.time_from_video)
        if cap.isOpened():
            vfps = cap.get(cv2.CAP_PROP_FPS)
            if vfps and vfps > 0:
                fps_time = float(vfps)
                print(f"[timing] Using video FPS for timestamps: {fps_time:.6f} (from {args.time_from_video})")
            else:
                print("[timing] Could not read FPS from video; falling back to --fps for timestamps.")
        else:
            print("[timing] Could not open video; falling back to --fps for timestamps.")
        if cap:
            cap.release()
    if abs(fps_time - float(args.fps)) > 0.2:
        est_drift = abs(fps_time - float(args.fps)) * 20.0  # seconds drift over ~20s
        print(f"[warn] video_fps ({fps_time:.3f}) != json_fps ({float(args.fps):.3f}); "
              f"expect ~{est_drift:.2f}s drift over 20s if you don't use video FPS.")

    # ---- Setup keypoints ----
    kpmap, model_name = choose_model_map(args.model)
    if model_name != "body25":
        # COCO has no toes/heels; fall back to ankles on both channels (works but less precise)
        idxs = {
            "right": {"toe": kpmap.get("R_ANKLE"), "heel": kpmap.get("R_ANKLE")},
            "left":  {"toe": kpmap.get("L_ANKLE"), "heel": kpmap.get("L_ANKLE")},
        }
    else:
        idxs = {
            "right": {"toe": kpmap.get("R_BIGTOE"), "heel": kpmap.get("R_HEEL")},
            "left":  {"toe": kpmap.get("L_BIGTOE"), "heel": kpmap.get("L_HEEL")},
        }

    # ---- Load JSONs ----
    files = list(json_dir.glob("*_keypoints.json"))
    if not files:
        raise SystemExit(f"[error] no JSON files in {json_dir}")
    files.sort(key=parse_frame_num)
    frame_nums = [parse_frame_num(p) for p in files]
    pts_per_frame = [load_openpose_frame(p) for p in files]

    # ---- Process per side ----
    strikes = []
    for side in ["left", "right"]:
        if args.side != "both" and side != args.side:
            continue

        # Separate series: toe & heel
        y_toe  = extract_series_single(pts_per_frame, idxs[side]["toe"],  conf_min=args.conf_min)
        y_heel = extract_series_single(pts_per_frame, idxs[side]["heel"], conf_min=args.conf_min)

        # Candidates for each (plateaus)
        peaks_toe,  y_toe_s,  y_toe_d1,  y_toe_d2  = detect_candidates_maxima(
            y_toe,  fps=float(args.fps), frame_nums=frame_nums,
            smooth_win=args.smooth_win, smooth_poly=args.smooth_poly,
            base_prominence_px=args.prominence_px,
            distance_frames=max(1, int(round(args.min_interval_s * float(args.fps)))),
            height_percentile=args.height_pct, min_width_s=args.min_width_s
        )
        peaks_heel, y_heel_s, y_heel_d1, y_heel_d2 = detect_candidates_maxima(
            y_heel, fps=float(args.fps), frame_nums=frame_nums,
            smooth_win=args.smooth_win, smooth_poly=args.smooth_poly,
            base_prominence_px=args.prominence_px,
            distance_frames=max(1, int(round(args.min_interval_s * float(args.fps)))),
            height_percentile=args.height_pct, min_width_s=args.min_width_s
        )

        # Refine to contact onsets (search back)
        contact_toe  = [refine_contact_from_peak(ip, y_toe_s,  y_toe_d1,  y_toe_d2,  float(args.fps),
                                                 args.contact_search_back_s, args.contact_within_px,
                                                 args.v_min_px_s, args.v_quiet_px_s, args.use_accel)
                        for ip in peaks_toe]
        contact_heel = [refine_contact_from_peak(ip, y_heel_s, y_heel_d1, y_heel_d2, float(args.fps),
                                                 args.contact_search_back_s, args.contact_within_px,
                                                 args.v_min_px_s, args.v_quiet_px_s, args.use_accel)
                        for ip in peaks_heel]
        contact_toe  = np.array(sorted(set(contact_toe)), dtype=int)   # dedup
        contact_heel = np.array(sorted(set(contact_heel)), dtype=int)

        # Merge toe & heel into single-step events (earliest kept)
        merged_idx = merge_toe_heel_contacts(contact_toe, contact_heel, frame_nums,
                                             fps_time=fps_time, merge_window_s=args.toe_heel_merge_s)

        # Combined foot trace for hysteresis: take per-frame MAX(y_toe_s, y_heel_s)
        if y_toe_s.size == 0 and y_heel_s.size == 0:
            continue
        if y_toe_s.size == 0:
            y_comb = y_heel_s
        elif y_heel_s.size == 0:
            y_comb = y_toe_s
        else:
            y_comb = np.nanmax(np.vstack([y_toe_s, y_heel_s]), axis=0)

        # Lift-off gating so one step == one strike
        kept_idx = filter_strikes_by_lift(
            merged_idx, y_comb, frame_nums, float(args.fps),
            min_interval_s=args.min_interval_s,
            lift_drop_px=args.lift_drop_px,
            lift_drop_iqr_frac=args.lift_drop_iqr_frac
        )

        # Emit strikes (timestamps from fps_time)
        for i in kept_idx:
            t = frame_nums[i] / float(fps_time)
            strikes.append((t, side, int(frame_nums[i]), float(y_comb[i])))

    # ---- Output ----
    strikes.sort(key=lambda r: r[0])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "side", "frame_num", "y_smoothed"])
        for t, side, fnum, ypix in strikes:
            w.writerow([f"{t:.6f}", side, fnum, f"{ypix:.2f}"])

    print(f"✅ wrote {len(strikes)} strikes → {out_csv}")
    if frame_nums:
        last_f = frame_nums[-1]
        est_dur = last_f / float(args.fps)
        print(f"   JSONs cover frames 0…{last_f} ⇒ ~{est_dur:.2f}s at {float(args.fps):.3f} fps")

if __name__ == "__main__":
    main()
