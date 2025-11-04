import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your beats file
csv_path = "data/beats/emily_mov_1_short.beats.csv"

# First line has tempo; skip that row when loading times
with open(csv_path, "r") as f:
    header = f.readline().strip().split(",")
    global_bpm = float(header[1])
    df = pd.read_csv(f)

# Compute instantaneous BPM between beats
beat_times = df["beat_time_s"].values
intervals = np.diff(beat_times)
inst_bpm = 60.0 / intervals  # BPM = 60 / seconds per beat
inst_times = beat_times[1:]  # time at which each interval ends

# --- Plot ---
plt.figure(figsize=(10,5))
plt.plot(inst_times, inst_bpm, 'o-', label="Instantaneous BPM", alpha=0.7)
plt.axhline(global_bpm, color="red", linestyle="--", label=f"Global tempo = {global_bpm:.1f} BPM")
plt.xlabel("Time (s)")
plt.ylabel("BPM")
plt.title("Tempo Stability Across Song")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.savefig("plots/bpm_plot.png", dpi=150)

