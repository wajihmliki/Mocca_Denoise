import os
import csv
import json
import numpy as np

from mocca_denoise import denoise_signal
from qc_metrics import qc_area_check


TIME_COL = "Time (min)"
INTENSITY_COL = "Intensity (mAU)"


def load_chromatogram_csv(path):
    t, y = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row[TIME_COL]))
            y.append(float(row[INTENSITY_COL]))
    return np.array(t), np.array(y)


def load_peaks_csv(path):
    peaks = []
    if not os.path.exists(path):
        return peaks

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            peaks.append({
                "begin_time": float(row["Begin Time (min)"]),
                "end_time": float(row["End Time (min)"])
            })
    return peaks


def baseline_mask(t, peak_windows):
    mask = np.ones_like(t, dtype=bool)
    for pw in peak_windows:
        mask &= ~((t >= pw["begin_time"]) & (t <= pw["end_time"]))
    return mask


def evaluate_file(chrom_path, peaks_path, preset):
    t, y_raw = load_chromatogram_csv(chrom_path)
    peak_windows = load_peaks_csv(peaks_path)

    y_dn = denoise_signal(y_raw, preset=preset)

    # ---- area QC ----
    qc = qc_area_check(t, y_raw, y_dn, peak_windows)
    area_errors = [r["relative_error"] for r in qc]

    # ---- noise reduction ----
    mask = baseline_mask(t, peak_windows)
    noise_raw = np.std(y_raw[mask])
    noise_dn = np.std(y_dn[mask])
    noise_ratio = noise_raw / noise_dn if noise_dn > 0 else np.inf

    return {
        "area_errors": area_errors,
        "noise_ratio": noise_ratio
    }
