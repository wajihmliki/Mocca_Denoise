import os
import csv
import json
import numpy as np

from mocca_denoise import denoise_signal
from qc_metrics import qc_area_check, qc_pass


# -------------------------------
# Locked / accepted ADE headers
# -------------------------------

TIME_COL = "Time (min)"

INTENSITY_COL_CANDIDATES = [
    "Intensity (mAU)",
    "Signal",
    "Intensity",
    "Absorbance",
    "Response"
]

PEAK_BEGIN_COL_CANDIDATES = [
    "Begin Time (min)",
    "Begin Time",
    "begin_time",
    "start"
]

PEAK_END_COL_CANDIDATES = [
    "End Time (min)",
    "End Time",
    "end_time",
    "end"
]


# -------------------------------
# Small utilities
# -------------------------------

def pick_column(fieldnames, candidates, kind, path):
    for c in candidates:
        if c in fieldnames:
            return c
    raise KeyError(
        f"{path}: cannot find {kind} column. "
        f"Tried {candidates}. Found headers: {fieldnames}"
    )


# -------------------------------
# Loaders
# -------------------------------

def load_chromatogram_csv(path):
    t, y = [], []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row")

        if TIME_COL not in reader.fieldnames:
            raise KeyError(
                f"{path}: expected time column '{TIME_COL}'. "
                f"Found: {reader.fieldnames}"
            )

        inten_col = pick_column(
            reader.fieldnames,
            INTENSITY_COL_CANDIDATES,
            "intensity",
            path
        )

        for row in reader:
            if row[TIME_COL] == "" or row[inten_col] == "":
                continue
            t.append(float(row[TIME_COL]))
            y.append(float(row[inten_col]))

    if len(t) < 2:
        raise ValueError(f"{path}: too few data points")

    return np.array(t, dtype=np.float64), np.array(y, dtype=np.float64), inten_col


def load_peaks_csv(path):
    peaks = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row")

        bcol = pick_column(
            reader.fieldnames,
            PEAK_BEGIN_COL_CANDIDATES,
            "begin_time",
            path
        )
        ecol = pick_column(
            reader.fieldnames,
            PEAK_END_COL_CANDIDATES,
            "end_time",
            path
        )

        for row in reader:
            if row[bcol] == "" or row[ecol] == "":
                continue
            peaks.append({
                "begin_time": float(row[bcol]),
                "end_time": float(row[ecol])
            })

    return peaks


# -------------------------------
# Core logic
# -------------------------------

def denoise_one_sample(chrom_path, peaks_path, out_dir, preset):
    base = os.path.basename(chrom_path).replace("_chromatogram.csv", "")

    # Load chromatogram
    t, y_raw, inten_col = load_chromatogram_csv(chrom_path)

    # Load peaks (optional)
    peak_windows = (
        load_peaks_csv(peaks_path)
        if os.path.exists(peaks_path)
        else []
    )

    # Denoise
    y_dn = denoise_signal(y_raw, preset=preset)

    # QC
    qc = None
    passed = True
    if peak_windows:
        qc = qc_area_check(t, y_raw, y_dn, peak_windows)
        passed = qc_pass(qc)

    os.makedirs(out_dir, exist_ok=True)

    # Write denoised chromatogram
    out_chrom = os.path.join(out_dir, f"{base}_chromatogram_denoised.csv")
    with open(out_chrom, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([TIME_COL, inten_col])
        for ti, yi in zip(t, y_dn):
            writer.writerow([float(ti), float(yi)])

    # Write QC report
    out_qc = os.path.join(out_dir, f"{base}_denoise_qc.json")
    with open(out_qc, "w") as f:
        json.dump(
            {
                "sample": base,
                "preset": preset,
                "qc_pass": passed,
                "qc_results": qc
            },
            f,
            indent=2
        )


def denoise_folder(in_dir, out_dir, preset="medium"):
    os.makedirs(out_dir, exist_ok=True)

    chrom_files = [
        f for f in os.listdir(in_dir)
        if f.endswith("_chromatogram.csv")
    ]

    if not chrom_files:
        raise RuntimeError(
            f"No '*_chromatogram.csv' files found in {in_dir}. "
            f"Files present: {os.listdir(in_dir)}"
        )

    for chrom in chrom_files:
        base = chrom.replace("_chromatogram.csv", "")
        chrom_path = os.path.join(in_dir, chrom)
        peaks_path = os.path.join(in_dir, f"{base}_peaks.csv")

        denoise_one_sample(
            chrom_path=chrom_path,
            peaks_path=peaks_path,
            out_dir=out_dir,
            preset=preset
        )
