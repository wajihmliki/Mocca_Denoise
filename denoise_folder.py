import os
import csv
import json
import numpy as np
from datetime import datetime

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
    "Response",
]

PEAK_BEGIN_COL_CANDIDATES = [
    "Begin Time (min)",
    "Begin Time",
    "begin_time",
    "start",
]

PEAK_END_COL_CANDIDATES = [
    "End Time (min)",
    "End Time",
    "end_time",
    "end",
]


# -------------------------------
# Utilities
# -------------------------------

def pick_column(fieldnames, candidates, kind, path):
    for c in candidates:
        if c in fieldnames:
            return c
    raise KeyError(
        f"{path}: cannot find {kind} column. Tried {candidates}. Found headers: {fieldnames}"
    )


def infer_peaks_path_from_chrom(chrom_path: str) -> str:
    """
    Given .../<base>_chromatogram.csv  -> .../<base>_peaks.csv
    """
    base = os.path.basename(chrom_path)
    if not base.endswith("_chromatogram.csv"):
        raise ValueError(f"{chrom_path}: expected *_chromatogram.csv")
    base = base.replace("_chromatogram.csv", "")
    return os.path.join(os.path.dirname(chrom_path), f"{base}_peaks.csv")


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
                f"{path}: expected time column '{TIME_COL}'. Found: {reader.fieldnames}"
            )

        inten_col = pick_column(
            reader.fieldnames,
            INTENSITY_COL_CANDIDATES,
            "intensity",
            path,
        )

        for row in reader:
            if row.get(TIME_COL, "") == "" or row.get(inten_col, "") == "":
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

        bcol = pick_column(reader.fieldnames, PEAK_BEGIN_COL_CANDIDATES, "begin_time", path)
        ecol = pick_column(reader.fieldnames, PEAK_END_COL_CANDIDATES, "end_time", path)

        for row in reader:
            if row.get(bcol, "") == "" or row.get(ecol, "") == "":
                continue
            peaks.append({"begin_time": float(row[bcol]), "end_time": float(row[ecol])})

    return peaks


# -------------------------------
# Core logic
# -------------------------------

def denoise_one_sample(chrom_path: str, out_dir: str, preset: str = "medium"):
    """
    Denoise exactly one chromatogram file.
    Peaks file is inferred automatically if present in same folder.
    """
    base = os.path.basename(chrom_path).replace("_chromatogram.csv", "")
    peaks_path = infer_peaks_path_from_chrom(chrom_path)

    # Load chromatogram
    t, y_raw, inten_col = load_chromatogram_csv(chrom_path)

    # Load peaks (optional)
    peak_windows = load_peaks_csv(peaks_path) if os.path.exists(peaks_path) else []

    # Denoise
    y_dn = denoise_signal(y_raw, preset=preset)

    # QC (only if peaks exist)
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
                "chrom_path": chrom_path,
                "peaks_path": peaks_path if os.path.exists(peaks_path) else None,
                "qc_pass": passed,
                "qc_results": qc,
            },
            f,
            indent=2,
        )


def denoise_folder(in_dir: str, out_dir: str, preset: str = "medium"):
    """
    Batch denoise a folder recursively.
    - Never crashes the full run for one bad file
    - Logs failures to _errors.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    err_path = os.path.join(out_dir, "_errors.csv")
    write_header = not os.path.exists(err_path)

    import glob
    chrom_paths = glob.glob(os.path.join(in_dir, "**", "*_chromatogram.csv"), recursive=True)
    if len(chrom_paths) == 0:
        raise RuntimeError(f"No chromatogram files found in {in_dir}")

    n_ok, n_fail = 0, 0

    with open(err_path, "a", newline="", encoding="utf-8") as f_err:
        writer = csv.writer(f_err)
        if write_header:
            writer.writerow(["timestamp", "chrom_path", "error_type", "error_message"])

        for chrom_path in chrom_paths:
            try:
                denoise_one_sample(chrom_path, out_dir, preset=preset)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                writer.writerow([datetime.now().isoformat(), chrom_path, type(e).__name__, str(e)])

    print(f"[DONE] preset={preset} ok={n_ok} fail={n_fail} (see {err_path})")
