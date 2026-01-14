import os, glob, json
import numpy as np

from evaluate_denoising import load_chromatogram_csv, load_peaks_csv, baseline_mask
from mocca_denoise import denoise_signal
from qc_metrics import qc_area_check


def run_evaluation(in_dir: str, out_path: str, preset: str = "medium"):
    chrom_files = glob.glob(os.path.join(in_dir, "**", "*_chromatogram.csv"), recursive=True)
    if len(chrom_files) == 0:
        raise RuntimeError(f"No chromatogram files found under {in_dir}")

    rows = []
    for chrom_path in chrom_files:
        base = os.path.basename(chrom_path).replace("_chromatogram.csv", "")
        peaks_path = os.path.join(os.path.dirname(chrom_path), f"{base}_peaks.csv")

        try:
            t, y_raw = load_chromatogram_csv(chrom_path)
            peaks = load_peaks_csv(peaks_path) if os.path.exists(peaks_path) else []

            y_dn = denoise_signal(y_raw, preset=preset)

            # baseline noise proxy
            mask = baseline_mask(t, peaks) if peaks else np.ones_like(t, dtype=bool)
            noise_raw = float(np.std(y_raw[mask])) if np.any(mask) else float("nan")
            noise_dn  = float(np.std(y_dn[mask]))  if np.any(mask) else float("nan")

            if not np.isfinite(noise_raw) or not np.isfinite(noise_dn) or noise_dn <= 1e-12:
                noise_ratio = float("nan")
            else:
                noise_ratio = noise_raw / noise_dn

            # QC area distortion if peaks exist
            qc = qc_area_check(t, y_raw, y_dn, peaks) if peaks else None

            rows.append({
                "sample": base,
                "chrom_path": chrom_path,
                "preset": preset,
                "noise_ratio": noise_ratio,
                "qc": qc,
            })

        except Exception as e:
            rows.append({
                "sample": base,
                "chrom_path": chrom_path,
                "preset": preset,
                "error": f"{type(e).__name__}: {e}",
            })

    # summarize
    ok = [r for r in rows if "error" not in r]
    #noise = [r["noise_ratio"] for r in ok if np.isfinite(r["noise_ratio"])]
    noise = [r["noise_ratio"] for r in ok if np.isfinite(r.get("noise_ratio", np.nan))]

    summary = {
        "preset": preset,
        "num_total": len(rows),
        "num_ok": len(ok),
        "num_fail": len(rows) - len(ok),
        "noise_ratio_mean": float(np.mean(noise)) if noise else None,
        "noise_ratio_p50": float(np.percentile(noise, 50)) if noise else None,
        "noise_ratio_p95": float(np.percentile(noise, 95)) if noise else None,
        "rows": rows[:200],  # keep output size reasonable
        "note": "rows truncated to first 200; increase if needed"
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[EVAL] preset={preset} total={summary['num_total']} ok={summary['num_ok']} fail={summary['num_fail']}")
    return summary


if __name__ == "__main__":
    run_evaluation("input", "phase3_classical_summary.json", preset="medium")
