import os
import json
import numpy as np

from denoise_router import DenoiseRouter
from evaluate_denoising import load_chromatogram_csv, load_peaks_csv, baseline_mask


def run_router_evaluation(input_dir, output_path, method="classical", preset="medium"):
    """
    Evaluate the *router pipeline* instead of calling denoise_signal directly.
    method: "classical" or "cnn"
    """
    router = DenoiseRouter(
        default_method=method,
        classical_preset=preset,
        qc_enabled=True,
        qc_threshold=0.005,
        cnn_enabled=(method == "cnn")
    )

    chrom_files = [f for f in os.listdir(input_dir) if f.endswith("_chromatogram.csv")]
    rows = []
    for chrom in chrom_files:
        base = chrom.replace("_chromatogram.csv", "")
        chrom_path = os.path.join(input_dir, chrom)
        peaks_path = os.path.join(input_dir, f"{base}_peaks.csv")

        t, y_raw = load_chromatogram_csv(chrom_path)
        peaks = load_peaks_csv(peaks_path)

        y_out, meta = router.denoise(t, y_raw, peaks)

        # baseline noise ratio
        mask = baseline_mask(t, peaks)
        noise_raw = float(np.std(y_raw[mask]))
        noise_out = float(np.std(y_out[mask]))
        noise_ratio = noise_raw / noise_out if noise_out > 0 else float("inf")

        # summarize QC if present
        qc_passed = meta.get("qc", {}).get("passed", None)
        fallback_used = meta.get("fallback", {}).get("used", False)

        rows.append({
            "sample": base,
            "method": method,
            "preset": preset,
            "qc_passed": qc_passed,
            "fallback_used": fallback_used,
            "noise_ratio": noise_ratio,
        })

    # summary stats
    noise_ratios = [r["noise_ratio"] for r in rows if np.isfinite(r["noise_ratio"])]
    fallback_rate = float(np.mean([1.0 if r["fallback_used"] else 0.0 for r in rows]))

    summary = {
        "method": method,
        "preset": preset,
        "num_files": len(rows),
        "fallback_rate": fallback_rate,
        "noise_ratio_mean": float(np.mean(noise_ratios)) if noise_ratios else None,
        "noise_ratio_p95": float(np.percentile(noise_ratios, 95)) if noise_ratios else None,
        "rows": rows
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
