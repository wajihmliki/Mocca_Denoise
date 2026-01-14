import os, glob, json
import numpy as np

from denoise_router import DenoiseRouter
from evaluate_denoising import load_chromatogram_csv, load_peaks_csv, baseline_mask


def peaks_to_windows(peaks_df):
    """
    Convert MOCCA peaks CSV into [(t_start, t_end), ...].
    Returns empty list if peaks_df is None or empty.
    """
    if peaks_df is None or len(peaks_df) == 0:
        return []

    windows = []
    for _, r in peaks_df.iterrows():
        if "start_time" in r and "end_time" in r:
            windows.append((float(r["start_time"]), float(r["end_time"])))
        elif "Start" in r and "End" in r:
            windows.append((float(r["Start"]), float(r["End"])))
    return windows


def run_router_evaluation(input_dir, output_path, method="classical", preset="medium"):
    """
    Evaluate the router end-to-end over a directory tree (recursive).
    method: "classical" or "cnn"
    """
    router = DenoiseRouter(
        default_method=method,
        classical_preset=preset,
        qc_enabled=True,
        qc_threshold=0.005,
        cnn_enabled=(method == "cnn"),
    )

    chrom_paths = glob.glob(os.path.join(input_dir, "**", "*_chromatogram.csv"), recursive=True)
    if len(chrom_paths) == 0:
        raise RuntimeError(f"No chromatogram files found under {input_dir}")

    rows = []
    for chrom_path in chrom_paths:
        base = os.path.basename(chrom_path).replace("_chromatogram.csv", "")
        peaks_path = os.path.join(os.path.dirname(chrom_path), f"{base}_peaks.csv")

        try:
            # Your loader returns 3 values (t, y, intensity_col)
            #t, y_raw, _ = load_chromatogram_csv(chrom_path)
            out = load_chromatogram_csv(chrom_path)
            if len(out) == 2:
                t, y_raw = out
            elif len(out) == 3:
                t, y_raw, _ = out
            else:
                raise ValueError(f"Unexpected return from load_chromatogram_csv: {len(out)} values")

            #peaks = load_peaks_csv(peaks_path) if os.path.exists(peaks_path) else []

            #y_out, meta = router.denoise(t, y_raw, peak_windows=peaks)
            peaks_df = load_peaks_csv(peaks_path) if os.path.exists(peaks_path) else None
            peak_windows = peaks_to_windows(peaks_df)

            y_out, meta = router.denoise(t, y_raw, peak_windows=peak_windows)

            # baseline noise ratio (NaN-safe)
            #mask = baseline_mask(t, peaks) if peaks else np.ones_like(t, dtype=bool)
            # baseline noise ratio (NaN-safe)
            mask = baseline_mask(t, peak_windows) if peak_windows else np.ones_like(t, dtype=bool)
            if mask is None or np.sum(mask) < 10:
                mask = np.ones_like(t, dtype=bool)

            noise_raw = float(np.std(y_raw[mask])) if np.any(mask) else float("nan")
            noise_out = float(np.std(y_out[mask])) if np.any(mask) else float("nan")

            if (not np.isfinite(noise_raw)) or (not np.isfinite(noise_out)) or noise_out <= 1e-12:
                noise_ratio = float("nan")
            else:
                noise_ratio = noise_raw / noise_out


            qc_passed = meta.get("qc", {}).get("passed", None)
            fallback_used = meta.get("fallback", {}).get("used", False)

            rows.append({
                "sample": base,
                "chrom_path": chrom_path,
                "method": method,
                "preset": preset,
                "executed": True,                 # router succeeded
                "qc_passed": qc_passed,
                "fallback_used": fallback_used,
                "noise_ratio": noise_ratio,       # may be NaN
                "noise_valid": bool(np.isfinite(noise_ratio)),
            })


        except Exception as e:
            rows.append({
                "sample": base,
                "chrom_path": chrom_path,
                "method": method,
                "preset": preset,
                "error": f"{type(e).__name__}: {e}",
            })
        # Crash
        #except Exception as e:
        #    print("\n=== EVALUATION CRASH ===")
        #    print("File:", chrom_path)
        #    print("Exception:", type(e).__name__)
        #    print("Message:", e)
        #    raise

    ok = [r for r in rows if "error" not in r]
    executed = [r for r in rows if r.get("executed", False)]
    noise_valid = [r for r in executed if r.get("noise_valid", False)]

    fallback_rate = float(np.mean([1.0 if r.get("fallback_used", False) else 0.0 for r in executed])) if executed else None
    noise_vals = [r["noise_ratio"] for r in noise_valid]

    summary = {
        "method": method,
        "preset": preset,
        "num_total": len(rows),
        "num_executed": len(executed),
        "num_exec_fail": len(rows) - len(executed),

        # noise metric applicability
        "num_noise_valid": len(noise_valid),
        "noise_valid_rate": (len(noise_valid) / len(executed)) if executed else None,

        # noise statistics (computed only on valid subset)
        "noise_ratio_mean": float(np.mean(noise_vals)) if len(noise_vals) > 0 else None,
        "noise_ratio_p50": float(np.percentile(noise_vals, 50)) if len(noise_vals) > 0 else None,
        "noise_ratio_p95": float(np.percentile(noise_vals, 95)) if len(noise_vals) > 0 else None,

        # safety
        "fallback_rate": fallback_rate,

        # preview rows
        "rows": rows,
        #"rows_preview": rows[:200],
        #"note": "rows truncated to first 200; increase if needed"
    }



    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    #print(f"[ROUTER EVAL] method={method} preset={preset} total={summary['num_total']} ok={summary['num_ok']} fail={summary['num_fail']} fallback_rate={summary['fallback_rate']}")
    print(
        f"[ROUTER EVAL] method={method} preset={preset} "
        f"total={summary['num_total']} executed={summary['num_executed']} exec_fail={summary['num_exec_fail']} "
        f"noise_valid={summary['num_noise_valid']} fallback_rate={summary['fallback_rate']}"
    )

    return summary


if __name__ == "__main__":
    run_router_evaluation("input", "router_eval_classical.json", method="classical", preset="medium")
    run_router_evaluation("input", "router_eval_cnn.json", method="cnn", preset="medium")
