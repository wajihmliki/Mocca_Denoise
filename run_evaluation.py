import os
import json
import numpy as np
from evaluate_denoising import evaluate_file


PRESETS = ["gentle", "medium", "strong"]


def run_evaluation(input_dir, output_path):
    results = {p: [] for p in PRESETS}

    chrom_files = [
        f for f in os.listdir(input_dir)
        if f.endswith("_chromatogram.csv")
    ]

    for preset in PRESETS:
        for chrom in chrom_files:
            base = chrom.replace("_chromatogram.csv", "")
            chrom_path = os.path.join(input_dir, chrom)
            peaks_path = os.path.join(input_dir, f"{base}_peaks.csv")

            r = evaluate_file(chrom_path, peaks_path, preset)
            results[preset].append(r)

    summary = {}
    for preset, items in results.items():
        all_errors = [e for r in items for e in r["area_errors"]]
        noise_ratios = [r["noise_ratio"] for r in items]

        summary[preset] = {
            "num_files": len(items),
            "area_error_mean": float(np.mean(all_errors)),
            "area_error_p95": float(np.percentile(all_errors, 95)),
            "area_error_max": float(np.max(all_errors)),
            "noise_ratio_mean": float(np.mean(noise_ratios))
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
