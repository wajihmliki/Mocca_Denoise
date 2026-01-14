import json
import numpy as np

def load_rows(path):
    with open(path, "r") as f:
        d = json.load(f)
    rows = d.get("rows", d.get("rows_preview", []))
    return rows

def main():
    cls_rows = load_rows("router_eval_classical.json")
    cnn_rows = load_rows("router_eval_cnn.json")

    cls = {r["sample"]: r for r in cls_rows if r.get("executed") and r.get("noise_valid")}
    cnn = {r["sample"]: r for r in cnn_rows if r.get("executed") and r.get("noise_valid")}

    shared = sorted(set(cls.keys()) & set(cnn.keys()))
    if not shared:
        print("No shared noise-valid samples found.")
        return

    cls_vals = np.array([cls[k]["noise_ratio"] for k in shared], dtype=float)
    cnn_vals = np.array([cnn[k]["noise_ratio"] for k in shared], dtype=float)

    # lower is better (noise_out smaller relative to noise_raw)
    diff = cnn_vals - cls_vals

    def stats(x):
        return {
            "mean": float(np.mean(x)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
        }

    print("Shared samples:", len(shared))
    print("Classical noise_ratio:", stats(cls_vals))
    print("CNN noise_ratio:", stats(cnn_vals))
    print("CNN - Classical (negative is better):", stats(diff))
    print("Win-rate (CNN < Classical):", float(np.mean(cnn_vals < cls_vals)))

if __name__ == "__main__":
    main()
