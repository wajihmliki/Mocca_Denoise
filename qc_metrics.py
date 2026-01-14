import numpy as np


def integrate_area(t, y, t0, t1):
    """
    Numerical integration inside a peak window.
    """
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 2:
        return 0.0

    # NumPy ≥2.0 compatible
    return np.trapezoid(y[mask], t[mask])


def qc_area_check(t, y_raw, y_dn, peak_windows):
    """
    Compare peak areas before vs after denoising.
    """
    results = []

    for pw in peak_windows:
        t0 = pw["begin_time"]
        t1 = pw["end_time"]

        a_raw = integrate_area(t, y_raw, t0, t1)
        a_dn  = integrate_area(t, y_dn,  t0, t1)

        rel_err = 0.0 if a_raw == 0 else abs(a_dn - a_raw) / a_raw

        results.append({
            "begin_time": t0,
            "end_time": t1,
            "area_raw": a_raw,
            "area_denoised": a_dn,
            "relative_error": rel_err
        })

    return results


def qc_pass(qc_results, max_rel_error=0.02):
    """
    Decide whether denoising is acceptable.
    """
    for r in qc_results:
        if r["relative_error"] > max_rel_error:
            return False
    return True
