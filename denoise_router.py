"""
denoise_router.py

Central routing logic for MOCCA denoising.

Responsibilities:
- Select denoising method (classical / cnn)
- Run denoiser
- Run QC
- Enforce fallback on QC failure
- Emit audit metadata

This is the ONLY entry point for denoising in the pipeline.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from mocca_denoise import denoise_signal
from qc_metrics import qc_area_check, qc_pass
from cnn_denoiser import CNNDenoiser


class DenoiseRouter:
    """
    Central denoising controller.
    """

    def __init__(
        self,
        default_method: str = "classical",
        classical_preset: str = "medium",
        qc_enabled: bool = True,
        qc_threshold: float = 0.005,  # 0.5%
        cnn_enabled: bool = False,
        device: str = "cpu"
    ):
        """
        Parameters
        ----------
        default_method : str
            "classical" or "cnn"
        classical_preset : str
            Preset for classical denoiser
        qc_enabled : bool
            Whether QC gating is active
        qc_threshold : float
            Max allowed relative area error
        cnn_enabled : bool
            Safety switch for CNN (must be True to allow CNN usage)
        device : str
            Torch device for CNN
        """

        if default_method not in ("classical", "cnn"):
            raise ValueError("default_method must be 'classical' or 'cnn'")

        self.default_method = default_method
        self.classical_preset = classical_preset
        self.qc_enabled = qc_enabled
        self.qc_threshold = qc_threshold
        self.cnn_enabled = cnn_enabled

        # CNN scaffold (identity forward for now)
        #self.cnn = CNNDenoiser(device=device)
        #self.cnn = CNNDenoiser(model_path="models/cnn_medium.pt")
        self.cnn = None
        if self.cnn_enabled:
            self.cnn = CNNDenoiser(model_path="models/cnn_medium.pt", device=device)

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _run_classical(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        y_out = denoise_signal(y, preset=self.classical_preset)
        meta = {
            "method": "classical",
            "preset": self.classical_preset
        }
        return y_out, meta


    def _run_cnn(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        y_out = self.cnn.denoise(y)
        if self.cnn is None:
            raise RuntimeError("CNN requested but cnn_enabled=False or model not initialized")

        meta = {
            "method": "cnn",
            "model_path": getattr(self.cnn, "model_path", None),
            "device": getattr(self.cnn, "device", None),
            "ready": getattr(self.cnn, "ready", False),
        }
        return y_out, meta



    def _run_qc(
        self,
        t: np.ndarray,
        y_raw: np.ndarray,
        y_dn: np.ndarray,
        peak_windows: List[Dict]
    ) -> Dict:
        qc = qc_area_check(t, y_raw, y_dn, peak_windows)

        passed = qc_pass(
            qc,
            max_relative_error=self.qc_threshold
        )

        return {
            "passed": passed,
            "threshold": self.qc_threshold,
            "results": qc
        }


    # -----------------------------
    # Public API
    # -----------------------------

    def denoise(
        self,
        t: np.ndarray,
        y_raw: np.ndarray,
        peak_windows: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Denoise a chromatogram with routing + QC + fallback.

        Returns
        -------
        y_final : np.ndarray
            Final denoised signal
        meta : dict
            Full audit metadata
        """

        if peak_windows is None:
            peak_windows = []

        meta: Dict = {
            "router": {
                "default_method": self.default_method,
                "cnn_enabled": self.cnn_enabled,
                "qc_enabled": self.qc_enabled
            }
        }

        # -----------------------------
        # Step 1: choose method
        # -----------------------------

        method = self.default_method

        if method == "cnn" and not self.cnn_enabled:
            method = "classical"
            meta["router"]["note"] = "cnn disabled, falling back to classical"

        # -----------------------------
        # Step 2: run denoiser
        # -----------------------------

        if method == "cnn":
            y_dn, dn_meta = self._run_cnn(y_raw)
        else:
            y_dn, dn_meta = self._run_classical(y_raw)

        if method == "cnn" and not dn_meta.get("ready", False):
            meta["router"]["note"] = "cnn selected but model not ready; falling back to classical"
            y_dn, dn_meta = self._run_classical(y_raw)
            method = "classical"

        meta["denoiser"] = dn_meta
        meta["selected_method"] = method

        # -----------------------------
        # Step 3: QC gate
        # -----------------------------

        if self.qc_enabled and peak_windows:
            qc_meta = self._run_qc(t, y_raw, y_dn, peak_windows)
            meta["qc"] = qc_meta

            if not qc_meta["passed"]:
                # ---- fallback ----
                y_fb, fb_meta = self._run_classical(y_raw)

                meta["fallback"] = {
                    "used": True,
                    "reason": "qc_failed",
                    "fallback_method": "classical",
                    "fallback_meta": fb_meta
                }

                return y_fb, meta

        else:
            meta["qc"] = {
                "skipped": True,
                "reason": "qc_disabled_or_no_peaks"
            }

        # -----------------------------
        # Step 4: success
        # -----------------------------

        meta["fallback"] = {"used": False}
        return y_dn, meta
