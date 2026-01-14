"""
cnn_denoiser.py

CNN denoiser scaffold for MOCCA.
Currently implements an identity forward pass (NO signal modification).

This file defines the final interface and data flow.
Training / real inference will be enabled later.
"""

from typing import Optional, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class CNNDenoiser(nn.Module):
    """
    CNN-based denoiser scaffold.

    IMPORTANT:
    - Forward pass is identity
    - No learnable behavior is active
    - Safe to integrate into pipeline
    """

    def __init__(
        self,
        normalize: bool = True,
        device: str = "cpu"
    ):
        super().__init__()

        self.normalize = normalize
        self.device = device

        # ---- placeholder layers (NOT USED YET) ----
        # These exist to lock architecture and interfaces
        if nn is not None:
            self.placeholder = nn.Identity()
        else:
            self.placeholder = None

        self.eval()  # force eval mode


    def _normalize(self, y: np.ndarray):
        """
        Normalize signal to zero mean / unit std.
        Returns normalized signal + stats for inversion.
        """
        mean = float(np.mean(y))
        std = float(np.std(y))
        if std == 0:
            std = 1.0
        return (y - mean) / std, mean, std


    def _denormalize(self, y: np.ndarray, mean: float, std: float):
        return y * std + mean


    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Identity forward pass.
        """
        return y


    def denoise(
        self,
        y: np.ndarray,
        return_metadata: bool = False
    ):
        """
        Public API used by the pipeline.

        Parameters
        ----------
        y : np.ndarray
            1D chromatogram intensity array
        return_metadata : bool
            If True, returns metadata dict

        Returns
        -------
        y_out : np.ndarray
            Denoised signal (currently identical to input)
        meta : dict (optional)
            Metadata for audit trail
        """

        if y.ndim != 1:
            raise ValueError("CNNDenoiser expects 1D signal")

        meta: Dict[str, Optional[str]] = {
            "method": "cnn",
            "status": "identity_forward",
            "normalized": False,
            "note": "CNN scaffold active, no learning applied"
        }

        y_proc = y.astype(np.float32, copy=True)

        # ---- optional normalization (NO EFFECT since identity) ----
        if self.normalize:
            y_proc, mean, std = self._normalize(y_proc)
            meta["normalized"] = True
        else:
            mean, std = 0.0, 1.0

        # ---- torch pass (identity) ----
        if torch is not None:
            yt = torch.from_numpy(y_proc).to(self.device)
            with torch.no_grad():
                yt_out = self.forward(yt)
            y_out = yt_out.cpu().numpy()
        else:
            y_out = y_proc

        # ---- de-normalize ----
        if self.normalize:
            y_out = self._denormalize(y_out, mean, std)

        # Guarantee exact shape & type
        y_out = y_out.astype(np.float64)

        if return_metadata:
            return y_out, meta
        else:
            return y_out
