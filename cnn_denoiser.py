import os
import numpy as np
import torch
import torch.nn as nn


class Small1DCNN(nn.Module):
    def __init__(self, channels=32, k=9):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, k, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, k, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, 1, k, padding=pad),
        )

    def forward(self, x):
        return self.net(x)


class CNNDenoiser:
    """
    CPU-safe 1D CNN denoiser with per-sample normalization.
    Loads a trained model checkpoint (state_dict) if present.
    """

    def __init__(self, model_path="models/cnn_medium.pt", device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Small1DCNN().to(self.device)
        self.model.eval()

        self.ready = False
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.ready = True

    def denoise(self, y_raw: np.ndarray) -> np.ndarray:
        """
        Returns denoised signal with the same length as input.
        If model isn't ready, returns input unchanged.
        """
        y_raw = np.asarray(y_raw, dtype=np.float32)

        if not self.ready:
            return y_raw.copy()

        m = float(np.mean(y_raw))
        s = float(np.std(y_raw))
        if s <= 1e-12:
            return y_raw.copy()

        x = (y_raw - m) / s
        x_t = torch.from_numpy(x[None, None, :]).to(self.device)  # (1,1,L)

        with torch.no_grad():
            y_hat = self.model(x_t).cpu().numpy()[0, 0, :]

        y_dn = y_hat * s + m
        return y_dn.astype(np.float32)
