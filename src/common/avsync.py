# common/avsync.py
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class AVSyncResult:
    ok: bool
    reason: str
    lag_ms: float
    peak_corr: float
    samples_used: int

class AVSyncChecker:
    """
    Build two time-aligned 1D signals:
      - mouth_open_series: (timestamp_sec, value in [0,1])
      - audio_energy_series: (timestamp_sec, normalized energy)
    Then compute normalized cross-correlation over a sliding window.

    Decide "real" when |lag_ms| <= max_allowed_lag_ms and peak_corr >= min_corr.
    """
    def __init__(self,
                 window_seconds: float = 3.0,
                 max_allowed_lag_ms: float = 120.0,
                 min_corr: float = 0.45,
                 resample_hz: float = 50.0):
        assert window_seconds > 0
        assert resample_hz > 0
        self.window_seconds = window_seconds
        self.max_allowed_lag_ms = max_allowed_lag_ms
        self.min_corr = min_corr
        self.resample_hz = resample_hz

        self._mouth = deque()   # (t, v)
        self._audio = deque()   # (t, v)
        self._maxlen = int(window_seconds * resample_hz) + 8

    def push_mouth(self, t_sec: float, mouth_open: float):
        if not np.isfinite(mouth_open):
            return
        mouth_open = float(np.clip(mouth_open, 0.0, 1.0))
        self._mouth.append((float(t_sec), mouth_open))
        if len(self._mouth) > self._maxlen:
            self._mouth.popleft()

    def push_audio_energy(self, t_sec: float, energy: float):
        if not np.isfinite(energy):
            return
        # Robust log/normalize
        energy = float(np.log1p(max(0.0, energy)))
        self._audio.append((float(t_sec), energy))
        if len(self._audio) > self._maxlen:
            self._audio.popleft()

    def _resample(self, series):
        if len(series) < 3:
            return None, None
        times = np.array([t for t, _ in series], dtype=np.float64)
        vals  = np.array([v for _, v in series], dtype=np.float64)
        t0 = max(times[0], times[-1] - self.window_seconds)
        t1 = times[-1]
        if t1 <= t0:
            return None, None
        grid = np.arange(t0, t1, 1.0/self.resample_hz, dtype=np.float64)
        if grid.size < 8:
            return None, None
        # linear interpolate
        resampled = np.interp(grid, times, vals)
        # standardize
        s = resampled.std()
        if s < 1e-6:
            return grid, None
        resampled = (resampled - resampled.mean()) / s
        return grid, resampled

    def compute(self) -> AVSyncResult:
        # get aligned, standardized series
        t_m, m = self._resample(self._mouth)
        t_a, a = self._resample(self._audio)
        if (m is None) or (a is None) or (t_m is None) or (t_a is None):
            return AVSyncResult(False, "insufficient_data", 0.0, 0.0, 0)

        # align by time overlap
        t0 = max(t_m[0], t_a[0])
        t1 = min(t_m[-1], t_a[-1])
        if t1 - t0 < 0.8:  # need at least ~0.8s overlap
            return AVSyncResult(False, "insufficient_overlap", 0.0, 0.0, 0)

        # cut both to overlap window
        sel_m = (t_m >= t0) & (t_m <= t1)
        sel_a = (t_a >= t0) & (t_a <= t1)
        m2 = m[sel_m]
        a2 = a[sel_a]

        # re-grid to identical length (protect against off-by-one)
        N = min(len(m2), len(a2))
        if N < 20:
            return AVSyncResult(False, "too_short", 0.0, 0.0, 0)
        m2 = m2[-N:]
        a2 = a2[-N:]

        # cross-correlation (full) → lag where corr is max
        corr = np.correlate(m2, a2, mode='full')
        lags = np.arange(-N+1, N, dtype=np.int32)
        peak_idx = int(np.argmax(corr))
        peak_corr = float(corr[peak_idx] / (N))  # simple normalization

        # convert lag (in samples) to ms
        lag_samples = lags[peak_idx]
        lag_sec = lag_samples / self.resample_hz
        lag_ms = float(lag_sec * 1000.0)

        # decision
        ok_lag = abs(lag_ms) <= self.max_allowed_lag_ms
        ok_corr = peak_corr >= self.min_corr
        ok = bool(ok_lag and ok_corr)

        reason = "ok" if ok else (
            "low_corr" if not ok_corr else "excessive_lag"
        )
        return AVSyncResult(ok, reason, lag_ms, peak_corr, N)
