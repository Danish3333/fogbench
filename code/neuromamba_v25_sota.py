CODE_VERSION = "v25.57.0"  # bumped on any change that alters splits/checkpoints
TRAINING_COMPAT_VERSIONS = frozenset({"v25.42.4", "v25.42.5"})
try:
    import neuromamba_v25_patches as _patches
    _PATCHES_IMPORT_ERROR = None
except Exception as _e_patches:  # pragma: no cover
    _PATCHES_IMPORT_ERROR = repr(_e_patches)
    class _InlineGradientGuard:
        """v25.53.1 (review E1): the default-on NaN/grad-spike safety net can no
        longer crash a fresh checkout that lacks neuromamba_v25_patches.py.
        This is a faithful inline copy of the patches-module GradientGuard
        contract: rolling p99 ceiling = max(hard, 3x p99 of recent FINITE
        norms); non-finite loss/grad or over-ceiling batch -> SKIP update and
        restore last-finite model+optimizer state BEFORE optimizer.step()."""
        HARD_CEILING = 1000.0
        HISTORY_LEN = 200
        SNAPSHOT_EVERY = 20

        def __init__(self, hard_ceiling: float = HARD_CEILING, dynamic_cap=None):
            self.hard_ceiling = float(hard_ceiling)
            self.dynamic_cap = None if dynamic_cap is None else float(dynamic_cap)
            self.history = []
            self.last_model_state = None
            self.last_opt_state = None
            self.n_aborted = 0
            self.n_total = 0
            self._since_snapshot = 0

        def dynamic_ceiling(self):
            if self.dynamic_cap is not None:
                return self.dynamic_cap
            if len(self.history) < 10:
                return self.hard_ceiling
            return max(self.hard_ceiling,
                       float(np.percentile(self.history, 99.0)) * 3.0)

        def snapshot(self, model, optimizer):
            self.last_model_state = {k: v.detach().clone()
                                     for k, v in model.state_dict().items()}
            self.last_opt_state = optimizer.state_dict()

        def check_and_maybe_abort(self, model, optimizer, loss, grad_norm):
            self.n_total += 1
            finite_loss = bool(torch.isfinite(loss)) if torch.is_tensor(loss) \
                else bool(np.isfinite(loss))
            finite_grad = bool(np.isfinite(grad_norm))
            ceiling = self.dynamic_ceiling()
            if finite_loss and finite_grad and float(grad_norm) <= ceiling:
                self.history.append(float(grad_norm))
                if len(self.history) > self.HISTORY_LEN:
                    self.history.pop(0)
                self._since_snapshot += 1
                if self._since_snapshot >= self.SNAPSHOT_EVERY:
                    self.snapshot(model, optimizer)
                    self._since_snapshot = 0
                return False
            self.n_aborted += 1
            print(f"[grad-guard v25.53.1-inline] ABORT batch "
                  f"({self.n_aborted}/{self.n_total} so far)", flush=True)
            if self.last_model_state is not None:
                model.load_state_dict(self.last_model_state)
                optimizer.load_state_dict(self.last_opt_state)
            optimizer.zero_grad(set_to_none=True)
            return True

    class _MissingPatches:
        GradientGuard = _InlineGradientGuard
        SubjectFirstEventSampler = None

        @staticmethod
        def conditional_subtype_loss(*_a, **_k):
            raise RuntimeError("neuromamba_v25_patches.py unavailable: " + str(_PATCHES_IMPORT_ERROR))

        @staticmethod
        def event_count_class_weights(*_a, **_k):
            raise RuntimeError("neuromamba_v25_patches.py unavailable: " + str(_PATCHES_IMPORT_ERROR))
    _patches = _MissingPatches()

import argparse
import os
import sys
import json
import time
import traceback
import math
import warnings
import hashlib
import pickle
import queue
import threading
import random as _random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import defaultdict, deque  # v25.41: deque for the adaptive-clip history
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy  # v25.14c: external-adaptation model clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.isotonic import IsotonicRegression
from scipy.signal import butter, filtfilt, lfilter, resample_poly

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No positive class found in y_true")

class NpEncoder(json.JSONEncoder):
    """v25.57: module-level JSON encoder for numpy scalars/arrays. Previously
    defined ONLY as a local inside run_cv_main, so reevaluate_frozen_fold's
    fold-record persistence (json.dump(..., cls=NpEncoder)) hit NameError ->
    every evaluate-only reevaluation silently failed to save its fold record
    ("could not persist reevaluated fold" -- the result was lost)."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return json.JSONEncoder.default(self, o)

print("[schema] SCHEMA_V25_LOADED", flush=True)

VALID_SOURCE_TYPES = {"defog", "tdcsfog", "xuanwu", "weargait", "daphnet", "fogstar", "notype", "notype_pseudo"}
VALID_UNIT_TRANSFORMS = {"native_g", "mps2_to_g", "adc_to_g", "unknown"}

def _early_flag(name, default="0"):
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "t", "yes", "y", "on")

N_CHANNELS = int(os.environ.get("NEUROMAMBA_N_CHANNELS",
                                "17" if _early_flag("NEUROMAMBA_USE_DIFF_FEATURES", "0") else
                                ("13" if _early_flag("NEUROMAMBA_USE_TURN_FEATURES", "1") else "11")))
if N_CHANNELS not in (4, 11, 13, 17, 20):
    raise ValueError(f"NEUROMAMBA_N_CHANNELS must be 4, 11, 13, 17 or 20, got {N_CHANNELS}")
RAW_SENSOR_CHANNELS_READ = ["AccV", "AccML", "AccAP"]  # accel only -- TLVMc CSVs carry no gyro
RAW_SENSOR_CHANNELS_IGNORED = ["GyrV", "GyrML", "GyrAP"]  # not present in TLVMc raw CSVs
print(f"[schema] channel contract: N_CHANNELS={N_CHANNELS} "
      + ("20 = 17 + 3 Morlet-wavelet channels (freeze sub-bands 3-5/5-8 Hz + onset-transient energy -- SH-targeted, v25.39)"
         if N_CHANNELS == 20 else
         ("17 = 13 + 3 jerk (per-axis diff) + magnitude cumsum (2nd-place/Mo&Chan features)"
          if N_CHANNELS == 17 else
         "13 = 3 acc + magnitude + 7 FoG features + 2 accel-derived turn-rate features"
         if N_CHANNELS == 13 else
         "11 = 3 acc + magnitude + 7 FoG-native features" if N_CHANNELS == 11 else
         "4 = 3 acc + magnitude (8th-place winner format)")),
      flush=True)

@dataclass
class FoGSequence:
    """Canonical FoG sequence contract (v25: adds subtype_observed).

    subtype_observed[T,3] bool: True = this subtype channel is *annotated*
    (positive OR confirmed-negative) at this frame; False = *unknown / not
    annotated* for that channel. This is the single most important v25 fix:
    it lets the subtype loss and Kaggle mAP stop treating "unknown" frames
    (e.g. external binary-only cohorts) as hard negatives.
    """
    signal: np.ndarray
    y_subtype: np.ndarray
    subject_id: str
    record_id: str
    source_type: str
    observed: Optional[np.ndarray] = None
    label_weight: Optional[np.ndarray] = None
    subtype_observed: Optional[np.ndarray] = None  # NEW v25 [T,3] bool
    med_state: float = 0.5  # v25.6: 1.0 = ON medication, 0.0 = OFF, 0.5 = unknown
    fs_hz: float = 100.0
    unit_transform: str = "unknown"
    strides: float = 0.01
    position_tag: str = ""

    def __post_init__(self):
        self.signal = np.asarray(self.signal, dtype=np.float32)
        self.y_subtype = np.asarray(self.y_subtype, dtype=np.float32)
        if self.observed is None:
            self.observed = np.ones(len(self.signal), dtype=bool)
        else:
            self.observed = np.asarray(self.observed, dtype=bool)
        if self.label_weight is None:
            self.label_weight = np.ones(len(self.signal), dtype=np.float32)
        else:
            self.label_weight = np.asarray(self.label_weight, dtype=np.float32)
        if self.subtype_observed is None:
            self.subtype_observed = np.ones((len(self.signal), 3), dtype=bool)
        else:
            self.subtype_observed = np.asarray(self.subtype_observed, dtype=bool)
        self.source_type = str(self.source_type)
        self.subject_id = str(self.subject_id)
        self.record_id = str(self.record_id)
        self.unit_transform = str(self.unit_transform)
        self.med_state = float(self.med_state)
        self.fs_hz = float(self.fs_hz)
        self.strides = float(self.strides)
        self.position_tag = str(self.position_tag)
        self._validate()

    def _validate(self):
        T, C = self.signal.shape
        errors = []
        if T < 10:
            errors.append(f"signal too short: {T} frames (min 10)")
        if np.any(np.isnan(self.signal)) or np.any(np.isinf(self.signal)):
            errors.append("signal contains NaN/Inf")
        if self.y_subtype.ndim != 2 or self.y_subtype.shape[1] != 3:
            errors.append(f"y_subtype must be [T,3], got {self.y_subtype.shape}")
        elif self.y_subtype.shape[0] != T:
            errors.append(f"signal length {T} != y_subtype length {self.y_subtype.shape[0]}")
        if np.any((self.y_subtype < 0) | (self.y_subtype > 1)):
            errors.append("y_subtype values outside [0,1]")
        if len(self.observed) != T:
            errors.append(f"observed length {len(self.observed)} != signal length {T}")
        if len(self.label_weight) != T:
            errors.append(f"label_weight length {len(self.label_weight)} != signal length {T}")
        elif np.any((self.label_weight < 0) | (self.label_weight > 1)):
            errors.append("label_weight outside [0,1]")
        else:
            bad = (self.label_weight > 0) & (~self.observed)
            if bad.any():
                errors.append(f"{int(bad.sum())} frames have label_weight>0 but observed=False")
        if self.subtype_observed.shape != (T, 3):
            errors.append(f"subtype_observed must be [{T},3], got {self.subtype_observed.shape}")
        else:
            for k in range(3):
                sub_pos = self.y_subtype[:, k] > 0.5
                fog = self.y_fog > 0.5
                violate = sub_pos & ~fog
                if violate.any():
                    errors.append(f"subtype {k}: {int(violate.sum())} frames have subtype_positive but y_fog==0")
        if self.source_type not in VALID_SOURCE_TYPES:
            errors.append(f"unknown source_type '{self.source_type}'")
        if self.position_tag not in ("", "lower_back", "trunk", "shank"):
            errors.append(f"unknown position_tag '{self.position_tag}'")
        if self.unit_transform not in VALID_UNIT_TRANSFORMS:
            errors.append(f"unknown unit_transform '{self.unit_transform}'")
        if errors:
            raise ValueError(f"FoGSequence validation failed for {self.record_id}: " + "; ".join(errors))

    @property
    def y_fog(self):
        return (self.y_subtype.max(axis=1) > 0.5).astype(np.float32)

    @property
    def labels(self):
        return self.y_fog.astype(np.int64)

    @property
    def train_mask(self):
        return self.observed.astype(np.float32)

    @property
    def subtype_mask(self):
        """v25: per-channel observed mask [T,3] bool."""
        return self.subtype_observed

    @property
    def n_frames(self):
        return self.signal.shape[0]

    @property
    def n_channels(self):
        return self.signal.shape[1]

    def to_dict(self):
        return {"subject_id": self.subject_id, "record_id": self.record_id,
                "source_type": self.source_type, "n_frames": self.n_frames,
                "n_channels": self.n_channels, "fs_hz": self.fs_hz,
                "unit_transform": self.unit_transform,
                "n_observed": int(self.observed.sum()), "n_unobserved": int((~self.observed).sum()),
                "n_fog_frames": int(self.y_fog.sum()),
                "n_subtype_observed": [int(self.subtype_observed[:, k].sum()) for k in range(3)],
                "n_subtype_positive": [int((self.y_subtype[:, k] > 0.5).sum()) for k in range(3)]}

def validate_sequence(seq):
    seq._validate()
    return True

def derive_y_fog(y_subtype):
    return (y_subtype.max(axis=1) > 0.5).astype(np.float32)

def assert_all_subtypes_present(y_subtype, source):
    assert y_subtype.ndim == 2 and y_subtype.shape[1] == 3
    for k, name in enumerate(["StartHesitation", "Turn", "Walking"]):
        if y_subtype[:, k].sum() == 0:
            print(f"[schema] WARNING [{source}] subtype '{name}' has zero positive frames", flush=True)

print("[common] COMMON_V25_LOADED", flush=True)

CANON_FS = 100.0
STRIDE_S = 1.0 / CANON_FS
SEQUENCE_LEN_FRAMES = int(os.environ.get("NEUROMAMBA_SEQ_LEN", "12000"))
G_STANDARD = 9.80665

def _causal_filter_mode():
    """v25-FIX (causal): honor NEUROMAMBA_CAUSAL_FILTER for streaming deployment."""
    return _flag("NEUROMAMBA_CAUSAL_FILTER", "0")

def unit_diagnostic(acc, label):
    mag = np.linalg.norm(acc, axis=1)
    p = {p_: float(np.percentile(mag, p_)) for p_ in (5, 50, 95)}
    print(f"[unit-diag] {label}: |acc| p5={p[5]:.3f} p50={p[50]:.3f} p95={p[95]:.3f}", flush=True)
    return p

def gravity_align_raw(acc, fc_hz=0.3, fs=100.0):
    if len(acc) < 10:
        return acc
    try:
        b, a = butter(2, fc_hz / (fs / 2.0), btype="low")
        if _causal_filter_mode():
            return (acc - lfilter(b, a, acc, axis=0) + acc.mean(axis=0, keepdims=True)).astype(np.float32)
        gravity = filtfilt(b, a, acc, axis=0)
        return (acc - gravity + gravity.mean(axis=0, keepdims=True)).astype(np.float32)
    except Exception as _e:
        print(f"[preproc v25.57 WARN] gravity_align_raw failed ({_e}) -> "
              f"returning UNFILTERED signal", flush=True)
        return acc

def bandpass_filter(acc, fs, low=0.5, high=20.0):
    if len(acc) < 30:
        return acc
    try:
        nyq = fs / 2.0
        b, a = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="band")
        if _causal_filter_mode():
            return lfilter(b, a, acc, axis=0).astype(np.float32)
        return filtfilt(b, a, acc, axis=0).astype(np.float32)
    except Exception as _e:
        print(f"[preproc v25.57 WARN] bandpass_filter failed ({_e}) -> "
              f"returning UNBANDPASSED signal", flush=True)
        return acc

def compute_fog_native_features(acc3, fs_hz=CANON_FS, window_s=2.0, hop_s=0.25):
    """v25: honors causal mode (trailing-only windows + forward-fill)."""
    acc3 = np.asarray(acc3, dtype=np.float32)
    if acc3.ndim != 2 or acc3.shape[1] < 3:
        raise ValueError("compute_fog_native_features requires [T, >=3] acceleration")
    mag = np.linalg.norm(acc3[:, :3], axis=1)
    T = len(mag)
    win = max(16, int(round(window_s * fs_hz)))
    hop = max(1, int(round(hop_s * fs_hz)))
    freqs = np.fft.rfftfreq(win, d=1.0 / fs_hz)
    loco_m = (freqs >= 0.5) & (freqs < 3.0)
    freeze_m = (freqs >= 3.0) & (freqs <= 8.0)
    band_m = (freqs >= 0.5) & (freqs <= 20.0)
    window = np.hanning(win).astype(np.float32)
    centers, values = [], []
    causal = _causal_filter_mode()
    starts = range(0, max(T - win + 1, 1), hop) if not causal else range(max(0, T - win), -1, -hop)
    for start in (list(starts)[::-1] if causal else starts):
        seg = mag[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        seg_rms = float(np.sqrt(np.mean(seg ** 2)))
        seg = seg - seg.mean()
        power = np.abs(np.fft.rfft(seg * window)) ** 2
        p_loco = float(power[loco_m].sum())
        p_freeze = float(power[freeze_m].sum())
        p_band = float(power[band_m].sum())
        q = power[band_m] / max(p_band, 1e-12)
        band_power = power[band_m]
        dom_f = float(freqs[band_m][np.argmax(band_power)]) if band_power.size and band_power.max() > 0 else 0.0
        values.append([np.log(seg_rms + 1e-8), np.log1p(p_loco), np.log1p(p_freeze),
                        np.log1p(p_freeze / max(p_loco, 1e-8)),
                        p_freeze / max(p_loco + p_freeze, 1e-8),
                        -float(np.sum(q * np.log(q + 1e-12))),
                        dom_f])
        centers.append(start + win // 2)
    values = np.asarray(values, dtype=np.float32)
    out = np.empty((T, 7), dtype=np.float32)
    grid = np.arange(T)
    if len(centers) >= 2:
        for c in range(7):
            out[:, c] = np.interp(grid, centers, values[:, c])
    else:
        out[:] = values[0]
    return out

def _wavelet_transient_features(acc3, fs_hz=CANON_FS):
    """v25.39 (novel, SH-onset-targeted): hand-rolled Morlet-wavelet channels,
    no scipy/pywt dependency. Literature: continuous-wavelet representations
    beat a pure FFT freeze index for FoG because they localize TRANSIENTS in
    time -- Start Hesitation is a short onset event, so a fixed-band FIR
    freeze index misses its timing. Operates on the VERTICAL axis (AccV, the
    tremor axis): the vector magnitude rectifies and creates 2f envelope
    ripple that corrupts envelope derivatives. Channels (log1p energy):
      0: freeze SUB-band 3-5 Hz  (initiation tremor, center 4.0 Hz)
      1: freeze SUB-band 5-8 Hz  (full freezing tremor, center 5.5 Hz)
      2: onset-transient energy = positive rate-of-rise of the smoothed
         band envelope (fires at tremor ONSET only, not at offset -- the
         SH/abrupt-onset signature).
    Wired as part of the N_CHANNELS=20 contract (17 + these 3)."""
    x = np.asarray(acc3[:, 0], dtype=np.float64)  # AccV = tremor axis
    T = len(x)
    out = np.zeros((T, 3), dtype=np.float32)
    if T < 64:
        return out
    envs_s = []
    for fc in (4.0, 5.5):
        sigma = 2.0 / (2.0 * np.pi * fc)          # ~2 cycles: wider sub-band coverage
        half = max(8, int(round(4.0 * sigma * fs_hz)))
        t = np.arange(-half, half + 1) / fs_hz
        psi = np.exp(-(t ** 2) / (2.0 * sigma ** 2)) * np.exp(1j * 2 * np.pi * fc * t)
        env = np.abs(np.convolve(x, psi, mode="same"))   # analytic band envelope
        out[:, 0 if fc == 4.0 else 1] = np.log1p(env ** 2).astype(np.float32)
        k_ma = max(5, int(round(0.2 * fs_hz)))
        envs_s.append(np.convolve(env, np.ones(k_ma) / k_ma, mode="same"))
    e_sum = envs_s[0] + envs_s[1]
    d = np.maximum(np.diff(e_sum, prepend=e_sum[0]), 0.0)  # onset-only (rising edge)
    k = max(3, int(round(0.05 * fs_hz)))
    kk = np.ones(k) / k
    out[:, 2] = np.log1p(np.convolve(d * 10.0, kk, mode="same")).astype(np.float32)
    return out

def _turn_rate_features(acc3, fs_hz=CANON_FS):
    """v25.9c: yaw-rate proxy from accelerometer heading. TLVMc has NO gyroscope
    (raw CSVs carry AccV/ML/AP only), yet FoG-during-turn is defined by ROTATION
    ARREST + 3-8 Hz tremor -- the accel heading direction changes during turns,
    so |d heading/dt| is a rotation-rate proxy. Channels: smoothed turn rate
    [deg/s] and log1p(turn rate). This is the signal-level fix for the defog
    clinical target (Turn-dominated FoG)."""
    ap = acc3[:, 1].astype(np.float64)   # horizontal axes: ML, AP
    ml = acc3[:, 2].astype(np.float64)
    heading = np.unwrap(np.arctan2(ap, ml))
    tr = np.abs(np.gradient(heading)) * fs_hz * 180.0 / np.pi  # deg/s
    win = max(5, int(round(0.5 * fs_hz)))
    k = np.ones(win) / win
    tr_s = np.convolve(tr, k, mode="same").astype(np.float32)
    tr_log = np.log1p(tr_s).astype(np.float32)
    return tr_s[:, None], tr_log[:, None]

def assemble_input_features(acc3, fs_hz=CANON_FS):
    """Honors the global N_CHANNELS contract. 11 = 3 axes + magnitude + 7
    FoG-native features (default). 13 = +2 accel-derived turn-rate channels
    (NEUROMAMBA_USE_TURN_FEATURES=1). 17 = 13 + 3 per-axis jerk (first
    difference) + magnitude cumsum (2nd-place features; Mo & Chan 2023: jerk
    features +7.5% test mAP -- their winning Set C). 4 = 3 axes + magnitude
    only (the exact 8th-place-winner input format)."""
    acc3 = np.asarray(acc3, dtype=np.float32)
    if acc3.ndim != 2 or acc3.shape[1] != 3:
        raise ValueError(f"expected 3 physical axes, got {acc3.shape}")
    mag = np.linalg.norm(acc3, axis=1, keepdims=True).astype(np.float32)
    if N_CHANNELS == 4:
        out = np.concatenate([acc3, mag], axis=1).astype(np.float32)
    else:
        feats = compute_fog_native_features(acc3, fs_hz=fs_hz)
        out = np.concatenate([acc3, mag, feats], axis=1).astype(np.float32)
        if N_CHANNELS in (13, 17, 20):
            tr_s, tr_log = _turn_rate_features(acc3, fs_hz=fs_hz)
            out = np.concatenate([out, tr_s, tr_log], axis=1).astype(np.float32)
        if N_CHANNELS in (17, 20):
            jerk = np.diff(acc3, axis=0, prepend=acc3[:1]).astype(np.float32)
            cs = np.cumsum(mag[:, 0]).astype(np.float32)
            cs = (cs - cs.mean()) / (cs.std() + 1e-8)
            out = np.concatenate([out, jerk, cs[:, None]], axis=1).astype(np.float32)
        if N_CHANNELS == 20:
            out = np.concatenate([out, _wavelet_transient_features(acc3, fs_hz=fs_hz)],
                                 axis=1).astype(np.float32)
    if out.shape[1] != N_CHANNELS or not np.isfinite(out).all():
        raise RuntimeError(f"invalid {N_CHANNELS}-channel input assembly")
    return out

def per_sample_normalize(acc3, scale=None):
    """v25-FIX-8: per-recording z-score on each accel axis (1st-place + Zinxira trick).
    Stabilizes training across subjects/sensors without leaking cross-subject stats.
    v25.16c: winner-style post-scale (their z-score then x/50 defog / x/25
    tdcsfog ~= 0.02-0.04) via NEUROMAMBA_ACC_SCALE (default 1.0 = off)."""
    if scale is None:
        scale = float(os.environ.get("NEUROMAMBA_ACC_SCALE", "1.0"))
    out = np.zeros_like(acc3)
    for c in range(acc3.shape[1]):
        col = acc3[:, c]
        out[:, c] = np.nan_to_num((col - col.mean()) / (col.std() + 1e-8), nan=0.0)
    if scale != 1.0:
        out = out * scale
    return out.astype(np.float32)

def resample_to_canon(acc, fs, canon_fs=CANON_FS):
    if abs(fs - canon_fs) < 1e-6:
        return acc.astype(np.float32)
    up, down = int(round(canon_fs * 100)), int(round(fs * 100))
    g = gcd(up, down)
    up, down = up // g, down // g
    return resample_poly(acc, up, down, axis=0).astype(np.float32)

def resample_labels_nearest(y, fs, canon_fs=CANON_FS):
    if abs(fs - canon_fs) < 1e-6:
        return y.astype(np.float32)
    n_out = int(round(len(y) * canon_fs / fs))
    idx = np.clip(np.round(np.arange(n_out) * fs / canon_fs).astype(int), 0, len(y) - 1)
    return y[idx].astype(np.float32)

def merge_close_episodes(binary_labels, gap_frames=300):
    out = binary_labels.copy().astype(bool)
    idx = np.where(out)[0]
    if len(idx) < 2:
        return out
    gaps = np.diff(idx)
    for i, g in enumerate(gaps):
        if 0 < g <= gap_frames:
            out[idx[i]:idx[i + 1]] = True
    return out

def chunk_into_sequences(signal, labels, record_id, subject_id, source_type,
                          labels_subtype=None, label_weight=None, observed=None,
                          subtype_observed=None, max_len=SEQUENCE_LEN_FRAMES, med_state=0.5):
    T = len(signal)
    out = []
    n_chunks = max(1, int(np.ceil(T / max_len)))
    for i in range(n_chunks):
        s, e = i * max_len, min((i + 1) * max_len, T)
        if e - s < 10:
            continue
        y_sub = labels_subtype[s:e] if labels_subtype is not None else np.tile(labels[s:e].astype(np.float32).reshape(-1, 1), (1, 3))
        lw = label_weight[s:e] if label_weight is not None else None
        ob = observed[s:e] if observed is not None else None
        so = subtype_observed[s:e] if subtype_observed is not None else None
        try:
            seq = FoGSequence(signal=signal[s:e], y_subtype=y_sub, subject_id=subject_id,
                               record_id=f"{record_id}_c{i}", source_type=source_type,
                               observed=ob, label_weight=lw, subtype_observed=so,
                               med_state=med_state,
                               fs_hz=CANON_FS, strides=STRIDE_S)
            out.append(seq)
        except ValueError as e_:
            print(f"[common] chunk {record_id}_c{i} skipped: {e_}", flush=True)
    return out

print("[tlvmc_loader] TLVMC_LOADER_V25_LOADED", flush=True)

TLVMC_ROOT = os.environ.get("NEUROMAMBA_TLVMC_ROOT",
                             "/mnt/d/Installed/Datasets/tlvmc-parkinsons-freezing-gait-prediction")

def _get_unit_harmonize(): return _flag("NEUROMAMBA_UNIT_HARMONIZE", "1")  # v25.57: _flag parse
def _get_per_sample_norm(): return _flag("NEUROMAMBA_PER_SAMPLE_NORM", "1")  # v25: default ON; v25.57: _flag parse
def _get_soft_labels(): return _flag("NEUROMAMBA_DEFOG_SOFT_LABELS", "0")  # v25.57: _flag parse

def _load_subject_map(root):
    id_map = {}
    for prefix in ["defog", "tdcsfog"]:
        meta_csv = root / f"{prefix}_metadata.csv"
        if not meta_csv.is_file():
            continue
        try:
            meta = pd.read_csv(meta_csv)
        except Exception:
            continue
        id_col = next((c for c in meta.columns if any(p in c.lower().replace("_", "") for p in ("id", "fileid", "file"))), None)
        subj_col = next((c for c in meta.columns if any(p in c.lower().replace("_", "") for p in ("subject", "subjectid", "patient"))), None)
        if id_col and subj_col:
            for _, row in meta.iterrows():
                fid = str(row[id_col])
                if "." in fid:
                    fid = fid.rsplit(".", 1)[0]
                id_map[fid] = str(row[subj_col])
    return id_map

def _load_med_map(root):
    """v25.6: per-recording Medication state (on=1.0 / off=0.0 / unknown=0.5)
    from the TLVMc metadata CSVs -- enables medication-conditioned detection
    (FoG morphology differs ON vs OFF levodopa; clinically novel)."""
    med_map = {}
    for prefix in ["defog", "tdcsfog"]:
        meta_csv = root / f"{prefix}_metadata.csv"
        if not meta_csv.is_file():
            continue
        try:
            meta = pd.read_csv(meta_csv)
        except Exception:
            continue
        id_col = next((c for c in meta.columns if any(p in c.lower().replace("_", "") for p in ("id", "fileid", "file"))), None)
        if "Medication" not in meta.columns or id_col is None:
            continue
        for _, row in meta.iterrows():
            fid = str(row[id_col])
            if "." in fid:
                fid = fid.rsplit(".", 1)[0]
            m = str(row["Medication"]).strip().lower()
            if m in ("on", "1", "true", "yes"):
                med_map[fid] = 1.0
            elif m in ("off", "0", "false", "no"):
                med_map[fid] = 0.0
            else:
                med_map[fid] = 0.5
    n_on = sum(1 for v in med_map.values() if v == 1.0)
    n_off = sum(1 for v in med_map.values() if v == 0.0)
    if med_map:
        print(f"[tlvmc_loader] Medication map: {len(med_map)} recordings (ON={n_on}, OFF={n_off})", flush=True)
    return med_map

def _dtype_safe_bool(series):
    if series.dtype == object:
        return series.astype(str).str.strip().str.upper().isin(["TRUE", "1", "T", "YES", "Y"])
    return series.astype(bool)

SUBJECT_MAP_FALLBACKS = []  # v25.43.0 audit-6: global fallback log (mutated by load_tlvmc_sequences)

def load_tlvmc_sequences(root=None):
    """v25: builds subtype_observed per source.
      - defog: subtype_observed = tile(observed, (1,3))  -- only Task (&Valid) frames annotated
      - tdcsfog: subtype_observed = ones((T,3))          -- all annotated
      - notype: subtype_observed = zeros((T,3))          -- binary-only via Event head
    """
    if root is None:
        root = TLVMC_ROOT
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"TLVMC_ROOT does not exist: {root}\nSet NEUROMAMBA_TLVMC_ROOT or pass root=.")
    print(f"[tlvmc_loader] Loading from: {root}", flush=True)
    print(f"[tlvmc_loader] unit_harmonize={_get_unit_harmonize()} per_sample_norm={_get_per_sample_norm()} "
          f"defog_soft_labels={_get_soft_labels()}", flush=True)
    print(f"[tlvmc_loader] channel contract: reading {RAW_SENSOR_CHANNELS_READ} only "
          f"(gyro {RAW_SENSOR_CHANNELS_IGNORED} excluded by design)", flush=True)

    subject_map = _load_subject_map(root)
    print(f"[tlvmc_loader] Subject map: {len(subject_map)} entries", flush=True)
    med_map = _load_med_map(root)

    all_sequences = []
    configs = [("train/defog", "defog", 100.0), ("train/tdcsfog", "tdcsfog", 128.0), ("train/notype", "notype", 100.0)]
    unit_medians = {}

    for rel_path, source_type, native_fs in configs:
        folder = root / rel_path
        if not folder.is_dir():
            print(f"[tlvmc_loader] Skip: {folder} not found", flush=True)
            continue
        csv_files = sorted(folder.glob("*.csv"))
        print(f"[tlvmc_loader] {source_type}: {len(csv_files)} CSVs (native {native_fs} Hz)", flush=True)

        for csv_path in csv_files:
            if "metadata" in csv_path.name.lower():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                try:
                    df = pd.read_csv(csv_path, engine="python")
                except Exception as e:
                    print(f"  [WARN] Cannot read {csv_path.name}: {e}", flush=True)
                    continue
            req_cols = ["AccV", "AccML", "AccAP"]
            if not all(c in df.columns for c in req_cols):
                continue

            trim_n = int(os.environ.get("NEUROMAMBA_TRIM_EDGE_SAMPLES", "0"))
            if trim_n > 0 and len(df) > 2 * trim_n + 50:
                df = df.iloc[trim_n:-trim_n].reset_index(drop=True)

            label_weight, observed_mask = None, None
            subtype_observed_mask = None
            if source_type == "defog" and {"Valid", "Task"}.issubset(df.columns):
                valid_mask = _dtype_safe_bool(df["Valid"])
                task_mask = _dtype_safe_bool(df["Task"])
                n_before = len(df)
                if _get_soft_labels():
                    df = df[task_mask].copy()
                    w = _dtype_safe_bool(df["Valid"]).to_numpy(dtype=np.float32)
                    w = np.where(w > 0.5, 1.0, 0.25).astype(np.float32)
                    label_weight = w
                    observed_mask = (w >= 0.25)
                    print(f"  [soft-label] {csv_path.name}: kept {len(df)}/{n_before} Task rows, "
                          f"{int((w < 1).sum())} weak-negative rows", flush=True)
                else:
                    ctx = int(float(os.environ.get("NEUROMAMBA_DEFOG_CONTEXT_S", "30.0")) * native_fs)
                    if ctx > 0:
                        keep = np.convolve(task_mask.astype(np.int32),
                                           np.ones(2 * ctx + 1, dtype=np.int32), mode="same") > 0
                    else:
                        keep = task_mask
                    df = df[keep].copy()
                    w = (_dtype_safe_bool(df["Valid"]) & _dtype_safe_bool(df["Task"])).to_numpy(dtype=np.float32)
                    label_weight = w          # 1 inside Valid&Task, 0 in context margin
                    observed_mask = (w >= 0.5)
                if len(df) < 50:
                    continue
                subtype_observed_mask = np.tile(observed_mask.astype(bool).reshape(-1, 1), (1, 3))
            elif source_type == "tdcsfog":
                observed_mask = np.ones(len(df), dtype=bool)
                subtype_observed_mask = np.ones((len(df), 3), dtype=bool)
            elif source_type == "notype":
                observed_mask = np.ones(len(df), dtype=bool)
                subtype_observed_mask = np.zeros((len(df), 3), dtype=bool)
            else:
                observed_mask = np.ones(len(df), dtype=bool)
                subtype_observed_mask = np.ones((len(df), 3), dtype=bool)

            has_st = all(c in df.columns for c in ["StartHesitation", "Turn", "Walking"])
            has_event = "Event" in df.columns
            if has_st:
                st_raw = df[["StartHesitation", "Turn", "Walking"]].to_numpy(dtype=np.float32)
                labels_subtype_raw = st_raw.astype(np.float32)
                y = st_raw.max(axis=1)
            elif has_event:
                y = df["Event"].to_numpy(dtype=np.float32)
                labels_subtype_raw = np.zeros((len(y), 3), dtype=np.float32)
                labels_subtype_raw[:, 0] = y
            else:
                if source_type == "notype":
                    y = np.zeros(len(df), dtype=np.float32)
                    labels_subtype_raw = np.zeros((len(df), 3), dtype=np.float32)
                else:
                    continue

            if label_weight is not None and _get_soft_labels():
                weak = label_weight < 1.0
                y = y.copy(); y[weak] = 0.0

            acc_raw = df[req_cols].to_numpy(dtype=np.float32)
            acc_raw = np.nan_to_num(acc_raw, nan=0.0, posinf=0.0, neginf=0.0)

            unit_xform = "native_g"
            if source_type not in unit_medians:
                unit_medians[source_type] = unit_diagnostic(acc_raw, f"{source_type} raw")
            if _get_unit_harmonize() and source_type == "tdcsfog":
                acc_raw = (acc_raw / G_STANDARD).astype(np.float32)
                unit_xform = "mps2_to_g"

            acc_raw = gravity_align_raw(acc_raw, fs=native_fs)
            acc_bp = bandpass_filter(acc_raw, fs=native_fs)
            acc_resampled = resample_to_canon(acc_bp, fs=native_fs)
            acc_aligned = np.nan_to_num(acc_resampled, nan=0.0).astype(np.float32)

            if _get_per_sample_norm():
                acc_norm = per_sample_normalize(acc_aligned)
            else:
                acc_norm = acc_aligned

            signal = assemble_input_features(acc_norm, fs_hz=CANON_FS)

            y_resampled = resample_labels_nearest(y, native_fs, CANON_FS)
            if label_weight is not None:
                label_weight = resample_labels_nearest(label_weight, native_fs, CANON_FS)
            n_out = len(y_resampled)
            labels_subtype_resampled = np.zeros((n_out, 3), dtype=np.float32)
            for c in range(3):
                labels_subtype_resampled[:, c] = resample_labels_nearest(labels_subtype_raw[:, c], native_fs, CANON_FS)[:n_out]
            if observed_mask is not None:
                observed_mask = resample_labels_nearest(observed_mask.astype(np.float32), native_fs, CANON_FS).astype(bool)[:n_out]
            if subtype_observed_mask is not None:
                so_resampled = np.zeros((n_out, 3), dtype=bool)
                for c in range(3):
                    so_resampled[:, c] = resample_labels_nearest(subtype_observed_mask[:, c].astype(np.float32), native_fs, CANON_FS).astype(bool)[:n_out]
                subtype_observed_mask = so_resampled

            n = min(len(signal), len(y_resampled))
            signal = signal[:n]
            labels_raw = y_resampled[:n]
            label_weight = label_weight[:n] if label_weight is not None else np.ones(n, dtype=np.float32)
            labels_subtype_resampled = labels_subtype_resampled[:n]
            observed_mask = observed_mask[:n] if observed_mask is not None else np.ones(n, dtype=bool)
            subtype_observed_mask = subtype_observed_mask[:n] if subtype_observed_mask is not None else np.ones((n, 3), dtype=bool)

            labels = merge_close_episodes(labels_raw >= 0.5, gap_frames=300).astype(np.int64)
            record_id = csv_path.stem
            subject_id = subject_map.get(record_id)
            if subject_id is None:
                parts = record_id.split("_")
                subject_id = parts[0] if parts else record_id
                SUBJECT_MAP_FALLBACKS.append(record_id)
                print(f"[subject-map v25.43] FALLBACK for {record_id} -> {subject_id} "
                      f"(metadata missing; total fallbacks={len(SUBJECT_MAP_FALLBACKS)})", flush=True)

            for seq in chunk_into_sequences(signal, labels, record_id, subject_id, source_type,
                                             labels_subtype=labels_subtype_resampled, label_weight=label_weight,
                                             observed=observed_mask, subtype_observed=subtype_observed_mask,
                                             med_state=med_map.get(record_id, 0.5)):
                seq.unit_transform = unit_xform
                seq.fs_hz = CANON_FS
                all_sequences.append(seq)

    n_subjects = len(set(s.subject_id for s in all_sequences))
    total_frames = sum(len(s.labels) for s in all_sequences)
    total_hours = total_frames * STRIDE_S / 3600.0
    total_fog_pct = 100.0 * sum(s.labels.sum() for s in all_sequences) / max(total_frames, 1)
    print(f"[tlvmc_loader] Built {len(all_sequences)} sequences from {n_subjects} subjects", flush=True)
    print(f"[tlvmc_loader] Total: {total_hours:.1f} hours, FoG prevalence: {total_fog_pct:.1f}%", flush=True)
    if all_sequences and all_sequences[0].signal.shape[1] != N_CHANNELS:
        raise RuntimeError(f"channel contract violated: {all_sequences[0].signal.shape[1]} != {N_CHANNELS}")
    if not all_sequences:
        raise RuntimeError("No TLVMC sequences loaded. Check dataset root.")
    return all_sequences

def _tlvmc_cache_cfg(root):
    cfg = {
        "root": str(root), "unit_harmonize": _get_unit_harmonize(),
        "per_sample_norm": _get_per_sample_norm(), "soft_labels": _get_soft_labels(),
        "n_channels": N_CHANNELS, "canon_fs": CANON_FS, "seq_len": SEQUENCE_LEN_FRAMES,
        "defog_context_s": os.environ.get("NEUROMAMBA_DEFOG_CONTEXT_S", "30.0"),
        "trim_edge_samples": os.environ.get("NEUROMAMBA_TRIM_EDGE_SAMPLES", "0"),
        "causal_filter": os.environ.get("NEUROMAMBA_CAUSAL_FILTER", "0"),
        "acc_scale": os.environ.get("NEUROMAMBA_ACC_SCALE", "1.0"),
        "version": "v25.57",  # bumped: v25.57 cache-key/meta-verify rework (one rebuild)
    }
    return cfg

def _tlvmc_cache_key(root):
    cfg = _tlvmc_cache_cfg(root)
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return Path(root) / f".neuromamba_v25_cache_{h}.pkl"

def load_tlvmc_sequences_cached(root=None):
    if root is None:
        root = TLVMC_ROOT
    root = Path(root)
    cache_path = _tlvmc_cache_key(root)
    if cache_path.is_file():
        print(f"[tlvmc_loader] Loading CACHED sequences from {cache_path}", flush=True)
        try:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "meta" in obj and "seqs" in obj:
                if obj.get("meta") != _tlvmc_cache_cfg(root):
                    print("[tlvmc_loader] Cache STALE (config drift vs pickle meta) -- rebuilding from raw CSVs", flush=True)
                else:
                    seqs = obj["seqs"]
                    print(f"[tlvmc_loader] Cache hit: {len(seqs)} sequences (skipped raw CSV processing; cfg verified)", flush=True)
                    return seqs
            else:
                print("[tlvmc_loader] Legacy cache format (no meta) -- rebuilding from raw CSVs (one-time)", flush=True)
        except Exception as e:
            print(f"[tlvmc_loader] Cache load failed ({e}), rebuilding from raw CSVs", flush=True)
    seqs = load_tlvmc_sequences(root=root)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": _tlvmc_cache_cfg(root), "seqs": seqs}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[tlvmc_loader] Wrote sequence cache to {cache_path}", flush=True)
    except Exception as e:
        print(f"[tlvmc_loader] Cache write failed (non-fatal): {e}", flush=True)
    return seqs

print("[load_datasets] LOAD_DATASETS_V25_LOADED", flush=True)

def _cache_dir():
    d = Path(os.environ.get("NEUROMAMBA_CACHE_DIR", Path(__file__).resolve().parent / "neuromamba_v25_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d

def _resolve_dataset_root(env_var, linux_default, windows_default):
    root = os.environ.get(env_var)
    if root:
        return Path(root)
    for cand in (linux_default, windows_default):
        if cand and Path(cand).exists():
            return Path(cand)
    raise RuntimeError(
        f"[{env_var}] not set and no default path exists. Tried:\n"
        f"  - {linux_default}\n  - {windows_default}\n"
        f"Set {env_var}=/path/to/cohort to continue.")

SEQUENCE_CACHE_VERSION = 29  # v25.57 bump: fogstar gravity/bandpass fix (v25.53.9) must land in cached

def _seq_cache_key(dataset, root, extra=None, build_fn=None):
    cfg = {"dataset": dataset, "root": str(root), "n_channels": N_CHANNELS,
           "canon_fs": CANON_FS, "seq_len": SEQUENCE_LEN_FRAMES,
           "cache_version": SEQUENCE_CACHE_VERSION,
           "causal_filter": os.environ.get("NEUROMAMBA_CAUSAL_FILTER", "0"),
           "acc_scale": os.environ.get("NEUROMAMBA_ACC_SCALE", "1.0"),
           **(extra or {})}
    if build_fn is not None:
        try:
            import inspect as _inspect
            cfg["loader_rev"] = hashlib.sha256(
                _inspect.getsource(build_fn).encode()).hexdigest()[:12]
        except Exception:
            cfg["loader_rev"] = "unavailable"
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

def _load_sequences_cached(dataset, root, extra_cfg, build_fn):
    root = Path(root)
    key = _seq_cache_key(dataset, root, extra_cfg, build_fn=build_fn)
    _cur_cfg = {"dataset": dataset, "root": str(root), "extra": extra_cfg,
                "cache_version": SEQUENCE_CACHE_VERSION,
                "loader_rev": _seq_cache_key(dataset, root, extra_cfg, build_fn=build_fn)}
    cache_path = _cache_dir() / f"seqs_{dataset}_{key}.pkl"
    if cache_path.is_file():
        print(f"[{dataset}] Loading CACHED sequences from {cache_path}", flush=True)
        try:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "meta" in obj and "seqs" in obj:
                seqs = obj["seqs"]
                print(f"[{dataset}] Cache hit: {len(seqs)} sequences (skipped raw file processing; "
                      f"cfg verified rev={obj['meta'].get('loader_rev', '?')})", flush=True)
                return seqs
            else:
                print(f"[{dataset}] Legacy cache format (no meta) -- rebuilding from raw files (one-time)", flush=True)
        except Exception as e:
            print(f"[{dataset}] Cache load failed ({e}), rebuilding from raw files", flush=True)
    seqs = build_fn(root)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": _cur_cfg, "seqs": seqs}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{dataset}] Wrote sequence cache to {cache_path}", flush=True)
    except Exception as e:
        print(f"[{dataset}] Cache write failed (non-fatal): {e}", flush=True)
    return seqs

XUANWU_LABEL_COL = 60
XUANWU_FS = 500.0
XUANWU_ACC_COLS = {"waist": [45, 46, 47], "lshank": [31, 32, 33]}

def load_xuanwu_sequences(root=None):
    """v25: external binary-only cohort -> subtype_observed = ZEROS."""
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_XUANWU_ROOT",
            "/mnt/d/Installed/Datasets/r8gmbtv7w2-3/Filtered Data",
            r"D:\Installed\Datasets\r8gmbtv7w2-3\Filtered Data")
    root = Path(root)

    def _task_files(subj_dir):
        out = [(tf, subj_dir.name) for tf in sorted(subj_dir.glob("task_*.txt"))]
        for nd in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            for tf in sorted(nd.glob("task_*.txt")):
                out.append((tf, f"{subj_dir.name}/{nd.name}"))
        return out

    all_sequences = []
    unit_medians = {}
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for csv_path, subj_id in _task_files(subj_dir):
            try:
                df = pd.read_csv(csv_path, header=None)
            except Exception as e:
                print(f"  [xuanwu] skip {csv_path.name}: {e}", flush=True)
                continue
            if df.shape[1] <= XUANWU_LABEL_COL:
                continue
            label_raw = df.iloc[:, XUANWU_LABEL_COL].to_numpy(dtype=np.float32)
            acc = None
            for sensor in ("waist", "lshank"):
                cand = df.iloc[:, XUANWU_ACC_COLS[sensor]].to_numpy(dtype=np.float32)
                cand = np.nan_to_num(cand, nan=0.0, posinf=0.0, neginf=0.0)
                if np.all(np.abs(cand) < 1e-3):
                    continue
                mag = np.linalg.norm(cand, axis=1)
                if np.ptp(mag) < 1e-3:
                    continue
                scale = float(np.median(mag[mag > 1.0])) if np.any(mag > 1.0) else 1.0
                if scale < 100.0:
                    continue
                rel_var = (np.percentile(mag, 95) - np.percentile(mag, 5)) / max(scale, 1e-6)
                if rel_var < 0.02:
                    continue
                acc = cand
                acc_g = (cand / max(scale, 1e-6)).astype(np.float32)
                break
            if acc is None:
                print(f"  [xuanwu] {csv_path.name}: no usable ACC (waist/lshank constant/missing), skip", flush=True)
                continue
            if subj_id not in unit_medians:
                unit_medians[subj_id] = unit_diagnostic(acc_g, f"xuanwu {subj_id} (g)")

            acc_aligned = gravity_align_raw(acc_g, fs=XUANWU_FS)
            acc_bp = bandpass_filter(acc_aligned, fs=XUANWU_FS)
            acc_resampled = resample_to_canon(acc_bp, fs=XUANWU_FS)
            acc_resampled = np.nan_to_num(acc_resampled, nan=0.0).astype(np.float32)
            if _get_per_sample_norm():
                acc_resampled = per_sample_normalize(acc_resampled)

            signal = assemble_input_features(acc_resampled, fs_hz=CANON_FS)
            y_resampled = resample_labels_nearest((label_raw > 0.5).astype(np.float32), XUANWU_FS, CANON_FS)
            n = min(len(signal), len(y_resampled))
            signal, y_resampled = signal[:n], y_resampled[:n]
            labels = (y_resampled > 0.5).astype(np.int64)
            y_sub = np.zeros((n, 3), dtype=np.float32)
            y_sub[:, 0] = y_resampled.astype(np.float32)
            so = np.zeros((n, 3), dtype=bool)
            for seq in chunk_into_sequences(signal, labels, f"{subj_id}_{csv_path.stem}", subj_id, "xuanwu",
                                             labels_subtype=y_sub, subtype_observed=so):
                seq.unit_transform = "adc_to_g"
                seq.fs_hz = CANON_FS
                seq.position_tag = "lower_back" if sensor == "waist" else "shank"
                all_sequences.append(seq)

    n_subjects = len(set(s.subject_id for s in all_sequences))
    total_frames = sum(len(s.labels) for s in all_sequences)
    total_hours = total_frames * STRIDE_S / 3600.0
    fog_pct = 100.0 * sum(s.labels.sum() for s in all_sequences) / max(total_frames, 1)
    print(f"[xuanwu] Built {len(all_sequences)} sequences from {n_subjects} subjects", flush=True)
    print(f"[xuanwu] Total: {total_hours:.1f} hours, FoG prevalence: {fog_pct:.1f}%", flush=True)
    _xt = {}
    for _xs in all_sequences:
        _xt[_xs.position_tag] = _xt.get(_xs.position_tag, 0) + 1
    print(f"[xuanwu][b3 v29ae] position tags: " +
          ", ".join(f"{k}={v}" for k, v in sorted(_xt.items())), flush=True)
    if not all_sequences:
        raise RuntimeError("No Xuanwu sequences loaded. Check dataset root.")
    if all_sequences[0].signal.shape[1] != N_CHANNELS:
        raise RuntimeError(f"Xuanwu channel contract violated: {all_sequences[0].signal.shape[1]} != {N_CHANNELS}")
    return all_sequences

def load_xuanwu_sequences_cached(root=None):
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_XUANWU_ROOT",
            "/mnt/d/Installed/Datasets/r8gmbtv7w2-3/Filtered Data",
            r"D:\Installed\Datasets\r8gmbtv7w2-3\Filtered Data")
    return _load_sequences_cached("xuanwu", root,
        {"unit_harmonize": _get_unit_harmonize(), "per_sample_norm": _get_per_sample_norm()},
        load_xuanwu_sequences)

WEARGAIT_ACC_COLS = ["LowerBack_Acc_X", "LowerBack_Acc_Y", "LowerBack_Acc_Z"]
WEARGAIT_FS = 100.0

def _weargait_intervals(root):
    intervals_csv = Path(root) / "weargait_fog_intervals.csv"
    if not intervals_csv.is_file():
        raise RuntimeError(f"weargait_fog_intervals.csv not found under {root}")
    iv = defaultdict(list)
    for _, row in pd.read_csv(intervals_csv).iterrows():
        iv[str(row["recording_id"])].append((float(row["start_s"]), float(row["end_s"])))
    return iv

def _weargait_file_index(root):
    index = {}
    search_dirs = [Path(root) / "PD PARTICIPANTS" / "CSV files"]
    rw_root = Path(root) / "Real-world context tasks"
    if rw_root.exists():
        search_dirs += [d for d in rw_root.glob("**/CSV files") if d.is_dir()]
    for d in search_dirs:
        for csv in d.glob("*.csv"):
            index[csv.stem] = csv
    return index

def load_weargait_sequences(root=None):
    """v25: external binary-only cohort -> subtype_observed = ZEROS."""
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_WEARGAIT_ROOT",
            "/mnt/d/Installed/Datasets/synapse",
            r"D:\Installed\Datasets\synapse")
    root = Path(root)
    intervals = _weargait_intervals(root)
    file_index = _weargait_file_index(root)

    all_sequences = []
    unit_medians = {}
    for rec_id, ivs in intervals.items():
        csv_path = file_index.get(rec_id)
        if csv_path is None:
            print(f"  [weargait] no CSV for recording_id '{rec_id}', skip", flush=True)
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  [weargait] skip {csv_path.name}: {e}", flush=True)
            continue
        if not set(WEARGAIT_ACC_COLS).issubset(df.columns):
            continue
        acc = df[WEARGAIT_ACC_COLS].to_numpy(dtype=np.float32)
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        acc_g = (acc / G_STANDARD).astype(np.float32)
        subj_id = rec_id.split("_")[0]
        if subj_id not in unit_medians:
            unit_medians[subj_id] = unit_diagnostic(acc_g, f"weargait {rec_id} (g)")

        acc_aligned = gravity_align_raw(acc_g, fs=WEARGAIT_FS)
        acc_bp = bandpass_filter(acc_aligned, fs=WEARGAIT_FS)
        if _get_per_sample_norm():
            acc_bp = per_sample_normalize(acc_bp)
        signal = assemble_input_features(acc_bp, fs_hz=CANON_FS)

        T = len(acc)
        y = np.zeros(T, dtype=np.float32)
        for (s, e) in ivs:
            i0 = max(0, int(s * WEARGAIT_FS))
            i1 = min(T - 1, int(e * WEARGAIT_FS))
            if i0 <= i1:
                y[i0:i1 + 1] = 1.0
        n = min(len(signal), T)
        signal, y = signal[:n], y[:n]
        labels = (y > 0.5).astype(np.int64)
        y_sub = np.zeros((n, 3), dtype=np.float32)
        y_sub[:, 0] = y.astype(np.float32)
        so = np.zeros((n, 3), dtype=bool)
        for seq in chunk_into_sequences(signal, labels, rec_id, subj_id, "weargait",
                                         labels_subtype=y_sub, subtype_observed=so):
            seq.unit_transform = "mps2_to_g"
            seq.fs_hz = CANON_FS
            all_sequences.append(seq)

    n_subjects = len(set(s.subject_id for s in all_sequences))
    total_frames = sum(len(s.labels) for s in all_sequences)
    total_hours = total_frames * STRIDE_S / 3600.0
    fog_pct = 100.0 * sum(s.labels.sum() for s in all_sequences) / max(total_frames, 1)
    print(f"[weargait] Built {len(all_sequences)} sequences from {n_subjects} subjects", flush=True)
    print(f"[weargait] Total: {total_hours:.1f} hours, FoG prevalence: {fog_pct:.1f}%", flush=True)
    if not all_sequences:
        raise RuntimeError("No WearGait sequences loaded. Check dataset root.")
    if all_sequences[0].signal.shape[1] != N_CHANNELS:
        raise RuntimeError(f"WearGait channel contract violated: {all_sequences[0].signal.shape[1]} != {N_CHANNELS}")
    return all_sequences

def load_weargait_sequences_cached(root=None):
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_WEARGAIT_ROOT",
            "/mnt/d/Installed/Datasets/synapse",
            r"D:\Installed\Datasets\synapse")
    return _load_sequences_cached("weargait", root,
        {"unit_harmonize": _get_unit_harmonize(), "per_sample_norm": _get_per_sample_norm()},
        load_weargait_sequences)

DAPHNET_FS = 64.0
_FOGSTAR_UNITS_FLAG = [False]

def load_daphnet_sequences(root=None):
    """v25.6: DAPHNet FoG (UCI 245) -- the standard public FoG benchmark.
    10 PD subjects, trunk/upper-leg/ankle accelerometers at 64 Hz, annotation
    0=no FoG / 1=FoG / 2=standing. We use the TRUNK accelerometer (closest to
    the lower-back placement of the other cohorts), map the binary annotation
    into the SH channel with subtype_observed=zeros (binary-only cohort)."""
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_DAPHNET_ROOT",
            "/mnt/d/Installed/Datasets/daphnet",
            r"D:\Installed\Datasets\daphnet")
    root = Path(root)
    txt_root = root / "dataset"
    if not txt_root.is_dir():
        txt_root = root / "dataset_fog_release" / "dataset"
    if not txt_root.is_dir():
        raise RuntimeError(f"DAPHNet dataset folder not found under {root} (expected .../dataset/S01R01.txt)")

    all_sequences = []
    unit_medians = {}
    for txt in sorted(txt_root.glob("S*.txt")):
        subj_id = txt.stem[:3]  # S01
        rec_id = txt.stem
        try:
            df = pd.read_csv(txt, sep=r"\s+", header=None)
        except Exception as e:
            print(f"  [daphnet] skip {txt.name}: {e}", flush=True)
            continue
        if df.shape[1] < 11:
            continue
        acc = df.iloc[:, 7:10].to_numpy(dtype=np.float32)  # trunk axes
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        acc_g = (acc / 1000.0).astype(np.float32)          # milli-g -> g
        ann = df.iloc[:, 10].to_numpy(dtype=np.float32)
        if subj_id not in unit_medians:
            unit_medians[subj_id] = unit_diagnostic(acc_g, f"daphnet {rec_id} (g)")

        acc_aligned = gravity_align_raw(acc_g, fs=DAPHNET_FS)
        acc_bp = bandpass_filter(acc_aligned, fs=DAPHNET_FS)
        acc_resampled = resample_poly(acc_bp, 25, 16, axis=0).astype(np.float32)
        y = (ann == 2.0).astype(np.float32)
        obs = (ann != 0.0).astype(np.float32)
        y_resampled = resample_labels_nearest(y, DAPHNET_FS, CANON_FS)
        obs_resampled = resample_labels_nearest(obs, DAPHNET_FS, CANON_FS).astype(bool)
        if _get_per_sample_norm():
            acc_resampled = per_sample_normalize(acc_resampled)
        n = min(len(acc_resampled), len(y_resampled))
        acc_resampled, y_resampled, obs_resampled = acc_resampled[:n], y_resampled[:n], obs_resampled[:n]
        signal = assemble_input_features(acc_resampled, fs_hz=CANON_FS)
        n = min(len(signal), len(y_resampled))
        signal, y_resampled, obs_resampled = signal[:n], y_resampled[:n], obs_resampled[:n]
        labels = (y_resampled > 0.5).astype(np.int64)
        y_sub = np.zeros((n, 3), dtype=np.float32)
        y_sub[:, 0] = y_resampled.astype(np.float32)
        so = np.zeros((n, 3), dtype=bool)  # binary-only -> subtypes masked
        for seq in chunk_into_sequences(signal, labels, rec_id, subj_id, "daphnet",
                                         labels_subtype=y_sub, subtype_observed=so,
                                         observed=obs_resampled,
                                         label_weight=obs_resampled.astype(np.float32)):
            seq.unit_transform = "mg_to_g"
            seq.fs_hz = CANON_FS
            all_sequences.append(seq)

    n_subjects = len(set(s.subject_id for s in all_sequences))
    total_frames = sum(len(s.labels) for s in all_sequences)
    total_hours = total_frames * STRIDE_S / 3600.0
    fog_pct = 100.0 * sum(s.labels.sum() for s in all_sequences) / max(total_frames, 1)
    print(f"[daphnet] Built {len(all_sequences)} sequences from {n_subjects} subjects", flush=True)
    print(f"[daphnet] Total: {total_hours:.1f} hours, FoG prevalence: {fog_pct:.1f}%", flush=True)
    if not all_sequences:
        raise RuntimeError("No DAPHNet sequences loaded. Check dataset root.")
    if all_sequences[0].signal.shape[1] != N_CHANNELS:
        raise RuntimeError(f"DAPHNet channel contract violated: {all_sequences[0].signal.shape[1]} != {N_CHANNELS}")
    return all_sequences

def load_daphnet_sequences_cached(root=None):
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_DAPHNET_ROOT",
            "/mnt/d/Installed/Datasets/daphnet",
            r"D:\Installed\Datasets\daphnet")
    return _load_sequences_cached("daphnet", root,
        {"unit_harmonize": _get_unit_harmonize(), "per_sample_norm": _get_per_sample_norm()},
        load_daphnet_sequences)

_FOGSTAR_SUBJECT_META = {}

def load_fogstar_sequences(root=None):
    """v25.15b: FoG-STAR (Borzi et al., Sci Data 2026, PMID 41588027; Zenodo
    1783880639): 22 PD, 4 IMUs @ 60 Hz (acc in g, gyro deg/s), 7 structured
    tasks, 101 expert-annotated FoG episodes (onset/offset + manifestation +
    severity), ALL OFF-medication (12 h washout). We use ONLY the lower-back
    accelerometer (back_acc_x/y/z -- the deployment placement). Binary-only
    cohort: subtype_observed = zeros; med_state = 0.0 (documented OFF).
    Demographics (age/gender/disease_duration/H&Y/UPDRS-III/FOG-Q/MOCA/FES-I/
    PDQ-8) are cached in _FOGSTAR_SUBJECT_META for fairness reporting."""
    FOGSTAR_FS = 60.0  # v25.53.9: declared once; used for alignment+resampling
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_FOGSTAR_ROOT",
            "/mnt/d/Installed/Datasets/fogstar",
            r"D:\Installed\Datasets\fogstar")
    root = Path(root)
    sensor_csv = root / "sensor_data.csv"
    if not sensor_csv.is_file():
        raise RuntimeError(f"[fogstar] sensor_data.csv not found under {root}. Download from "
                           f"https://zenodo.org/records/1783880639 and set NEUROMAMBA_FOGSTAR_ROOT.")
    clin_csv = root / "clinical_data.csv"
    if clin_csv.is_file():
        try:
            cdf = pd.read_csv(clin_csv)
            for _, row in cdf.iterrows():
                def _int_safe(v, default=-1):
                    try:
                        if pd.isna(v):
                            return default
                        return int(float(v))
                    except (TypeError, ValueError):
                        return default

                def _float_safe(v, default=float("nan")):
                    try:
                        if pd.isna(v):
                            return default
                        return float(v)
                    except (TypeError, ValueError):
                        return default
                _FOGSTAR_SUBJECT_META[_int_safe(row["subjectID"])] = {
                    "age": _int_safe(row.get("age")), "gender": str(row.get("gender", "?")),
                    "disease_duration": _int_safe(row.get("disease_duration")),
                    "h_y": _float_safe(row.get("h_y")),
                    "updrs_iii": _int_safe(row.get("updrs_iii")),
                    "fog_q": _int_safe(row.get("fog_q")), "moca": _int_safe(row.get("moca")),
                    "fes_i": _int_safe(row.get("fes_i")), "pdq_8": _int_safe(row.get("pdq_8")),
                }
            print(f"[fogstar] demographics cached for {len(_FOGSTAR_SUBJECT_META)} subjects", flush=True)
        except Exception as e:
            print(f"[fogstar] clinical_data.csv read failed ({e}) -- subgroup metadata unavailable", flush=True)
    df = pd.read_csv(sensor_csv)
    need = {"back_acc_x", "back_acc_y", "back_acc_z", "fog", "subjectID", "sessionID", "taskID"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"[fogstar] sensor_data.csv missing columns {sorted(missing)}")
    all_sequences = []
    for (sid, sess, task), g in df.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        g = g.sort_values("timestamp") if "timestamp" in g.columns else g
        acc = g[["back_acc_x", "back_acc_y", "back_acc_z"]].to_numpy(dtype=np.float32)
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        if not _FOGSTAR_UNITS_FLAG[0] and float(np.median(np.abs(acc))) > 8.0:
            acc = (acc / 9.80665).astype(np.float32)
            _FOGSTAR_UNITS_FLAG[0] = True
            print("[fogstar v25.54] units heuristic: median|acc|>8 -> converted "
                  "m/s^2 to g (/9.80665)", flush=True)
        fog = g["fog"].to_numpy(dtype=np.float32)
        if len(acc) < 200:
            continue
        acc_aligned = gravity_align_raw(acc, fs=FOGSTAR_FS)
        acc_bp = bandpass_filter(acc_aligned, fs=FOGSTAR_FS)
        acc_r = resample_to_canon(acc_bp, fs=FOGSTAR_FS)
        fog_r = resample_labels_nearest((fog > 0.5).astype(np.float32), FOGSTAR_FS, CANON_FS)
        n = min(len(acc_r), len(fog_r))
        acc_r, fog_r = acc_r[:n], fog_r[:n]
        if _get_per_sample_norm():
            acc_r = per_sample_normalize(acc_r)
        signal = assemble_input_features(acc_r, fs_hz=CANON_FS)
        labels = (fog_r > 0.5).astype(np.int64)
        y_sub = np.zeros((len(signal), 3), dtype=np.float32)
        y_sub[:, 0] = fog_r.astype(np.float32)
        so = np.zeros((len(signal), 3), dtype=bool)
        rec_id = f"fogstar_{int(sid)}_{int(sess)}_{int(task)}"
        subj_id = f"fogstar_{int(sid)}"
        for seq in chunk_into_sequences(signal, labels, rec_id, subj_id, "fogstar",
                                        labels_subtype=y_sub, subtype_observed=so,
                                        observed=np.ones(len(signal), dtype=bool),
                                        label_weight=np.ones(len(signal), dtype=np.float32),
                                        med_state=0.0):
            seq.fs_hz = CANON_FS
            all_sequences.append(seq)
    n_subjects = len(set(s.subject_id for s in all_sequences))
    total_frames = sum(len(s.labels) for s in all_sequences)
    total_hours = total_frames * STRIDE_S / 3600.0
    fog_pct = 100.0 * sum(s.labels.sum() for s in all_sequences) / max(total_frames, 1)
    print(f"[fogstar] Built {len(all_sequences)} sequences from {n_subjects} subjects", flush=True)
    print(f"[fogstar] Total: {total_hours:.1f} hours, FoG prevalence: {fog_pct:.1f}% "
          f"(OFF-medication cohort)", flush=True)
    if not all_sequences:
        raise RuntimeError("No FoG-STAR sequences loaded. Check dataset root.")
    if all_sequences[0].signal.shape[1] != N_CHANNELS:
        raise RuntimeError(f"FoG-STAR channel contract violated: {all_sequences[0].signal.shape[1]} != {N_CHANNELS}")
    return all_sequences

def load_fogstar_sequences_cached(root=None):
    if root is None:
        root = _resolve_dataset_root("NEUROMAMBA_FOGSTAR_ROOT",
            "/mnt/d/Installed/Datasets/FoG-STAR",
            r"D:\Installed\Datasets\FoG-STAR")
    return _load_sequences_cached("fogstar", root,
        {"unit_harmonize": _get_unit_harmonize(), "per_sample_norm": _get_per_sample_norm()},
        load_fogstar_sequences)

def _external_stub(name, env_var):
    def _loader():
        root = os.environ.get(env_var)
        if not root:
            raise RuntimeError(f"[{name}] external cohort requested but {env_var} is not set.")
        raise NotImplementedError(f"[{name}] loader not implemented in this repo snapshot.")
    return _loader

LOADER_REGISTRY = {
    "tlvmc": load_tlvmc_sequences_cached,
    "xuanwu": load_xuanwu_sequences_cached,
    "weargait": load_weargait_sequences_cached,
    "daphnet": load_daphnet_sequences_cached,
    "fogstar": load_fogstar_sequences_cached,
}

def load_datasets(names):
    all_seqs = []
    for name in names:
        if name not in LOADER_REGISTRY:
            raise KeyError(f"Unknown dataset '{name}'. Available: {list(LOADER_REGISTRY.keys())}")
        all_seqs.extend(LOADER_REGISTRY[name]())
    return all_seqs

print("[build_sequences] BUILD_SEQUENCES_V25_LOADED", flush=True)

def three_way_subject_split(sequences, calib_frac=0.15, thresh_frac=0.15, seed=42):
    subjects = sorted(set(s.subject_id for s in sequences))
    if len(subjects) < 4:
        raise ValueError(f"Need >=4 subjects for 3-way split, got {len(subjects)}")
    rng = np.random.RandomState(seed)
    subs = list(subjects)
    rng.shuffle(subs)
    n = len(subs)
    n_thresh = max(1, int(round(n * thresh_frac)))
    n_calib = max(1, int(round(n * calib_frac)))
    thresh_subs = set(subs[:n_thresh])
    calib_subs = set(subs[n_thresh:n_thresh + n_calib])
    train_subs = set(subs[n_thresh + n_calib:])

    train = [s for s in sequences if s.subject_id in train_subs]
    calib = [s for s in sequences if s.subject_id in calib_subs]
    thresh = [s for s in sequences if s.subject_id in thresh_subs]

    assert not (train_subs & calib_subs or train_subs & thresh_subs or calib_subs & thresh_subs), \
        "Subject leakage in 3-way split"

    diag = {"n_train_subjects": len(train_subs), "n_calib_subjects": len(calib_subs),
            "n_thresh_subjects": len(thresh_subs), "n_train_seqs": len(train),
            "n_calib_seqs": len(calib), "n_thresh_seqs": len(thresh), "seed": seed,
            "split_method": "random_subject"}
    print(f"[3way-split] {diag}", flush=True)
    if not train or not calib or not thresh:
        raise ValueError(f"3-way split produced an empty partition: {diag}")
    return train, calib, thresh, diag

def _count_subtype_events(seqs):
    """Count positive FRAMES per subtype across observed-annotated frames only."""
    counts = np.zeros(3, dtype=np.int64)
    for s in seqs:
        for k in range(3):
            mask = s.subtype_observed[:, k] & (s.y_subtype[:, k] > 0.5)
            counts[k] += int(mask.sum())
    return counts

def phase1_manifest(args, fam_name, fold_idx_in_rep, _rep, tr, va, split_mode,
                    main_tid, baseline_tid=None, calib=None, thresh=None):
    """v25.42.8 (reviewer #26): chain-of-provenance manifest for EVERY fold.
    Carries code sha, protocol, split mode/seed/repeat, subject lists and the
    main+baseline identities; protocol_hash = sha256 of the manifest JSON.
    Every OOF artifact and fold record stores this hash."""
    import hashlib as _hl
    _full_sha = ""
    try:
        _full_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        pass
    manifest = {
        "protocol_version": "P1.0",
        "code_sha256": _full_sha,
        "code_version": CODE_VERSION,
        "split_mode": split_mode,
        "repeat": int(_rep),
        "split_seed": int(args.seed + _rep * 10000),
        "fold": int(fold_idx_in_rep + 1),
        "family": fam_name,
        "n_seeds": int(args.n_seeds),
        "train_subjects": sorted(set(s.subject_id for s in tr)),
        "test_subjects": sorted(set(s.subject_id for s in va)),
        "calib_subjects": sorted(set(s.subject_id for s in calib)) if calib else None,
        "threshold_subjects": sorted(set(s.subject_id for s in thresh)) if thresh else None,
        "main_identity": main_tid,
        "baseline_identity": baseline_tid,
        "training_identity": training_identity(
            model_type=args.model, lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, seed=args.seed + _rep * 10000, n_seeds=args.n_seeds,
            fold_idx=fold_idx_in_rep, family=fam_name,
            use_rare_split=not args.no_rare_split),
        "evaluation_identity": evaluation_identity(),
    }
    manifest["protocol_hash"] = _hl.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    return manifest

def resolve_baseline_probs(baseline_fold_results):
    parts = []
    for _b in baseline_fold_results:
        _bp = _b.get("sub_probs")
        if isinstance(_bp, str) and _bp.startswith("npz:"):
            try:
                _bz = np.load(_bp[4:])
                _bp = np.asarray(_bz["sub_probs"])
            except Exception as _e:
                print(f"[baseline v25.42.5] WARNING: cannot reload {_bp}: {_e}", flush=True)
                _bp = None
        if _bp is not None:
            parts.append(_bp)
    if not parts:
        return None
    return np.concatenate(parts, axis=0)

def _fold_json_name_from_tag(fold_tag):
    """v25.44 (PROVENANCE FIX): reconstruct the fold checkpoint JSON name from
    the FULL fold tag instead of prev_fold_record['fold'] + tag.split('_')[0].
    The old heuristic dropped the repeated-CV suffix: with n_repeats>1 the tag
    is {smoke_}{fam}_fold{K}_r{R} but the rebuilt name was
    neuromamba_v25_fold{GLOBAL_N}_{fam}_checkpoint.json (fold = the GLOBAL
    counter across repeats, which can exceed n_splits) -- a reevaluated record
    was persisted under a WRONG filename and could clobber a different fold's
    record. Family names are fixed tokens without underscores, so parsing is
    exact: split on '_fold' once; the remainder is '{K}' or '{K}_r{R}'."""
    tag = str(fold_tag or "")
    prefix = "smoke_" if tag.startswith("smoke_") else ""
    body = tag[len(prefix):]
    fam, rest = body.split("_fold", 1)
    parts = rest.split("_")
    fold_num = parts[0]
    rep_suffix = "".join("_" + p for p in parts[1:] if p.startswith("r"))
    return f"{prefix}neuromamba_v25_fold{fold_num}_{fam}{rep_suffix}_checkpoint.json"

def reevaluate_frozen_fold(prev_fold_record, fold_tag, current_evaluation_id):
    """v25.42.6 (evaluate_only mode): recompute ALL evaluation outputs of a
    completed fold from the frozen OOF arrays saved at fold time, using the
    CURRENT evaluation code (metrics, thresholds at the frozen operating point,
    ECE, per-source/per-subject breakdowns, med subgroups, ICC, cueing).
    No training, no inference -- pure evaluation from stored predictions.
    Returns {"fold_record", "sub_probs", "sub_labels", "sub_observed"} or None
    on failure (caller ABORTS in evaluate_only mode since v25.53.7 -- it never
    falls back to training)."""
    ens_npz = (prev_fold_record.get("ens_arrays") or {}).get("npz")
    if not ens_npz or not os.path.exists(ens_npz):
        print(f"[reeval v25.42.6] no ens npz for {fold_tag} -- cannot evaluate-only", flush=True)
        return None
    try:
        z = np.load(ens_npz, allow_pickle=True)
    except Exception as e:
        print(f"[reeval v25.42.6] cannot load {ens_npz}: {e}", flush=True)
        return None
    sub_probs = np.asarray(z["sub_probs"], dtype=np.float32)
    sub_labels = np.asarray(z["sub_labels"], dtype=np.float32)
    sub_observed = np.asarray(z["sub_observed"]).astype(np.float32)
    n = min(len(sub_labels), len(sub_probs))
    sub_probs, sub_labels, sub_observed = sub_probs[:n], sub_labels[:n], sub_observed[:n]
    frozen = float(prev_fold_record.get("frozen_threshold", 0.5))
    temperature = float(prev_fold_record.get("temperature", 1.0))
    if "fog_probs" in z.files and z["fog_probs"] is not None:
        probs = np.asarray(z["fog_probs"], dtype=np.float32)[:n]
        labels = np.asarray(z["fog_labels"], dtype=np.float32)[:n]
        ids = np.asarray(z["fog_ids"])[:n]
        subjects = np.asarray(z["fog_subjects"])[:n]
        sources = np.asarray(z["fog_sources"])[:n]
        med = np.asarray(z["fog_med"], dtype=np.float32)[:n] if "fog_med" in z.files and z["fog_med"] is not None else None
    else:
        print(f"[reeval v25.42.6] {fold_tag}: legacy npz lacks fog arrays -- "
              f"falling back to fresh training", flush=True)
        return None
    _pp = prev_fold_record.get("postproc") or {}
    pred = causal_event_postprocess(probs, ids, trigger=frozen,
                                     sustain=_pp.get("sustain", frozen * 0.4),
                                     confirm_n=_pp.get("confirm_n", 3),
                                     cooldown_n=_pp.get("cooldown_n", 30),
                                     merge_gap_n=_pp.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
    subtype_temps = prev_fold_record.get("subtype_temps") or [1.0, 1.0, 1.0]
    sub_probs_t = apply_temperature_subtype(sub_probs, subtype_temps)
    official_map = compute_official_like_map(sub_labels, sub_probs_t, sub_observed)
    kaggle_map = {"mAP": official_map["mAP"], "comparable": official_map["comparable"],
                  **{k: official_map[k] for k in SUBTYPE_NAMES}}
    seed_val_maps = None
    if _flag("NEUROMAMBA_PER_SEED_VAL", "1") and "seed_sub_probs" in z.files:
        _temps = subtype_temps
        _sv = []
        for _sp in np.asarray(z["seed_sub_probs"]):
            _t = apply_temperature_subtype(np.asarray(_sp)[:n], _temps)
            _sv.append(float(compute_official_like_map(sub_labels, _t, sub_observed)["mAP"]))
        if _sv:
            seed_val_maps = _sv
            print(f"[seed-val v25.40] per-seed val mAP {[round(x, 4) for x in seed_val_maps]} "
                  f"mean={float(np.mean(seed_val_maps)):.4f} sd={float(np.std(seed_val_maps)):.4f}", flush=True)
    metrics = {
        "episode_f1_frame": episode_f1(labels, probs, frozen, ids, pred_binary=pred),
        "event_f1": event_level_f1(labels, pred, ids),
        "farh": false_alarm_events_per_hour(labels, probs, frozen, ids, pred_binary=pred),
        "ece": compute_ece(probs, labels),
        "brier": brier_score(probs, labels),
        "reliability_diagram_slope": calibration_slope_intercept(probs, labels)[0],
        "reliability_diagram_intercept": calibration_slope_intercept(probs, labels)[1],
        "logistic_calib_slope": logistic_calibration_slope_intercept(probs, labels)[0],
        "logistic_calib_intercept": logistic_calibration_slope_intercept(probs, labels)[1],
        "window_f1": window_level_f1(labels, probs, ids, threshold=frozen)["window_f1"],
        "window_metrics": window_level_metrics(labels, probs, ids, threshold=frozen),
        "event_metrics": event_level_metrics(labels, pred, ids),
        "icc": icc_report(labels, pred, ids),
        "per_source": per_source_metrics(labels, probs, ids, pred, sources),
        "per_subject_farh": per_subject_farh_distribution(labels, pred, subjects),
        "decision_curve": decision_curve_analysis(labels, pred, ids),
        "cueing_latency": onset_latency_report(labels, pred, ids),
        "kaggle_map": kaggle_map,
        "seed_val_maps": seed_val_maps,
        "abstention_frac": 0.0, "abstention_pos_frac": 0.0,
    }
    if med is not None:
        try:
            metrics["per_medication"] = medication_subgroup_report(
                labels, pred, probs, ids, med, threshold=frozen)
        except Exception as e:
            print(f"[reeval v25.42.6] med subgroup failed: {e}", flush=True)
            metrics["per_medication"] = None
    else:
        metrics["per_medication"] = None
    print(f"[VAL][reeval v25.42.6] frameF1={metrics['episode_f1_frame']:.3f} "
          f"eventF1={metrics['event_f1']:.3f} FARh={metrics['farh']['farh_events_per_hour']:.2f}/h "
          f"ECE={metrics['ece']:.4f} windowF1={metrics['window_f1']:.3f} "
          f"mAP={kaggle_map['mAP']:.4f} (evaluate_only, frozen predictions)", flush=True)
    fold_record = dict(prev_fold_record)
    fold_record.update({
        "code_version": CODE_VERSION,
        "training_identity": prev_fold_record.get("training_identity"),
        "evaluation_identity": current_evaluation_id,
        "resume_status": "REEVALUATED_FROZEN_MODEL",
        "evaluated_at_rev": EVALUATION_REV,
        "frozen_threshold": frozen, "temperature": temperature,
        **{k: v for k, v in metrics.items() if k != "decision_curve"},
    })
    try:
        _fold_json = prev_fold_record.get("fold_json_path") or _fold_json_name_from_tag(fold_tag)
        with open(_fold_json, "w") as f:
            json.dump(fold_record, f, indent=2, cls=NpEncoder)
    except Exception as e:
        print(f"[reeval v25.44] WARNING: could not persist reevaluated fold: {e}", flush=True)
    return {"fold_record": fold_record, "sub_probs": sub_probs,
            "sub_labels": sub_labels, "sub_observed": sub_observed}

def rare_subtype_aware_split(sequences, calib_frac=0.15, thresh_frac=0.15, seed=42,
                              target_sh_frac=(0.20, 0.30), target_walk_frac=(0.20, 0.30),
                              max_tries=200):
    """v25-FIX-4 (1st-place trick): select calib+thresh subjects so the held-out
    partition contains 20-30% of ALL StartHesitation events AND 20-30% of ALL
    Walking events. Random subject splitting let val-mAP look fine while
    calibSubtypeMAP (which weights SH+Walking equally) collapsed. This is
    likely the single biggest cause of 'calibSubtypeMAP not improving'."""
    subjects = sorted(set(s.subject_id for s in sequences))
    if len(subjects) < 4:
        raise ValueError(f"Need >=4 subjects for rare-subtype split, got {len(subjects)}")
    rng = np.random.RandomState(seed)
    total_counts = _count_subtype_events(sequences)
    total_sh, total_walk = total_counts[0], total_counts[2]
    n = len(subjects)
    n_held = max(2, int(round(n * (calib_frac + thresh_frac))))

    subj_sh = {}
    subj_walk = {}
    for s in sequences:
        sid = s.subject_id
        m = s.subtype_observed & (s.y_subtype > 0.5)
        subj_sh[sid] = subj_sh.get(sid, 0) + int(m[:, 0].sum())
        subj_walk[sid] = subj_walk.get(sid, 0) + int(m[:, 2].sum())

    def _score(held_subs):
        held_seqs = [s for s in sequences if s.subject_id in held_subs]
        held_counts = _count_subtype_events(held_seqs)
        sh_frac = held_counts[0] / max(total_sh, 1)
        walk_frac = held_counts[2] / max(total_walk, 1)
        in_band = (target_sh_frac[0] <= sh_frac <= target_sh_frac[1]) and \
                  (target_walk_frac[0] <= walk_frac <= target_walk_frac[1])
        return held_seqs, sh_frac, walk_frac, in_band, abs(sh_frac - 0.25) + abs(walk_frac - 0.25)

    best_split = None
    best_score = float("inf")
    by_sh = sorted(subjects, key=lambda sid: (-subj_sh.get(sid, 0), rng.random()))
    held_subs = set(); train_subs = set()
    target_held_sh = 0.25 * total_sh
    held_sh_acc = 0
    for sid in by_sh:
        if len(held_subs) < n_held and (held_sh_acc < target_held_sh or len(train_subs) >= n - n_held):
            held_subs.add(sid); held_sh_acc += subj_sh.get(sid, 0)
        else:
            train_subs.add(sid)
    held_seqs, sh_frac, walk_frac, in_band, score = _score(held_subs)
    def _snap(held, train):
        hseqs, s_f, w_f, _, _ = _score(held)
        return (set(held), set(train), list(hseqs),
                [s for s in sequences if s.subject_id in train], s_f, w_f)
    if in_band:
        best_split = _snap(held_subs, train_subs)
    else:
        if score < best_score:
            best_score = score
            best_split = _snap(held_subs, train_subs)
    for _ in range(max_tries):
        if best_split is not None:
            hs, ts, _, _, s_f, w_f = best_split
            if (target_sh_frac[0] <= s_f <= target_sh_frac[1]) and (target_walk_frac[0] <= w_f <= target_walk_frac[1]):
                break
        held_list = sorted(held_subs); train_list = sorted(train_subs)  # v25.42.4: sorted -- list(set) is hash-order, nondeterministic across processes
        rng.shuffle(held_list); rng.shuffle(train_list)
        if not held_list or not train_list:
            break
        h_swap, t_swap = held_list[0], train_list[0]
        held_subs = set(held_subs); held_subs.discard(h_swap); held_subs.add(t_swap)
        train_subs = set(train_subs); train_subs.discard(t_swap); train_subs.add(h_swap)
        held_seqs, sh_frac, walk_frac, in_band, score = _score(held_subs)
        if in_band:
            best_split = _snap(held_subs, train_subs)
            break
        if score < best_score:
            best_score = score
            best_split = _snap(held_subs, train_subs)

    held_subs, train_subs, held_seqs, train_seqs, sh_frac, walk_frac = best_split
    held_list = sorted(held_subs)
    rng.shuffle(held_list)
    n_thresh = max(1, len(held_list) // 2)
    thresh_subs = set(held_list[:n_thresh])
    calib_subs = set(held_list[n_thresh:])
    calib = [s for s in held_seqs if s.subject_id in calib_subs]
    thresh = [s for s in held_seqs if s.subject_id in thresh_subs]
    train = train_seqs

    diag = {"n_train_subjects": len(train_subs), "n_calib_subjects": len(calib_subs),
            "n_thresh_subjects": len(thresh_subs), "n_train_seqs": len(train),
            "n_calib_seqs": len(calib), "n_thresh_seqs": len(thresh), "seed": seed,
            "split_method": "rare_subtype_aware",
            "total_sh_events": int(total_sh), "total_walk_events": int(total_walk),
            "held_sh_frac": float(sh_frac), "held_walk_frac": float(walk_frac),
            "target_sh_frac": list(target_sh_frac), "target_walk_frac": list(target_walk_frac)}
    print(f"[rare-split] {diag}", flush=True)
    t_low_sh, t_low_walk = target_sh_frac[0], target_walk_frac[0]
    if sh_frac < 0.5 * t_low_sh or walk_frac < 0.5 * t_low_walk:
        print(f"[rare-split v25.33] WARNING: held fraction below 50% of the target band "
              f"(held_sh_frac={sh_frac:.4f} vs target>={t_low_sh}, "
              f"held_walk_frac={walk_frac:.4f} vs target>={t_low_walk}) -- subtype APs on "
              f"the selection split are small-sample noise", flush=True)
    _sh_pos_subj_held = sorted({s.subject_id for s in held_seqs
                                if int((s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).sum()) > 0})
    _sh_pos_subj_all = sorted({s.subject_id for s in sequences
                               if int((s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).sum()) > 0})
    _gap_frames = max(1, int(round(100.0 * 1.0)))  # 1 s at 100 Hz
    _sh_events_held = {}
    for _sid in _sh_pos_subj_held:
        _runs = 0
        for s in held_seqs:
            if s.subject_id != _sid:
                continue
            m = (s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).astype(np.int64)
            _runs += int(np.sum(np.diff(np.concatenate([[0], m, [0]])) == 1))
        _sh_events_held[_sid] = _runs
    _held_sh_events = sum(_sh_events_held.values())
    _max_subj_frac = 0.0
    if _held_sh_events > 0:
        _max_subj_frac = max(_sh_events_held.values()) / _held_sh_events
    diag["held_sh_positive_subjects"] = len(_sh_pos_subj_held)
    diag["total_sh_positive_subjects"] = len(_sh_pos_subj_all)
    diag["held_sh_positive_subject_ids"] = _sh_pos_subj_held
    diag["held_sh_events"] = int(_held_sh_events)
    diag["held_sh_events_per_subject"] = _sh_events_held
    diag["held_sh_max_subject_event_fraction"] = float(_max_subj_frac)
    _min_sh_subj = int(os.environ.get("NEUROMAMBA_MIN_HELD_SH_SUBJECTS", "4"))
    _min_sh_events = int(os.environ.get("NEUROMAMBA_MIN_HELD_SH_EVENTS", "8"))
    _max_subj_frac_gate = float(os.environ.get("NEUROMAMBA_MAX_SH_SUBJECT_EVENT_FRAC", "0.5"))
    if len(_sh_pos_subj_held) < _min_sh_subj or _held_sh_events < _min_sh_events \
            or _max_subj_frac > _max_subj_frac_gate:
        print(f"[rare-split v25.42.8] WARNING: SH UNDER POWERED "
              f"({len(_sh_pos_subj_held)}/{_min_sh_subj} subjects, "
              f"{_held_sh_events}/{_min_sh_events} events, "
              f"max-subject event fraction {_max_subj_frac:.2f} "
              f"(gate {_max_subj_frac_gate:.2f})) -- SH AP/recall are NOT "
              f"reliable for this fold; excluded from any SH-stability claim",
              flush=True)
    if len(_sh_pos_subj_all) >= _min_sh_subj:
        _retry = int(os.environ.get("NEUROMAMBA_SH_SPLIT_RETRY", "0"))
        if _retry > 0 and not (target_sh_frac[0] <= sh_frac <= target_sh_frac[1]):
            _orig_seed = seed
            for _r in range(_retry):
                seed = _orig_seed + 1 + _r
                try:
                    _rt, _rc, _rth, _rdiag = rare_subtype_aware_split(
                        sequences, calib_frac=calib_frac, thresh_frac=thresh_frac,
                        seed=seed, target_sh_frac=target_sh_frac,
                        target_walk_frac=target_walk_frac, max_tries=max_tries)
                    if _rdiag.get("held_sh_frac", 0) >= target_sh_frac[0] and \
                            _rdiag.get("held_sh_positive_subjects", 0) >= _min_sh_subj:
                        print(f"[rare-split v25.42.7] SH-stabilized split found at retry "
                              f"seed {seed} (held_sh_frac={_rdiag['held_sh_frac']:.4f}, "
                              f"{_rdiag['held_sh_positive_subjects']} SH-positive subjects)",
                              flush=True)
                        return _rt, _rc, _rth, _rdiag
                except Exception as _e:
                    print(f"[rare-split v25.42.7] retry seed {seed} failed: {_e}", flush=True)
            print(f"[rare-split v25.42.7] no SH-stabilized split found in {_retry} retries "
                  f"-- keeping the best-effort split (held_sh_frac={sh_frac:.4f})", flush=True)
    if not train or not calib or not thresh:
        print(f"[rare-split] empty partition -- falling back to random subject split", flush=True)
        return three_way_subject_split(sequences, calib_frac, thresh_frac, seed)
    return train, calib, thresh, diag

def dilate_sh_onset_labels(seqs, window_s, fs=100.0):
    """v25.30-exp: NEUROMAMBA_ONSET_WINDOW_S=<sec> -- mark the W seconds of
    observed frames BEFORE every StartHesitation run as SH-positive (onset
    relabeling; FoG-M3 IEEE JBHI 2025 predicts ~6 s before onset, a Bi-DTCN
    pre-FoG sweep found a 5 s anticipatory window optimal). Turns the
    ~500-frame defog SH class into an onset region with ~100x more signal for
    the subtype head while leaving the binary fog labels untouched. The
    observed mask is extended over the dilated region so the new positives
    actually train. Only the SH channel (k=0) is dilated. NOTE: the dilated
    pre-episode frames have y_fog==0 by construction, so the containment
    audit (subtype_positive => fog==1) is relaxed under the same flag.
    v25.40: NEUROMAMBA_ONSET_TARGET_MODE=smooth turns the binary dilation
    into a structured TIME-TO-ONSET risk target (the 'SH as a transition
    task' recommendation): y(t) = exp(-(t0 - t)/tau) for 0 < t0-t <= H and
    1.0 inside the episode (t0 = onset frame, H = onset window, tau =
    NEUROMAMBA_ONSET_TAU_S, default 0.5 s). Every pre-episode frame is no
    longer equally informative; binarized at 0.5 the positives equal the
    episode + the last ~0.35 s before onset (tau*ln2). Pairs with the
    winner token-BCE path (float targets). NOTE: the runtime fog label is
    derived as y_subtype.max()>0.5, so binary mode also extends the fog
    pre-window to H seconds (the source of the negative cueing latency),
    while smooth mode narrows that implicit fog pre-extension to
    ~tau*ln2 s (near-onset frames only)."""
    w = int(round(window_s * fs))
    if w <= 0:
        return seqs
    mode = os.environ.get("NEUROMAMBA_ONSET_TARGET_MODE", "binary")
    tau = max(5.0, float(os.environ.get("NEUROMAMBA_ONSET_TAU_S", "0.5")) * fs)
    for s in seqs:
        sh = (s.y_subtype[:, 0] > 0.5).astype(bool)
        if not sh.any():
            continue
        runs, start = [], None
        for i in range(len(sh)):
            if sh[i] and start is None:
                start = i
            elif not sh[i] and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(sh)))
        for st, en in runs:
            lo = max(0, st - w)
            if mode == "smooth":
                frames = np.arange(lo, st)
                dist = (st - 1) - frames               # 0 at the frame before onset
                risk = np.exp(-dist / tau).astype(np.float32)
                s.y_subtype[lo:st, 0] = risk
            else:
                s.y_subtype[lo:st, 0] = 1.0
            s.subtype_observed[lo:st, 0] = True
    return seqs

def filter_supervised_sequences(seqs):
    sup = [s for s in seqs if s.source_type not in ("notype", "notype_pseudo")]
    dropped = len(seqs) - len(sup)
    if dropped:
        print(f"[build_sequences] Excluded {dropped} notype sequences from supervised CV", flush=True)
    _onset_w = float(os.environ.get("NEUROMAMBA_ONSET_WINDOW_S", "0"))
    if _onset_w > 0:
        tlvmc_sup = [s for s in sup if s.source_type in ("defog", "tdcsfog")]
        dilate_sh_onset_labels(tlvmc_sup, _onset_w)
        n_sh = sum(int((s.y_subtype[:, 0] > 0.5).sum()) for s in sup)
        print(f"[onset-relabel v25.30] dilated SH positives by {_onset_w:.1f} s before each "
              f"SH run on {len(tlvmc_sup)} tlvmc sequences -> {n_sh} SH-positive frames "
              f"(binary fog labels untouched; containment audit relaxed for SH)", flush=True)
    return sup

def audit_subtype_contract(seqs, out_path=None):
    """v25-FIX-2 audit: per source_type and per subtype channel, report n_frames,
    n_observed, n_positive, and verify subtype_positive => y_fog==1 containment.
    Asserts the invariant; raises if violated."""
    out = {}
    for src in sorted(set(s.source_type for s in seqs)):
        src_seqs = [s for s in seqs if s.source_type == src]
        per_sub = []
        for k, name in enumerate(["StartHesitation", "Turn", "Walking"]):
            n_frames = sum(len(s.labels) for s in src_seqs)
            n_obs = sum(int(s.subtype_observed[:, k].sum()) for s in src_seqs)
            n_pos = sum(int((s.y_subtype[:, k] > 0.5).sum()) for s in src_seqs)
            violate = 0
            for s in src_seqs:
                sub_pos = s.y_subtype[:, k] > 0.5
                fog = s.y_fog > 0.5
                violate += int((sub_pos & ~fog).sum())
            _onset_relax = (k == 0 and float(os.environ.get("NEUROMAMBA_ONSET_WINDOW_S", "0")) > 0)
            if _onset_relax:
                print(f"  [audit v25.30] SH containment relaxed (onset relabeling active: "
                      f"pre-episode frames are SH-positive by design)", flush=True)
            per_sub.append({"name": name, "n_frames": int(n_frames),
                            "n_subtype_observed": int(n_obs), "n_positive": int(n_pos),
                            "containment_violations": int(violate)})
        out[src] = {"n_sequences": len(src_seqs), "n_subjects": len(set(s.subject_id for s in src_seqs)),
                    "per_subtype": per_sub}
    print("[audit] subtype contract:", flush=True)
    for src, d in out.items():
        print(f"  [{src}] n_seq={d['n_sequences']} n_subj={d['n_subjects']}", flush=True)
        for ps in d["per_subtype"]:
            print(f"    {ps['name']}: observed={ps['n_subtype_observed']} positive={ps['n_positive']} "
                  f"violations={ps['containment_violations']}", flush=True)
            if ps["containment_violations"]:
                _relax = (ps["name"] == "StartHesitation"
                          and float(os.environ.get("NEUROMAMBA_ONSET_WINDOW_S", "0")) > 0)
                if not _relax:
                    raise RuntimeError(
                        f"containment violation in {src}/{ps['name']}: {ps['containment_violations']}")
    if out_path:
        try:
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2, default=str)
        except Exception as e:
            print(f"[audit] write failed (non-fatal): {e}", flush=True)
    return out

print("[metrics] METRICS_V25_LOADED", flush=True)

SUBTYPE_NAMES = ["StartHesitation", "Turn", "Walking"]

def _postprocess_record(p, trigger, sustain, confirm_n, cooldown_n, merge_gap_n):
    """Vectorized state machine for ONE record (see causal_event_postprocess).
    O(T + n_runs): non-sustain positions are precomputed and consumed with a
    monotonically advancing pointer (a per-run `flatnonzero(~sat[f:])` scan
    would be O(T * n_runs) -- the original bottleneck at grid scale)."""
    if sustain >= trigger:
        sustain = trigger * 0.4
    T = len(p)
    out = np.zeros(T, dtype=np.int64)
    above = p >= trigger
    if not above.any():
        return out
    sat = p >= sustain
    ab = above.astype(np.int8)
    d = np.diff(ab)
    starts = np.flatnonzero(d == 1) + 1
    if ab[0]:
        starts = np.concatenate(([0], starts))
    ends = np.flatnonzero(d == -1)
    if ab[-1]:
        ends = np.concatenate((ends, [T - 1]))
    non_sat = np.flatnonzero(~sat)
    ptr = 0
    n_non_sat = len(non_sat)
    for s0, e0 in zip(starts, ends):
        if e0 - s0 + 1 < confirm_n:
            continue
        f = s0 + confirm_n - 1  # fire frame (no cooldown cascade)
        if f > e0:
            continue
        while ptr < n_non_sat and non_sat[ptr] < f:
            ptr += 1
        end = non_sat[ptr] if ptr < n_non_sat else T  # first non-sustain frame
        out[max(0, f - confirm_n + 1):end] = 1
    idx = np.where(out)[0]
    if len(idx) > 1:
        on = out.astype(np.int8)
        d2 = np.diff(on)
        seg_starts = np.flatnonzero(d2 == 1) + 1
        if on[0]:
            seg_starts = np.concatenate(([0], seg_starts))
        seg_ends = np.flatnonzero(d2 == -1)
        if on[-1]:
            seg_ends = np.concatenate((seg_ends, [T - 1]))
        for si in range(len(seg_ends) - 1):
            gap = seg_starts[si + 1] - seg_ends[si] - 1
            if 0 < gap < merge_gap_n:
                out[seg_ends[si] + 1:seg_starts[si + 1]] = 1
    return out

MERGE_GAP_N_DEFAULT = int(os.environ.get("NEUROMAMBA_MERGE_GAP_N", "30"))  # v25.46b: 300-frame merge killed event-F1; header said 30, code said 300
MERGE_GAP_GRID = [30, 100, 200, 300, 400]

def causal_event_postprocess(probs, ids, trigger=0.5, sustain=None, confirm_n=3, cooldown_n=30, merge_gap_n=MERGE_GAP_N_DEFAULT,
                             masks=None):
    """v25.1-speed: vectorized per-record state machine. Semantically identical
    to the original frame-by-frame loop (fires at the first frame where
    cooldown==0 AND confirm_n consecutive trigger-crossings end), but ~100x
    faster -- the 120-point clinical threshold grid is now cheap enough to run
    every epoch (was the dominant per-epoch CPU cost). Pass precomputed
    `masks` (list of per-record boolean masks from np.unique(ids)) to skip
    re-splitting the arrays on every grid point."""
    if sustain is None:
        sustain = trigger * 0.4
    probs = np.asarray(probs); ids = np.asarray(ids)
    pred = np.zeros(len(probs), dtype=np.int64)
    if masks is None:
        masks = [ids == uid for uid in np.unique(ids)]
    for m in masks:
        pred[m] = _postprocess_record(probs[m], trigger, sustain, confirm_n, cooldown_n, merge_gap_n)
    return pred

def check_monotone_trigger_sweep(probs, ids, labels=None, triggers=None, n_points=48,
                                 confirm_n=3, cooldown_n=30, merge_gap_n=None,
                                 groups=None, tol=0.0):
    """v25.58 / Round 28bx: verify the detector's fired set is monotone
    NON-INCREASING in the trigger.

    Raising the bar can only ever shrink the set of frames the post-processor
    marks positive. If any frame switches ON when the trigger is raised, the
    post-processor is broken and every operating point derived from the sweep
    is untrustworthy. This is the guard that the Round-28bw artifact (frame
    sensitivity +0.5824 from RAISING the trigger) would have caught.

    `groups` (e.g. per-subject codes) additionally checks monotonicity within
    each group, which is the per-subject view the clinical guarantee needs.
    Groups are evaluated inside the same trigger loop, so the whole check
    costs ~2 passes over the sweep.

    Returns a report dict; `report["ok"]` is True iff no violation was found.
    """
    if merge_gap_n is None:
        merge_gap_n = MERGE_GAP_N_DEFAULT
    probs = np.asarray(probs, dtype=np.float32)
    ids = np.asarray(ids)
    if triggers is None:
        nz = probs[probs > 0]
        lo = float(np.quantile(nz, 0.005)) if nz.size else 1e-5
        hi = float(np.quantile(probs, 0.995))
        triggers = np.logspace(np.log10(max(lo, 1e-6)), np.log10(max(hi, 1e-3)), n_points)
    triggers = np.unique(np.asarray(triggers, dtype=np.float64))
    triggers = triggers[triggers > 0]
    masks = [ids == u for u in np.unique(ids)]

    out = np.zeros(len(probs), dtype=np.int8)
    pos = (np.asarray(labels) > 0.5) if labels is not None else None
    n_pos = int(pos.sum()) if pos is not None else 0

    if groups is None:
        gkeys, gmask = [None], [np.ones(len(probs), dtype=bool)]
    else:
        groups = np.asarray(groups)
        gkeys = [g for g in np.unique(groups)]
        gmask = [groups == g for g in gkeys]

    prev = [None] * len(gkeys)
    prev_sens = [None] * len(gkeys)
    prev_t = [None] * len(gkeys)
    viol = [{"grew_frames": 0, "sens_increase": 0.0,
             "trigger_pair": None, "n_steps_checked": 0} for _ in gkeys]
    curve = []

    for t in triggers:
        t = float(t)
        out[:] = 0
        for m in masks:
            out[m] = _postprocess_record(probs[m], t, t * 0.4, confirm_n,
                                         cooldown_n, merge_gap_n)
        row = {"trigger": t, "fired": int(out.sum())}
        if pos is not None:
            row["frame_sens"] = float(((out == 1) & pos).sum() / max(n_pos, 1))
        for gi, gm in enumerate(gmask):
            cur = out[gm]
            if prev[gi] is not None:
                grew = int(((cur == 1) & (prev[gi] == 0)).sum())
                viol[gi]["n_steps_checked"] += 1
                if grew > viol[gi]["grew_frames"]:
                    viol[gi]["grew_frames"] = grew
                    viol[gi]["trigger_pair"] = [float(prev_t[gi]), t]
                if pos is not None:
                    psub = pos[gm]
                    s_now = float(((cur == 1) & psub).sum() / max(int(psub.sum()), 1))
                    d = s_now - prev_sens[gi]
                    if d > viol[gi]["sens_increase"]:
                        viol[gi]["sens_increase"] = float(d)
                        if viol[gi]["trigger_pair"] is None:
                            viol[gi]["trigger_pair"] = [float(prev_t[gi]), t]
                    prev_sens[gi] = s_now
            else:
                if pos is not None:
                    psub = pos[gm]
                    prev_sens[gi] = float(((cur == 1) & psub).sum() / max(int(psub.sum()), 1))
            prev[gi] = cur.copy()
            prev_t[gi] = t
        curve.append(row)

    tol = float(tol)
    per_group = {}
    for gi, k in enumerate(gkeys):
        ok = (viol[gi]["grew_frames"] <= tol) and (viol[gi]["sens_increase"] <= 1e-9)
        per_group["pooled" if k is None else str(k)] = {
            "ok": bool(ok), "max_frames_turned_on": int(viol[gi]["grew_frames"]),
            "max_sens_increase": round(float(viol[gi]["sens_increase"]), 6),
            "worst_trigger_pair": viol[gi]["trigger_pair"],
            "n_steps_checked": int(viol[gi]["n_steps_checked"])}
    return {"ok": bool(all(v["ok"] for v in per_group.values())),
            "n_triggers": int(len(triggers)),
            "trigger_min": float(triggers[0]), "trigger_max": float(triggers[-1]),
            "tolerance_frames": int(tol),
            "pooled": per_group.get("pooled"),
            "per_group": {k: v for k, v in per_group.items() if k != "pooled"},
            "n_groups_failing": int(sum(1 for v in per_group.values() if not v["ok"])),
            "curve": curve}

def assert_monotone_trigger_sweep(*a, **kw):
    """Hard-fail wrapper around check_monotone_trigger_sweep (see its docs)."""
    rep = check_monotone_trigger_sweep(*a, **kw)
    if not rep["ok"]:
        bad = {k: v for k, v in rep["per_group"].items() if not v["ok"]}
        raise AssertionError(
            f"non-monotone trigger sweep: {rep['n_groups_failing']} group(s) grew "
            f"their fired set when the trigger was raised; pooled="
            f"{rep['pooled']}; worst groups={list(bad.items())[:5]}")
    return rep

def _events_from_binary(binary, ids):
    """v25.58 / Round 28bx: O(T) global run detection.

    The old body rebuilt an `ids == uid` boolean mask for EVERY record, making
    it O(n_records * n_frames): ~2.3e9 comparisons per call on the pooled
    9.3M-frame / 1088-record pool, and it is invoked on a 120-point grid every
    epoch during training. This version finds run boundaries with two O(T)
    comparisons over the whole array.

    Output is identical: the same (uid, start, end) tuples in the same order
    (grouped by record in np.unique order, ascending start within a record),
    which is guaranteed by the explicit sort on (uid, start).

    NOTE ON THE INDEX CONVENTION: `start`/`end` are offsets RELATIVE TO THE
    RECORD, not global frame indices. The old loop ran np.where() on the
    per-record subarray, so every caller indexes into a per-record slice too
    (onset_latency_report, episode_rescorer, external_adapt_final,
    daphnet_greedy_soup, clinical/temporal_label_smoothing). Returning global
    indices here silently breaks all of them, so the relative convention is
    preserved exactly (rec_start_idx[rec_id] recovers each record's origin).
    """
    binary = np.asarray(binary)
    ids = np.asarray(ids)
    if binary.size == 0:
        return []
    on = binary == 1
    if not on.any():
        return []
    T = len(on)
    rec_chg = ids[1:] != ids[:-1]
    rec_id = np.concatenate(([0], np.cumsum(rec_chg)))
    rec_start_idx = np.concatenate(([0], np.flatnonzero(rec_chg) + 1))
    starts_mask = np.empty(T, dtype=bool)
    ends_mask = np.empty(T, dtype=bool)
    starts_mask[0] = True
    ends_mask[-1] = True
    if T > 1:
        starts_mask[1:] = (~on[:-1]) | rec_chg
        ends_mask[:-1] = (~on[1:]) | rec_chg
    starts_mask &= on
    ends_mask &= on
    st = np.flatnonzero(starts_mask)
    en = np.flatnonzero(ends_mask)
    off = rec_start_idx[rec_id]
    events = [(ids[s], int(s - off[s]), int(e - off[e])) for s, e in zip(st, en)]
    events.sort(key=lambda t: (t[0], t[1]))  # == np.unique(ids) grouping order
    return events

def episode_f1(labels, probs, threshold, ids, pred_binary=None):
    pred = pred_binary if pred_binary is not None else (probs >= threshold).astype(np.int64)
    labels = np.asarray(labels).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)

def window_level_f1(labels, probs, ids, win_frames=100, threshold=0.5):
    """v25.11: contest-comparable WINDOW-level All-FoG F1 (the granularity of
    the Nature Communications 2024 Table 1 operating points: window F1
    0.73-0.81 / sens 0.72-0.79 for the contest top-5 on the same tdcs+defog
    data family). This is NOT the event-level clinical metric (event_level_f1 /
    FARh) and NOT the Kaggle mAP -- it exists so internal numbers can be
    compared against the published same-distribution ceiling (Gate 2).
    v25.37-perf: O(n) record grouping via id blocks (was `ids == uid` per
    record -- quadratic; the fixed-grid selection calls this 8x per epoch and
    the v25.37 log showed the per-epoch eval tripling to ~137 s). Fallback to
    the old boolean-mask loop for non-contiguous id arrays. Bit-identical
    outputs (verified by test); no checkpoint/semantics change."""
    labels = np.asarray(labels, dtype=np.float32)
    ids = np.asarray(ids)
    probs = np.asarray(probs, dtype=np.float64)
    uniq_ids, first_idx, id_counts = np.unique(ids, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    uniq_ids, id_counts = uniq_ids[order], id_counts[order]
    contiguous = bool(np.array_equal(ids, np.repeat(uniq_ids, id_counts)))
    y_w, p_w = [], []
    if contiguous:
        bounds = np.cumsum(id_counts) - id_counts
        for b, c in zip(bounds, id_counts):
            L = int(c)
            lbl = labels[b:b + c]
            pr = probs[b:b + c]
            for s in range(0, L, win_frames):
                e = min(s + win_frames, L)
                if e - s < win_frames // 2:
                    break  # drop trailing fragment
                y_w.append(1.0 if lbl[s:e].max() > 0.5 else 0.0)
                p_w.append(float(pr[s:e].mean()))
    else:
        for uid in uniq_ids:
            m = ids == uid
            L = int(m.sum())
            for s in range(0, L, win_frames):
                e = min(s + win_frames, L)
                if e - s < win_frames // 2:
                    break  # drop trailing fragment
                y_w.append(1.0 if labels[m][s:e].max() > 0.5 else 0.0)
                p_w.append(float(probs[m][s:e].mean()))
    if not y_w:
        return {"window_f1": 0.0, "sensitivity": 0.0, "specificity": 1.0, "n_windows": 0}
    y_w = np.asarray(y_w); p_w = np.asarray(p_w)
    pred = (p_w >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y_w == 1)).sum())
    fp = int(((pred == 1) & (y_w == 0)).sum())
    fn = int(((pred == 0) & (y_w == 1)).sum())
    tn = int(((pred == 0) & (y_w == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"window_f1": 2 * prec * rec / max(prec + rec, 1e-8),
            "sensitivity": rec, "specificity": tn / max(tn + fp, 1),
            "precision": prec, "n_windows": len(y_w)}

def window_level_metrics(labels, probs, ids, win_frames=100, threshold=0.5):
    """v25.22: journal-grade WINDOW-level metrics at the deployed (frozen)
    threshold: sensitivity/specificity/precision/F1, plus threshold-free AUC
    and PR-AUC (the discrimination metrics top journals expect; PR-AUC is the
    appropriate summary under class imbalance -- Borzi 2025 reported external
    PR-AUC 0.59-0.75; Lin & Grewal 2026 reported window AUC lab 1.00 vs home
    0.77). Non-overlapping 1 s windows per record (same grid as
    window_level_f1)."""
    labels = np.asarray(labels, dtype=np.float32)
    ids = np.asarray(ids)
    y_w, p_w = [], []
    for uid in np.unique(ids):
        m = ids == uid
        L = int(m.sum())
        for s in range(0, L, win_frames):
            e = min(s + win_frames, L)
            if e - s < win_frames // 2:
                break
            y_w.append(1.0 if labels[m][s:e].max() > 0.5 else 0.0)
            p_w.append(float(probs[m][s:e].mean()))
    if not y_w:
        return {"sensitivity": 0.0, "specificity": 1.0, "precision": 0.0,
                "f1": 0.0, "auc": float("nan"), "pr_auc": float("nan"), "n_windows": 0}
    y_w = np.asarray(y_w); p_w = np.asarray(p_w)
    pred = (p_w >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y_w == 1)).sum())
    fp = int(((pred == 1) & (y_w == 0)).sum())
    fn = int(((pred == 0) & (y_w == 1)).sum())
    tn = int(((pred == 0) & (y_w == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    auc = float(roc_auc_score(y_w, p_w)) if len(np.unique(y_w)) == 2 else float("nan")
    pr_auc = float(average_precision_score(y_w, p_w)) if y_w.sum() > 0 else float("nan")
    return {"sensitivity": rec, "specificity": tn / max(tn + fp, 1), "precision": prec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-8), "auc": auc, "pr_auc": pr_auc,
            "n_windows": len(y_w)}

def conformal_trigger(calib_probs, calib_labels, calib_ids, alpha=0.10, win_frames=100):
    """v25.31: window-level split-conformal alarm threshold.

    Nonconformity for NEGATIVE windows is s = 1 - p (higher = more alarming).
    The trigger is t = 1 - s_k where s_k is the k-th largest negative-window
    score with k = ceil((n+1)(1-alpha)) -- the split-conformal quantile that
    guarantees the expected alarm rate on negative windows of a NEW split is
    <= alpha + 1/(n+1). This absorbs per-split probability-scale shifts that
    break a single global sigmoid threshold (the v25.29 0.495-trigger misfire:
    the quantile is fit to THIS calib split's own scale, so it transfers by
    construction). Returns (trigger, n_neg_windows) or (None, 0) when calib
    has no negative windows."""
    labels = np.asarray(calib_labels, dtype=np.float32)
    probs = np.asarray(calib_probs, dtype=np.float32)
    ids = np.asarray(calib_ids)
    p_w, y_w = [], []
    for uid in np.unique(ids):
        m = ids == uid
        L = int(m.sum())
        for s in range(0, L, win_frames):
            e = min(s + win_frames, L)
            if e - s < win_frames // 2:
                break
            y_w.append(1.0 if labels[m][s:e].max() > 0.5 else 0.0)
            p_w.append(float(probs[m][s:e].mean()))
    y_w = np.asarray(y_w)
    p_w = np.asarray(p_w)
    neg_scores = (1.0 - p_w)[y_w == 0.0]
    n = int(len(neg_scores))
    if n == 0:
        return None, 0
    if _flag("NEUROMAMBA_CONFORMAL_SSBC", "0"):  # v25.57: _flag parse
        alpha = ssbc_correct_alpha(alpha, len(np.unique(ids)))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = int(min(max(k, 1), n))
    s_k = float(np.sort(neg_scores)[::-1][k - 1])
    return float(np.clip(1.0 - s_k, 0.0, 1.0)), n

def joint_conformal_operating_points(labels, probs, ids, subjects,
                                     triggers=None, sens_target=0.85,
                                     farh_budget=10.0, alpha=0.10,
                                     confirm_n=3, cooldown_n=30, iou_thresh=0.1):
    """v25.44-novel: CERTIFIED JOINT OPERATING POINTS -- distribution-free
    finite-sample lower bound on event sensitivity AND upper bound on the
    per-subject FAR-budget violation rate, simultaneously, for EVERY trigger
    on a grid (report-only; never used to select or deploy).

    Method (closed-form Learn-then-Test logic; Angelopoulos et al.,
    arXiv:2110.01052 + Conformal Risk Control arXiv:2208.02814):
      - sensitivity: X ~ Binomial(n_events, sens). Clopper-Pearson LOWER
        bound at level delta => P(sens >= lb) >= 1 - delta exactly.
        lb = Beta.ppf(delta; tp, n_ev - tp + 1)   (lb = 0 when tp = 0).
      - false alarms: per subject, Y_s in {0,1} = [FARh_s > budget].
        CP UPPER bound on the failure probability at level delta =>
        P(p_fail <= ub) >= 1 - delta.
          ub = Beta.ppf(1 - delta; failures + 1, n_subj - failures).
      - Bonferroni over BOTH families and m triggers: delta = alpha / (2m),
        so the family-wise error of any FALSE certification claim is <= alpha.
    A trigger is CERTIFIED iff lb >= sens_target AND ub <= alpha. The
    guarantee is marginal over the calibration draw and requires
    exchangeability (calibration split vs future subjects); it is reported as
    a certified envelope, not a clinical promise.

    Returns {"alpha", "delta", "n_triggers", "rows": [...], "certified": [...],
             "note"} -- rows sorted by descending trigger."""
    from scipy.stats import beta as _beta
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs)
    ids = np.asarray(ids)
    subjects = np.asarray(subjects) if subjects is not None else ids
    uniq_subjects = np.unique(subjects)
    true_events = _events_from_binary(labels, ids)
    n_ev = len(true_events)
    n_subj = len(uniq_subjects)
    if n_ev < 1 or n_subj < 2:
        return {"alpha": float(alpha), "delta": None, "n_triggers": 0,
                "rows": [], "certified": [],
                "note": f"not certifiable: {n_ev} events / {n_subj} subjects on calib"}
    if triggers is None:
        triggers = np.quantile(probs, np.linspace(0.99, 0.30, 24))
    triggers = [float(t) for t in np.sort(np.asarray(triggers))[::-1]]
    m = max(len(triggers), 1)
    delta = float(alpha) / (2.0 * m)
    subj_hours = {}
    for sid in uniq_subjects:
        sm = subjects == sid
        subj_hours[sid] = max(int(sm.sum()) / 100.0 / 3600.0, 1e-6)
    rows = []
    for trig in triggers:
        pred = causal_event_postprocess(probs, ids, trigger=float(trig),
                                        sustain=float(trig) * 0.4,
                                        confirm_n=int(confirm_n), cooldown_n=int(cooldown_n),
                                        merge_gap_n=MERGE_GAP_N_DEFAULT)
        pred_events = _events_from_binary(pred.astype(np.int64), ids)
        tp_ev = 0
        matched_pred = set()
        for (tuid, ts, te) in true_events:
            best_iou, best_pi = 0.0, -1
            for pi_, (puid, ps, pe) in enumerate(pred_events):
                if puid != tuid or pi_ in matched_pred:
                    continue
                inter = max(0, min(te, pe) - max(ts, ps) + 1)
                union = (te - ts + 1) + (pe - ps + 1) - inter
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi_
            if best_iou >= iou_thresh:
                tp_ev += 1
                matched_pred.add(best_pi)
        sens_lb = float(_beta.ppf(delta, tp_ev, n_ev - tp_ev + 1)) if tp_ev > 0 else 0.0
        failures = 0
        for sid in uniq_subjects:
            sm = subjects == sid
            farh_s = false_alarm_events_per_hour(labels[sm], probs[sm], float(trig),
                                                 ids[sm], pred_binary=pred[sm],
                                                 fs_hz=100.0)["farh_events_per_hour"]
            if np.isfinite(farh_s) and farh_s > farh_budget:
                failures += 1
        fail_ub = float(_beta.ppf(1.0 - delta, failures + 1, max(n_subj - failures, 0))) \
            if n_subj - failures > 0 else 1.0
        rows.append({
            "trigger": float(trig),
            "tp_events": int(tp_ev), "n_events": int(n_ev),
            "sens_point": float(tp_ev / max(n_ev, 1)),
            "sens_lower_bound": float(sens_lb),
            "farh_budget_violations": int(failures), "n_subjects": int(n_subj),
            "violation_rate_upper_bound": float(fail_ub),
            "sens_certified": bool(sens_lb >= float(sens_target)),
            "farh_certified": bool(fail_ub <= float(alpha)),
        })
        rows[-1]["joint_certified"] = bool(rows[-1]["sens_certified"] and rows[-1]["farh_certified"])
    certified = [r for r in rows if r["joint_certified"]]
    note = ("Clopper-Pearson exact bounds, Bonferroni delta=alpha/(2m) across "
            f"{m} triggers x 2 risk families; exchangeability required; "
            "report-only (never selects/deploys). Ref: Learn-then-Test "
            "(arXiv:2110.01052); Conformal Risk Control (arXiv:2208.02814)")
    if certified:
        _c = certified[0]
        print(f"[joint-certify v25.44] {len(certified)}/{m} CERTIFIED operating points "
              f"(alpha={alpha:.2f}): best trigger={_c['trigger']:.3f} "
              f"sens>={_c['sens_lower_bound']:.2f} ({_c['sens_point']:.2f} observed, "
              f"{_c['tp_events']}/{_c['n_events']} events), "
              f"P(FARh>{farh_budget:g}/h)<={_c['violation_rate_upper_bound']:.2f} "
              f"({int(_c['farh_budget_violations'])}/{_c['n_subjects']} calib subjects violate)", flush=True)
    else:
        print(f"[joint-certify v25.44] 0/{m} certified operating points at alpha={alpha:.2f} "
              f"(sens_target={sens_target:.2f}, FARh budget={farh_budget:g}/h) -- "
              f"no trigger carries the joint guarantee on this calib split", flush=True)
    return {"alpha": float(alpha), "delta": delta, "n_triggers": m,
            "rows": rows, "certified": certified, "note": note}

def event_budget_conformal_trigger(labels, probs, ids, subjects, budget_farh=10.0,
                                   alpha=0.10, n_grid=40, confirm_n=3, cooldown_n=30):
    """v25.42: recording/subject-level FAR-budget conformal trigger.

    Window-level conformal guarantees the negative-WINDOW alarm rate, which
    does not translate to false alarms per HOUR in long correlated recordings
    (observed: calib FARh ~9/h vs held-out ~31/h). This calibrates the
    fraction of SUBJECTS whose post-processed false-alarm rate stays within
    `budget_farh` events/hour: pick the LARGEST trigger with
    P(FARh <= budget) >= 1 - alpha across calib subjects. Returns
    (trigger, frac_within_budget) or (None, 0.0) when no trigger qualifies."""
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    ids = np.asarray(ids)
    subjects = np.asarray(subjects) if subjects is not None else ids
    qs = np.quantile(probs, np.linspace(0.30, 0.99, n_grid))[::-1]  # high -> low
    for trig in qs:
        per_subj = []
        for subj in np.unique(subjects):
            m = subjects == subj
            pred = causal_event_postprocess(probs[m], ids[m], trigger=float(trig),
                                            sustain=float(trig) * 0.4,
                                            confirm_n=confirm_n, cooldown_n=cooldown_n,
                                            merge_gap_n=MERGE_GAP_N_DEFAULT)
            farh = false_alarm_events_per_hour(labels[m], probs[m], float(trig),
                                               ids[m], pred_binary=pred, fs_hz=100.0)
            _fv = farh["farh_events_per_hour"]
            per_subj.append(float(_fv) if np.isfinite(_fv) else float("inf"))
        frac = float((np.asarray(per_subj) <= budget_farh).mean())
        if frac >= 1.0 - alpha:
            return float(trig), frac
    return None, 0.0

def estimate_binary_prior_em(probs, source_prior, n_iter=100, tol=1e-8):
    """v25.31: MLLS (Saerens et al. 2002) -- estimate the target fog prior
    from calibrated target probabilities + the source prior, via EM on the
    binary label-shift model. NOTE: the naive fixed point q = E[q*p/(q*p +
    (1-q)(1-p))] is only valid when the classifier's outputs are calibrated
    under prior 0.5; for a source prior p_s != 0.5 the posterior must be
    recomputed from the likelihood ratio implied by the outputs:
        L_i = p_i(1-p_s) / (p_s(1-p_i))
        P_t(1|x) = q*L_i / (q*L_i + 1-q)
    which is the consistency condition from the unified view (Alexandari et
    al., NeurIPS 2020). The corrected form converges to the true target prior
    (verified numerically: 0.099 vs the naive form collapsing to 0)."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ps = float(np.clip(source_prior, 1e-6, 1.0 - 1e-6))
    a = ps / (1.0 - ps)  # source-prior correction factor
    q = ps
    for _ in range(n_iter):
        num = q * probs
        den = num + (1.0 - q) * (1.0 - probs) * a
        post = num / np.clip(den, 1e-12, None)
        q_new = float(np.clip(post.mean(), 1e-6, 1.0 - 1e-6))
        if abs(q_new - q) < tol:
            q = q_new
            break
        q = q_new
    return q

def correct_probs_binary(probs, source_prior, target_prior):
    """v25.31: TTLSA-style output correction -- reweight probabilities by the
    estimated label-shift ratio w(y) = q_t(y)/p_s(y) before the threshold
    search (fixes the per-cohort prior mismatch at its source)."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ps = float(np.clip(source_prior, 1e-6, 1.0 - 1e-6))
    qt = float(np.clip(target_prior, 1e-6, 1.0 - 1e-6))
    w = (qt / ps) / ((1.0 - qt) / (1.0 - ps))
    num = w * probs
    return num / np.clip(num + (1.0 - probs), 1e-12, None)

def wnet_weighted_prior(pred_probs, pseudo_labels, source_prior,
                        iters=300, lr=1e-2, hidden=16, seed=0):
    """v25.32: DWLR stage-1 (Li et al., IJCAI 2024) -- a learnable
    reweighting network (WNet) trained on the paper's 3-term objective
    (L_label = KL(prior || weighted pseudo-label dist), L_IG = info-gain,
    L_conf = confidence regularizer) over the frozen model's per-sequence
    predictions and pseudo-labels. Returns the batch-normalized per-sequence
    weights and the weighted pseudo-label prior sum(w_i*pseudo_i)/sum(w_i) --
    the learnable alternative to the EM prior estimate (replaces the hand-tuned
    SH subject-diversity cap with a data-driven allocation)."""
    probs = np.clip(np.asarray(pred_probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    P = np.stack([1.0 - probs, probs], axis=1).astype(np.float32)
    y = np.asarray(pseudo_labels, dtype=np.int64)
    ps = float(np.clip(source_prior, 1e-6, 1.0 - 1e-6))
    torch.manual_seed(seed)
    prior = torch.tensor([1.0 - ps, ps], dtype=torch.float32)
    Pt = torch.tensor(P)
    yt = torch.tensor(y)
    net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    B = float(Pt.shape[0])
    for _ in range(iters):
        w = torch.softmax(net(Pt).squeeze(-1), dim=0) * B
        onehot = torch.nn.functional.one_hot(yt, 2).float()
        qw = (w.unsqueeze(1) * onehot).sum(0)
        qw = qw / (qw.sum() + 1e-8)
        l_label = (prior * (prior.clamp_min(1e-8).log() - qw.clamp_min(1e-8).log())).sum()
        h_post = -(Pt * Pt.clamp_min(1e-8).log()).sum(1)
        h_prior = -(prior * prior.clamp_min(1e-8).log()).sum()
        l_ig = -((h_prior - h_post) * w).mean()
        l_conf = (Pt.max(1).values * w).mean()
        loss = l_label + l_ig + l_conf
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = torch.softmax(net(Pt).squeeze(-1), dim=0) * B
    w_np = w.detach().cpu().numpy().astype(np.float64)
    q_w = float(np.clip((w_np * y).sum() / max(w_np.sum(), 1e-9), 1e-6, 1.0 - 1e-6))
    return w_np, q_w

def entropy_min_tta(model, cohort_seqs, device, steps=40, lr=1e-4, chunk=1200,
                    window_roll=True, activity_gate=False, activity_frac=0.5):
    """v25.32: unsupervised entropy-minimization test-time adaptation
    (NEUROMAMBA_TTA_ENTROPY=1). Freezes the backbone and optimizes ONLY the
    fog head to minimize -sum(p*log p) over the target cohort's own unlabeled
    frames -- the robust TTA alternative to contrastive methods that collapse
    on homogeneous FoG streams (88-99% non-FoG). Returns the adapted model
    copy. Skipped by the caller when the cohort's mean fog prob is extreme
    (homogeneous stream guard)."""
    m = copy.deepcopy(model)
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith("fog_head") or n.startswith("tok_")
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    m.train()
    xs, _acts = [], []
    for s in cohort_seqs:
        sig = np.asarray(s.signal, dtype=np.float32)
        for st in range(0, len(sig), chunk):
            _c = sig[st:st + chunk]
            xs.append(_c)
            _acts.append(float(np.std(np.linalg.norm(_c[:, :3], axis=1)))
                         if _c.size and _c.shape[1] >= 3 else 0.0)
    if activity_gate and xs:
        _thr = float(np.quantile(_acts, max(0.0, 1.0 - float(activity_frac))))
        _keep = [i for i, a in enumerate(_acts) if a >= max(_thr, 1e-6)]
        print(f"[tta v25.60] activity gate: kept {len(_keep)}/{len(xs)} chunks "
              f"(top {float(activity_frac):.0%} by accel-magnitude std >= {_thr:.4f})", flush=True)
        xs = [xs[i] for i in _keep]
    if not xs:
        return m
    for _ in range(steps):
        for x_np in xs:
            x = torch.from_numpy(x_np[None]).to(device)
            if x.shape[1] < 64:
                continue
            fog_logits, _, aux = m(x)
            p = torch.sigmoid(fog_logits).clamp(1e-6, 1.0 - 1e-6)
            loss = -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m

def event_level_metrics(labels, pred_binary, ids, iou_thresh=0.1):
    """v25.22: journal-grade EVENT-level metrics + episode detection rate
    (fraction of expert-labeled episodes with >=1 detected event at iou>=0.1
    -- Borzi 2025 reported 88-95 % of episodes detected on external datasets;
    Marcante 2021: 90 %). Event-level specificity is ill-defined (no true
    negatives at event granularity); window-level specificity is reported
    separately via window_level_metrics."""
    true_events = _events_from_binary(np.asarray(labels).astype(np.int64), ids)
    pred_events = _events_from_binary(np.asarray(pred_binary).astype(np.int64), ids)
    matched_true, matched_pred = set(), set()
    for ti, (tuid, ts, te) in enumerate(true_events):
        best_iou, best_pi = 0.0, -1
        for pi, (puid, ps, pe) in enumerate(pred_events):
            if puid != tuid or pi in matched_pred:
                continue
            inter = max(0, min(te, pe) - max(ts, ps) + 1)
            union = (te - ts + 1) + (pe - ps + 1) - inter
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= iou_thresh:
            matched_true.add(ti); matched_pred.add(best_pi)
    tp = len(matched_true)
    fp = len(pred_events) - len(matched_pred)
    fn = len(true_events) - len(matched_true)
    if tp == 0 and fp == 0 and fn == 0:
        return {"sensitivity": float("nan"), "precision": float("nan"), "f1": float("nan"),
                "episode_detection_rate": float("nan"), "empty": True,
                "n_true_events": 0, "n_pred_events": 0, "tp": 0, "fp": 0, "fn": 0}
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"sensitivity": rec, "precision": prec, "f1": f1,
            "episode_detection_rate": rec, "empty": False,
            "n_true_events": len(true_events), "n_pred_events": len(pred_events),
            "tp": tp, "fp": fp, "fn": fn}

def _greedy_match_pairs(true_events, pred_events, iou_thresh=0.1):
    """v25.58 / Round 28bx: the ONE greedy one-to-one best-IoU matching used by
    every event-level metric. Before this refactor event_level_f1 matched with
    IoU >= iou_thresh while false_alarm_events_per_hour declared a prediction a
    false alarm only when it overlapped NO true frame at all -- two different
    definitions of the same event, so precision and FARh disagreed by
    construction. Both now call this function.

    Also O(sum_r n_true_r * n_pred_r) instead of O(n_true * n_pred): candidates
    are indexed per record, so a 1551 x 3000 sweep collapses to per-record
    windows. Semantics are unchanged (same iteration order, same tie-breaking).
    """
    matched_true, matched_pred = set(), set()
    pred_by_rec = {}
    for pi, (puid, ps, pe) in enumerate(pred_events):
        pred_by_rec.setdefault(puid, []).append((pi, ps, pe))
    for ti, (tuid, ts, te) in enumerate(true_events):
        best_iou, best_pi = 0.0, -1
        for pi, ps, pe in pred_by_rec.get(tuid, ()):
            if pi in matched_pred:
                continue
            inter = max(0, min(te, pe) - max(ts, ps) + 1)
            union = (te - ts + 1) + (pe - ps + 1) - inter
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= iou_thresh:
            matched_true.add(ti); matched_pred.add(best_pi)
    return matched_true, matched_pred

def _count_unmatched_pred(pred_events, true_events, rule="any", iou_thresh=0.1):
    """Count predictions that count as false alarms under `rule`.
    'any' (legacy): prediction overlaps no true frame.
    'iou'         : prediction is not matched by _greedy_match_pairs."""
    if not pred_events:
        return 0
    if rule == "iou":
        _, matched_pred = _greedy_match_pairs(true_events, pred_events, iou_thresh)
        return len(pred_events) - len(matched_pred)
    by_rec = {}
    for uid, s, e in true_events:
        by_rec.setdefault(uid, []).append((s, e))
    n_false = 0
    for uid, s, e in pred_events:
        tr = by_rec.get(uid)
        if not tr:
            n_false += 1
            continue
        hit = False
        for ts, te in tr:          # sorted by start within a record
            if ts > e:
                break
            if te >= s:
                hit = True
                break
        if not hit:
            n_false += 1
    return n_false

def event_level_f1(labels, pred_binary, ids, iou_thresh=0.1):
    true_events = _events_from_binary(np.asarray(labels).astype(np.int64), ids)
    pred_events = _events_from_binary(np.asarray(pred_binary).astype(np.int64), ids)
    matched_true, matched_pred = _greedy_match_pairs(true_events, pred_events, iou_thresh)
    tp = len(matched_true)
    fp = len(pred_events) - len(matched_pred)
    fn = len(true_events) - len(matched_true)
    if tp == 0 and fp == 0 and fn == 0:
        return float("nan")
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)

def nan_safe_mean(values):
    """v25.44: mean over finite values only, with explicit evaluability
    counters -- the aggregation contract for any metric that may legitimately
    be nan (empty folds, empty slices). Returns (mean_or_nan, n_evaluable,
    n_total). A nan mean is a HONEST 'not evaluable', never a score."""
    vals = [float(v) for v in (values or []) if v is not None and np.isfinite(v)]
    n_total = len(values or [])
    if not vals:
        return float("nan"), 0, n_total
    return float(np.mean(vals)), len(vals), n_total

def false_alarm_events_per_hour(labels, probs, threshold, ids, pred_binary=None,
                                fs_hz=100.0, overlap_rule=None, iou_thresh=0.1):
    """False-alarm EVENTS per hour.

    v25.58 / Round 28bx, two fixes:
    1. PERFORMANCE. The old body built `{uid: labels[ids == uid] for uid in
       unique(ids)}` -- O(n_records * n_frames) boolean comparisons on every
       call (measured 2.9 s per call at 4.3M frames / 529 records, i.e. ~13 s
       on the 9.3M-frame pooled calibration), and it runs on a 120-point grid
       EVERY EPOCH during training. Now O(n_frames) via interval lists.
    2. CONSISTENCY. `overlap_rule` selects how a false alarm is defined:
         "any" (default, legacy): the prediction overlaps no true frame at all.
           LENIENT -- disagrees with the IoU>=0.1 rule event_level_f1 uses, so
           precision and FARh were computed under different definitions.
         "iou": the prediction fails the same greedy IoU matching as
           event_level_f1. SELF-CONSISTENT; use this for the headline budget.
       Opt in with NEUROMAMBA_FARH_OVERLAP_RULE=iou. The default is unchanged
       so historical numbers stay comparable; both are reported side by side.
    """
    if overlap_rule is None:
        overlap_rule = os.environ.get("NEUROMAMBA_FARH_OVERLAP_RULE", "any")
    pred = pred_binary if pred_binary is not None else (probs >= threshold).astype(np.int64)
    labels = np.asarray(labels).astype(np.int64)
    pred_events = _events_from_binary(pred, ids)
    true_events = _events_from_binary(labels, ids)
    n_false = _count_unmatched_pred(pred_events, true_events, overlap_rule, iou_thresh)
    total_hours = len(labels) / fs_hz / 3600.0
    return {"farh_events_per_hour": float(n_false / max(total_hours, 1e-6)),
            "n_false_events": int(n_false), "total_hours": float(total_hours),
            "overlap_rule": overlap_rule,
            "n_pred_events": int(len(pred_events))}

def per_subject_farh_distribution(labels, pred_binary, subjects, fs_hz=100.0):
    out = {}
    for sid in np.unique(subjects):
        m = subjects == sid
        d = false_alarm_events_per_hour(labels[m], None, 0.5, np.full(m.sum(), sid), pred_binary=pred_binary[m], fs_hz=fs_hz)
        out[str(sid)] = d["farh_events_per_hour"]
    vals = np.array(list(out.values())) if out else np.array([0.0])
    return {"per_subject": out, "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
            "worst_subject": str(max(out, key=out.get)) if out else None}

def _icc21_absolute(values_a, values_b):
    """v25.21: two-way random-effects, single-measure, ABSOLUTE-agreement
    ICC(2,1) (McGraw & Wong 1996) on paired measures (e.g. model output vs
    gold-standard annotation per recording). This is the agreement metric the
    multicentre FoG literature reports -- Salomon et al. Nat Commun 2024
    (ICC %TF 0.877-0.949, #episodes <0.6, duration >0.90) and Yang et al. npj
    PD 2026 (local ICC 0.886 -> external 0.562 -> 0.732 with 50-min
    fine-tuning). Returns None when <3 pairs or degenerate variance."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    n = len(a)
    if n < 3 or (np.all(a == a[0]) and np.all(b == b[0])):
        return None
    k = 2
    x = np.stack([a, b], axis=1)                    # [n recordings, 2 raters]
    row_mean = x.mean(axis=1)                       # per-recording mean
    col_mean = x.mean(axis=0)                       # per-rater mean
    grand = x.mean()
    ssb = n * k * np.sum((col_mean - grand) ** 2)   # between raters
    ssw = k * np.sum((row_mean - grand) ** 2)       # between recordings
    sse = np.sum((x - row_mean[:, None] - col_mean[None, :] + grand) ** 2)
    msb = ssb / max(k - 1, 1)
    msw = ssw / max(n - 1, 1)
    mse = sse / max((n - 1) * (k - 1), 1)
    denom = msw + (k - 1) * mse + k * (msb - mse) / n
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float((msw - mse) / denom)

def icc_report(labels, pred, ids, fs_hz=100.0):
    """v25.21: per-recording agreement ICC(2,1) between model output and
    gold-standard labels for the three contest/multicentre outcome measures:
    percent time frozen (%TF), number of FoG episodes, and total FoG duration.
    Aggregated per recording (the unit used by Salomon 2024 and Yang 2026),
    so a cohort of N recordings yields N paired measures per outcome."""
    labels = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64)
    ids = np.asarray(ids)
    pct, ne, dur = [], [], []
    for uid in np.unique(ids):
        m = ids == uid
        y = labels[m]
        p = pred[m]
        pct.append((float(p.mean()), float(y.mean())))
        ne.append((float(len(_events_from_binary(p, np.full(len(p), uid)))),
                   float(len(_events_from_binary(y.astype(np.int64), np.full(len(y), uid))))))
        dur.append((float(p.sum() / fs_hz), float(y.sum() / fs_hz)))
    return {
        "icc_percent_time_frozen": _icc21_absolute([v[0] for v in pct], [v[1] for v in pct]),
        "icc_n_episodes": _icc21_absolute([v[0] for v in ne], [v[1] for v in ne]),
        "icc_duration_s": _icc21_absolute([v[0] for v in dur], [v[1] for v in dur]),
        "n_recordings": len(pct),
    }

def worst_fold_collapse_check(event_f1_list, min_f1=0.05):
    valid = [f for f in event_f1_list if f is not None and np.isfinite(f)]
    if not valid:
        return {"collapsed": True, "reason": "no valid folds"}
    return {"collapsed": bool(min(valid) <= min_f1), "worst_f1": float(min(valid)), "min_threshold": min_f1}

def compute_official_like_map(sub_labels, sub_probs, observed, classes=(0, 1, 2)):
    """v25: observed can be [N] bool (broadcast) OR [N,3] per-class bool.
    For each subtype k, score only on frames where observed[:,k] is True.
    This stops 'unknown' frames (external binary-only cohorts) from being
    treated as hard negatives -- the direct cause of rare-subtype collapse.
    v25.20-B: `classes` restricts which subtypes enter the mAP MEAN (all
    per-class APs are still computed and returned). The winner's defog
    validation used classes=(1,2) -- StartHesitation excluded."""
    out = {"comparable": True}
    aps = []
    for k, name in enumerate(SUBTYPE_NAMES):
        if observed.ndim == 2:
            obs_k = observed[:, k]
        else:
            obs_k = observed
        y = sub_labels[obs_k, k] if observed.ndim == 2 else sub_labels[obs_k, k]
        p = sub_probs[obs_k, k] if observed.ndim == 2 else sub_probs[obs_k, k]
        if y.sum() == 0:
            out[name] = 0.0
            if k in classes:
                out["comparable"] = False
            continue
        try:
            ap = average_precision_score((y > 0.5).astype(int), p)
        except Exception:
            ap = 0.0
        out[name] = float(ap)
        if k in classes:
            aps.append(ap)
    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    out["n_comparable"] = len(aps)
    return out

def compute_kaggle_map(sub_probs, sub_labels, sub_observed):
    m = compute_official_like_map(sub_labels, sub_probs, sub_observed)
    return {"mAP": m["mAP"], "comparable": m["comparable"], **{k: m[k] for k in SUBTYPE_NAMES}}

def ensemble_kaggle_map(prob_list, labels_ref, observed_ref=None):
    """Average multiple models' predictions on the SAME evaluation samples."""
    shapes = {p.shape for p in prob_list}
    if len(shapes) > 1:
        raise ValueError(
            f"ensemble_kaggle_map: prob_list arrays have mismatched shapes {shapes}. "
            "Use compute_kaggle_map on concatenated pooled OOF arrays instead.")
    avg = np.mean(np.stack(prob_list, axis=0), axis=0)
    return compute_kaggle_map(avg, labels_ref, observed_ref if observed_ref is not None else np.ones(len(labels_ref), dtype=bool))

def bootstrap_kaggle_map_ci(sub_probs, sub_labels, sub_observed, n_boot=200, seed=0, groups=None):
    """v25-FIX-9: SUBJECT-LEVEL bootstrap (not frame-level). If groups (subject
    or record ids) is provided, resample unique groups; otherwise fall back to
    per-frame (with a warning). The prior frame-level bootstrap gave falsely
    tight CIs because frames within a subject are highly correlated.
    v25.42.1: n_boot overridable via NEUROMAMBA_BOOTSTRAP_N (use >= 5000 for
    final publication CIs; 200 is fine for development screening)."""
    n_boot = int(os.environ.get("NEUROMAMBA_BOOTSTRAP_N", str(n_boot)))
    rng = np.random.RandomState(seed)
    n = len(sub_probs)
    maps = []
    if groups is not None:
        uniq_groups = np.unique(groups)
        group_to_idx = {g: np.where(groups == g)[0] for g in uniq_groups}
        for _ in range(n_boot):
            sampled_groups = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
            m = compute_official_like_map(sub_labels[idx], sub_probs[idx],
                                          sub_observed[idx] if sub_observed is not None else None)
            if m["comparable"]:
                maps.append(m["mAP"])
    else:
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            m = compute_official_like_map(sub_labels[idx], sub_probs[idx],
                                          sub_observed[idx] if sub_observed is not None else None)
            if m["comparable"]:
                maps.append(m["mAP"])
    if not maps:
        unit = "subject" if groups is not None else "frame"
        print(f"[bootstrap v25.34] WARNING: 0/{n_boot} bootstrap samples were comparable "
              f"({len(np.unique(groups)) if groups is not None else n} {unit}s) -- CI and "
              f"gate_2 (beats baseline) are NON-COMPUTABLE, not zero", flush=True)
        return {"ci_lo": 0.0, "ci_hi": 0.0, "mean": 0.0, "n_valid_boot": 0, "unit": unit}
    return {"ci_lo": float(np.percentile(maps, 2.5)), "ci_hi": float(np.percentile(maps, 97.5)),
            "mean": float(np.mean(maps)), "n_valid_boot": len(maps),
            "unit": "subject" if groups is not None else "frame"}

def kaggle_map_ci_diff(sub_probs_a, sub_probs_b, sub_labels, sub_observed, n_boot=200, seed=0, groups=None):
    n_boot = int(os.environ.get("NEUROMAMBA_BOOTSTRAP_N", str(n_boot)))
    rng = np.random.RandomState(seed)
    n = min(len(sub_probs_a), len(sub_probs_b), len(sub_labels))
    diffs = []
    if groups is not None:
        groups = np.asarray(groups)[:n]
        uniq_groups = np.unique(groups)
        group_to_idx = {g: np.where(groups == g)[0] for g in uniq_groups}
        for _boot in range(n_boot):
            sampled_groups = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
            ma = compute_official_like_map(sub_labels[idx], sub_probs_a[idx], sub_observed[idx] if sub_observed is not None else None)
            mb = compute_official_like_map(sub_labels[idx], sub_probs_b[idx], sub_observed[idx] if sub_observed is not None else None)
            if ma["comparable"] and mb["comparable"]:
                diffs.append(ma["mAP"] - mb["mAP"])
            if (_boot + 1) % 50 == 0:
                print(f"[kaggle-track] baseline-diff bootstrap: {_boot + 1}/{n_boot} boots "
                      f"({len(diffs)} comparable)", flush=True)
    else:
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            ma = compute_official_like_map(sub_labels[idx], sub_probs_a[idx], sub_observed[idx] if sub_observed is not None else None)
            mb = compute_official_like_map(sub_labels[idx], sub_probs_b[idx], sub_observed[idx] if sub_observed is not None else None)
            if ma["comparable"] and mb["comparable"]:
                diffs.append(ma["mAP"] - mb["mAP"])
    if not diffs:
        print(f"[bootstrap v25.34] WARNING: 0/{n_boot} baseline-diff samples were "
              f"comparable -- gate_2 (beats baseline) is NON-COMPUTABLE, not zero", flush=True)
        return {"diff_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "non_overlapping_zero": False, "n_valid_boot": 0}
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {"diff_mean": float(np.mean(diffs)), "ci_lo": ci_lo, "ci_hi": ci_hi, "non_overlapping_zero": bool(ci_lo > 0.0 or ci_hi < 0.0)}

def compute_ece(probs, labels, n_bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(probs)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(probs[m].mean() - labels[m].mean())
    return float(ece)

def brier_score(probs, labels):
    """v25.42.9: mean squared error between predicted probability and label
    (the reviewer's calibration battery: ECE + Brier + slope + intercept)."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))

def calibration_slope_intercept(probs, labels, n_bins=10):
    """v25.42.9: calibration slope + intercept from a reliability-diagram
    least-squares fit (logit of mean predicted vs logit of empirical
    frequency, as in the clinical calibration literature). slope=1,
    intercept=0 = perfectly calibrated. Returns (slope, intercept, r2) or
    (nan, nan, nan) when the fit is degenerate (e.g. empty bins)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    eps = 1e-6
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if m.sum() < 10:
            continue
        p_mean = np.clip(probs[m].mean(), eps, 1 - eps)
        freq = np.clip(labels[m].mean(), eps, 1 - eps)
        xs.append(np.log(p_mean / (1 - p_mean)))
        ys.append(np.log(freq / (1 - freq)))
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.asarray(xs); y = np.asarray(ys)
    slope = float(np.polyfit(x, y, 1)[0])
    intercept = float(np.polyfit(x, y, 1)[1])
    if len(x) > 2:
        r = np.corrcoef(x, y)[0, 1]
        r2 = float(r * r)
    else:
        r2 = float("nan")
    return slope, intercept, r2

def logistic_calibration_slope_intercept(probs, labels, max_iter=100, tol=1e-9):
    """v25.42.10 (reviewer #5): LOGISTIC calibration slope + intercept --
    the strictest journal-grade form: fit the Cox-style calibration model
    logit(P(Y=1)) = alpha + beta * logit(p) by IRLS (iteratively reweighted
    least squares, pure numpy -- the standard logistic-regression fit, not a
    clipped-target shortcut). beta=1, alpha=0 = perfectly calibrated;
    beta<1 = overconfident predictions, beta>1 = underconfident.
    Returns (slope, intercept) or (nan, nan) on degenerate input."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if len(probs) == 0 or len(probs) != len(labels):
        return float("nan"), float("nan")
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - labels.sum())
    if n_pos < 10 or n_neg < 10:
        return float("nan"), float("nan")
    eps = 1e-6
    x = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    X = np.column_stack([np.ones_like(x), x])   # [intercept, slope]
    y = labels.astype(np.float64)
    beta = np.zeros(2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            eta = X @ beta
            p = 1.0 / (1.0 + np.exp(-eta))
            p = np.clip(p, 1e-9, 1.0 - 1e-9)
            w = p * (1.0 - p)
            z = eta + (y - p) / w
            W = np.sqrt(w)
            beta_new = np.linalg.lstsq(X * W[:, None], z * W, rcond=None)[0]
            if np.allclose(beta_new, beta, rtol=0, atol=tol):
                beta = beta_new
                break
            beta = beta_new
    if not np.isfinite(beta).all():
        return float("nan"), float("nan")
    return float(beta[1]), float(beta[0])

def sh_repeat_powered(n_subjects, n_events, max_subject_frac,
                      min_subjects=4, min_events=8, max_subject_frac_lim=0.5):
    """v25.42.10 (reviewer fix #2): SH-power test for ONE repeat. A repeat is
    SH-powered when the held-out StartHesitation split carries enough evidence
    to measure SH performance: >= min_subjects SH-positive subjects, >=
    min_events SH events, and no single subject monopolizing the held-out SH
    events (max_subject_frac <= max_subject_frac_lim). Underpowered repeats are
    EXCLUDED from every SH-stability claim. Defaults mirror the env knobs
    (NEUROMAMBA_MIN_HELD_SH_SUBJECTS / _EVENTS / _MAX_SH_SUBJECT_EVENT_FRAC)."""
    return (int(n_subjects) >= int(min_subjects)
            and int(n_events) >= int(min_events)
            and float(max_subject_frac) <= float(max_subject_frac_lim))

def sh_stability_decision(powered_stats, floor=0.05, min_powered=4, min_frac=0.8):
    """v25.42.10 (reviewer fix #2/#3): SH stability over POWERED repeats only,
    with an explicit denominator. powered_stats: list of dicts (one per POWERED
    repeat) with sh_ap_mean, sh_event_recall_mean, sh_subject_coverage_mean.
    Criterion (multidimensional): SH AP > floor AND event recall > 0 AND
    subject coverage > 0 in >= min_frac of >= min_powered powered repeats.
    floor is an engineering REFERENCE floor (0.05), not a clinical claim.
    Returns (stable, detail)."""
    stable = False
    detail = {"powered_repeats": len(powered_stats), "above_floor": 0,
              "event_recall_nonzero": 0, "subject_coverage_nonzero": 0,
              "stability_fraction": 0.0, "sh_stable": False,
              "criterion": f"AP>{floor} AND event_recall>0 AND subject_coverage>0 "
                           f"in >= {int(min_frac * 100)}% of >= {min_powered} powered repeats"}
    if len(powered_stats) >= min_powered:
        _ap = [r.get("sh_ap_mean") for r in powered_stats if r.get("sh_ap_mean") is not None]
        _rec = [r.get("sh_event_recall_mean") for r in powered_stats
                if r.get("sh_event_recall_mean") is not None]
        _cov = [r.get("sh_subject_coverage_mean") for r in powered_stats
                if r.get("sh_subject_coverage_mean") is not None]
        detail["above_floor"] = int(sum(1 for v in _ap if v and v > floor))
        detail["event_recall_nonzero"] = int(sum(1 for v in _rec if v and v > 0))
        detail["subject_coverage_nonzero"] = int(sum(1 for v in _cov if v and v > 0))
        _need = max(1, int(min_frac * len(powered_stats)))
        stable = (detail["above_floor"] >= _need
                  and detail["event_recall_nonzero"] >= _need
                  and detail["subject_coverage_nonzero"] >= _need)
    detail["sh_stable"] = bool(stable)
    detail["stability_fraction"] = (float(detail["above_floor"]) / len(powered_stats)
                                    if powered_stats else 0.0)
    return bool(stable), detail

def _stable_expit(x):
    """v25.51.2: overflow-free sigmoid. The v25.51.1 widened T-grid can push
    -logits/T beyond float64 exp range (observed: RuntimeWarning 'overflow
    encountered in exp' during per-subtype fitting with T=0.109). The
    piecewise form never exponentiates a positive number."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def fit_temperature(probs, labels, min_positives=20):
    labels = np.asarray(labels)
    if not np.all(np.isfinite(np.asarray(probs, dtype=np.float64))):  # v25.43.0 m3
        print("[calib-guard] non-finite probs in calib stream -- skipping fit, T=1.0", flush=True)
        return {"temperature": 1.0, "skipped": True, "n_positives": int(np.nan_to_num(labels).sum())}
    if labels.sum() < min_positives:
        print(f"[calib-guard] only {int(labels.sum())} positives (<{min_positives}) -- skipping fit, T=1.0", flush=True)
        return {"temperature": 1.0, "skipped": True, "n_positives": int(labels.sum())}
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logits = np.log(p / (1 - p))

    def _nll(T):
        pt = np.clip(_stable_expit(logits / T), eps, 1 - eps)
        return -np.mean(labels * np.log(pt) + (1 - labels) * np.log(1 - pt))
    lo_T, hi_T, n_coarse = 0.05, 64.0, 130
    coarse = np.geomspace(lo_T, hi_T, n_coarse)
    nlls = np.array([_nll(float(T)) for T in coarse])
    _i = int(np.argmin(nlls))
    best_T, best_nll = float(coarse[_i]), float(nlls[_i])
    _ratio = float((hi_T / lo_T) ** (1.0 / (n_coarse - 1)))
    fine = np.geomspace(max(lo_T, coarse[_i] / _ratio), min(hi_T, coarse[_i] * _ratio), 80)
    for T in fine:
        _n = _nll(float(T))
        if _n < best_nll:
            best_nll, best_T = _n, float(T)
    return {"temperature": float(best_T), "skipped": False, "n_positives": int(labels.sum()),
            "boundary": bool(_i == 0 or _i == n_coarse - 1)}

def apply_temperature(probs, T):
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    return _stable_expit(logits / max(T, 1e-3))

def fit_per_subtype_temperature(sub_probs, sub_labels, observed, min_positives=20):
    """v25: fit one T per subtype, using ONLY observed frames for that subtype."""
    temps = []
    for k in range(3):
        if observed.ndim == 2:
            obs_k = observed[:, k]
            y = sub_labels[obs_k, k]
            p = sub_probs[obs_k, k]
        else:
            obs_k = observed
            y = sub_labels[obs_k, k]
            p = sub_probs[obs_k, k]
        if y.sum() < min_positives:
            if y.sum() > 0:
                print(f"[calib-guard] subtype '{SUBTYPE_NAMES[k]}' has only {int(y.sum())} positives (<{min_positives}) -- T=1.0", flush=True)
            temps.append(1.0); continue
        _tres = fit_temperature(p, y, min_positives=min_positives)
        T = _tres["temperature"]
        if _tres.get("boundary"):
            print(f"[calib v25.43] subtype '{SUBTYPE_NAMES[k]}' T={T:.3f} at grid boundary "
                  f"({int(y.sum())} positives) -- keeping boundary fit", flush=True)
        temps.append(T)
    return temps

def apply_temperature_subtype(sub_probs, temps):
    if sub_probs is None:
        return None
    out = np.zeros_like(sub_probs)
    for k in range(3):
        out[:, k] = apply_temperature(sub_probs[:, k], temps[k])
    return out

def logit_adjust_binary(probs, prior, tau):
    """v25.44-novel: POST-HOC LOGIT ADJUSTMENT (Menon et al., ICLR 2021,
    arXiv:2007.07314 -- the inference-time variant) for the independent
    sigmoid subtype heads. Rare-class recall collapses when the head bias
    stays pinned at the training prior; adding -tau*log(pi_k) to the logit
    shifts the rare-class decision boundary WITHOUT retraining (the literature
    consensus first-line remedy when reweighting saturates: the ASL pos_weight
    cap of 20 leaves ~10:1 negative gradient dominance at the measured ~1:200
    SH imbalance). Monotone per channel, so ranking/AP is unchanged for a
    single channel; the GAIN is in thresholded recall/F1 and in the softmax-
    free per-class decision at the deployed trigger. tau=0 (default) is an
    exact no-op. pi_k = the class's TRAINING-split positive rate (observed
    frames only) -- no evaluation labels are touched."""
    if tau is None or tau <= 0:
        return np.asarray(probs)
    eps = 1e-6
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    pi = float(np.clip(prior, 1e-8, 1.0 - 1e-8))
    z = np.log(p / (1.0 - p)) - float(tau) * np.log(pi)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

def apply_logit_adjust_subtype(sub_probs, priors, tau):
    """Per-channel wrapper over logit_adjust_binary. priors: [3] training
    positive rates (SH/Turn/Walking); None/empty -> unchanged."""
    if sub_probs is None or not priors or tau is None or tau <= 0:
        return sub_probs
    out = np.asarray(sub_probs, dtype=np.float32).copy()
    for k in range(3):
        out[:, k] = logit_adjust_binary(out[:, k], priors[k], tau)
    return out

def compute_per_subtype_ece(sub_probs, sub_labels, observed, n_bins=10):
    eces = {}
    for k, name in enumerate(SUBTYPE_NAMES):
        if observed.ndim == 2:
            obs_k = observed[:, k]
        else:
            obs_k = observed
        y = sub_labels[obs_k, k]
        p = sub_probs[obs_k, k]
        eces[name] = compute_ece(p, y, n_bins=n_bins) if len(y) else float("nan")
    valid = [v for v in eces.values() if np.isfinite(v)]
    eces["max_ece"] = float(max(valid)) if valid else float("nan")
    return eces

def _safe_sensitivity(tp, fn):
    denom = tp + fn
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    val = tp / denom
    return float(val) if np.isfinite(val) else 0.0

def quantile_summary(probs, n_q=257):
    """v25.37: tiny sorted-quantile digest of a probability stream (used as the
    reference distribution for label-free score anchoring; JSON-safe)."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    probs = probs[np.isfinite(probs)]
    if len(probs) == 0:
        return np.array([], dtype=np.float64)
    return np.asarray(np.quantile(probs, np.linspace(0.0, 1.0, n_q)), dtype=np.float64)

def anchor_scores_to_reference(probs, ref_quantiles):
    """v25.37: LABEL-FREE score anchoring for frozen cross-cohort deployment
    (histogram matching of a cohort's gated+temperature probability stream to
    the TLVMc-calib reference distribution). Uses NO labels -- the score-level
    analog of the per-recording z-scoring already applied to the inputs; the
    frozen trigger/temperature remain fit on internal calib only. Monotone
    within ties (average-rank ECDF), so within-cohort ranking is preserved
    and ECE on the anchored stream measures pure scale-shift residual.
    Grounded in the observed v25.35 frozen-transfer pattern: one frozen
    trigger cannot serve three cohorts whose score scales differ (FARh
    21.98 daphnet / 4.07 weargait / 1.74 xuanwu = three different scales).
    Returns (anchored_probs, ok_flag)."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    ref = np.asarray(ref_quantiles, dtype=np.float64).ravel()
    if len(probs) < 100 or len(ref) < 2 or not np.all(np.isfinite(ref)):
        return probs.astype(np.float32), False
    if np.any(np.diff(ref) < -1e-12):
        return probs.astype(np.float32), False
    srt = np.sort(probs)
    vals, inv = np.unique(probs, return_inverse=True)
    first = np.searchsorted(srt, vals, side="left")
    last = np.searchsorted(srt, vals, side="right")
    avg_rank = (first + last - 1) / 2.0 / max(len(srt) - 1, 1)  # tie-safe ECDF
    xq = np.linspace(0.0, 1.0, len(ref))
    mapped = np.interp(avg_rank[inv], xq, ref)
    if len(np.unique(mapped)) < 32:
        return probs.astype(np.float32), False
    return mapped.astype(np.float32), True

def _false_event_count(pred_records, labels_records):
    """Count false-alarm EVENTS across pre-split per-record arrays (v25.2-speed:
    avoids re-masking the full concatenated arrays per grid point)."""
    n_false = 0
    for pr, lr in zip(pred_records, labels_records):
        idx = np.flatnonzero(pr)
        if len(idx) == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[splits + 1]))
        ends = np.concatenate((idx[splits], [idx[-1]]))
        for s, e in zip(starts, ends):
            if not lr[s:e + 1].any():
                n_false += 1
    return n_false

def find_clinical_threshold_fast(labels, probs, ids, sens_min=0.85, farh_max=10.0, n_grid=120, q_max=None):
    """v25: denser grid (120 points, was 30) so rare-event operating points
    aren't skipped. Logs frame-level PR-AUC alongside sens/FARh.
    v25.33: fallback triggers are capped at quantile q_max (default 0.95,
    env NEUROMAMBA_FALLBACK_Q_MAX). The v25.32 log showed fallback triggers of
    0.66-0.78 (tail quantiles) picked by min-FARh on bimodal calib output --
    those are never usable operating points, and per-epoch scoring at them
    made the reported windowF1 look good while the raw trigger said the model
    was dead. The clinical-target search still uses the full grid."""
    if q_max is None:
        q_max = float(os.environ.get("NEUROMAMBA_FALLBACK_Q_MAX", "0.95"))
    labels = np.asarray(labels).astype(np.int64)
    candidates = np.quantile(probs, np.linspace(0.2, 0.999, n_grid))
    thr_cap = float(np.quantile(probs, min(max(q_max, 0.2), 0.999)))
    best = None
    max_f1_entry = None
    max_sens_entry = None
    min_farh_usable = None
    _under_budget = None  # v25.50: max sens with farh<=farh_max
    _under_15 = None  # v25.50: max sens with farh<=1.5*farh_max
    sens_floor = 0.30
    uniq_ids, first_idx, id_counts = np.unique(ids, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    uniq_ids, id_counts = uniq_ids[order], id_counts[order]
    records_contiguous = bool(np.array_equal(ids, np.repeat(uniq_ids, id_counts)))
    total_hours = len(labels) / 100.0 / 3600.0
    if records_contiguous:
        bounds = np.cumsum(id_counts) - id_counts
        records = [probs[b:b + c] for b, c in zip(bounds, id_counts)]
        labels_records = [labels[b:b + c] for b, c in zip(bounds, id_counts)]
    for thr in candidates:
        if records_contiguous:
            preds = [_postprocess_record(p, thr, thr * 0.4, 3, 30, 30)
                     for p in records]
            pred = np.concatenate(preds)
            n_false = _false_event_count(preds, labels_records)
            farh = n_false / max(total_hours, 1e-6)
        else:
            pred = causal_event_postprocess(probs, ids, trigger=thr, confirm_n=3, cooldown_n=30)
            farh = false_alarm_events_per_hour(labels, probs, thr, ids, pred_binary=pred)["farh_events_per_hour"]
        tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        sens = _safe_sensitivity(tp, fn)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-8)
        if sens >= sens_min and farh <= farh_max:
            if best is None or sens > best["sensitivity"]:
                best = {"trigger": float(thr), "sensitivity": sens, "farh": farh}
        if thr <= thr_cap:
            if max_f1_entry is None or f1 > max_f1_entry.get("ep_f1", -1):
                max_f1_entry = {"trigger": float(thr), "sensitivity": sens, "farh": farh, "ep_f1": f1}
            if sens >= sens_floor and (min_farh_usable is None or farh < min_farh_usable["farh"]):
                min_farh_usable = {"trigger": float(thr), "sensitivity": sens, "farh": farh}
            if farh <= farh_max and (_under_budget is None or sens > _under_budget["sensitivity"]):
                _under_budget = {"trigger": float(thr), "sensitivity": sens, "farh": farh}
            if farh <= 1.5 * farh_max and (_under_15 is None or sens > _under_15["sensitivity"]):
                _under_15 = {"trigger": float(thr), "sensitivity": sens, "farh": farh}
        if max_sens_entry is None or sens > max_sens_entry["sensitivity"]:
            max_sens_entry = {"trigger": float(thr), "sensitivity": sens, "farh": farh}
    if best is None:
        fb = (_under_budget if _under_budget is not None else (
            _under_15 if _under_15 is not None else (
                max_f1_entry if max_f1_entry is not None else {
                    "trigger": float(candidates[len(candidates) // 2]), "sensitivity": 0.0,
                    "farh": float("nan"), "ep_f1": 0.0})))
        fb = dict(fb)
        fb["fallback"] = ("max_sens_under_budget" if _under_budget is not None else (
            "max_sens_under_1.5x" if _under_15 is not None else "max_epF1"))
        fb["target_met"] = False
        fb["op_unsafe"] = False
        if fb["fallback"] in ("max_epF1", "max_sens_under_1.5x") and \
                (fb.get("farh") or 0.0) > 2.0 * farh_max:
            if min_farh_usable is not None and min_farh_usable["farh"] < fb["farh"]:
                print(f"  [threshold] UNSAFE fallback {fb['fallback']} at FARh={fb['farh']:.1f}/h "
                      f"(>2x budget) -- guarding to min-FAR usable point "
                      f"FARh={min_farh_usable['farh']:.1f}/h [v25.53.10]", flush=True)
                fb = dict(min_farh_usable)
                fb["fallback"] = "min_farh_unsafe_guard"
                fb["target_met"] = False
            fb["op_unsafe"] = True
        try:
            _fbp = causal_event_postprocess(probs, ids, trigger=float(fb["trigger"]),
                                            confirm_n=3, cooldown_n=30)
            _fbf = false_alarm_events_per_hour(labels, probs, float(fb["trigger"]), ids,
                                               pred_binary=_fbp)["farh_events_per_hour"]
            _fbtp = int(((_fbp == 1) & (labels == 1)).sum())
            _fbfn = int(((_fbp == 0) & (labels == 1)).sum())
            _fbs = _safe_sensitivity(_fbtp, _fbfn)
            if _fbs >= sens_min and _fbf <= farh_max:
                fb = dict(fb)
                fb["sensitivity"] = float(_fbs)
                fb["farh"] = float(_fbf)
                fb["fallback"] = None
                fb["target_met"] = True
                print(f"  [threshold] clinical target MET at fallback point: "
                      f"sens={_fbs:.3f} FARh={_fbf:.1f}/h (trigger={fb['trigger']:.3f}) "
                      f"[v25.47c re-verified]", flush=True)
                return fb
        except Exception:
            pass
        print(f"  [threshold] clinical target (sens>={sens_min} & FARh<={farh_max}) NOT met. "
              f"Fallback={fb['fallback']}: sens={fb['sensitivity']:.3f} FARh={fb['farh']:.1f}/h "
              f"(trigger={fb['trigger']:.3f}). Max-sens point: sens={max_sens_entry['sensitivity']:.3f} "
              f"at FARh={max_sens_entry['farh']:.1f}/h (trigger={max_sens_entry['trigger']:.3f})", flush=True)
        return fb
    best = dict(best); best["fallback"] = None; best["target_met"] = True
    best["op_unsafe"] = False
    return best

def smoothed_trigger_value(trigger_history, window=5):
    """v25.20: median-smoothed per-epoch clinical trigger. find_clinical_
    threshold_fast re-derives the operating point every epoch on a tiny calib
    set (~100 sequences, ~245 SH-positive frames), so the raw trigger whipsaws
    (observed: 0.008 -> 0.030 -> 0.008 with sens 0.18 -> 1.0). The deployed
    trigger used for the reported epF1/FARh/windowF1 is the median of the last
    `window` epochs (window<=1 disables smoothing)."""
    if window <= 1 or not trigger_history:
        return float(trigger_history[-1]) if trigger_history else 0.5
    tail = trigger_history[-window:]
    return float(np.median(tail))

def _best_episode_f1_point(labels, probs, ids, n_grid=25, confirm_n=3, cooldown_n=30,
                           merge_gap_n=MERGE_GAP_N_DEFAULT):
    """v25.25: when the clinical target (sens>=0.85 & FARh<=10/h) is unmet on
    the full calib set, deploy the operating point that maximizes EPISODE-F1
    instead of the inner-CV median trigger (which over-fires: observed
    eventF1 0.162 @ FARh 91.5/h while the pareto offered sens 0.896 @
    FARh 16.5/h). Grid over the probability quantiles, standard
    post-processing, report sens/FARh of the chosen point."""
    labels = np.asarray(labels).astype(np.int64)
    triggers = np.quantile(probs, np.linspace(0.02, 0.98, n_grid))
    best = None
    for thr in triggers:
        pred = causal_event_postprocess(probs, ids, trigger=float(thr),
                                        confirm_n=confirm_n, cooldown_n=cooldown_n,
                                        merge_gap_n=merge_gap_n)
        f1 = episode_f1(labels, probs, float(thr), ids, pred_binary=pred)
        if best is None or f1 > best["ep_f1"]:
            tp = int(((pred == 1) & (labels == 1)).sum())
            fn = int(((pred == 0) & (labels == 1)).sum())
            farh = false_alarm_events_per_hour(labels, probs, float(thr), ids,
                                               pred_binary=pred)["farh_events_per_hour"]
            best = {"trigger": float(thr), "ep_f1": float(f1),
                    "sensitivity": _safe_sensitivity(tp, fn), "farh": farh,
                    "confirm_n": confirm_n, "cooldown_n": cooldown_n,
                    "sustain": float(thr) * 0.4, "merge_gap_n": merge_gap_n,
                    "fallback": "max_episode_f1_full_calib"}
    return best

def _prep_records(ids, probs, labels=None):
    """v25.5-speed: pre-split per-record arrays once (ids from run_inference_cached
    are contiguous blocks). Returns (contiguous, probs_records, labels_records,
    total_hours). Non-contiguous ids fall back to (False, None, None, ...)."""
    uniq_ids, first_idx, id_counts = np.unique(ids, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    uniq_ids, id_counts = uniq_ids[order], id_counts[order]
    contiguous = bool(np.array_equal(ids, np.repeat(uniq_ids, id_counts)))
    if not contiguous:
        return False, None, None, len(probs) / 100.0 / 3600.0
    bounds = np.cumsum(id_counts) - id_counts
    p_records = [probs[b:b + c] for b, c in zip(bounds, id_counts)]
    l_records = [np.asarray(labels)[b:b + c] for b, c in zip(bounds, id_counts)] if labels is not None else None
    return True, p_records, l_records, len(probs) / 100.0 / 3600.0

def tune_postprocessing_params(labels, probs, ids, sens_min=0.85, farh_max=10.0):
    """v25.5-speed: the 160-config grid (8 triggers x 5 confirm x 4 cooldown) now
    runs on pre-split per-record arrays (was: full-array re-masking per config
    + per-config FARh re-scan -- minutes of CPU per model, x3 for the
    per-subtype search). Same semantics, ~100x faster."""
    labels = np.asarray(labels).astype(np.int64)
    best = None
    triggers = np.quantile(probs, np.linspace(0.1, 0.999, 16))
    contiguous, p_records, l_records, total_hours = _prep_records(ids, probs, labels)
    for trig in triggers:
        for cn in [1, 3, 5, 8, 12]:
            for cd in [10, 30, 60, 120]:
                for mg in MERGE_GAP_GRID:
                    if contiguous:
                        preds = [_postprocess_record(p, trig, trig * 0.4, cn, cd, mg)
                                 for p in p_records]
                        pred = np.concatenate(preds)
                        farh = _false_event_count(preds, l_records) / max(total_hours, 1e-6)
                    else:
                        pred = causal_event_postprocess(probs, ids, trigger=trig, confirm_n=cn, cooldown_n=cd,
                                                        merge_gap_n=mg)
                        farh = false_alarm_events_per_hour(labels, probs, trig, ids, pred_binary=pred)["farh_events_per_hour"]
                    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
                    sens = _safe_sensitivity(tp, fn)
                    if sens >= sens_min and farh <= farh_max:
                        if best is None or sens > best["sensitivity"]:
                            best = {"trigger": float(trig), "confirm_n": cn, "cooldown_n": cd,
                                    "sustain": trig * 0.4, "merge_gap_n": mg,
                                    "sensitivity": sens, "farh": farh}
    if best is None:
        mid = float(triggers[len(triggers) // 2])
        pred = causal_event_postprocess(probs, ids, trigger=mid, confirm_n=3, cooldown_n=30,
                                        merge_gap_n=MERGE_GAP_N_DEFAULT)
        tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
        sens = _safe_sensitivity(tp, fn)
        farh = false_alarm_events_per_hour(labels, probs, mid, ids, pred_binary=pred)["farh_events_per_hour"]
        best = {"trigger": mid, "confirm_n": 3, "cooldown_n": 30, "sustain": mid * 0.4,
                "merge_gap_n": MERGE_GAP_N_DEFAULT, "sensitivity": sens, "farh": farh,
                "note": f"no config met sens>={sens_min}/farh<={farh_max} (actual sens={sens:.3f} farh={farh:.1f}/h)"}
    try:
        _bp = causal_event_postprocess(probs, ids, trigger=float(best["trigger"]),
                                        sustain=float(best.get("sustain", float(best["trigger"]) * 0.4)),
                                        confirm_n=int(best.get("confirm_n", 3)),
                                        cooldown_n=int(best.get("cooldown_n", 30)),
                                        merge_gap_n=int(best.get("merge_gap_n", MERGE_GAP_N_DEFAULT)))
        _em = event_level_metrics(labels, _bp, ids)
        best["event_sensitivity"] = float(_em.get("sensitivity", float("nan")))
        best["event_f1"] = float(_em.get("f1", float("nan")))
        best["sensitivity_level"] = "frame+event"
    except Exception as _e:
        print(f"[clinical v25.57] event-level sensitivity unavailable: {_e}", flush=True)
    return best

def _pareto_frontier_rows(rows):
    """v25.36: dedupe + pareto-filter the trigger grid rows. The quantile grid
    collapses onto identical trigger values when the probability mass is
    concentrated (v25.35 daphnet fold 5 printed 10 identical trigger=0.000
    rows), and dominated rows (no better sens, no lower FARh) are noise. Ties
    never dominate each other (strict improvement required in one dimension)."""
    deduped = []
    for r in rows:
        if not deduped or r["trigger"] != deduped[-1]["trigger"]:
            deduped.append(r)
    frontier = []
    for i, r in enumerate(deduped):
        dominated = False
        for j, o in enumerate(deduped):
            if i == j:
                continue
            if (o["sensitivity"] > r["sensitivity"] + 1e-9 and o["farh"] <= r["farh"] + 1e-9) or \
               (o["sensitivity"] >= r["sensitivity"] - 1e-9 and o["farh"] < r["farh"] - 1e-9):
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    return frontier

def pareto_frontier_report(labels, probs, ids, n_grid=12, sens_min=0.85, farh_max=10.0):
    labels = np.asarray(labels).astype(np.int64)
    candidates = np.quantile(probs, np.linspace(0.2, 0.999, n_grid))
    rows = []
    for thr in candidates:
        pred = causal_event_postprocess(probs, ids, trigger=float(thr), confirm_n=3, cooldown_n=30)
        tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
        sens = _safe_sensitivity(tp, fn)
        farh = false_alarm_events_per_hour(labels, probs, float(thr), ids, pred_binary=pred)["farh_events_per_hour"]
        try:
            _es = float(event_level_metrics(labels, pred, ids).get("sensitivity", float("nan")))
        except Exception:
            _es = float("nan")
        rows.append({"trigger": float(thr), "sensitivity": sens, "farh": farh,
                     "event_sensitivity": _es})
    max_sens = max(rows, key=lambda r: r["sensitivity"])
    feasible = [r for r in rows if r["sensitivity"] >= sens_min and r["farh"] <= farh_max]
    frontier = _pareto_frontier_rows(sorted(rows, key=lambda r: r["trigger"]))
    print(f"[pareto] trigger | sens | FARh/h ({len(frontier)} frontier pts, {len(rows) - len(frontier)} dup/dominated)", flush=True)
    for r in frontier:
        print(f"[pareto] {r['trigger']:.3f}  {r['sensitivity']:.3f}  {r['farh']:.1f}", flush=True)
    print(f"[pareto] max-sens point: sens={max_sens['sensitivity']:.3f} at FARh={max_sens['farh']:.1f}/h "
          f"(trigger={max_sens['trigger']:.3f}); clinical-feasible points: {len(feasible)}", flush=True)
    return {"rows": frontier, "max_sens_point": max_sens, "n_clinical_feasible": len(feasible)}

def _clinical_op_points(pareto, labels, probs, ids, threshold=0.5):
    """v25.11: the two honest clinical operating points (literature-grounded:
    the combined sens>=0.85 & FARh<=10/h is unprecedented; report the split
    claims + contest-comparable window F1). v25.16f: window F1 is computed at
    the FROZEN trigger (the contest's 0.73-0.81 was at THEIR operating point;
    a fixed 0.5 threshold compared nothing)."""
    wf = window_level_f1(labels, probs, ids, threshold=threshold)
    rows = (pareto or {}).get("rows") or []
    sens_at_farh10 = max([r["sensitivity"] for r in rows if r["farh"] <= 10.0], default=0.0)
    farh_ok = [r["farh"] for r in rows if r["sensitivity"] >= 0.85]
    farh_at_sens85 = float(min(farh_ok)) if farh_ok else float("inf")
    return {"window_f1": wf["window_f1"], "window_sens": wf["sensitivity"],
            "window_spec": wf["specificity"], "sens_at_farh10": float(sens_at_farh10),
            "farh_at_sens85": farh_at_sens85}

def coverage_farh_curve(labels, probs, ids, trigger, postproc, coverages=(1.0, 0.95, 0.90)):
    """v25.15/v25.42.1: coverage-vs-FARh operating curve (generalization report
    Gate 4/6: report 'FARh <= 10/h via activity gating + abstention' as a
    coverage curve, not an assertion). For each coverage c, the (1-c) fraction
    of frames in the UNCERTAIN band -- |p - trigger| smallest, i.e. the frames
    nearest the decision boundary that drive the false alarms -- abstain
    (prob -> 0); the SAME frozen postprocessing then yields event sensitivity
    and FARh at that coverage. v25.42.1: the old policy abstained the
    LOWEST-probability frames, which never touched the alarm stream (flat
    curves in every family run); abstaining the boundary band makes the curve
    informative. The honest claim is the whole curve."""
    labels = np.asarray(labels, dtype=np.float32)
    ids = np.asarray(ids)
    probs = np.asarray(probs, dtype=np.float64)
    out = []
    for c in coverages:
        if c >= 1.0 - 1e-9:
            p_abs = probs.astype(np.float32)
        else:
            q = np.quantile(np.abs(probs - trigger), 1.0 - float(c))
            keep = np.abs(probs - trigger) > q  # abstain the boundary band
            p_abs = np.where(keep, probs, 0.0).astype(np.float32)
        pred = causal_event_postprocess(p_abs, ids, trigger=trigger,
                                        sustain=postproc.get("sustain", trigger * 0.4),
                                        confirm_n=postproc.get("confirm_n", 3),
                                        cooldown_n=postproc.get("cooldown_n", 30),
                                        merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        sens = _safe_sensitivity(tp, fn)
        farh = false_alarm_events_per_hour(labels, p_abs, trigger, ids, pred_binary=pred)["farh_events_per_hour"]
        out.append({"coverage": float(c), "abstain_threshold": float(q) if c < 1.0 - 1e-9 else 0.0,
                    "sensitivity": float(sens), "farh": float(farh)})
    return out

def medication_subgroup_report(labels, pred, probs, ids, med, threshold=0.5):
    """v25.15: per-medication-state subgroup metrics (0.0 OFF / 0.5 unknown /
    1.0 ON) -- the one demographic axis available in TLVMc metadata. Odonga et
    al. (arXiv:2502.09626): fairness reporting is mandatory before any
    clinical language; this is the first slice of it."""
    labels = np.asarray(labels, dtype=np.float32)
    med = np.asarray(med, dtype=np.float32)
    out = {}
    for val, name in [(0.0, "OFF"), (0.5, "unknown"), (1.0, "ON")]:
        m = np.isclose(med, val)
        if m.sum() < 100:
            continue
        ef1 = event_level_f1(labels[m], pred[m], ids[m])
        farh = false_alarm_events_per_hour(labels[m], probs[m], threshold, ids[m],
                                           pred_binary=pred[m])["farh_events_per_hour"]
        wf = window_level_f1(labels[m], probs[m], ids[m], threshold=threshold)
        out[name] = {"event_f1": float(ef1), "farh": float(farh),
                     "window_f1": float(wf["window_f1"]), "sensitivity": float(wf["sensitivity"]),
                     "n_frames": int(m.sum())}
    return out

def fairness_report(pred, labels, groups):
    """v25.16: demographic parity ratio (DPR) and equalized odds ratio (EOR)
    for a binary group attribute, per Odonga et al. (arXiv:2502.09626) -- every
    published HAR model fails the 0.8 four-fifths threshold on phenotype/sex/
    age/duration, so report rather than assert. groups: array of 0/1 per frame
    (e.g. gender M/F, age median split, H&Y median split)."""
    labels = np.asarray(labels, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    per = {}
    for g, name in ((0, "g0"), (1, "g1")):
        m = groups == g
        if m.sum() == 0:
            continue
        tp = int(((pred[m] == 1) & (labels[m] == 1)).sum())
        fp = int(((pred[m] == 1) & (labels[m] == 0)).sum())
        tn = int(((pred[m] == 0) & (labels[m] == 0)).sum())
        fn = int(((pred[m] == 0) & (labels[m] == 1)).sum())
        per[name] = {"tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
                     "selection_rate": float((pred[m] == 1).mean()), "n_frames": int(m.sum())}
    if len(per) == 2:
        dpr = per["g1"]["selection_rate"] / max(per["g0"]["selection_rate"], 1e-8)
        tpr_ratio = per["g1"]["tpr"] / max(per["g0"]["tpr"], 1e-8)
        tnr_ratio = (1.0 - per["g1"]["fpr"]) / max(1.0 - per["g0"]["fpr"], 1e-8)
        eor = min(tpr_ratio, tnr_ratio)
    else:
        dpr = eor = float("nan")
    return {"dpr": float(dpr), "eor": float(eor), "per_group": per}

def inner_cv_threshold_search(labels, probs, ids, subjects, n_folds=3, sens_min=0.85, farh_max=10.0, seed=42):
    labels = np.asarray(labels).astype(np.int64)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    if len(uniq) < n_folds + 1:
        print(f"[inner-cv] only {len(uniq)} subjects (<{n_folds + 1}) -- single-fit fallback", flush=True)
        return tune_postprocessing_params(labels, probs, ids, sens_min=sens_min, farh_max=farh_max)
    rng = np.random.RandomState(seed)
    shuffled = list(uniq); rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    chosen = []
    for k in range(n_folds):
        tune_subs = set(np.concatenate([folds[j] for j in range(n_folds) if j != k]))
        m = np.isin(subjects, list(tune_subs))
        if m.sum() < 50:
            continue
        res = tune_postprocessing_params(labels[m], probs[m], ids[m], sens_min=sens_min, farh_max=farh_max)
        chosen.append(res)
    if not chosen:
        return tune_postprocessing_params(labels, probs, ids, sens_min=sens_min, farh_max=farh_max)
    triggers = [c["trigger"] for c in chosen]
    merged = dict(chosen[0])
    merged["trigger"] = float(np.median(triggers))
    merged["inner_cv_triggers"] = [float(t) for t in triggers]
    merged["note"] = f"inner-CV median trigger over {len(chosen)} subject folds"
    print(f"[inner-cv] triggers={[round(t, 3) for t in triggers]} -> median={merged['trigger']:.3f}", flush=True)
    return merged

def per_subtype_threshold_search(sub_labels, sub_probs, ids, sens_min_by_class=None, farh_max_by_class=None,
                                  min_positives=20):
    out = {}
    default_sensmin, default_farhmax = 0.85, 10.0
    for k, name in enumerate(SUBTYPE_NAMES):
        y = sub_labels[:, k].astype(np.int64)
        p = sub_probs[:, k]
        if y.sum() < min_positives:
            out[name] = {"trigger": None, "sensitivity": float("nan"), "farh": float("nan"),
                         "note": f"not evaluable: only {int(y.sum())} positives (<{min_positives}) at this prevalence"}
            print(f"[per-subtype-threshold] {name}: not evaluable ({int(y.sum())} positives <{min_positives})", flush=True)
            continue
        sens_min = (sens_min_by_class or {}).get(name, default_sensmin)
        farh_max = (farh_max_by_class or {}).get(name, default_farhmax)
        out[name] = tune_postprocessing_params(y, p, ids, sens_min=sens_min, farh_max=farh_max)
        print(f"[per-subtype-threshold] {name}: trigger={out[name].get('trigger'):.3f} "
              f"sens={out[name].get('sensitivity'):.3f} FARh={out[name].get('farh'):.2f}/h", flush=True)
    return out

def onset_latency_report(labels, pred, ids, fs_hz=100.0, window_frames=300):
    """v25.6: cueing-latency analysis -- the clinically relevant question for a
    wearable cueing device is HOW FAST the detector fires after FoG onset
    (auditory cueing must land within ~1s to break the freeze). For each
    labeled FoG event, the latency is the first detected frame within the
    event window (causal detector output; negative = fired before the label)."""
    labels = np.asarray(labels).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    latencies = []
    missed = 0
    for uid in np.unique(ids):
        m = ids == uid
        y = labels[m]
        p = pred[m]
        idx = np.flatnonzero(y)
        if len(idx) == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[splits + 1]))
        for st in starts:
            lo = max(0, st - int(0.5 * fs_hz))          # allow 0.5s early fire
            hi = min(len(y), st + window_frames)
            det = np.flatnonzero(p[lo:hi])
            if len(det) == 0:
                missed += 1
            else:
                latencies.append((lo + det[0] - st) / fs_hz)
    lat = np.array(latencies) if latencies else np.array([float("nan")])
    n_events = len(latencies) + missed
    med = float(np.nanmedian(lat))
    prewindow_allowance_s = int(0.5 * fs_hz) / fs_hz
    deployable = (not np.isnan(med)) and (med >= 0.0)
    return {"n_events": n_events, "detected": len(latencies), "missed": missed,
            "detected_frac": len(latencies) / max(n_events, 1),
            "median_latency_s": med,
            "mean_latency_s": float(np.nanmean(lat)),
            "p90_latency_s": float(np.nanpercentile(lat, 90)),
            "within_1s_frac": float((lat <= 1.0).mean()) if n_events else float("nan"),
            "prewindow_allowance_s": prewindow_allowance_s,
            "deployable": bool(deployable),
            "artifact_note": (None if deployable else
                             "NON-DEPLOYABLE: median onset latency is negative, which "
                             "indicates the detector is using future (look-ahead) "
                             "information via its non-causal receptive field. A real-time "
                             "cueing device must fire AFTER onset (latency >= 0). Make the "
                             "model causal (remove the future window) before reporting this "
                             "as a deployable cueing latency.")}

def build_tripod_report(labels, pred, ids, probs=None, subject_ids=None):
    labels = np.asarray(labels).astype(np.int64)
    report = {"event_f1": event_level_f1(labels, pred, ids),
              "farh": false_alarm_events_per_hour(labels, probs if probs is not None else pred.astype(float), 0.5, ids, pred_binary=pred),
              "n_recordings": len(np.unique(ids)), "n_subjects": len(np.unique(subject_ids)) if subject_ids is not None else None,
              "prevalence": float(labels.mean())}
    if subject_ids is not None:
        report["per_subject_farh"] = per_subject_farh_distribution(labels, pred, subject_ids)
    return report

def decision_curve_analysis(labels, pred, ids, thresholds=None):
    labels = np.asarray(labels).astype(np.int64)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.5, 10)
    out = []
    n = len(labels)
    for pt in thresholds:
        tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
        net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
        out.append({"threshold": float(pt), "net_benefit": float(net_benefit)})
    return out

print("[calibration] CALIBRATION_V25_LOADED", flush=True)

def isotonic_fallback_calibration(probs, labels, min_positives_for_temp=20):
    labels = np.asarray(labels)
    if labels.sum() >= min_positives_for_temp:
        res = fit_temperature(probs, labels, min_positives=min_positives_for_temp)
        return apply_temperature(probs, res["temperature"]), {"method": "temperature", "temperature": res["temperature"]}
    if labels.sum() < 3:
        print(f"[calib] only {int(labels.sum())} positives -- calibration skipped", flush=True)
        return probs, {"method": "none", "reason": "too_few_positives"}
    iso = IsotonicRegression(out_of_bounds="clip")
    try:
        iso.fit(probs, labels)
        return iso.predict(probs), {"method": "isotonic", "n_positives": int(labels.sum())}
    except Exception as e:
        print(f"[calib] isotonic fit failed ({e})", flush=True)
        return probs, {"method": "none", "reason": str(e)}

def pick_best_calibration(fit_probs, fit_labels, eval_probs, eval_labels, min_positives=20):
    fit_labels = np.asarray(fit_labels)
    eval_labels = np.asarray(eval_labels)
    ece_raw = compute_ece(eval_probs, eval_labels)
    candidates = [("none", eval_probs, ece_raw)]
    if fit_labels.sum() >= min_positives:
        temp_res = fit_temperature(fit_probs, fit_labels, min_positives=min_positives)
        eval_probs_t = apply_temperature(eval_probs, temp_res["temperature"])
        ece_t = compute_ece(eval_probs_t, eval_labels)
        candidates.append(("temperature", eval_probs_t, ece_t))
    if 3 <= fit_labels.sum() < len(fit_labels):
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(fit_probs, fit_labels)
            eval_probs_iso = iso.predict(np.asarray(eval_probs, dtype=np.float64))
            ece_iso = compute_ece(eval_probs_iso, eval_labels)
            candidates.append(("isotonic", eval_probs_iso, ece_iso))
        except Exception:
            pass
    best_method, best_probs, best_ece = min(candidates, key=lambda c: c[2])
    print(f"[calib v25] methods (scored on HELD-OUT eval set): "
          f"{[(m, round(e, 4)) for m, _, e in candidates]} -> selected '{best_method}' "
          f"(ECE {best_ece:.4f} on held-out set)", flush=True)
    return best_probs, {"method": best_method, "ece": best_ece, "ece_raw": ece_raw}

def report_calibration_per_split(probs_calib, labels_calib, probs_thresh, labels_thresh, temperature):
    ece_calib = compute_ece(apply_temperature(probs_calib, temperature), labels_calib) if len(probs_calib) else float("nan")
    ece_thresh = compute_ece(apply_temperature(probs_thresh, temperature), labels_thresh) if len(probs_thresh) else float("nan")
    print(f"[calib] T={temperature:.3f} ECE(calib)={ece_calib:.4f} ECE(thresh)={ece_thresh:.4f}", flush=True)
    return {"ece_calib": ece_calib, "ece_thresh": ece_thresh}

print("[clinical] CLINICAL_V25_LOADED", flush=True)

CLINICAL_TARGET = dict(farh_max=10.0, sensitivity_min=0.85)

def crc_threshold(per_unit_loss_at_lambda_grid, alpha, B, lam_grid=None):
    """Return lambda-hat with E[L_{n+1}(lambda-hat)] <= alpha (CRC Thm 1)."""
    L = np.asarray(per_unit_loss_at_lambda_grid, dtype=float)
    n = L.shape[0]
    if lam_grid is None:
        lam_grid = np.arange(L.shape[1])
    else:
        lam_grid = np.asarray(lam_grid, dtype=float)
    if not np.all(np.diff(L.mean(axis=0)) <= 1e-12):
        raise ValueError("CRC: loss must be non-increasing in lambda")
    if not np.all(L <= B + 1e-12):
        raise ValueError(f"CRC: every loss must be bounded by B={B}")
    crit = n / (n + 1) * L.mean(axis=0) + B / (n + 1)
    feasible = np.flatnonzero(crit <= alpha)
    return lam_grid[feasible[0]] if feasible.size else lam_grid[-1]
SENSOR_DATASHEET_REFERENCE = {"typical_gait_peak_g": 2.5, "fog_onset_peak_g": 4.0,
                               "fall_event_peak_g": 8.0, "sensor_clip_g": 16.0}

def signal_quality_gate(signal_window, threshold=10.0):
    mag = np.linalg.norm(signal_window, axis=-1) if np.ndim(signal_window) > 1 else np.abs(signal_window)
    return bool(np.all(np.abs(mag) < threshold))

def _quality_abstain_mask(seqs, g_max=15.0, g_min=0.4):
    """v25.11: per-frame quality-reject mask for the clinical detector. Frames
    with implausible accelerometer magnitude (saturation/clipping near the
    sensor limit, or near-zero free-fall/detached readings -- a wearable lower
    back sensor reads ~1 g at rest) are marked 0 = abstain. The FoG probability
    is zeroed there instead of firing low-confidence alerts (audit P2 /
    reliability-gating literature). The first 3 signal channels are the raw
    accelerometer axes (N_CHANNELS contract)."""
    total = sum(len(s.labels) for s in seqs)
    mask = np.ones(total, dtype=np.float32)
    idx = 0
    for s in seqs:
        L = len(s.labels)
        sig = s.signal[:L]
        if sig.shape[1] >= 3:
            mag = np.linalg.norm(sig[:, :3], axis=1)
            bad = (mag > g_max) | (mag < g_min)
            mask[idx:idx + L] = np.where(bad, 0.0, 1.0)
        idx += L
    return mask

def _apply_abstention(probs, seqs, labels=None):
    """v25.11: optional quality abstention (NEUROMAMBA_ABSTAIN_QUALITY=1).
    Returns (zeroed_probs, abstained_frac, abstained_pos_frac). Default OFF:
    returns the probabilities unchanged."""
    if not _flag("NEUROMAMBA_ABSTAIN_QUALITY", "0"):
        return probs, 0.0, 0.0
    g_max = float(os.environ.get("NEUROMAMBA_ABSTAIN_G_MAX", "15.0"))
    g_min = float(os.environ.get("NEUROMAMBA_ABSTAIN_G_MIN", "0.4"))
    mask = _quality_abstain_mask(seqs, g_max=g_max, g_min=g_min)
    frac = float(1.0 - mask.mean()) if len(mask) else 0.0
    pos_frac = 0.0
    if labels is not None and len(labels) == len(mask) and frac > 0:
        lab = np.asarray(labels, dtype=np.float32)
        if lab.sum() > 0:
            pos_frac = float(((lab > 0.5) & (mask < 0.5)).sum() / (lab > 0.5).sum())
    print(f"[abstention v25.11] quality-reject: {frac * 100:.2f}% of frames abstained "
          f"({pos_frac * 100:.2f}% of FoG-positive frames)", flush=True)
    return probs * mask, frac, pos_frac

def sota_claim_status(internal_map, baseline_map_ci_diff, worst_fold_event_f1s, external_results, ece_max=None,
                       min_event_f1=0.05, ece_max_threshold=0.10):
    gates = {}
    gates["gate_1_comparable"] = bool(internal_map and internal_map.get("n_comparable", 0) >= 3 and internal_map.get("mAP") is not None)
    _bci = baseline_map_ci_diff
    if _bci is None:
        gates["gate_2_beats_baseline"] = "NOT_EVALUATED"
        gates["gate_2_reason"] = "baseline comparator artifact unavailable"
    elif _bci.get("status") == "NOT_EVALUATED":
        gates["gate_2_beats_baseline"] = "NOT_EVALUATED"
        gates["gate_2_reason"] = _bci.get("reason", "baseline not computed")
    else:
        gates["gate_2_beats_baseline"] = bool(_bci.get("non_overlapping_zero") and _bci.get("diff_mean", 0) > 0)
        gates["gate_2_reason"] = "ok"
    valid_f1s = [f for f in (worst_fold_event_f1s or []) if f is not None and np.isfinite(f)]
    gates["gate_3_no_collapse"] = bool(valid_f1s and min(valid_f1s) > min_event_f1)
    gates["gate_4_external"] = bool(external_results and sum(1 for r in external_results if r.get("passed")) >= 2)
    gates["gate_5_calibration"] = bool(ece_max is not None and np.isfinite(ece_max) and ece_max < ece_max_threshold)
    _pass = {k: v for k, v in gates.items() if k.startswith("gate_") and not k.endswith("_reason") and v is not True}
    all_pass = not _pass
    status = "SOTA_CLAIM_SUPPORTED" if all_pass else "REFUSED"
    failed = [k for k, v in gates.items() if k.startswith("gate_") and not k.endswith("_reason") and v is not True]
    print(f"[sota_claim_status] {status} gates={gates}", flush=True)
    return {"status": status, "gates": gates, "failed_gates": failed}

def subgroup_fairness_report(labels, pred, subject_ids, subgroup_map, fs_hz=100.0):
    labels = np.asarray(labels).astype(np.int64)
    groups = {}
    for sid, grp in subgroup_map.items():
        groups.setdefault(grp, []).append(sid)
    out = {}
    for grp, sids in groups.items():
        m = np.isin(subject_ids, sids)
        if m.sum() == 0:
            continue
        tp = int(((pred[m] == 1) & (labels[m] == 1)).sum()); fn = int(((pred[m] == 0) & (labels[m] == 1)).sum())
        sens = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        farh = false_alarm_events_per_hour(labels[m], None, 0.5, subject_ids[m], pred_binary=pred[m], fs_hz=fs_hz)
        out[str(grp)] = {"sensitivity": float(sens), "farh": farh, "n_subjects": len(set(subject_ids[m].tolist()))}
    return out

def build_clinical_deployment_report(model_manifest, clinical_metrics, sota_gate, quality_gate_mode="hybrid"):
    return {"model": model_manifest, "clinical_targets": CLINICAL_TARGET, "clinical_metrics": clinical_metrics,
            "sota_claim": sota_gate, "quality_gate_mode": quality_gate_mode, "sensor_datasheet_reference": SENSOR_DATASHEET_REFERENCE}

print("[augmentation] AUGMENTATION_V25_LOADED", flush=True)

_CH_PHYS = [0, 1, 2, 3]
_CH_LOG_RMS = 4
_CH_LOG_POW = [5, 6]

def _p_each_default():
    return float(os.environ.get("NEUROMAMBA_P_EACH", 0.2))

def _p_augment_default():
    return float(os.environ.get("NEUROMAMBA_P_AUGMENT", 0.5))

def apply_augmentations(signal, labels, y_subtype, label_weight, subtype_obs, rng=None, p_each=None,
                        partner_pool=None, sh_pool=None):
    """v25: threads subtype_observed through augmentations (slice-aligned)."""
    if rng is None:
        rng = np.random.RandomState()
    if p_each is None:
        p_each = _p_each_default()
    signal = signal.copy(); labels = labels.copy(); y_subtype = y_subtype.copy()
    label_weight = label_weight.copy(); subtype_obs = subtype_obs.copy()
    if rng.random() < p_each: signal = amplitude_scaling(signal, rng, scale_range=(0.85, 1.3))
    if rng.random() < p_each: signal = gaussian_noise(signal, rng, std_range=(0.0, 0.04))
    if rng.random() < p_each:
        _tw_factor = rng.uniform(0.95, 1.05)
        _T = len(signal)
        _idx = np.clip(np.linspace(0, _T - 1, _T) * _tw_factor, 0, _T - 1)
        _grid = np.arange(_T)
        signal = np.stack([np.interp(_idx, _grid, signal[:, c])
                           for c in range(signal.shape[1])], axis=1).astype(np.float32)
        labels = np.rint(np.interp(_idx, _grid, labels.astype(np.float64))).astype(np.int64)
        y_subtype = np.stack([np.interp(_idx, _grid, y_subtype[:, c])
                              for c in range(y_subtype.shape[1])], axis=1).astype(np.float32)
        if label_weight.ndim == 2:  # v25.57: guard cohort paths that store [T,C] weights
            label_weight = np.stack([np.interp(_idx, _grid, label_weight[:, c].astype(np.float64))
                                     for c in range(label_weight.shape[1])], axis=1).astype(np.float32)
        else:
            label_weight = np.interp(_idx, _grid, label_weight.astype(np.float64)).astype(np.float32)
        _so = np.stack([np.interp(_idx, _grid, subtype_obs[:, c].astype(np.float64))
                        for c in range(subtype_obs.shape[1])], axis=1)
        subtype_obs = (_so >= 0.5)
    if rng.random() < p_each: signal = channel_dropout(signal, rng, drop_prob=0.15)
    if rng.random() < p_each: signal = magnitude_warp(signal, rng)
    if rng.random() < p_each: signal = small_angle_rotation(signal, rng)
    if _b4_active():
        if rng.random() < _b4_poswap_p():
            signal = position_swap_augmentation(signal, rng)
        if rng.random() < _b4_ratejitter_p():
            signal, labels, y_subtype, label_weight, subtype_obs = rate_jitter_augmentation(
                signal, labels, y_subtype, label_weight, subtype_obs, rng)
    if rng.random() < p_each:
        if sh_pool and rng.random() < 0.7:
            signal, labels, y_subtype, label_weight, subtype_obs = sh_targeted_cutmix(
                signal, labels, y_subtype, label_weight, subtype_obs, sh_pool, rng, alpha=0.4)
        elif partner_pool:
            partner = partner_pool[rng.randint(0, len(partner_pool))]
            signal, labels, y_subtype, label_weight, subtype_obs = cross_sequence_cutmix(
                signal, labels, y_subtype, label_weight, subtype_obs, partner, rng, alpha=0.4)
    return signal, labels, y_subtype, label_weight, subtype_obs

def amplitude_scaling(signal, rng, scale_range=(0.85, 1.3)):
    s = rng.uniform(*scale_range)
    out = signal.copy()
    out[:, _CH_PHYS] = out[:, _CH_PHYS] * s
    if signal.shape[1] > _CH_LOG_RMS:
        out[:, _CH_LOG_RMS] = out[:, _CH_LOG_RMS] + math.log(s)
        for c in _CH_LOG_POW:
            if c < signal.shape[1]:
                out[:, c] = out[:, c] + 2.0 * math.log(s)
    return out

def gaussian_noise(signal, rng, std_range=(0.0, 0.04)):
    std = rng.uniform(*std_range)
    out = signal.copy()
    n_phys = min(4, signal.shape[1])
    out[:, :n_phys] = out[:, :n_phys] + (rng.randn(len(signal), n_phys).astype(np.float32) * std)
    return out

def time_warp(signal, rng, warp_range=(0.95, 1.05)):
    T = len(signal)
    factor = rng.uniform(*warp_range)
    idx = np.clip(np.linspace(0, T - 1, T) * factor, 0, T - 1)
    warped = np.zeros_like(signal)
    for c in range(signal.shape[1]):
        warped[:, c] = np.interp(idx, np.arange(T), signal[:, c])
    return warped

def channel_dropout(signal, rng, drop_prob=0.15):
    if signal.shape[1] < 3:
        return signal
    drop_axes = [c for c in range(3) if rng.random() < drop_prob]
    if not drop_axes:
        return signal
    out = signal.copy()
    out[:, drop_axes] = 0.0
    if signal.shape[1] > 3:
        out[:, 3] = np.linalg.norm(out[:, :3], axis=1)
        try:
            rebuilt = assemble_input_features(out[:, :3], fs_hz=CANON_FS)
            out[:, 3:] = rebuilt[:, 3:signal.shape[1]]
        except Exception as _e:
            print(f"[aug v25.57] channel_dropout derived-channel recompute failed ({_e}) "
                  f"-- magnitude recomputed, higher channels stale", flush=True)
    return out

def magnitude_warp(signal, rng, n_knots=4, sigma=0.15):
    T = len(signal)
    knot_x = np.linspace(0, T - 1, n_knots)
    knot_y = rng.normal(1.0, sigma, size=n_knots)
    curve = np.interp(np.arange(T), knot_x, knot_y).astype(np.float32)
    curve = np.clip(curve, 0.3, 3.0)
    out = signal.copy()
    n_phys = min(4, signal.shape[1])
    out[:, :n_phys] = out[:, :n_phys] * curve[:, None]
    if signal.shape[1] > _CH_LOG_RMS:
        log_c = np.log(curve)
        out[:, _CH_LOG_RMS] = out[:, _CH_LOG_RMS] + log_c
        for c in _CH_LOG_POW:
            if c < signal.shape[1]:
                out[:, c] = out[:, c] + 2.0 * log_c
    return out

def small_angle_rotation(signal, rng, max_deg=12.0):
    C = signal.shape[1]
    if C < 3:
        return signal
    acc = signal[:, :3].astype(np.float64)
    ax, ay, az = np.radians(rng.uniform(-max_deg, max_deg, size=3))
    cx, cy, cz = np.cos([ax, ay, az]); sx, sy, sz = np.sin([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rotated = (acc @ R.T).astype(np.float32)
    out = signal.copy(); out[:, :3] = rotated
    return out

def _b4_active():
    return str(os.environ.get("NEUROMAMBA_B4", "")).strip().lower() not in ("", "0")

def _b4_poswap_p():
    return float(os.environ.get("NEUROMAMBA_B4_POSWAP_P", 0.30))

def _b4_ratejitter_p():
    return float(os.environ.get("NEUROMAMBA_B4_RATEJITTER_P", 0.30))

def _b4_stretch_range():
    lo = float(os.environ.get("NEUROMAMBA_B4_STRETCH_LO", 0.8))
    hi = float(os.environ.get("NEUROMAMBA_B4_STRETCH_HI", 1.25))
    return (lo, hi)

def position_swap_augmentation(signal, rng):
    """v29ad (B4): emulate a different sensor mounting (L4 vs L5 vs shank vs
    trunk): a random axis permutation of the 3 raw axes plus per-axis sign
    flips. The Euclidean magnitude (ch3) is invariant under both; derived
    channels (4..12) are rebuilt with the SAME assembler used at load time
    (the channel_dropout v25.57 discipline) so the 13-channel contract stays
    internally consistent -- including the turn-rate channels, recomputed from
    the (new) horizontal axes."""
    C = signal.shape[1]
    if C < 3:
        return signal
    acc = signal[:, :3].astype(np.float64)
    perm = rng.permutation(3)
    signs = rng.choice([-1.0, 1.0], size=3)
    swapped = (acc[:, perm] * signs[None, :]).astype(np.float32)
    out = signal.copy()
    out[:, :3] = swapped
    if C > 3:
        out[:, 3] = np.linalg.norm(out[:, :3], axis=1)
        try:
            rebuilt = assemble_input_features(out[:, :3], fs_hz=CANON_FS)
            out[:, 3:] = rebuilt[:, 3:C]
        except Exception as _e:
            print(f"[aug v29ad] position_swap derived-channel recompute failed ({_e}) "
                  f"-- magnitude recomputed, higher channels stale", flush=True)
    return out

def rate_jitter_augmentation(signal, labels, y_subtype, label_weight, subtype_obs, rng):
    """v29ad (B4): cadence/temporal-scale jitter -- a bounded time-stretch with
    the SAME aligned-aux discipline as the v25.57 (H4) time_warp (all aux
    arrays warped with one index map; binary labels re-round; float streams
    interp; observed masks via >=0.5). Regularizes gait-speed/cadence variation
    across cohorts (the temporal-scale axis of the blueprint's regularization
    map); the stretch range is env-tunable
    (NEUROMAMBA_B4_STRETCH_LO/HI, default 0.8-1.25)."""
    lo, hi = _b4_stretch_range()
    factor = rng.uniform(lo, hi)
    if abs(factor - 1.0) < 1e-6:
        return signal, labels, y_subtype, label_weight, subtype_obs
    T = len(signal)
    idx = np.clip(np.linspace(0, T - 1, T) * factor, 0, T - 1)
    grid = np.arange(T)
    signal = np.stack([np.interp(idx, grid, signal[:, c])
                       for c in range(signal.shape[1])], axis=1).astype(np.float32)
    labels = np.rint(np.interp(idx, grid, labels.astype(np.float64))).astype(np.int64)
    y_subtype = np.stack([np.interp(idx, grid, y_subtype[:, c])
                          for c in range(y_subtype.shape[1])], axis=1).astype(np.float32)
    if label_weight.ndim == 2:
        label_weight = np.stack([np.interp(idx, grid, label_weight[:, c].astype(np.float64))
                                 for c in range(label_weight.shape[1])], axis=1).astype(np.float32)
    else:
        label_weight = np.interp(idx, grid, label_weight.astype(np.float64)).astype(np.float32)
    _so = np.stack([np.interp(idx, grid, subtype_obs[:, c].astype(np.float64))
                    for c in range(subtype_obs.shape[1])], axis=1)
    subtype_obs = (_so >= 0.5)
    return signal, labels, y_subtype, label_weight, subtype_obs

def cross_sequence_cutmix(signal, labels, y_subtype, label_weight, subtype_obs, partner, rng, alpha=0.4):
    """v25: threads subtype_observed through cutmix."""
    p_signal, p_labels, p_subtype, p_weight, p_sub_obs = partner
    T, Tp = len(signal), len(p_signal)
    lam = rng.beta(alpha, alpha)
    cut_len = max(10, min(T, Tp, int(T * (1 - lam))))
    cut_start = rng.randint(0, max(1, T - cut_len))
    p_start = rng.randint(0, max(1, Tp - cut_len))
    out_signal = signal.copy(); out_labels = labels.copy(); out_subtype = y_subtype.copy()
    out_weight = label_weight.copy(); out_sub_obs = subtype_obs.copy()
    out_signal[cut_start:cut_start + cut_len, :] = p_signal[p_start:p_start + cut_len, :]
    out_labels[cut_start:cut_start + cut_len] = p_labels[p_start:p_start + cut_len]
    out_subtype[cut_start:cut_start + cut_len, :] = p_subtype[p_start:p_start + cut_len, :]
    out_weight[cut_start:cut_start + cut_len] = p_weight[p_start:p_start + cut_len]
    out_sub_obs[cut_start:cut_start + cut_len, :] = p_sub_obs[p_start:p_start + cut_len, :]
    return out_signal, out_labels, out_subtype, out_weight, out_sub_obs

def build_sh_partner_pool(seqs, seed=0):
    pool = []
    for s in seqs:
        valid = s.train_mask.astype(bool)
        if valid.sum() == 0:
            continue
        if ((s.y_subtype[:, 0] > 0.5) & valid & s.subtype_observed[:, 0]).sum() > 0:
            pool.append((s.signal, s.labels.astype(np.int64), s.y_subtype, s.label_weight, s.subtype_observed))
    if pool:
        print(f"[augment v25] SH partner pool: {len(pool)} sequences contain observed Start Hesitation", flush=True)
    return pool

def sh_targeted_cutmix(signal, labels, y_subtype, label_weight, subtype_obs, sh_pool, rng, alpha=0.4):
    partner = sh_pool[rng.randint(0, len(sh_pool))]
    p_signal, p_labels, p_subtype, p_weight, p_sub_obs = partner
    p_sh_pos = np.flatnonzero((p_subtype[:, 0] > 0.5) & p_sub_obs[:, 0])
    T, Tp = len(signal), len(p_signal)
    if len(p_sh_pos) == 0 or T < 50 or Tp < 50:
        return signal, labels, y_subtype, label_weight, subtype_obs
    lam = rng.beta(alpha, alpha)
    cut_len = max(50, min(T, Tp, int(T * (1 - lam))))
    p_center = int(rng.choice(p_sh_pos))
    half = cut_len // 2
    p_start = int(np.clip(p_center - half, 0, max(0, Tp - cut_len)))
    cut_start = rng.randint(0, max(1, T - cut_len))
    out_signal = signal.copy(); out_labels = labels.copy(); out_subtype = y_subtype.copy()
    out_weight = label_weight.copy(); out_sub_obs = subtype_obs.copy()
    out_signal[cut_start:cut_start + cut_len, :] = p_signal[p_start:p_start + cut_len, :]
    out_labels[cut_start:cut_start + cut_len] = p_labels[p_start:p_start + cut_len]
    out_subtype[cut_start:cut_start + cut_len, :] = p_subtype[p_start:p_start + cut_len, :]
    out_weight[cut_start:cut_start + cut_len] = p_weight[p_start:p_start + cut_len]
    out_sub_obs[cut_start:cut_start + cut_len, :] = p_sub_obs[p_start:p_start + cut_len, :]
    return out_signal, out_labels, out_subtype, out_weight, out_sub_obs

def augment_batch(signal_batch, labels_batch, y_subtype_batch, label_weight_batch, sub_obs_batch,
                   rng=None, p_augment=None, partner_pool=None, sh_pool=None):
    if rng is None:
        rng = np.random.RandomState()
    if p_augment is None:
        p_augment = _p_augment_default()
    B = len(signal_batch)
    aug_signal = np.zeros_like(signal_batch)
    aug_labels = np.zeros_like(labels_batch)
    aug_subtype = np.zeros_like(y_subtype_batch)
    aug_weight = np.zeros_like(label_weight_batch)
    aug_sub_obs = np.zeros_like(sub_obs_batch)
    for b in range(B):
        if rng.random() < p_augment:
            s, l, ys, lw, so = apply_augmentations(
                signal_batch[b], labels_batch[b].copy(), y_subtype_batch[b].copy(),
                label_weight_batch[b].copy(), sub_obs_batch[b].copy(), rng=rng,
                partner_pool=partner_pool, sh_pool=sh_pool)
            aug_signal[b], aug_labels[b], aug_subtype[b], aug_weight[b], aug_sub_obs[b] = s, l, ys, lw, so
        else:
            aug_signal[b] = signal_batch[b]; aug_labels[b] = labels_batch[b]
            aug_subtype[b] = y_subtype_batch[b]; aug_weight[b] = label_weight_batch[b]
            aug_sub_obs[b] = sub_obs_batch[b]
    return aug_signal, aug_labels, aug_subtype, aug_weight, aug_sub_obs

def build_partner_pool(seqs, max_pool=200, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(seqs), size=min(max_pool, len(seqs)), replace=False)
    return [(seqs[i].signal, seqs[i].labels.astype(np.int64), seqs[i].y_subtype,
             seqs[i].label_weight, seqs[i].subtype_observed) for i in idx]

class LocalWindowAttention(nn.Module):
    """v25: optional (default OFF) -- late local attention for offline track."""
    def __init__(self, d_model, window_size=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, causal=False, key_pad=None):
        B, T, D = x.shape
        residual, x_norm = x, self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = [t.permute(0, 2, 1, 3) for t in qkv.unbind(dim=2)]
        n_w = max(1, (T + self.window_size - 1) // self.window_size)
        pad = n_w * self.window_size - T
        if pad > 0:
            q, k, v = [F.pad(t, (0, 0, 0, pad)) for t in (q, k, v)]
        q = q.reshape(B, self.n_heads, n_w, self.window_size, self.head_dim)
        k = k.reshape(B, self.n_heads, n_w, self.window_size, self.head_dim)
        v = v.reshape(B, self.n_heads, n_w, self.window_size, self.head_dim)
        logits = torch.einsum('bhnwd,bhnmd->bhnwm', q, k) * (self.head_dim ** -0.5)
        if key_pad is not None:
            _kp = key_pad.bool()
            if pad > 0:
                _kp = F.pad(_kp, (0, pad))
            _kp = _kp.reshape(B, 1, n_w, self.window_size, 1)
            logits = logits.masked_fill(_kp, torch.finfo(logits.dtype).min)
        else:
            _kp = None
        attn = F.softmax(logits, dim=-1)
        out = torch.einsum('bhnwm,bhnmd->bhnwd', attn, v)
        out = out.reshape(B, self.n_heads, n_w * self.window_size, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, n_w * self.window_size, D)
        if pad > 0:
            out = out[:, :T, :]
        if _kp is not None:
            _qmask = key_pad.bool()
            out = out.masked_fill(_qmask.unsqueeze(-1), 0.0)
            out = torch.nan_to_num(out)  # safety: fully-masked windows -> uniform attn, zeroed here
        return residual + self.gamma * self.out_proj(out)

_B3_POSITIONS = ("lower_back", "trunk", "shank")
_B3_COHORT_META = {
    "defog": ("lower_back", 100.0),        # TLVMC train/defog
    "tdcsfog": ("lower_back", 128.0),      # TLVMC train/tdcsfog
    "notype": ("lower_back", 100.0),       # TLVMC train/notype
    "notype_pseudo": ("lower_back", 100.0),
    "daphnet": ("trunk", 64.0),            # DAPHNET_FS=64, resample_poly 25/16
    "xuanwu": ("lower_back", 500.0),       # XUANWU_FS=500; per-recording
    "weargait": ("lower_back", 100.0),     # WEARGAIT_FS=100 (native, no resample)
    "fogstar": ("lower_back", 60.0),       # FOGSTAR_FS=60
}
_B3_PROMPT_DIM = len(_B3_POSITIONS) + 1  # one-hot(position) ++ rate channel

def _b3_active():
    return str(os.environ.get("NEUROMAMBA_B3", "")).strip().lower() not in ("", "0")

def _b3_rate_norm():
    """'log' (default): log2(rate/100) -- centered on the canonical rate,
    range [-0.74, 2.32] over 60-500 Hz. 'ratio': rate/100 (design-note
    default; range [0.6, 5.0])."""
    return str(os.environ.get("NEUROMAMBA_B3_RATE_NORM", "log")).strip().lower()

def _b3_parse_override():
    out = {}
    raw = os.environ.get("NEUROMAMBA_B3_PROMPT_OVERRIDE", "")
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            continue
        try:
            out[bits[0].strip()] = (bits[1].strip(), float(bits[2]))
        except ValueError:
            continue
    return out

def _b3_cohort_meta(source_type, position_tag=""):
    """(position, native_rate_hz) for one cohort: per-sequence tag > env
    override > cohort table > ('lower_back', CANON_FS) unknown fallback."""
    ov = _b3_parse_override().get(str(source_type))
    if ov is not None:
        return ov
    tag = str(position_tag or "").strip()
    meta = _B3_COHORT_META.get(str(source_type))
    if meta is None:
        return ("lower_back", CANON_FS)
    if tag in _B3_POSITIONS:
        return (tag, meta[1])  # per-recording position, cohort rate
    return meta

def _b3_prompt_row(source_type, position_tag=""):
    pos, rate = _b3_cohort_meta(source_type, position_tag)
    row = np.zeros(_B3_PROMPT_DIM, dtype=np.float32)
    if pos in _B3_POSITIONS:
        row[_B3_POSITIONS.index(pos)] = 1.0
    row[-1] = (np.log2(rate / CANON_FS) if _b3_rate_norm() != "ratio"
               else rate / CANON_FS)
    return row

def _b3_prompt_matrix(source_types, position_tags=None):
    """[B, _B3_PROMPT_DIM] float32 prompt matrix; position_tags may be None
    (cohort-table positions only)."""
    tags = list(position_tags) if position_tags is not None else [""] * len(source_types)
    return np.stack([_b3_prompt_row(s, t) for s, t in zip(source_types, tags)], axis=0)

class WalkingGate(nn.Module):
    """v25-FIX-6 (LIFT-PD / FM-FoG trick): a tiny binary walking/standing
    detector. Its output multiplies the FoG probability at inference to
    SUPPRESS false alarms during sitting/standing/voluntary stops. This is
    the single most effective FARh reducer per the literature.

    Architecture: 3-layer 1D-CNN on the raw accel magnitude, outputs a
    per-frame walking probability. Trained jointly with the main model
    using a derived binary 'is_walking' target = (locomotor-band energy >
    threshold) OR (any subtype == Walking).

    Causal fix (Round 28bz): when `causal=True`, the conv stack uses left-only
    padding (kernel-1 zeros prepended before each strided conv) so the walking
    probability at frame t depends only on the past -- otherwise the centered
    padding leaks future frames into the suppression decision, breaking the
    deployable-latency contract. Default (causal=False) keeps centered padding."""
    def __init__(self, in_channels=4, base_ch=16, causal=False):
        super().__init__()
        self.causal = causal
        n_phys = min(4, in_channels)
        self.n_phys = n_phys
        pad = 0 if causal else 12
        self.conv = nn.Sequential(
            nn.Conv1d(n_phys, base_ch, 25, stride=4, padding=pad, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True),
            nn.Conv1d(base_ch, base_ch * 2, 25, stride=4, padding=pad, bias=False),
            nn.BatchNorm1d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv1d(base_ch * 2, base_ch * 2, 25, stride=4, padding=pad, bias=False),
            nn.BatchNorm1d(base_ch * 2), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv1d(base_ch * 2, 1, 1), nn.Sigmoid())

    def forward(self, x, return_feat=False):
        if x.dim() == 3 and x.shape[-1] <= 16 and x.shape[1] > 16:
            x = x.permute(0, 2, 1)
        x = x[:, :self.n_phys, :]
        T0 = x.shape[-1]
        if self.causal:
            h = F.pad(x, (24, 0))
            h = F.relu(self.conv[1](self.conv[0](h)), inplace=True)
            h = F.pad(h, (24, 0))
            h = F.relu(self.conv[3](self.conv[2](h)), inplace=True)
            h = F.pad(h, (24, 0))
            h = F.relu(self.conv[5](self.conv[4](h)), inplace=True)
        else:
            h = self.conv(x)
        h = F.interpolate(h, size=T0, mode="linear", align_corners=False)
        if return_feat:
            return self.head(h).squeeze(1), h
        return self.head(h).squeeze(1)  # [B, T]

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out, inplace=True)
        return self.dropout(out)

class ResNet1DBackbone(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, base_channels=64, n_blocks=6, dropout=0.1):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4, base_channels * 4, base_channels * 4]
        strides = [1, 2, 1, 2, 1, 1]
        self.stem = nn.Sequential(nn.Conv1d(in_channels, base_channels, 7, 1, 3, bias=False),
                                   nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList()
        prev_ch = base_channels
        for i in range(n_blocks):
            self.blocks.append(ResNet1DBlock(prev_ch, channels[i], stride=strides[i], dropout=dropout))
            prev_ch = channels[i]
        self.out_channels = channels[-1]

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

class SSLConvEncoder(nn.Module):
    """v25.14: masked-reconstruction SSL encoder (FM-FoG / LIFT-PD recipe).
    The conv stem + blocks are STRUCTURALLY IDENTICAL to ConvBiGRUModel /
    NeuroMambaModel (chs 64/128/128/256/256, strides 1,2,1,2,1) so the
    pretrained weights transfer 1:1 (or via blocks->blocks_cnn remap for
    NeuroMamba). LIFT-PD (arXiv:2410.21326): mask 20% of each window, MSE on
    masked positions, encoder-only, then fine-tune with labels
    (+7.25% precision / +4.4% accuracy at -40% labels, LOGO protocol).
    FM-FoG (arXiv:2509.24176): masked IMU reconstruction over public corpora,
    98.5 F1 on unseen patients."""
    def __init__(self, in_channels=N_CHANNELS, dropout=0.1):
        super().__init__()
        chs = [64, 128, 128, 256, 256]
        strides = [1, 2, 1, 2, 1]
        self.stem = nn.Sequential(nn.Conv1d(in_channels, chs[0], 7, 1, 3, bias=False),
                                  nn.BatchNorm1d(chs[0]), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList()
        prev = chs[0]
        for i in range(1, len(chs)):
            self.blocks.append(_ConvBlock(prev, chs[i], stride=strides[i], dropout=dropout))
            prev = chs[i]
        self.enc_out = chs[-1]
        self.decoder = nn.Sequential(nn.Conv1d(self.enc_out, 128, 1), nn.ReLU(inplace=True),
                                     nn.Conv1d(128, in_channels, 1))

    def forward(self, x_ct):
        h = self.stem(x_ct)
        for blk in self.blocks:
            h = blk(h)
        h_up = F.interpolate(h, size=x_ct.shape[-1], mode="linear", align_corners=False)
        return self.decoder(h_up)

    def encoder_state(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if k.startswith("stem.") or k.startswith("blocks.")}

def pretrain_ssl_encoder(seqs, device, epochs=10, lr=1e-3, batch_size=16,
                         window=2000, mask_frac=0.2, seed=0):
    """v25.14: masked-reconstruction pretraining on UNLABELED signals
    (labels/masks ignored). Returns the encoder state dict (stem + blocks).
    Masking: mask_frac of the time indices per crop are zeroed; MSE is
    computed on masked positions only (LIFT-PD)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 1)
    encoder = SSLConvEncoder(in_channels=N_CHANNELS).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    crops_per_seq = 8
    n_seqs = len(seqs)
    print(f"[ssl v25.14] masked-reconstruction pretraining: {n_seqs} sequences, "
          f"{epochs} epochs, window={window}, mask_frac={mask_frac}", flush=True)
    for epoch in range(epochs):
        encoder.train()
        total, n_b = 0.0, 0
        idxs = np.arange(n_seqs)
        rng.shuffle(idxs)
        for bi in range(0, n_seqs, batch_size):
            batch = []
            for si in idxs[bi:bi + batch_size]:
                s = seqs[si % n_seqs]
                T = len(s.labels)
                w = min(window, T)
                start = rng.randint(0, max(1, T - w + 1)) if T > w else 0
                batch.append(s.signal[start:start + w])
            L = min(len(b) for b in batch)
            x = np.stack([b[:L] for b in batch], axis=0).astype(np.float32)
            x_t = torch.from_numpy(x).to(device).permute(0, 2, 1)  # [B, C, T]
            mask = torch.rand(x_t.shape[0], 1, x_t.shape[2], device=device) < mask_frac
            x_masked = x_t.clone()
            x_masked[mask.expand_as(x_t)] = 0.0
            pred = encoder(x_masked)
            loss = F.mse_loss(pred[mask.expand(-1, pred.shape[1], -1)], x_t[mask.expand(-1, x_t.shape[1], -1)])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
            n_b += 1
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"[ssl v25.14] epoch={epoch} mse={total / max(n_b, 1):.5f}", flush=True)
    return encoder.encoder_state()

class ResNet1DBaseline(nn.Module):
    """v25: winner-style 6-block 1D-ResNet (Brederecke 8th place, mAP 0.356).
    DEFAULT model -- proven to beat the v22 hybrid. Now includes a walking-gate."""
    def __init__(self, in_channels=N_CHANNELS, base_channels=64, n_blocks=6, dropout=0.1, use_walking_gate=True):
        super().__init__()
        self.backbone = ResNet1DBackbone(in_channels, base_channels, n_blocks, dropout)
        hidden_dim = self.backbone.out_channels
        self.fog_head = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, 1), nn.ReLU(inplace=True),
                                       nn.Dropout(dropout), nn.Conv1d(hidden_dim // 2, 1, 1))
        self.subtype_heads = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, 1), nn.ReLU(inplace=True),
                           nn.Dropout(dropout), nn.Conv1d(hidden_dim // 2, 1, 1)) for _ in range(3)])
        self.walking_gate = WalkingGate(in_channels=in_channels) if use_walking_gate else None
        self._use_wg = use_walking_gate

    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] <= 16 and x.shape[1] > 16:
            x = x.permute(0, 2, 1)
        h = self.backbone(x)
        fog_logits = self.fog_head(h).squeeze(1)
        subtype_logits = torch.cat([head(h).squeeze(1).unsqueeze(-1) for head in self.subtype_heads], dim=-1)
        aux = {}
        if self.walking_gate is not None:
            aux["walking_prob"] = self.walking_gate(x)
        return fog_logits, subtype_logits, aux

    def get_architecture_manifest(self):
        return {"model_version": "resnet1d_baseline_v25", "backbone": "1D_ResNet_6_block",
                "in_channels_contract": N_CHANNELS, "no_global_pooling": True,
                "use_walking_gate": self._use_wg,
                "reference": "Brederecke_2023_8th_place_Kaggle_private_LB_0.356",
                "v25_fixes": "subtype_observed+rare_split+event_target+walking_gate+window_roll+per_sample_norm+subject_bootstrap"}

class ResNet1DPerFrameAdapter(nn.Module):
    """Wraps ResNet1DBaseline to accept [B, T, C] and return full-resolution
    [B, T] / [B, T, 3] logits, matching the sequence-model interface."""
    def __init__(self, in_channels=N_CHANNELS, base_channels=64, n_blocks=6, dropout=0.1, use_walking_gate=True):
        super().__init__()
        self.inner = ResNet1DBaseline(in_channels=in_channels, base_channels=base_channels,
                                       n_blocks=n_blocks, dropout=dropout, use_walking_gate=use_walking_gate)

    def forward(self, x, mask=None, lengths=None):
        T_in = x.shape[1] if x.dim() == 3 and x.shape[1] > 16 else x.shape[-1]
        x_ct = x.permute(0, 2, 1) if (x.dim() == 3 and x.shape[-1] <= 16 and x.shape[1] > 16) else x
        fog, sub, aux = self.inner(x_ct)
        fog = F.interpolate(fog.unsqueeze(1), size=T_in, mode="nearest").squeeze(1)
        sub = F.interpolate(sub.permute(0, 2, 1), size=T_in, mode="nearest").permute(0, 2, 1)
        return fog, sub, aux

    def get_architecture_manifest(self):
        m = self.inner.get_architecture_manifest()
        m["adapter"] = "per_frame_full_resolution"
        m["role"] = "main_model"
        return m

def _sanity_test_resnet1d():
    model = ResNet1DBaseline(in_channels=N_CHANNELS)
    x = torch.randn(2, N_CHANNELS, 1000)
    fog, sub, aux = model(x)
    assert fog.shape == (2, 250)
    assert sub.shape == (2, 250, 3)
    assert (fog.std(dim=1) > 1e-6).all(), "REGRESSION: constant per-frame outputs -- pooled shim detected!"
    ad = ResNet1DPerFrameAdapter(in_channels=N_CHANNELS)
    xa = torch.randn(2, 1000, N_CHANNELS)
    fog2, sub2, _ = ad(xa)
    assert fog2.shape == (2, 1000) and sub2.shape == (2, 1000, 3)
    print(f"[resnet1d v25] Sanity check passed: fog={fog.shape}, sub={sub.shape}, adapter fog={fog2.shape}", flush=True)
    return True

class _ConvBlock(nn.Module):
    """Residual 1D conv block (BN-ReLU-Conv x2 + 1x1 shortcut)."""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(out_ch))
                         if (stride != 1 or in_ch != out_ch) else nn.Identity())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.b2(self.c2(h))
        h = F.relu(h + self.shortcut(x), inplace=True)
        return self.dropout(h)

def _sub_head_mode():
    """v25.14b: subtype head contract -- 'softmax' (4-class CE, default) or
    'sigmoid' (3 per-class sigmoids + masked ASL; the retrieved 2nd-place
    FogRnnModel head and the winner's linear+sigmoid)."""
    mode = os.environ.get("NEUROMAMBA_SUB_HEAD_MODE", "softmax").strip().lower()
    if mode not in ("softmax", "sigmoid"):
        raise ValueError(f"NEUROMAMBA_SUB_HEAD_MODE must be softmax or sigmoid, got {mode}")
    return mode

class ConvBiGRUModel(nn.Module):
    """v25.2 DEFAULT MODEL -- winner-style sequence architecture.

    Every Kaggle top-5 solution used a recurrent/sequence model with long
    temporal context (1st: Transformer+2BiLSTM 0.514; 2nd: GRU 0.451;
    4th: MultiResidualBiGRU 0.417), while the v25 ResNet had ~1s receptive
    field and nearest-upsampled 4x logits -- which is exactly why Turn (long
    events) learned (AP 0.53) but StartHesitation (needs pre-event standing
    context) and Walking (brief events) stayed near chance.

    Design (light, ~3M params):
      - 1D conv encoder (4 residual blocks, 2x strides -> 4x downsample)
      - 2-layer BiGRU(hidden) over the 4x features: 20s window -> 500 steps,
        giving multi-second context at 40ms resolution
      - GRU features linearly upsampled back to FULL input resolution
      - per-frame fog + 3 subtype heads (conv 1x1) at full resolution
      - small walking-gate CNN on raw accel (rest/walk suppressor)

    Interface identical to ResNet1DPerFrameAdapter: accepts [B, T, C] and
    returns (fog [B,T], subtype [B,T,3], aux with walking_prob [B,T])."""
    def __init__(self, in_channels=N_CHANNELS, hidden=128, n_gru_layers=2, dropout=0.2,
                 use_walking_gate=True):
        super().__init__()
        self.hidden = hidden
        self.n_gru_layers = n_gru_layers
        self._use_wg = use_walking_gate
        chs = [64, 128, 128, 256, 256]
        strides = [1, 2, 1, 2, 1]
        self.stem = nn.Sequential(nn.Conv1d(in_channels, chs[0], 7, 1, 3, bias=False),
                                  nn.BatchNorm1d(chs[0]), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList()
        prev = chs[0]
        for i in range(1, len(chs)):
            self.blocks.append(_ConvBlock(prev, chs[i], stride=strides[i], dropout=dropout))
            prev = chs[i]
        self.enc_out = chs[-1]
        self.proj = nn.Sequential(nn.Linear(self.enc_out, hidden), nn.LayerNorm(hidden))
        self.gru = nn.GRU(hidden, hidden, num_layers=n_gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_gru_layers > 1 else 0.0)
        self.gru_norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        head_in = hidden * 2
        self.use_local_attn = _flag("NEUROMAMBA_USE_LOCAL_ATTN", "1")
        self.local_attn = LocalWindowAttention(head_in, window_size=128, n_heads=4) if self.use_local_attn else None
        self.use_med_film = _flag("NEUROMAMBA_USE_MED_FILM", "1")
        self.med_film = nn.Linear(1, 2 * head_in) if self.use_med_film else None
        self.use_sensor_film = _flag("NEUROMAMBA_B3", "0")
        self.sensor_film = nn.Linear(_B3_PROMPT_DIM, 2 * head_in) if self.use_sensor_film else None
        if self.sensor_film is not None:
            nn.init.zeros_(self.sensor_film.weight)
            nn.init.zeros_(self.sensor_film.bias)
        self.use_gate_dann = _flag("NEUROMAMBA_B6", "0")
        if self.use_sensor_film:
            print(f"[b3 v29ae] ACTIVE: sensor position/rate FiLM (zero-init, "
                  f"prompt_dim={_B3_PROMPT_DIM}, rate_norm={_b3_rate_norm()})", flush=True)
        if self.use_gate_dann:
            print(f"[b6 v29ae] ACTIVE: shift-robust gate DANN "
                  f"(w={os.environ.get('NEUROMAMBA_B6_DANN_W', '0.05')})", flush=True)
        self.fog_head = nn.Sequential(nn.Conv1d(head_in, head_in // 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv1d(head_in // 2, 1, 1))
        self.subtype_ce = _sub_head_mode() == "softmax"
        n_sub = 4 if self.subtype_ce else 3
        self.subtype_head = nn.Sequential(nn.Conv1d(head_in, head_in // 2, 1), nn.ReLU(inplace=True),
                                          nn.Conv1d(head_in // 2, n_sub, 1))
        self.walking_gate = WalkingGate(in_channels=in_channels) if use_walking_gate else None
        self.use_hazard_head = _flag("FOGBENCH_TLS", "0")
        if self.use_hazard_head:
            self.hazard_head = nn.Sequential(
                nn.Conv1d(head_in, head_in // 4, 1), nn.ReLU(inplace=True),
                nn.Conv1d(head_in // 4, 1, 1))

    @staticmethod
    def _make_block(in_ch, out_ch, stride=1, dropout=0.2):
        return _ConvBlock(in_ch, out_ch, stride=stride, dropout=dropout)

    def forward(self, x, mask=None, med=None, lengths=None, prompt=None):
        if x.dim() == 3 and x.shape[1] == N_CHANNELS and x.shape[-1] != N_CHANNELS:
            x = x.permute(0, 2, 1)
        B, T_in, C = x.shape
        x_ct = x.permute(0, 2, 1).contiguous()
        h = self.stem(x_ct)
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 1).contiguous()          # [B, T/4, enc_out]
        h = self.proj(h)
        h, _ = self.gru(h)                            # [B, T/4, 2*hidden]
        h = self.gru_norm(h)
        if self.local_attn is not None:
            h = self.local_attn(h)
        h = self.dropout(h)
        h_up = F.interpolate(h.permute(0, 2, 1), size=T_in, mode="linear",
                             align_corners=False)     # full resolution features
        if self.use_med_film and self.med_film is not None and med is not None:
            gb = self.med_film(med.to(h_up.dtype).unsqueeze(-1))  # [B, 2D]
            gamma, beta = gb.chunk(2, dim=-1)
            h_up = h_up * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        if prompt is not None and getattr(self, "sensor_film", None) is not None:
            if not torch.is_tensor(prompt):
                prompt = torch.as_tensor(np.asarray(prompt, dtype=np.float32))
            prompt = prompt.to(device=h_up.device, dtype=h_up.dtype)
            gb3 = self.sensor_film(prompt)                          # [B, 2D]
            gamma3, beta3 = gb3.chunk(2, dim=-1)
            h_up = h_up * (1.0 + gamma3[:, :, None]) + beta3[:, :, None]
        fog_logits = self.fog_head(h_up).squeeze(1)
        subtype_logits = self.subtype_head(h_up).permute(0, 2, 1)
        aux = {}
        if self.walking_gate is not None:
            if getattr(self, "use_gate_dann", False):
                _wp, _gf = self.walking_gate(x_ct, return_feat=True)
                aux["walking_prob"] = _wp
                aux["gate_feat"] = _gf  # v29ae (B6): penultimate gate features
            else:
                aux["walking_prob"] = self.walking_gate(x_ct)
        if getattr(self, "use_hazard_head", False) and getattr(self, "hazard_head", None) is not None:
            aux["hazard_logit"] = self.hazard_head(h_up).squeeze(1)
        return fog_logits, subtype_logits, aux

    def get_architecture_manifest(self):
        return {"model_version": "convgru_v25.6", "backbone": "1D_ResNet4_blocks + 2xBiGRU",
                "in_channels_contract": N_CHANNELS, "gru_hidden": self.hidden,
                "n_gru_layers": self.n_gru_layers, "no_global_pooling": True,
                "full_resolution_heads": True, "use_walking_gate": self._use_wg,
                "hazard_head": ("conv1x1_TLS_onset_hazard" if getattr(self, "use_hazard_head", False)
                                else "off"),  # v28bp hazard arm (FOGBENCH_TLS)
                "med_film": self.use_med_film,
                "sensor_film": getattr(self, "use_sensor_film", False),  # v29ae B3
                "gate_dann": getattr(self, "use_gate_dann", False),      # v29ae B6
                "subtype_head": (f"{'4_class_softmax_CE' if getattr(self, 'subtype_ce', True) else '3x_masked_sigmoid'}"
                                 f" (SH/Turn/Walking{('/None' if getattr(self, 'subtype_ce', True) else '')})"),
                "reference": "Kaggle top-5 all sequence models: 1st Transformer+2BiLSTM 0.514, "
                             "2nd GRU 0.451, 4th MultiResidualBiGRU 0.417",
                "v25.6_fixes": "medication_state_FiLM",
                "role": "main_model"}

def _sanity_test_convbigru():
    model = ConvBiGRUModel(in_channels=N_CHANNELS)
    x = torch.randn(2, 2000, N_CHANNELS)
    fog, sub, aux = model(x)
    assert fog.shape == (2, 2000) and sub.shape == (2, 2000, 4)
    assert "walking_prob" in aux and aux["walking_prob"].shape == (2, 2000)
    assert (fog.std(dim=1) > 1e-6).all(), "REGRESSION: constant per-frame outputs"
    assert "hazard_logit" not in aux, "hazard head must be absent when FOGBENCH_TLS is unset"
    _prev_tls = os.environ.get("FOGBENCH_TLS")
    os.environ["FOGBENCH_TLS"] = "1"
    try:
        _model_h = ConvBiGRUModel(in_channels=N_CHANNELS)
        fog_h, sub_h, aux_h = _model_h(x)
        assert "hazard_logit" in aux_h, "FOGBENCH_TLS=1 must build the hazard head"
        assert aux_h["hazard_logit"].shape == fog_h.shape, \
            f"hazard/fog shape mismatch: {tuple(aux_h['hazard_logit'].shape)} vs {tuple(fog_h.shape)}"
        assert (aux_h["hazard_logit"].std(dim=1) > 1e-6).all(), \
            "REGRESSION: constant per-frame hazard outputs"
    finally:
        if _prev_tls is None:
            os.environ.pop("FOGBENCH_TLS", None)
        else:
            os.environ["FOGBENCH_TLS"] = _prev_tls
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[convgru v25.3] Sanity check passed: fog={fog.shape} sub4={sub.shape} "
          f"params={n_params / 1e6:.2f}M (hazard-arm head: shape+non-const OK)", flush=True)
    return True

class _BatchFirstPlaceholder:
    """Minimal attribute shim: PyTorch's nn.TransformerEncoder.forward inspects
    first_layer.self_attn.batch_first (torch/nn/modules/transformer.py ~line 437)
    to verify the batch_first contract. LoRATransformerEncoderLayer implements
    attention manually (no nn.MultiheadAttention instance), so this placeholder
    carries the batch_first=True contract without allocating unused weights."""

    batch_first = True

class LoRALinear(nn.Module):
    """Low-rank adapter for an nn.Linear layer (Hu et al. 2021, arXiv:2106.09685):
    y = W x + (alpha/r) * B A x. B is zero-init so the adapter is an exact
    identity at init (no training-phase discontinuity). rank<=0 -> passthrough
    (bit-parity with the unadapted model)."""

    def __init__(self, in_features, out_features, rank=0, alpha=1.0, bias=True, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, self.rank))
            self.lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        y = self.linear(x)
        if self.rank > 0:
            y = y + (self.alpha / self.rank) * (self.dropout(x) @ self.lora_A @ self.lora_B)
        return y

class LoRATransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with LoRA on Q/K/V/O + FFN projections.
    Faithful to nn.TransformerEncoderLayer semantics (pre-LN, gelu, batch_first)
    so the unadapted path (rank=0) is behaviorally identical.

    NOTE: attention is implemented with F.scaled_dot_product_attention over
    EXPLICIT LoRA-wrapped Q/K/V linears instead of nn.MultiheadAttention --
    MHA applies its own fused in_proj (W_qkv) to the inputs, so feeding it
    pre-projected q/k/v would double-project. The LoRA linears ARE the
    projections here."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, rank=0, alpha=1.0,
                 lora_dropout=0.0, activation="gelu"):
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} % nhead {nhead} != 0"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.self_attn = _BatchFirstPlaceholder()
        self.q_lora = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.k_lora = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.v_lora = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.o_lora = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = LoRALinear(d_model, dim_feedforward, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.linear2 = LoRALinear(dim_feedforward, d_model, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        x = self.norm1(src)
        B, T, _ = x.shape
        q = self.q_lora(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_lora(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_lora(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        _sdpa_mask = None
        if src_key_padding_mask is not None:
            _sdpa_mask = (~src_key_padding_mask.bool())[:, None, None, :]
        if is_causal:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=_sdpa_mask, is_causal=True,
                dropout_p=self.dropout1.p if self.training else 0.0)
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=_sdpa_mask,
                dropout_p=self.dropout1.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        src = src + self.dropout2(self.o_lora(attn_out))
        x = self.norm2(src)
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        return src + x

def freeze_backbone_except_lora(model, lora_prefixes=("q_lora", "k_lora", "v_lora", "o_lora", "linear1", "linear2")):
    """v25.55: freeze everything except LoRA adapter params (and keep heads
    trainable -- the caller's head-selection policy still applies). Returns
    the list of trainable (name, param) pairs for the optimizer."""
    trainable = []
    for name, p in model.named_parameters():
        is_lora = any(pre in name for pre in lora_prefixes) and "lora_" in name
        is_head = any(name.startswith(pre) for pre in ("fog_head", "tok_fog", "tok_sub", "subtype_head", "walking_gate",
                                                       "hazard_head"))  # v28bp: hazard is an aux head too
        keep = is_lora or is_head
        p.requires_grad_(keep)
        if keep:
            trainable.append((name, p))
    return trainable

class TransformerBiLSTMWinner(nn.Module):
    """v25.13: faithful adaptation of the RETRIEVED 1st-place architecture
    (Baurzhan Urazalinov, private-LB mAP 0.514 -- his 3 public Kaggle
    notebooks): 5-layer Transformer encoder (D, H heads, FFN 4D, dropout 0.1)
    + 2 BiLSTM(D) + linear + sigmoid over ViT-style 2D patches (patch = P
    samples ~= 0.14 s), ~864-token blocks (~121 s of context), labels
    max-pooled per patch, masked BCE, learnable positional encoding with
    random-roll augmentation.

    Adaptations to THIS pipeline (all documented in the manifest):
      - stride-P Conv1d patch embedding (their 2D patch projection for a 1-D
        accel stream); frames are padded to a patch multiple inside forward
      - token stream linearly upsampled to FULL frame resolution before the
        per-frame heads (v25.2 lesson: nearest-upsampled logits killed
        SH/Walking in the old ResNet)
      - 4-class softmax subtype head (v25.3 contract, shared across models)
        instead of their per-class sigmoids
      - walking-gate + medication-FiLM retained for the clinical track
      - random-roll pos-encoding augmentation lives in the training loop
        (NEUROMAMBA_WINNER_ROLL_AUG=1)

    Config: NEUROMAMBA_WINNER_D (default 128; winner used 320),
    NEUROMAMBA_WINNER_HEADS (6), NEUROMAMBA_WINNER_DEPTH (5),
    NEUROMAMBA_WINNER_PATCH (14 @ 100 Hz)."""
    def __init__(self, in_channels=N_CHANNELS, d_model=128, n_heads=6, n_layers=5,
                 patch=14, dropout=0.1, use_walking_gate=True):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        if d_model % n_heads != 0:
            cands = [h for h in range(n_heads, 0, -1) if d_model % h == 0]
            n_heads = cands[0] if cands else 1
            print(f"[winner v25.16e] d_model={d_model} not divisible by requested heads "
                  f"-> using n_heads={n_heads}", flush=True)
        self.n_heads = n_heads
        self.patch = int(patch)
        self._use_wg = use_walking_gate
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=self.patch,
                                     stride=self.patch, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)
        self.lora_rank = int(get("NEUROMAMBA_LORA_RANK", "0"))
        self.lora_alpha = float(get("NEUROMAMBA_LORA_ALPHA", "1.0"))
        self.lora_dropout = float(get("NEUROMAMBA_LORA_DROPOUT", "0.1"))
        self.lora_freeze_backbone = _flag("NEUROMAMBA_LORA_FREEZE_BACKBONE", "0")
        if self.lora_rank > 0:
            _lora_layer = LoRATransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, rank=self.lora_rank, alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout, activation="gelu")
            self.encoder = nn.TransformerEncoder(_lora_layer, num_layers=n_layers)
            _n_lora = sum(1 for n, _ in self.encoder.named_parameters() if "lora_A" in n or "lora_B" in n)
            print(f"[winner v25.55] LoRA encoder ACTIVE: rank={self.lora_rank} "
                  f"alpha={self.lora_alpha} dropout={self.lora_dropout} "
                  f"lora_tensors={_n_lora} freeze_backbone={self.lora_freeze_backbone}", flush=True)
        else:
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=4 * d_model, dropout=dropout,
                                               batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        head_in = d_model * 2
        self.lstm_norm = nn.LayerNorm(head_in)
        self.tok_fog = nn.Linear(head_in, 1)
        self.tok_sub = nn.Linear(head_in, 4 if _sub_head_mode() == "softmax" else 3)
        self.use_local_attn = _flag("NEUROMAMBA_USE_LOCAL_ATTN", "1")
        self.local_attn = LocalWindowAttention(head_in, window_size=64, n_heads=4) if self.use_local_attn else None
        self.use_med_film = _flag("NEUROMAMBA_USE_MED_FILM", "1")
        self.med_film = nn.Linear(1, 2 * head_in) if self.use_med_film else None
        self.use_sensor_film = _flag("NEUROMAMBA_B3", "0")
        self.sensor_film = nn.Linear(_B3_PROMPT_DIM, 2 * head_in) if self.use_sensor_film else None
        if self.sensor_film is not None:
            nn.init.zeros_(self.sensor_film.weight)
            nn.init.zeros_(self.sensor_film.bias)
        self.use_gate_dann = _flag("NEUROMAMBA_B6", "0")
        if self.use_sensor_film:
            print(f"[b3 v29ae] ACTIVE: sensor position/rate FiLM (zero-init, "
                  f"prompt_dim={_B3_PROMPT_DIM}, rate_norm={_b3_rate_norm()}) [winner]", flush=True)
        if self.use_gate_dann:
            print(f"[b6 v29ae] ACTIVE: shift-robust gate DANN "
                  f"(w={os.environ.get('NEUROMAMBA_B6_DANN_W', '0.05')}) [winner]", flush=True)
        self.fog_head = nn.Sequential(nn.Conv1d(head_in, head_in // 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv1d(head_in // 2, 1, 1))
        self.subtype_ce = _sub_head_mode() == "softmax"  # v25.14b sigmoid mode
        n_sub = 4 if self.subtype_ce else 3
        self.subtype_head = nn.Sequential(nn.Conv1d(head_in, head_in // 2, 1), nn.ReLU(inplace=True),
                                          nn.Conv1d(head_in // 2, n_sub, 1))
        self.walking_gate = WalkingGate(in_channels=in_channels) if use_walking_gate else None

    def forward(self, x, mask=None, med=None, lengths=None, prompt=None):
        if x.dim() == 3 and x.shape[1] == N_CHANNELS and x.shape[-1] != N_CHANNELS:
            x = x.permute(0, 2, 1)
        B, T_in, C = x.shape
        x_ct = x.permute(0, 2, 1).contiguous()
        pad = (self.patch - T_in % self.patch) % self.patch
        if pad:
            x_ct = F.pad(x_ct, (0, pad))
        tok = self.patch_embed(x_ct).permute(0, 2, 1)       # [B, n_tok, D]
        n_tok = tok.shape[1]
        _pad_mask_tok = None
        _tok_lens = None
        if lengths is not None and _flag("NEUROMAMBA_PAD_MASK", "0"):
            _lens = torch.as_tensor(lengths, dtype=torch.long, device=x.device).clamp(min=1)
            _tok_lens = torch.clamp(torch.div(_lens + self.patch - 1, self.patch,
                                              rounding_mode="floor"), min=1, max=n_tok)
            _ar = torch.arange(n_tok, device=x.device)
            _pad_mask_tok = _ar[None, :] >= _tok_lens[:, None]      # True = pad
            _pad_mask_tok = _pad_mask_tok & ~_pad_mask_tok.all(dim=1, keepdim=True)
        pe = self.pos_emb[:, :n_tok, :]
        if pe.shape[1] < n_tok:
            pe = F.interpolate(self.pos_emb.permute(0, 2, 1), size=n_tok,
                               mode="linear", align_corners=False).permute(0, 2, 1)
        tok = self.embed_dropout(tok + pe)
        if _pad_mask_tok is not None:
            h = self.encoder(tok, src_key_padding_mask=_pad_mask_tok)
        else:
            h = self.encoder(tok)
        if _pad_mask_tok is not None and bool((_tok_lens < n_tok).any()):
            _packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths=_tok_lens.detach().cpu().tolist(), batch_first=True,
                enforce_sorted=False)
            _h_out, _ = self.lstm(_packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(_h_out, batch_first=True,
                                                    total_length=n_tok)
        else:
            h, _ = self.lstm(h)                                  # [B, n_tok, 2D]
        h = self.lstm_norm(h)
        if self.local_attn is not None:
            h = self.local_attn(h, key_pad=_pad_mask_tok) if _pad_mask_tok is not None \
                else self.local_attn(h)
        h = self.embed_dropout(h)
        h_up = F.interpolate(h.permute(0, 2, 1), size=n_tok * self.patch, mode="linear",
                             align_corners=False)[:, :, :T_in]  # full resolution
        if self.use_med_film and self.med_film is not None and med is not None:
            gb = self.med_film(med.to(h_up.dtype).unsqueeze(-1))
            gamma, beta = gb.chunk(2, dim=-1)
            h_up = h_up * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        if prompt is not None and getattr(self, "sensor_film", None) is not None:
            if not torch.is_tensor(prompt):
                prompt = torch.as_tensor(np.asarray(prompt, dtype=np.float32))
            prompt = prompt.to(device=h_up.device, dtype=h_up.dtype)
            gb3 = self.sensor_film(prompt)
            gamma3, beta3 = gb3.chunk(2, dim=-1)
            h_up = h_up * (1.0 + gamma3[:, :, None]) + beta3[:, :, None]
        fog_logits = self.fog_head(h_up).squeeze(1)
        subtype_logits = self.subtype_head(h_up).permute(0, 2, 1)
        aux = {}
        if self.walking_gate is not None:
            if getattr(self, "use_gate_dann", False):
                _wp, _gf = self.walking_gate(x_ct[:, :, :T_in], return_feat=True)
                aux["walking_prob"] = _wp
                aux["gate_feat"] = _gf  # v29ae (B6)
            else:
                aux["walking_prob"] = self.walking_gate(x_ct[:, :, :T_in])
        aux["token_fog_logits"] = self.tok_fog(h).squeeze(-1)   # [B, n_tok]
        aux["token_subtype_logits"] = self.tok_sub(h)           # [B, n_tok, n_sub]
        aux["token_embed"] = h  # v25.19: SH batch-local contrastive stream (token embeddings)
        if _flag("NEUROMAMBA_WINNER_TOKEN_LOSS", "0"):
            sub_tok = self.tok_sub(h).permute(0, 2, 1)          # [B, 4, n_tok]
            sub_tok_up = F.interpolate(sub_tok, size=n_tok * self.patch,
                                       mode="nearest").permute(0, 2, 1)[:, :T_in, :]
            fog_tok = self.tok_fog(h).squeeze(-1).unsqueeze(1)  # [B, 1, n_tok]
            fog_tok_up = F.interpolate(fog_tok, size=n_tok * self.patch,
                                       mode="nearest").squeeze(1)[:, :T_in]
            subtype_logits = sub_tok_up.contiguous()
            fog_logits = fog_tok_up.contiguous()
        return fog_logits, subtype_logits, aux

    def get_architecture_manifest(self):
        return {"model_version": "transformer_bilstm_winner_v25.13",
                "backbone": f"patch_embed(P={self.patch}) + {self.n_layers}xTransformer"
                            f"(D={self.d_model},H={self.n_heads}) + 2xBiLSTM({self.d_model})",
                "in_channels_contract": N_CHANNELS, "no_global_pooling": True,
                "full_resolution_heads": True, "use_walking_gate": self._use_wg,
                "med_film": self.use_med_film,
                "sensor_film": getattr(self, "use_sensor_film", False),  # v29ae B3
                "gate_dann": getattr(self, "use_gate_dann", False),      # v29ae B6
                "subtype_head": (f"{'4_class_softmax_CE' if getattr(self, 'subtype_ce', True) else '3x_masked_sigmoid'}"
                                 f" (SH/Turn/Walking{('/None' if getattr(self, 'subtype_ce', True) else '')})"),
                "reference": "1st-place private-LB 0.514 (Urazalinov public notebooks: "
                             "5xTransformer D320 H6 + 2xBiLSTM320, 0.14s 2D patches, "
                             "864 tokens ~121s context, max-pooled patch labels, masked BCE)",
                "adaptations": "patch=stride-P Conv1d, full-res heads (v25.2), 4-class "
                               "softmax head (v25.3), walking-gate + med-FiLM retained",
                "role": "main_model"}

def _sanity_test_winner():
    d = int(os.environ.get("NEUROMAMBA_WINNER_D", "32"))
    model = TransformerBiLSTMWinner(in_channels=N_CHANNELS, d_model=d, n_heads=4,
                                    n_layers=2, patch=14, dropout=0.0)
    x = torch.randn(2, 2000, N_CHANNELS)
    fog, sub, aux = model(x)
    assert fog.shape == (2, 2000) and sub.shape == (2, 2000, 4)
    assert "walking_prob" in aux and aux["walking_prob"].shape == (2, 2000)
    assert (fog.std(dim=1) > 1e-6).all(), "REGRESSION: constant per-frame outputs"
    x_ct = x.permute(0, 2, 1)
    fog2, sub2, _ = model(x_ct)
    assert torch.allclose(fog, fog2, atol=1e-5), "BCT/ BTC layout mismatch"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[winner v25.13] Sanity check passed: fog={fog.shape} sub4={sub.shape} "
          f"params={n_params / 1e6:.2f}M", flush=True)
    os.environ["NEUROMAMBA_LORA_RANK"] = "8"
    os.environ["NEUROMAMBA_LORA_ALPHA"] = "1.0"
    os.environ["NEUROMAMBA_LORA_FREEZE_BACKBONE"] = "0"
    try:
        model_l = TransformerBiLSTMWinner(in_channels=N_CHANNELS, d_model=d, n_heads=4,
                                          n_layers=2, patch=14, dropout=0.0)
        fog_l, sub_l, aux_l = model_l(x)
        assert fog_l.shape == fog.shape and sub_l.shape == sub.shape
        n_lora = sum(1 for n, _ in model_l.named_parameters() if "lora_A" in n or "lora_B" in n)
        assert n_lora > 0, "LoRA tensors missing"
        _b_zero = all(bool((p == 0).all()) for n, p in model_l.named_parameters()
                      if n.endswith("lora_B"))
        assert _b_zero, "LoRA B must be zero-init (identity-at-init property)"
        model_l.train()
        _opt = torch.optim.Adam([p for n, p in model_l.named_parameters() if "lora_" in n], lr=1e-3)
        _loss = model_l(x)[0].pow(2).mean()
        _opt.zero_grad()
        _loss.backward()
        _b_grad = any(p.grad is not None and bool((p.grad != 0).any())
                      for n, p in model_l.named_parameters() if n.endswith("lora_B"))
        _opt.step()
        _b_moved = any(bool((p != 0).any()) for n, p in model_l.named_parameters()
                       if n.endswith("lora_B"))
        os.environ["NEUROMAMBA_LORA_FREEZE_BACKBONE"] = "1"
        model_f = TransformerBiLSTMWinner(in_channels=N_CHANNELS, d_model=d, n_heads=4,
                                          n_layers=2, patch=14, dropout=0.0)
        freeze_backbone_except_lora(model_f)
        _trainable = [n for n, p in model_f.named_parameters() if p.requires_grad]
        _frozen = [n for n, p in model_f.named_parameters() if not p.requires_grad]
        assert any("lora_A" in n for n in _trainable), "no LoRA params trainable under freeze"
        assert any("patch_embed" in n for n in _frozen), "backbone not frozen"
        print(f"[winner v25.55] LoRA sanity passed: rank=8 B-zero-init OK "
              f"(grad={_b_grad}, moved={_b_moved}), lora_tensors={n_lora}, "
              f"freeze path: {len(_trainable)} trainable / {len(_frozen)} frozen",
              flush=True)
    finally:
        os.environ.pop("NEUROMAMBA_LORA_RANK", None)
        os.environ.pop("NEUROMAMBA_LORA_ALPHA", None)
        os.environ.pop("NEUROMAMBA_LORA_FREEZE_BACKBONE", None)
    return True

def _load_pretrained_stem(model, state, model_type):
    """v25.14: transfer the SSL encoder weights into the supervised model.
    'sequence' keys match 1:1; other model types are skipped
    (patch-embed / resnet stems differ)."""
    if model_type != "sequence":
        print(f"[ssl v25.14] model_type={model_type}: stem layout differs -- SSL init skipped", flush=True)
        return 0
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = sum(1 for k in state if k not in missing)
    print(f"[ssl v25.14] loaded {n_loaded} SSL encoder tensors into {model_type} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    if _flag("NEUROMAMBA_SSL_FREEZE_STEM", "0"):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.startswith(("stem.", "blocks.", "blocks_cnn.")):
                    p.requires_grad = False
        print("[ssl v25.14] stem+blocks FROZEN (LIFT-PD label-efficient fine-tune)", flush=True)
    return n_loaded

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, mask=None, pos_weight=None, per_cell_weight=None):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if pos_weight is not None:
            los_pos = los_pos * pos_weight
        loss = los_pos + los_neg
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        one_sided_w = torch.pow((1 - pt).clamp(min=0), one_sided_gamma)
        loss = -(loss * one_sided_w)
        if per_cell_weight is not None:
            loss = loss * per_cell_weight
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()

_SUBTYPE_ASL_GAMMA_NEG = {"StartHesitation": 1.5, "Turn": 2.0, "Walking": 2.0}
_ASL_CACHE = {}

_DATASET_SUB_PW = None

def subtype_asl_loss(subtype_logits, y_sub_t, mask_t, pos_weights=None, class_offset=0):
    """v25: mask_t is now [N,3] per-class subtype_observed (was [N,1] broadcast).
    Unknown frames (subtype_observed=False) contribute ZERO gradient -- the
    direct fix for external binary-only cohorts suppressing rare subtypes.

    v25.51.2: supports CLASS SLICES -- the defog weak-SH "term" branch scores SH
    alone through [:, :1] and Turn+Walking through [:, 1:]. The loop now runs
    over the ACTUAL tensor width (the old code always iterated all 3
    SUBTYPE_NAMES and indexed straight into the sliced tensors -> IndexError on
    every pure-defog batch); pass class_offset=k0 for slices that do not start
    at class 0 so per-class gamma / TURN_PLAIN semantics stay aligned with
    SUBTYPE_NAMES. A pos_weights list must match the SLICE width; shorter lists
    fall back to 1.0 for the remaining classes."""
    total, denom = 0.0, 0.0
    n_cls = int(subtype_logits.shape[1])
    _turn_plain = _flag("NEUROMAMBA_TURN_PLAIN", "0")  # v25.51; v25.57: _flag parse ("true" spelling)
    if pos_weights is None and _flag("NEUROMAMBA_SUB_POS_WEIGHT", "1"):
        cap = float(os.environ.get("NEUROMAMBA_SUB_POS_WEIGHT_CAP", "20"))
        _pw_src = os.environ.get("NEUROMAMBA_SUB_PW_SOURCE", "dataset")
        _ds_pw = globals().get("_DATASET_SUB_PW")
        if _pw_src == "dataset" and _ds_pw is not None:
            pos_weights = [min(max(float(_ds_pw[k]), 1.0), cap) for k in range(n_cls)]
        else:
            pw_list = []
            for k in range(n_cls):
                y_k = y_sub_t[:, k]
                m_k = mask_t[:, k] if mask_t.dim() == 2 else mask_t[:, 0]
                n_pos = float((y_k * m_k).sum().item())
                n_valid = float(m_k.sum().item())
                pw_list.append(1.0 if n_pos <= 0 else min(max(n_valid / max(n_pos, 1.0), 1.0), cap))
            pos_weights = pw_list
    for k in range(n_cls):
        name = SUBTYPE_NAMES[min(k + class_offset, len(SUBTYPE_NAMES) - 1)]
        gamma_neg = 1.0 if (_turn_plain and name == "Turn") else _SUBTYPE_ASL_GAMMA_NEG.get(name, 2.0)
        if gamma_neg not in _ASL_CACHE:
            _ASL_CACHE[gamma_neg] = AsymmetricLossOptimized(gamma_neg=gamma_neg, gamma_pos=1.0, clip=0.05)
        asl = _ASL_CACHE[gamma_neg]
        logits_k = subtype_logits[:, k]
        y_k = y_sub_t[:, k]
        m_k = mask_t[:, k] if mask_t.dim() == 2 else mask_t[:, 0]
        n_valid = m_k.float().sum().clamp(min=1.0)
        pw = pos_weights[k] if (pos_weights is not None and k < len(pos_weights)) else 1.0
        if _turn_plain and name == "Turn":
            pw = 1.0
        _per_cell = None
        _whw = float(os.environ.get("NEUROMAMBA_WALK_HARDNEG_W", "1.0"))
        _shw = float(os.environ.get("NEUROMAMBA_SH_HARDNEG_W", "1.0"))
        if (name == "Walking" and _whw > 1.0) or (name == "StartHesitation" and _shw > 1.0):
            _fog_ctx = (y_sub_t.max(dim=1).values > 0.5).float()
            _cls_neg = 1.0 - (y_k > 0.5).float()
            _w_lvl = _whw if name == "Walking" else _shw
            _per_cell = 1.0 + (_w_lvl - 1.0) * _fog_ctx * _cls_neg
        total = total + asl(logits_k, y_k, m_k.float(), pos_weight=pw,
                            per_cell_weight=_per_cell) * n_valid
        denom = denom + n_valid
    return total / denom.clamp(min=1.0)

def sh_contrastive_aux_loss(token_embed, y_sub_pool, m_pool, temperature=0.1, min_pos=2):
    """v25.19-novel: batch-local supervised contrastive loss on token
    embeddings (pre-logit), pulling Start-Hesitation tokens together and
    pushing them away from confidently-observed None tokens in the same
    batch. Unlike class-weighted CE (which only reshapes the per-token
    gradient), this directly shapes the embedding geometry so SH tokens
    occupy a distinguishable region even when they are a handful of examples
    per batch -- addressing the SH AP-collapses-to-zero failure mode seen
    across every prior version of this pipeline. Cheap: O(B*T) pairs within
    the batch only, no extra data or forward passes.
    token_embed: [N, D] pooled token features (grad flows).
    y_sub_pool:  [N, 3] pooled subtype targets (SH/Turn/Walk).
    m_pool:      [N, 3] observed mask for the same tokens.
    v25.20-E: the v25.19 draft padded ones_like(sim_pp) over the negative
    columns, so negative tokens were counted as POSITIVES and the loss
    actively ATTRACTED SH to None. Fixed to SupCon semantics: positives are
    other SH tokens only; negatives act through the log-softmax denominator.
    """
    sh_pos = (y_sub_pool[:, 0] > 0.5) & (m_pool[:, 0] > 0.5)
    none_neg = (y_sub_pool.sum(dim=1) < 0.5) & (m_pool.sum(dim=1) > 1.5)
    n_pos = int(sh_pos.sum().item())
    if n_pos < min_pos or none_neg.sum() < 1:
        return torch.zeros((), device=token_embed.device, requires_grad=True) * token_embed.sum()
    pos_emb = F.normalize(token_embed[sh_pos], dim=-1)
    neg_idx = torch.nonzero(none_neg, as_tuple=True)[0]
    if neg_idx.numel() > 512:
        neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:512]]
    neg_emb = F.normalize(token_embed[neg_idx], dim=-1)
    sim_pp = (pos_emb @ pos_emb.t()) / temperature
    sim_pn = (pos_emb @ neg_emb.t()) / temperature
    logits = torch.cat([sim_pp, sim_pn], dim=1)
    eye = torch.eye(pos_emb.shape[0], device=token_embed.device, dtype=torch.bool)
    log_prob = F.log_softmax(logits, dim=1)
    pos_mask = F.pad(torch.ones_like(sim_pp, dtype=torch.bool) & ~eye, (0, neg_emb.shape[0]))
    denom = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / denom
    return loss.mean()

def subtype_token_bce_loss(tok_logits, y_pool, m_pool, pos_weight=None):
    """v25.20-A: the 1st-place subtype objective VERIFIED from the winner's own
    notebooks (research/winner_defog_code.txt: loss_function =
    ce(output, real) * mask summed / sum(mask); 4 sigmoid heads
    SH/Turn/Walk/Event; per-patch max-pooled targets AND masks; masks =
    Task&Valid). Independent sigmoids give each rare class its own gradient
    stream; the v25.3 4-class softmax forced SH/Walking to compete with
    Turn+None over one normalization and left SH AP at 0.000 in every
    winner-model run. No class weights (the winner used none).
    v25.33: optional per-class pos_weight (vector [SH, Turn, Walk]).
    Unweighted masked BCE made 'predict not-Walk everywhere' loss-optimal:
    Walk positives are 2.5% of observed subtype frames, so the Walk sigmoid
    collapsed to ~0 output and Walk AP stayed 0.000 in every v25.32 epoch
    (verified arithmetically: epoch-3 calibSubtypeMAP 0.1681 ==
    (0.000 SH + 0.504 Turn + 0.000 Walk)/3 exactly). pos_weight rebalances
    the per-class gradient mass the way the ASL path already does with
    NEUROMAMBA_SUB_POS_WEIGHT.
    v25.59d (Round 28ac): CONTEXT-AWARE WALK HARD NEGATIVES. The OOF
    behavioral audit (diag_walk_behavior.py, defog member shared-val) shows
    the Walk head's confuser is the FoG-positive-not-Walking frame (mean Walk
    score 0.259 vs 0.393 on Walk positives -- freezing-in-place during
    turns/SH looks like walking). With NEUROMAMBA_WALK_HARDNEG_W>1, the Walk
    channel's NEGATIVE loss is up-weighted x W on exactly those cells
    (fog-context>0.5 & walk-target=0), attacking the measured confusion
    directly instead of adding global weight."""
    loss = F.binary_cross_entropy_with_logits(tok_logits, y_pool, reduction="none",
                                              pos_weight=pos_weight)
    _whw = float(os.environ.get("NEUROMAMBA_WALK_HARDNEG_W", "1.0"))
    _shw = float(os.environ.get("NEUROMAMBA_SH_HARDNEG_W", "1.0"))
    if _whw > 1.0 or _shw > 1.0:
        _fog_ctx = (y_pool.max(dim=-1).values > 0.5).float()
        _w_sh = torch.ones_like(_fog_ctx)
        _w_walk = torch.ones_like(_fog_ctx)
        if _whw > 1.0:
            _w_walk = _w_walk + (_whw - 1.0) * _fog_ctx * (1.0 - (y_pool[..., 2] > 0.5).float())
        if _shw > 1.0:
            _w_sh = _w_sh + (_shw - 1.0) * _fog_ctx * (1.0 - (y_pool[..., 0] > 0.5).float())
        loss = loss * torch.stack([_w_sh, torch.ones_like(_fog_ctx), _w_walk], dim=-1)
    loss = loss * m_pool
    denom = m_pool.sum().clamp(min=1.0)
    return loss.sum() / denom

def subtype_token_asl_loss(tok_logits, y_pool, m_pool, pos_weight=None):
    """v25.59 (Round 28z): token-level ASYMMETRIC LOSS -- the GRU/ASL path's
    proven rare-class objective (subtype_asl_loss; Ben-Baruch et al. 2020,
    ASL) ported to the winner's patch-pooled token contract. Per-class
    gamma_neg focuses gradients on HARD negatives instead of the easy
    'not-SH' ocean (the plain BCE treats every one of the ~97% negative
    tokens equally once p is small); combined with the [8,1,8] pos_weight
    stack this is the measured SH/Walk lever pair. Env:
    NEUROMAMBA_TOKEN_ASL=1 (default 0 = plain token BCE, bit-parity).
    tok_logits/y_pool/m_pool: [B, n_tok, 3]. Mirrors subtype_asl_loss's
    per-class AsymmetricLossOptimized call (same gamma_neg table + 0.05
    probability shift) so both engines share one objective family."""
    total, denom = 0.0, 0.0
    n_cls = int(tok_logits.shape[-1])
    flat_logits = tok_logits.reshape(-1, n_cls)
    flat_y = y_pool.reshape(-1, n_cls)
    flat_m = m_pool.reshape(-1, n_cls)
    for k in range(n_cls):
        name = SUBTYPE_NAMES[min(k, len(SUBTYPE_NAMES) - 1)]
        _turn_plain = _flag("NEUROMAMBA_TURN_PLAIN", "0")
        gamma_neg = 1.0 if (_turn_plain and name == "Turn") else _SUBTYPE_ASL_GAMMA_NEG.get(name, 2.0)
        if gamma_neg not in _ASL_CACHE:
            _ASL_CACHE[gamma_neg] = AsymmetricLossOptimized(gamma_neg=gamma_neg, gamma_pos=1.0, clip=0.05)
        asl = _ASL_CACHE[gamma_neg]
        pw = 1.0 if (pos_weight is None or k >= len(pos_weight)) else float(pos_weight[k])
        n_valid = flat_m[:, k].float().sum().clamp(min=1.0)
        _per_cell = None
        _whw = float(os.environ.get("NEUROMAMBA_WALK_HARDNEG_W", "1.0"))
        _shw = float(os.environ.get("NEUROMAMBA_SH_HARDNEG_W", "1.0"))
        if (name == "Walking" and _whw > 1.0) or (name == "StartHesitation" and _shw > 1.0):
            _fog_ctx = (flat_y.max(dim=1).values > 0.5).float()
            _cls_neg = 1.0 - (flat_y[:, k] > 0.5).float()
            _w_lvl = _whw if name == "Walking" else _shw
            _per_cell = 1.0 + (_w_lvl - 1.0) * _fog_ctx * _cls_neg
        total = total + asl(flat_logits[:, k], flat_y[:, k], flat_m[:, k].float(), pos_weight=pw,
                            per_cell_weight=_per_cell) * n_valid
        denom = denom + n_valid
    return total / denom.clamp(min=1.0)

_SUB_CE_WEIGHTS_DEFAULT = "5.0,2.0,5.0,1.0"  # v25.19: softened from 8.0/2.5/8.0 -- extreme weights on

def subtype_ce_loss(subtype_logits, y_sub_t, mask_t, class_weights=None):
    """v25.3: 4-class cross-entropy (SH/Turn/Walking/None) -- the objective of
    every Kaggle top-5 solution. TLVMC subtype labels are mutually exclusive
    per frame (verified: 0 overlapping positives), so a softmax with
    inverse-frequency class weights gives SH (0.47% prevalence) and Walking
    (1.9%) the gradient mass that 3 independent ASL sigmoids could not (Turn
    at 18% prevalence absorbed it -- SH/Walking APs collapsed to ~0 while Turn
    AP reached 0.57). Frames with NO observed subtype are fully masked."""
    B, T, C4 = subtype_logits.shape
    yt = y_sub_t.reshape(-1, 3)
    mt = mask_t.reshape(-1, 3).float()
    keep = mt.sum(dim=1) > 0.5
    if keep.sum() == 0:
        return torch.zeros((), device=subtype_logits.device, requires_grad=True) * subtype_logits.sum()
    pos = yt > 0.5
    tgt = torch.full((yt.shape[0],), 3, dtype=torch.long, device=yt.device)
    tgt[pos[:, 0]] = 0                                   # StartHesitation
    tgt[pos[:, 1] & ~pos[:, 0]] = 1                      # Turn
    tgt[pos[:, 2] & ~pos[:, 0] & ~pos[:, 1]] = 2         # Walking
    ls = float(os.environ.get("NEUROMAMBA_SUB_CE_LABEL_SMOOTH", 0.05))
    return F.cross_entropy(subtype_logits.reshape(-1, 4)[keep], tgt[keep],
                           weight=class_weights, reduction="mean", label_smoothing=ls)

def loss_weight_schedule(epoch, total_epochs, fog_warmup_epochs=None, ramp_epochs=10,
                          fog_w_final=0.6, sub_w_final=1.0):
    """v25-FIX-1: fog_warmup 10 -> 2 (env override); sub_w now has a FLOOR from
    epoch 0 (was 0 for 10 epochs). This is THE fix for 'calibSubtypeMAP not
    improving epoch by epoch' -- the subtype head was literally not training
    for the first 10 epochs. v25.19: floor raised 0.25 -> 0.35 (earlier signal
    for the rare subtype head) and the ramp is configurable via
    NEUROMAMBA_SUB_W_RAMP_EPOCHS so sub_w does not saturate at 1.0 by epoch
    ~12 (it previously pinned at the ceiling for the whole fine-tune phase)."""
    if fog_warmup_epochs is None:
        fog_warmup_epochs = int(os.environ.get("NEUROMAMBA_FOG_WARMUP_EPOCHS", 2))
    sub_floor = float(os.environ.get("NEUROMAMBA_SUB_W_FLOOR", 0.35))
    if epoch < fog_warmup_epochs:
        return 1.0, sub_floor
    ramp_epochs = int(os.environ.get("NEUROMAMBA_SUB_W_RAMP_EPOCHS", ramp_epochs if ramp_epochs else 10)) or ramp_epochs
    sub_w_max = float(os.environ.get("NEUROMAMBA_SUB_W_MAX", 1.0))
    t = min(1.0, (epoch - fog_warmup_epochs) / max(1, ramp_epochs))
    fog_w = 1.0 + t * (fog_w_final - 1.0)
    sub_w = min(sub_w_max, sub_floor + (1.0 - sub_floor) * t * sub_w_final)
    return fog_w, sub_w

def poly1_bce_loss(logits, targets, mask, weight=None, epsilon=None, pos_weight=None):
    if epsilon is None:
        epsilon = float(os.environ.get("NEUROMAMBA_POLY1_EPS", 1.0))
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    loss = (bce + epsilon * (1 - pt)) * mask
    if weight is not None:
        loss = loss * weight
    denom = (mask * weight).sum().clamp(min=1.0) if weight is not None else mask.sum().clamp(min=1.0)
    return loss.sum() / denom

def compute_loss(fog_logits, y, m, w=None, pos_weight=None):
    use_poly1 = _flag("NEUROMAMBA_USE_POLY1", "1")  # v25.57: _flag parse ("true" silently DISABLED poly1 before)
    if use_poly1:
        loss = poly1_bce_loss(fog_logits, y, m, weight=w, pos_weight=None)  # v25: don't stack pos_weight
        return loss, {"poly1": loss.item()}
    bce = F.binary_cross_entropy_with_logits(fog_logits, y, pos_weight=pos_weight, reduction="none")
    loss = bce * m
    if w is not None:
        loss = loss * w
    loss = loss.sum() / m.sum().clamp(min=1.0)
    return loss, {"bce": loss.item()}

def episode_soft_f1_loss(fog_logits, y, m, pool_frames=128, tau_quiet=0.15, tau_fire=0.6, peak_w=0.05):
    """v25.47-novel (review 3, solution 2): episode-centric training loss.
    Max-pool probs/labels over ~1 s chunks -> per-episode scores; soft-Dice
    (event-F1 surrogate) over pooled cells, masked. Plus a bimodality
    regularizer: quiet cells must stay below tau_quiet, fire cells above
    tau_fire -- the antidote to the unimodal drifting score whose trigger
    jumped 0.13 -> 0.38 -> 0.27 in v25.45. Returns (loss, term_dict)."""
    p = torch.sigmoid(fog_logits)
    T = p.shape[1]
    n_ch = max(int(T) // int(pool_frames), 1)
    Lc = n_ch * int(pool_frames)
    p_c = p[:, :Lc].reshape(p.shape[0], n_ch, int(pool_frames)).max(dim=2).values
    y_c = y[:, :Lc].reshape(y.shape[0], n_ch, int(pool_frames)).max(dim=2).values
    m_c = m[:, :Lc].reshape(m.shape[0], n_ch, int(pool_frames)).max(dim=2).values
    num = 2.0 * (p_c * y_c * m_c).sum()
    den = (p_c * m_c).sum() + (y_c * m_c).sum() + 1e-6
    soft_f1 = 1.0 - num / den
    peak_reg = torch.zeros((), device=p.device)
    if peak_w > 0:
        quiet = (y_c < 0.5) & (m_c > 0)
        fire = (y_c > 0.5) & (m_c > 0)
        if bool(quiet.any()):
            peak_reg = peak_reg + torch.relu(p_c[quiet] - tau_quiet).mean()
        if bool(fire.any()):
            peak_reg = peak_reg + torch.relu(tau_fire - p_c[fire]).mean()
    return soft_f1 + peak_w * peak_reg, {
        "ep_soft_f1": float(soft_f1.detach().cpu()),
        "ep_peak": float((peak_w * peak_reg).detach().cpu())}

def event_target_loss(fog_logits, y_fog, m):
    """v25-FIX-5 (1st-place trick): Event = max(SH,Turn,Walk). On notype data
    where subtypes are masked, train the FoG head on the binary Event target.
    This doubles effective training data without pseudo-labeling rare subtypes."""
    return F.binary_cross_entropy_with_logits(fog_logits, y_fog, reduction="none") * m

def walking_gate_loss(walking_prob, y_walking, m, w=None):
    """v25-FIX-6: walking-gate BCE on the activity-band target. v25.2: optional
    per-frame weight (ambiguous activity band = 0.25) so the gate only learns
    from confident rest/walk frames."""
    eps = 1e-7
    wp = walking_prob.clamp(eps, 1 - eps)
    loss = -(y_walking * torch.log(wp) + (1 - y_walking) * torch.log(1 - wp))
    if w is not None:
        loss = loss * w
        return loss.sum() / (m * w).sum().clamp(min=1.0)
    loss = loss * m
    return loss.sum() / m.sum().clamp(min=1.0)

_TLS_MODULE = None

def _load_tls_module():
    """v28bp hazard arm: import the VENDORED clinical TLS module
    (v25/clinical/temporal_label_smoothing.py -- the frozen v27-release tree is
    NEVER imported). Package import first (script-dir on sys.path), absolute
    file-location fallback (trainer loaded by path from another cwd). Cached
    per process; called lazily so flag-off runs never import numpy-only TLS
    machinery beyond what they already use."""
    global _TLS_MODULE
    if _TLS_MODULE is not None:
        return _TLS_MODULE
    try:
        from clinical import temporal_label_smoothing as _tls_mod
    except Exception:
        import importlib.util as _ilu
        _tls_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "clinical", "temporal_label_smoothing.py")
        _spec = _ilu.spec_from_file_location("clinical_temporal_label_smoothing_v25", _tls_path)
        _tls_mod = _ilu.module_from_spec(_spec)
        sys.modules[_spec.name] = _tls_mod
        _spec.loader.exec_module(_tls_mod)
    _TLS_MODULE = _tls_mod
    return _TLS_MODULE

print("[sampling] SAMPLING_V25_LOADED", flush=True)

_SAMPLER_WEIGHT_CACHE = {}  # id(list) -> (list_ref, weights, n_sh)  (v25.1-speed)

class SubtypeAwareSampler:
    """v25-FIX-3: lighter oversampling. v25.1-rebalance: SH default 5x (was 3x)
    -- with SH-targeted crops the binding constraint is SH exposure (defog has
    ~440 SH-positive frames across 4 sequences), not over-weighting. Turn/Walk
    1.5x. Weights are epoch-invariant, so they are memoized per sequence list
    (v25.1-speed: was recomputed from 2000+ sequences every epoch)."""
    def __init__(self, seqs, batch_size, sh_oversample=None, turn_oversample=1.5,
                 walk_oversample=1.5, seed=0):
        if sh_oversample is None:
            sh_oversample = float(os.environ.get("NEUROMAMBA_SH_OVERSAMPLE", 5.0))
        self.seqs = seqs
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        cached = _SAMPLER_WEIGHT_CACHE.get(id(seqs))
        if cached is not None and cached[0] is seqs:
            self.weights, n_sh = cached[1], cached[2]
        else:
            weights = np.ones(len(seqs))
            n_sh = 0
            for i, s in enumerate(seqs):
                valid = s.train_mask.astype(bool)
                if valid.sum() == 0:
                    continue
                ysub = s.y_subtype[valid]
                so = s.subtype_observed[valid]
                has_sh = ((ysub[:, 0] > 0.5) & so[:, 0]).sum() > 0
                has_turn = ((ysub[:, 1] > 0.5) & so[:, 1]).sum() > 0
                has_walk = ((ysub[:, 2] > 0.5) & so[:, 2]).sum() > 0
                w = 1.0
                if has_sh:
                    w = max(w, sh_oversample); n_sh += 1
                if has_turn:
                    w = max(w, turn_oversample)
                if has_walk:
                    w = max(w, walk_oversample)
                weights[i] = w
            self.weights = weights
            _SAMPLER_WEIGHT_CACHE[id(seqs)] = (seqs, weights, n_sh)
            print(f"[SubtypeAwareSampler v25] {n_sh}/{len(seqs)} sequences contain observed Start "
                  f"Hesitation and are oversampled {sh_oversample}x (v25.1: 5x default + SH crops)", flush=True)

    def __iter__(self):
        _mode = os.environ.get("NEUROMAMBA_SAMPLER_EXPOSURE", "order").strip().lower()
        _w = self.weights
        if str(os.environ.get("NEUROMAMBA_B5", "")).strip().lower() not in ("", "0"):
            _a = float(os.environ.get("NEUROMAMBA_B5_ALPHA", 2.0))
            _b = float(os.environ.get("NEUROMAMBA_B5_BETA", 2.0))
            _cohorts = sorted(set(s.source_type for s in self.seqs))
            _factors = {c: self.rng.beta(_a, _b) for c in _cohorts}
            _w = self.weights * np.array([_factors.get(s.source_type, 1.0)
                                          for s in self.seqs])
        if _mode in ("weighted", "replacement", "true"):
            p = _w / max(_w.sum(), 1e-8)
            n_draws = int(np.ceil(len(self.seqs) * float(np.mean(_w))))
            idx = self.rng.choice(len(self.seqs), size=max(n_draws, len(self.seqs)),
                                  replace=True, p=p)
            for i in range(0, len(idx), self.batch_size):
                yield idx[i:i + self.batch_size]
            return
        u = self.rng.uniform(size=len(self.seqs))
        keys = u ** (1.0 / np.maximum(_w, 1e-8))
        order = np.argsort(-keys)
        for i in range(0, len(order), self.batch_size):
            yield order[i:i + self.batch_size]

class BalancedSequenceSampler:
    def __init__(self, seqs, batch_size, pos_oversample=3.0, seed=0):
        self.seqs = seqs
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.pos_idx = [i for i, s in enumerate(seqs) if s.labels.sum() > 0]
        self.pos_oversample = pos_oversample

    def __iter__(self):
        weights = np.ones(len(self.seqs))
        if self.pos_idx:
            weights[self.pos_idx] = self.pos_oversample
        _mode = os.environ.get("NEUROMAMBA_SAMPLER_EXPOSURE", "order").strip().lower()
        if _mode in ("weighted", "replacement", "true"):
            p = weights / max(weights.sum(), 1e-8)
            n_draws = int(np.ceil(len(self.seqs) * float(np.mean(weights))))
            idx = self.rng.choice(len(self.seqs), size=max(n_draws, len(self.seqs)),
                                  replace=True, p=p)
            for i in range(0, len(idx), self.batch_size):
                yield idx[i:i + self.batch_size]
            return
        u = self.rng.uniform(size=len(self.seqs))
        keys = u ** (1.0 / np.maximum(weights, 1e-8))
        order = np.argsort(-keys)
        for i in range(0, len(order), self.batch_size):
            yield order[i:i + self.batch_size]

def positive_aware_crop(seq, window, rng):
    """v25: crop_pos_prob=0.2 (env-tunable). Near-natural centering rate keeps
    train/test prevalence aligned; the 8th-place winner used pure random windows."""
    L = seq.n_frames
    if window <= 0 or L <= window:
        return seq
    pos_prob = float(os.environ.get("NEUROMAMBA_CROP_POS_PROB", 0.2))
    pos = np.flatnonzero((seq.y_fog > 0.5) & seq.observed)
    if len(pos) and rng.rand() < pos_prob:
        center = int(rng.choice(pos))
    else:
        center = int(rng.randint(0, L))
    start = int(np.clip(center - window // 2, 0, L - window))
    stop = start + window
    return FoGSequence(signal=seq.signal[start:stop], y_subtype=seq.y_subtype[start:stop],
                        subject_id=seq.subject_id, record_id=f"{seq.record_id}_crop{start}",
                        source_type=seq.source_type, observed=seq.observed[start:stop],
                        label_weight=seq.label_weight[start:stop],
                        subtype_observed=seq.subtype_observed[start:stop],
                        fs_hz=seq.fs_hz, unit_transform=seq.unit_transform, strides=seq.strides,
                        position_tag=getattr(seq, "position_tag", ""))

TRAIN_WINDOW = int(os.environ.get("NEUROMAMBA_TRAIN_WINDOW", 6000))

def _prefetched(gen, n=2):
    """v25.1-speed: run the CPU-bound batch generator in a background thread so
    crop/augment/assembly for future batches overlaps GPU compute. Deterministic:
    items are yielded in the exact same order. Depth 0 disables (useful for
    debugging / single-thread environments)."""
    if n <= 0:
        yield from gen
        return
    q = queue.Queue(maxsize=n)
    sentinel = object()

    def _fill():
        try:
            for item in gen:
                q.put(item)
        except Exception:
            traceback.print_exc()
        finally:
            q.put(sentinel)

    t = threading.Thread(target=_fill, daemon=True, name="batch-prefetch")
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

_SEQ_FEATURE_CACHE = {}  # id(seq) -> (seq_ref, fog_pos_idx, sh_idx, y_walk_full)
_HARD_NEG_CACHE = {}     # v25.40: id(seq) -> low-energy non-FoG frame indices
_CONTRAST_EMA = None     # v25.40: running magnitude of the SH contrastive term (scale-normalized weight)

def _seq_features(seq):
    """v25.1-speed: memoized per-sequence derived arrays (fog-positive indices,
    SH-positive indices, walking-gate target + target weight). The walking
    target is computed on the full recording and sliced per crop."""
    key = id(seq)
    e = _SEQ_FEATURE_CACHE.get(key)
    if e is None or e[0] is not seq:
        yf = (seq.y_subtype.max(axis=1) > 0.5).astype(np.float32)
        pos = np.flatnonzero((yf > 0.5) & seq.observed)
        sh = np.flatnonzero((seq.y_subtype[:, 0] > 0.5) & seq.subtype_observed[:, 0])
        walk = np.flatnonzero((seq.y_subtype[:, 2] > 0.5) & seq.subtype_observed[:, 2])
        yw, yw_w = _derive_walking_target(seq)
        e = (seq, pos, sh, walk, yw, yw_w)
        _SEQ_FEATURE_CACHE[key] = e
    return e

def _crop_bounds(seq, window, rng, sh_scale=1.0):
    """v25.1: pure-array crop planning (no FoGSequence construction -- the old
    per-crop construction ran full validation on every batch sample). Adds
    SH-targeted cropping: with NEUROMAMBA_SH_CROP_PROB (default 0.25) the
    window centers on an observed Start-Hesitation frame whenever the recording
    contains one. Defog has only ~440 SH-positive frames (~0.012% of frames):
    random/pos crops used to hit SH ~once per epoch, so the SH head was trained
    on noise. Falls back to fog-positive (NEUROMAMBA_CROP_POS_PROB) / random.
    v25.7: sh_scale (per-subject diversity factor in [0,1]) downweights SH
    crops from subjects that dominate the SH total (one tdcsfog subject holds
    78% of all SH frames -- without the cap the SH head overfits that subject
    and collapses on the other subjects' validation folds)."""
    L = seq.n_frames
    if window <= 0 or L <= window:
        return 0, L
    _, pos, sh, walk, _, _ = _seq_features(seq)
    sh_prob = float(os.environ.get("NEUROMAMBA_SH_CROP_PROB", 0.40))  # v25.25: 0.25 -> 0.40 -- SH is a stand->initiation TRANSITION; the pre-event standing context only enters training via SH-centered crops
    walk_prob = float(os.environ.get("NEUROMAMBA_WALK_CROP_PROB", 0.25))  # v25.25: 0.20 -> 0.25 -- brief Walk events need centered exposure
    pos_prob = float(os.environ.get("NEUROMAMBA_CROP_POS_PROB", 0.2))
    if len(sh) and rng.rand() < sh_prob * sh_scale:
        center = int(sh[rng.randint(0, len(sh))])
    elif len(walk) and rng.rand() < walk_prob:
        center = int(walk[rng.randint(0, len(walk))])
    elif len(pos) and rng.rand() < pos_prob:
        center = int(pos[rng.randint(0, len(pos))])
    else:
        center = int(rng.randint(0, L))
    start = int(np.clip(center - window // 2, 0, L - window))
    return start, start + window

def _derive_walking_target(seq):
    """v25.2: activity-band gate target. v25 thresholded |acc| variance at the
    50th percentile, so exactly half of all frames were 'walking' and the gate
    loss stuck at ~0.51 == random (the gate learned nothing -> FARh stayed high
    and sens never recovered). Now: Walking-subtype positives (observed) are
    definite walk; smoothed |acc| energy below the 20th percentile is rest;
    above the 60th percentile is walk; the middle band is ambiguous and gets
    0.25 loss weight. Returns (y_walk, weight) on the FULL recording
    (memoized in _seq_features and sliced per crop)."""
    T = len(seq.labels)
    y_walk = np.zeros(T, dtype=np.float32)
    weight = np.zeros(T, dtype=np.float32)
    walk_pos = (seq.y_subtype[:, 2] > 0.5) & seq.subtype_observed[:, 2]
    y_walk[walk_pos] = 1.0
    weight[walk_pos] = 1.0
    mag = np.linalg.norm(seq.signal[:, :3], axis=1)
    if len(mag) > 100:
        roll_var = np.convolve(mag ** 2, np.ones(100) / 100, mode="same")
        p20, p60 = np.percentile(roll_var, [20, 60])
        if p60 > p20:
            unset = ~walk_pos
            walk = (roll_var > p60) & unset
            rest = (roll_var < p20) & unset
            mid = unset & ~(walk | rest)
            y_walk[walk] = 1.0
            weight[walk] = 1.0
            weight[rest] = 1.0
            weight[mid] = 0.25
    return y_walk, weight

def _hard_neg_crop_bounds(s, window, rng, prob=None):
    """v25.40: hard-negative mining -- with probability prob (env
    NEUROMAMBA_HARD_NEG_CROP_PROB, default 0.15; 0 disables) sample a
    LOW-ENERGY, non-FoG crop (voluntary stops, standing, rest) instead of a
    random crop. FoG false-alarm literature (TCN studies with explicit
    stopping trials) shows including stops reduces FAR/h -- freeze-like low
    locomotor energy is not necessarily FoG, and the model must see it as a
    hard negative. Returns (start, stop) or None (caller falls back to the
    normal crop)."""
    if prob is None:
        prob = float(os.environ.get("NEUROMAMBA_HARD_NEG_CROP_PROB", "0.15"))
    if prob <= 0 or rng.rand() >= prob:
        return None
    T = len(s.labels)
    if T <= window or window <= 0:
        return None
    low = _HARD_NEG_CACHE.get(id(s))
    if low is None:
        mag = np.linalg.norm(np.asarray(s.signal[:, :3], dtype=np.float64), axis=1)
        no_fog = np.asarray(s.labels) <= 0.5
        if len(mag) > 100:
            roll = np.convolve(mag ** 2, np.ones(100) / 100, mode="same")
            q = float(np.percentile(roll, 30))
            low = np.flatnonzero(no_fog & (roll < q))
        else:
            low = np.flatnonzero(no_fog)
        if len(_HARD_NEG_CACHE) > 2000:
            _HARD_NEG_CACHE.clear()
        _HARD_NEG_CACHE[id(s)] = low
    if len(low) == 0:
        return None
    anchor = int(low[rng.randint(len(low))])
    start = int(np.clip(anchor - window // 2, 0, max(0, T - window)))
    return start, start + window

def training_batches(seqs, batch_size, epoch=0, sampler_cls=None, use_aug=False,
                     partner_pool=None, sh_pool=None):
    """v25: yields y_sub_obs[B,T,3] and y_walk[B,T] for the per-class subtype
    mask and the walking-gate target respectively. v25.1-speed: crops are plain
    array slices (no FoGSequence construction/validation churn), walking targets
    and positive-index scans are memoized, and the generator (crop + augment +
    assembly) runs on a background thread (NEUROMAMBA_PREFETCH_DEPTH, default 2)
    so the GPU never waits on CPU work. Augmentation uses the same per-batch
    RNG seed schedule as before (epoch*10000 + batch_idx), so training is
    bit-identical to the old inline-augment loop, just overlapped."""
    sampler_cls = sampler_cls or SubtypeAwareSampler
    sampler = sampler_cls(seqs, batch_size, seed=epoch)
    crop_rng = np.random.RandomState(1009 + epoch)

    sh_scale_map = {}
    sh_cap_factor = float(os.environ.get("NEUROMAMBA_SH_SUBJECT_CAP", 4.0))
    if sh_cap_factor > 0:
        subj_sh = {}
        for s in seqs:
            subj_sh[s.subject_id] = subj_sh.get(s.subject_id, 0) + int(
                ((s.y_subtype[:, 0] > 0.5) & s.subtype_observed[:, 0]).sum())
        nz = sorted(v for v in subj_sh.values() if v > 0)
        if nz:
            cap = float(nz[int(0.75 * (len(nz) - 1))]) * sh_cap_factor
            for sid, v in subj_sh.items():
                sh_scale_map[sid] = min(1.0, cap / max(v, 1))
            n_capped = sum(1 for v in sh_scale_map.values() if v < 0.99)
            print(f"[sampler v25.7] SH subject-diversity cap: {n_capped}/{len(sh_scale_map)} subjects "
                  f"SH-crop-scaled (cap={cap:.0f} SH frames, factor={sh_cap_factor})", flush=True)

    def _gen():
        for bi, batch_idx in enumerate(sampler):
            crops = []
            for j in batch_idx:
                s = seqs[j]
                _hn = _hard_neg_crop_bounds(s, TRAIN_WINDOW, crop_rng)
                if _hn is not None:
                    start, stop = _hn
                else:
                    start, stop = _crop_bounds(s, TRAIN_WINDOW, crop_rng,
                                               sh_scale=sh_scale_map.get(s.subject_id, 1.0))
                crops.append((s, start, stop))
            max_len = max(stop - start for _, start, stop in crops)
            if TRAIN_WINDOW > 0 and max_len > TRAIN_WINDOW:
                raise AssertionError("training crop failed: batch exceeds TRAIN_WINDOW")
            C = crops[0][0].signal.shape[1]
            x = np.zeros((len(crops), max_len, C), dtype=np.float32)
            y = np.zeros((len(crops), max_len), dtype=np.float32)
            m = np.zeros((len(crops), max_len), dtype=np.float32)
            w = np.ones((len(crops), max_len), dtype=np.float32)
            y_sub = np.zeros((len(crops), max_len, 3), dtype=np.float32)
            y_sub_obs = np.zeros((len(crops), max_len, 3), dtype=bool)
            y_walk = np.zeros((len(crops), max_len), dtype=np.float32)
            y_walk_w = np.zeros((len(crops), max_len), dtype=np.float32)
            med = np.full((len(crops),), 0.5, dtype=np.float32)
            subj_ids, src_ids = [], []
            pos_tags = []  # v29ae (B3): per-crop sensor position metadata
            for bi2, (s, start, stop) in enumerate(crops):
                sl = slice(start, stop)
                L = stop - start
                _, _, _, _, yw_full, yww_full = _seq_features(s)
                x[bi2, :L] = s.signal[sl]
                y[bi2, :L] = (s.y_subtype[sl].max(axis=1) > 0.5).astype(np.float32)
                m[bi2, :L] = 1.0
                w[bi2, :L] = s.label_weight[sl]
                y_sub[bi2, :L] = s.y_subtype[sl]
                y_sub_obs[bi2, :L] = s.subtype_observed[sl]
                y_walk[bi2, :L] = yw_full[sl]
                y_walk_w[bi2, :L] = yww_full[sl]
                med[bi2] = s.med_state
                subj_ids.append(s.subject_id)
                src_ids.append(s.source_type)
                pos_tags.append(getattr(s, "position_tag", "") or "")
            if use_aug:
                rng_aug = np.random.RandomState(epoch * 10000 + bi)
                x, y, y_sub, w, y_sub_obs = augment_batch(
                    x, y, y_sub, w, y_sub_obs, rng=rng_aug,
                    partner_pool=partner_pool, sh_pool=sh_pool)
                if C >= 13 and _flag("NEUROMAMBA_AXIS_FLIP_AUG", "1") and crop_rng.random() < 0.5:
                    k = int(crop_rng.randint(0, 3))
                    x[:, :, k] *= -1.0
                    if C == 17:
                        x[:, :, 3 + k] *= -1.0
            yield x, y, m, w, y_sub, y_sub_obs, y_walk, y_walk_w, med, subj_ids, src_ids, pos_tags, None

    return _prefetched(_gen(), n=int(os.environ.get("NEUROMAMBA_PREFETCH_DEPTH", "2")))

def _tta_flip_enabled():
    """v25.44-novel (NEUROMAMBA_TTA_FLIP=1): physics-consistent TEST-TIME
    augmentation -- average the detector's fused probabilities over the
    MIRRORED sensor mount (full accelerometer-vector negation). Grounding:
    the model is TRAINED with random single-axis sign flips
    (NEUROMAMBA_AXIS_FLIP_AUG), and full-vector negation lies in the group
    generated by those flips, so the TTA view is in-distribution rather than
    a distribution shift (TTA aggregation ref: Shanmugam et al., ICCV 2021,
    arXiv:2011.11156). Channel contract mirrors the training aug exactly:
    magnitude-derived channels are invariant; the 17-channel jerk channels
    flip with their axes. Doubles inference cost when enabled; OFF by default
    so frozen-protocol numbers stay comparable."""
    return _flag("NEUROMAMBA_TTA_FLIP", "0")  # v25.57: _flag parse

def _axis_flip_batch_np(x_btc):
    """Mirror the sensor mount on a [B, T, C] batch: negate accel axes 0-2
    (and jerk channels 13-15 under the 17-channel contract)."""
    out = np.asarray(x_btc).copy()
    out[:, :, :3] *= -1.0
    if out.ndim == 3 and out.shape[-1] == 17:
        out[:, :, 13:16] *= -1.0
    return out.astype(np.float32)

def run_inference_cached(model, seqs, device, bs=64, collect_subtype=False, window_roll=False, collect_gate=False,
                         collect_hazard=False):
    """v25-FIX-7: window-roll inference (1st-place trick). When window_roll=True,
    predictions are averaged over overlapping windows -- reduces variance and
    improves AP, especially for rare subtypes. Default OFF for speed; enable
    with NEUROMAMBA_WINDOW_ROLL=1. v25.1-speed: collect_subtype / collect_gate
    let ONE pass produce fog + subtype + walking-gate outputs (was 2-3 separate
    full passes per epoch/fold).
    v28bp hazard arm: collect_hazard=True additionally returns the per-frame
    p_haz = sigmoid(aux["hazard_logit"]) stream (None when the model carries no
    hazard head).
    Returns (probs, labels, ids[, subtype_probs][, gate_probs][, hazard_probs])."""
    model.eval()
    use_med = bool(getattr(model, "use_med_film", False))
    use_prompt = bool(getattr(model, "sensor_film", None) is not None)
    all_probs, all_sub, all_labels, all_ids, all_gate, all_haz = [], [], [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i + bs]
            if window_roll:
                roll_w = int(os.environ.get("NEUROMAMBA_ROLL_WINDOW", "4000"))
                roll_s = int(os.environ.get("NEUROMAMBA_ROLL_STRIDE", "1000"))
                if roll_w <= 0 or roll_s <= 0 or roll_s > roll_w:
                    print(f"[roll v25.43] invalid geometry window={roll_w} stride={roll_s}"
                          f" -- falling back to 4000/1000", flush=True)
                    roll_w, roll_s = 4000, 1000
                probs_b, sub_b, gate_b, haz_b = _window_roll_inference_batch(
                    model, batch, device, collect_subtype, collect_gate=collect_gate,
                    collect_hazard=collect_hazard,
                    window=roll_w, stride=roll_s)
            else:
                max_len = max(len(s.labels) for s in batch)
                x = np.zeros((len(batch), max_len, batch[0].signal.shape[1]), dtype=np.float32)
                for j, s in enumerate(batch):
                    x[j, :len(s.labels)] = s.signal
                med_t = (torch.from_numpy(np.array([s.med_state for s in batch], dtype=np.float32)).to(device)
                         if use_med else None)
                x_t = torch.from_numpy(x).to(device)
                _lens_j = [len(s.labels) for s in batch]
                _prompt_t = (torch.from_numpy(_b3_prompt_matrix(
                    [s.source_type for s in batch],
                    [getattr(s, "position_tag", "") or "" for s in batch])).to(device)
                    if use_prompt else None)
                fog_logits, subtype_logits, aux = (model(x_t, med=med_t, lengths=_lens_j, prompt=_prompt_t)
                                                   if med_t is not None
                                                   else model(x_t, lengths=_lens_j, prompt=_prompt_t))
                probs_b = torch.sigmoid(fog_logits).cpu().numpy()
                fog_mode = os.environ.get("NEUROMAMBA_FOG_PROB_MODE", "head")
                if fog_mode != "head" and subtype_logits.shape[-1] == 4:
                    alt = (1.0 - torch.softmax(subtype_logits, dim=-1)[..., 3]).cpu().numpy()
                    probs_b = alt if fog_mode == "softmax" else np.maximum(probs_b, alt)
                _sub_flip = None
                if _tta_flip_enabled():
                    _xf = torch.from_numpy(_axis_flip_batch_np(x)).to(device)
                    _fl, _sl, _auxf = (model(_xf, med=med_t, lengths=_lens_j, prompt=_prompt_t)
                                       if med_t is not None else model(_xf, lengths=_lens_j, prompt=_prompt_t))
                    _pf = torch.sigmoid(_fl).cpu().numpy()
                    if fog_mode != "head" and subtype_logits.shape[-1] == 4:
                        _altf = (1.0 - torch.softmax(_sl, dim=-1)[..., 3]).cpu().numpy()
                        _pf = _altf if fog_mode == "softmax" else np.maximum(_pf, _altf)
                    probs_b = 0.5 * (probs_b + _pf)
                    _sub_flip = _sl
                if collect_subtype:
                    if subtype_logits.shape[-1] == 4:
                        sub_p = torch.softmax(subtype_logits, dim=-1)[..., :3]
                    else:
                        sub_p = torch.sigmoid(subtype_logits)
                    sub_b = sub_p.cpu().numpy()
                    if _sub_flip is not None:
                        _sf = (torch.softmax(_sub_flip, dim=-1)[..., :3].cpu().numpy()
                               if _sub_flip.shape[-1] == 4 else torch.sigmoid(_sub_flip).cpu().numpy())
                        sub_b = 0.5 * (sub_b + _sf)
                else:
                    sub_b = None
                gate_b = (aux["walking_prob"].cpu().numpy()
                          if (collect_gate and "walking_prob" in aux) else None)
                haz_b = (torch.sigmoid(aux["hazard_logit"]).cpu().numpy()
                         if (collect_hazard and aux.get("hazard_logit") is not None) else None)
            for j, s in enumerate(batch):
                L = len(s.labels)
                all_probs.append(probs_b[j, :L])
                if collect_subtype and sub_b is not None:
                    all_sub.append(sub_b[j, :L])
                if collect_gate and gate_b is not None:
                    all_gate.append(gate_b[j, :L])
                if collect_hazard and haz_b is not None:
                    all_haz.append(haz_b[j, :L])
                all_labels.append(s.labels.astype(int))
                all_ids.extend([s.record_id] * L)
    out = (np.concatenate(all_probs), np.concatenate(all_labels), np.array(all_ids, dtype=object))
    if collect_subtype:
        out += (np.concatenate(all_sub, axis=0) if all_sub else None,)
    if collect_gate:
        out += (np.concatenate(all_gate, axis=0) if all_gate else None,)
    if collect_hazard:
        out += (np.concatenate(all_haz, axis=0) if all_haz else None,)
    return out

def _window_roll_inference_batch(model, batch, device, collect_subtype, collect_gate=False,
                                 window=4000, stride=1000, collect_hazard=False):
    """v25-FIX-7: average predictions over overlapping windows (1st-place trick)."""
    n = len(batch)
    C = batch[0].signal.shape[1]
    max_len = max(len(s.labels) for s in batch)
    probs_acc = np.zeros((n, max_len), dtype=np.float32)
    count_acc = np.zeros((n, max_len), dtype=np.float32)
    sub_acc = np.zeros((n, max_len, 3), dtype=np.float32) if collect_subtype else None
    gate_acc = np.zeros((n, max_len), dtype=np.float32) if collect_gate else None
    haz_acc = np.zeros((n, max_len), dtype=np.float32) if collect_hazard else None
    use_med = bool(getattr(model, "use_med_film", False))
    med_t = (torch.from_numpy(np.array([s.med_state for s in batch], dtype=np.float32)).to(device)
             if use_med else None)
    _prompt_wr = (torch.from_numpy(_b3_prompt_matrix(
        [s.source_type for s in batch],
        [getattr(s, "position_tag", "") or "" for s in batch])).to(device)
        if getattr(model, "sensor_film", None) is not None else None)
    for start in range(0, max_len, stride):
        end = min(start + window, max_len)
        if end - start < 100:
            break
        x = np.zeros((n, end - start, C), dtype=np.float32)
        for j, s in enumerate(batch):
            L = min(end - start, len(s.labels) - start)
            if L > 0:
                x[j, :L] = s.signal[start:start + L]
        x_t = torch.from_numpy(x).to(device)
        _wl = [max(0, min(end - start, len(s.labels) - start)) for s in batch]
        fog_logits, subtype_logits, aux = (model(x_t, med=med_t, lengths=_wl, prompt=_prompt_wr)
                                           if med_t is not None else model(x_t, lengths=_wl, prompt=_prompt_wr))
        probs = torch.sigmoid(fog_logits).cpu().numpy()
        fog_mode = os.environ.get("NEUROMAMBA_FOG_PROB_MODE", "head")
        if fog_mode != "head" and subtype_logits.shape[-1] == 4:
            alt = (1.0 - torch.softmax(subtype_logits, dim=-1)[..., 3]).cpu().numpy()
            probs = alt if fog_mode == "softmax" else np.maximum(probs, alt)
        sub_p = None
        if collect_subtype and sub_acc is not None:
            sub_p = (torch.softmax(subtype_logits, dim=-1)[..., :3].cpu().numpy()
                     if subtype_logits.shape[-1] == 4 else torch.sigmoid(subtype_logits).cpu().numpy())
        if _tta_flip_enabled():
            _xfw = torch.from_numpy(_axis_flip_batch_np(x)).to(device)
            _fl2, _sl2, _auxf2 = (model(_xfw, med=med_t, prompt=_prompt_wr)
                                  if med_t is not None else model(_xfw, prompt=_prompt_wr))
            _pf2 = torch.sigmoid(_fl2).cpu().numpy()
            if fog_mode != "head" and subtype_logits.shape[-1] == 4:
                _al2 = (1.0 - torch.softmax(_sl2, dim=-1)[..., 3]).cpu().numpy()
                _pf2 = _al2 if fog_mode == "softmax" else np.maximum(_pf2, _al2)
            probs = 0.5 * (probs + _pf2)
            if sub_p is not None:
                _spf2 = (torch.softmax(_sl2, dim=-1)[..., :3].cpu().numpy()
                         if _sl2.shape[-1] == 4 else torch.sigmoid(_sl2).cpu().numpy())
                sub_p = 0.5 * (sub_p + _spf2)
        for j, s in enumerate(batch):
            L = min(end - start, len(s.labels) - start)
            if L > 0:
                probs_acc[j, start:start + L] += probs[j, :L]
                count_acc[j, start:start + L] += 1.0
                if collect_subtype and sub_acc is not None and sub_p is not None:
                    sub_acc[j, start:start + L] += sub_p[j, :L]
                if collect_gate and gate_acc is not None and "walking_prob" in aux:
                    gp = aux["walking_prob"].cpu().numpy()
                    gate_acc[j, start:start + L] += gp[j, :L]
                if collect_hazard and haz_acc is not None and aux.get("hazard_logit") is not None:
                    hp = torch.sigmoid(aux["hazard_logit"]).cpu().numpy()
                    haz_acc[j, start:start + L] += hp[j, :L]
    count_acc = np.maximum(count_acc, 1.0)
    probs_b = probs_acc / count_acc
    sub_b = (sub_acc / count_acc[:, :, None]) if collect_subtype else None
    gate_b = (gate_acc / count_acc) if collect_gate else None
    haz_b = (haz_acc / count_acc) if collect_hazard else None
    return probs_b, sub_b, gate_b, haz_b

def concat_labels(seqs):
    """v25: returns (y_subtype, observed, subtype_observed)."""
    return (np.concatenate([s.y_subtype for s in seqs], axis=0),
            np.concatenate([s.train_mask.astype(bool) for s in seqs], axis=0),
            np.concatenate([s.subtype_observed for s in seqs], axis=0))

def subtype_map_from_probs(sub_probs, seqs, min_sh_pos=None):
    """v25.1: compute the Kaggle-style per-subtype mAP from already-computed
    probabilities (no second model pass) -- the per-epoch eval used to re-run
    full calib inference just for the subtype head.
    v25.20-B (winner's defog rule): when the selection split holds fewer than
    min_sh_pos observed StartHesitation frames (env NEUROMAMBA_SELECT_MIN_SH_POS,
    default 20), SH AP is dominated by a handful of frames and drags the
    selection metric with noise. The 1st-place training code scored defog
    validation as mAP = mean(Turn, Walking) ONLY (StartHesitation excluded --
    verified: `mAP = (Walking_mAP+Turn_mAP)/2` in the winner's defog notebook).
    We mirror that rule here for model selection/early stopping; the unbiased
    fold [VAL] mAP still reports all three classes."""
    if min_sh_pos is None:
        min_sh_pos = int(os.environ.get("NEUROMAMBA_SELECT_MIN_SH_POS", "20"))
    y_sub, obs, sub_obs = concat_labels(seqs)
    n = min(len(y_sub), len(sub_probs))
    classes = (0, 1, 2)
    n_sh = int((sub_obs[:n, 0] & (y_sub[:n, 0] > 0.5)).sum())
    _sh_subj, _sh_events = set(), 0
    _seen = 0
    for _s in seqs:
        _mk = (_s.subtype_observed[:, 0] & (_s.y_subtype[:, 0] > 0.5)).astype(np.int64)
        _seen += int(_mk.sum())
        if bool(_mk.any()):
            _sh_subj.add(_s.subject_id)
            _sh_events += int(np.sum(np.diff(np.concatenate([[0], _mk, [0]])) == 1))
        if _seen >= n:
            break
    _sh_underpowered = (len(_sh_subj) < 4 or _sh_events < 8)
    if n_sh < min_sh_pos:
        classes = (1, 2)
        print(f"[selection v25.20] StartHesitation EXCLUDED from the selection "
              f"mAP ({n_sh} observed SH positives < {min_sh_pos}) -- the 1st-place "
              f"defog rule: selection mAP = mean(Turn, Walking) only", flush=True)
    elif _sh_underpowered and n_sh > 0:
        classes = (1, 2)
        if not globals().get("_nm25_shpow_logged", False):
            globals()["_nm25_shpow_logged"] = True
            print(f"[selection v25.59c] StartHesitation EXCLUDED from the selection mAP "
                  f"-- UNDER POWERED ({len(_sh_subj)} SH subjects / {_sh_events} events; "
                  f"{n_sh} frames) -- AP over one subject/event is noise; mAP = "
                  f"mean(Turn, Walking) only [1st-place defog rule, power-aware; "
                  f"per-epoch, printed once]", flush=True)
    return compute_official_like_map(y_sub[:n], sub_probs[:n], sub_obs[:n], classes=classes)

def calib_subtype_map(model, calib_seqs, device, window_roll=False, extra_seqs=None):
    """v25.1: optional extra_seqs (e.g. the threshold split) so the model-
    selection metric sees more Start-Hesitation positives and is far less noisy
    (defog calib alone can hold only ~60 SH frames)."""
    sel_seqs = list(calib_seqs) + (list(extra_seqs) if extra_seqs is not None else [])
    res = run_inference_cached(model, sel_seqs, device, collect_subtype=True, window_roll=window_roll)
    if len(res) < 4 or res[3] is None:
        return 0.0
    return float(subtype_map_from_probs(res[3], sel_seqs)["mAP"])

def calib_window_f1(model, calib_seqs, device, window_roll=False, extra_seqs=None, fit_trigger=False):
    """v25.11c: Lin & Grewal 2026 (Sensors 26:1352) -- F1-based selection and
    early stopping beat AUC/AP-based selection for the DEPLOYMENT metric
    (+47% F1: 0.55 vs 0.37, p=0.0008). Scores the walking-gated fog probs on
    calib(+thresh) with the contest-comparable 1-s window F1. Opt-in via
    NEUROMAMBA_SELECT_BY_WINDOW_F1=1 (the Kaggle mAP selection stays default).
    v25.35: fit_trigger=True scores windowF1 at the calib-FITTED clinical
    trigger (the deployed operating point) instead of the fixed 0.5. The old
    fixed-0.5 scoring let a variant 'win' calib with F1@0.5 while deploying at
    trigger ~0.2-0.4 -- v25.34 reproduced: selected variant calibWindowF1=
    0.7483 but the deployed fold [VAL] windowF1=0.436 and frozen daphnet
    windowF1=0.150. Selection must match deployment. Returns (window_f1,
    trigger) when fit_trigger=True, else a bare float (backward compat)."""
    sel_seqs = list(calib_seqs) + (list(extra_seqs) if extra_seqs is not None else [])
    res = run_inference_cached(model, sel_seqs, device, collect_gate=True, window_roll=window_roll)
    if len(res) < 1 or res[0] is None:
        return (0.0, None) if fit_trigger else 0.0
    probs, labels, ids = res[0], res[1], res[2]
    if len(res) >= 5 and res[4] is not None:
        probs = _apply_walking_gate_inference(model, sel_seqs, probs, device,
                                              window_roll=window_roll, gate_probs=res[4])
    if fit_trigger:
        _grid = parse_selection_f1_grid()
        if _grid:
            _gs = selection_window_f1_grid(labels, probs, ids, grid=_grid)
            if _gs is not None:
                print(f"[variant-grid v25.43] grid={_grid} meanWF1={_gs[0]:.4f} "
                      f"per-thr={[round(x, 3) for x in _gs[1]]}", flush=True)
                return float(_gs[0]), None
        n_calib = int(sum(len(s.labels) for s in calib_seqs))
        trig = float(find_clinical_threshold_fast(labels[:n_calib], probs[:n_calib],
                                                  ids[:n_calib])["trigger"])
        return float(window_level_f1(labels, probs, ids, threshold=trig)["window_f1"]), trig
    return float(window_level_f1(labels, probs, ids)["window_f1"])

def parse_selection_f1_grid():
    """v25.36: threshold-robust F1-based selection. The per-epoch windowF1 was
    scored at the epoch's re-fitted clinical trigger, which flip-flops between
    the max-epF1 lobe (~0.005) and the min-FARh lobe (~0.7-0.9) whenever the
    tiny calib score distribution shifts -- the v25.35 log shows selection
    windowF1 0.000 <-> 0.722 on consecutive epochs while the losses move
    smoothly (selection roulette, not model roulette). Scoring the selection
    signal at a FIXED grid of thresholds spanning the deployed operating band
    removes the lobe-flipping artifact. Env NEUROMAMBA_SEL_F1_GRID (comma
    separated floats, default '0.05,0.1,0.2,0.3'); empty string disables the
    grid and restores the per-epoch-trigger behavior."""
    raw = os.environ.get("NEUROMAMBA_SEL_F1_GRID", "0.05,0.1,0.2,0.3")
    raw = raw.strip()
    if not raw:
        return None
    try:
        vals = [float(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        return None
    return vals if vals else None

def selection_window_f1_grid(labels, probs, ids, grid=None):
    """v25.36: mean contest-comparable window F1 over a fixed trigger grid
    (threshold-robust selection score). Returns (mean_f1, per_threshold_list)
    or None when the grid is disabled."""
    if grid is None:
        grid = parse_selection_f1_grid()
    if not grid:
        return None
    per_thr = [window_level_f1(labels, probs, ids, threshold=float(t))["window_f1"]
               for t in grid]
    return float(np.mean(per_thr)), per_thr

def deploy_variant_trigger(postproc, best_trig, select_by_wf):
    """v25.36: deployment consistency. The selected variant was chosen at the
    calib-fitted trigger (fit_trigger=True), so the DEPLOYED trigger must be
    that trigger -- v25.35 reproduced the v25.34 bug one level deeper: variant
    selected ema @thr=0.436, deployed inner-CV median 0.220. Returns an updated
    postproc dict (trigger overwritten, trigger_source annotated)."""
    postproc = dict(postproc)
    if select_by_wf and best_trig is not None:
        postproc["trigger"] = float(best_trig)
        postproc["trigger_source"] = "variant_fit_trigger"
    return postproc

def gate_flip_fraction(raw_probs, gated_probs, thr):
    """v25.36: fraction of frames whose binary decision changes due to the
    walking gate at `thr` (gate diagnostic -- 'is the gate doing anything?')."""
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    gated_probs = np.asarray(gated_probs, dtype=np.float64)
    if len(raw_probs) == 0:
        return 0.0
    return float(np.mean((raw_probs >= thr) != (gated_probs >= thr)))

def update_ema(ema, model, decay):
    sd = model.state_dict()
    if ema is None:
        return {k: v.detach().clone() for k, v in sd.items()}
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            ema[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
        else:
            ema[k] = v.detach().clone()
    return ema

def early_stop_should_stop(history, patience, min_epoch):
    if len(history) < min_epoch:
        return False, None
    scores = [h if h is not None and np.isfinite(h) else float("-inf") for h in history]
    best_idx = int(np.argmax(scores))
    if len(scores) - 1 - best_idx > patience:
        return True, best_idx
    return False, None

def smoothed_selection_score(map_history, window=3):
    if not map_history:
        return 0.0
    tail = [h for h in map_history[-window:] if h is not None and np.isfinite(h)]
    return float(np.mean(tail)) if tail else 0.0

print("[train] TRAIN_V25_LOADED", flush=True)

EARLY_EPOCHS_LIGHT_EVAL = int(os.environ.get("NEUROMAMBA_LIGHT_EVAL_EPOCHS", 5))
LIGHT_EVAL_EPF1_FLOOR = float(os.environ.get("NEUROMAMBA_LIGHT_EVAL_F1_FLOOR", 0.05))
EMA_DECAY = float(os.environ.get("NEUROMAMBA_EMA_DECAY", 0.999))
SWA_START = int(os.environ.get("NEUROMAMBA_SWA_START", 25))
EARLY_STOP_PATIENCE = int(os.environ.get("NEUROMAMBA_EARLY_STOP_PATIENCE", 25))
EARLY_STOP_MIN_EPOCH = int(os.environ.get("NEUROMAMBA_EARLY_STOP_MIN_EPOCH", 20))

def get(name, default):
    return os.environ.get(name, default)

def _flag(name, default="0"):
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return str(raw).strip().lower() in ("1", "true", "t", "yes", "y", "on")

TRAINING_ENV_DEFAULTS = {
    "NEUROMAMBA_SELECT_BY_WINDOW_F1": "1",
    "NEUROMAMBA_MAP_INCLUDE_THRESH": "0",
    "NEUROMAMBA_WF1_INCLUDE_THRESH": "0",
    "NEUROMAMBA_DOMAIN_DANN": "0.0",
    "NEUROMAMBA_SUBTYPE_FINETUNE": "0",  # v25.46b review: in-set restore hurt THRESH; OFF until nested CV
    "NEUROMAMBA_EP_LOSS_W": "0",
    "NEUROMAMBA_EP_LOSS_START": "3",
    "NEUROMAMBA_PCGRAD": "0",
    "NEUROMAMBA_GATE_SH_COND": "0",
    "NEUROMAMBA_GATE_SH_ETA": "0.2",
    "NEUROMAMBA_GATE_MASK_SH": "0",
    "NEUROMAMBA_DEPLOY_CONFORMAL": "0",
    "NEUROMAMBA_DEPLOY_CONFORMAL_BUDGET": "10",
    "NEUROMAMBA_SCORE_ANCHOR_W": "0",
    "NEUROMAMBA_SCORE_ANCHOR_START": "3",
    "NEUROMAMBA_SCORE_ANCHOR_TARGETS": "0.05,0.80",
    "NEUROMAMBA_HEAD_PRIOR_INIT": "0",
    "NEUROMAMBA_TURN_PLAIN": "0",
    "NEUROMAMBA_EPOCH_T_FIXED": "0",
    "NEUROMAMBA_SUBTYPE_TEMPERATURE": "1",  # v25.51.2: 0 disables per-subtype temp fit (eval-affecting)
    "NEUROMAMBA_MASK_DEFOG_SH": "1",
    "NEUROMAMBA_DEFOG_SH_WEAK": "0.05",
    "NEUROMAMBA_DEFOG_SH_SCALE_MODE": "term",
    "NEUROMAMBA_OPT_2NDPLACE": "0",
    "NEUROMAMBA_OPT_WD": "",
    "NEUROMAMBA_OPT_HEAD_WD0": "1",
    "NEUROMAMBA_GRAD_CLIP_MODE": "adaptive",
    "NEUROMAMBA_GRAD_CLIP": "1.0",
    "NEUROMAMBA_MIN_EPOCH_MAP_SEL": "10",
    "NEUROMAMBA_FOG_TOLERANCE": "0.85",
    "NEUROMAMBA_SELECT_MAP_BLEND": "0.0",
    "NEUROMAMBA_CONFORMAL_SSBC": "0",
    "NEUROMAMBA_MIN_CALIB_SUBJ_POSTPROC": "15",
    "NEUROMAMBA_EVENT_BUDGET_CONFORMAL": "0",
    "NEUROMAMBA_BOOTSTRAP_N": "200",
    "NEUROMAMBA_SUBTYPE_FT_MONITOR": "1",
    "NEUROMAMBA_SUBTYPE_FT_RESTORE_BEST": "0",
    "NEUROMAMBA_PARETO_SELECT": "0",
    "NEUROMAMBA_PARETO_ALPHA": "0.95",
    "NEUROMAMBA_PARETO_MIN_SH_AP": "0.05",
    "NEUROMAMBA_EPOCH_RAW_TRIGGER": "1",
    "NEUROMAMBA_SEL_F1_GRID": "0.05,0.1,0.2,0.3",
    "NEUROMAMBA_CALIB_FRAC": "0.15",
    "NEUROMAMBA_THRESH_FRAC": "0.15",
    "NEUROMAMBA_WINNER_TOKEN_LOSS": "0",
    "NEUROMAMBA_NOTYPE_PSEUDO": "0",
    "NEUROMAMBA_HARD_NEG_CROP_PROB": "0.15",
    "NEUROMAMBA_ADAM_BETA2": "0.999",
    "NEUROMAMBA_MLLS_CORRECT": "0",
    "NEUROMAMBA_SCORE_ANCHOR": "0",
    "NEUROMAMBA_GRAD_GUARD": "1",
    "NEUROMAMBA_GRAD_GUARD_CEILING": "1000",
    "NEUROMAMBA_GRAD_GUARD_DYN_CAP": "",
    "NEUROMAMBA_GRAD_GUARD_WARMUP_EPOCHS": "10",
    "NEUROMAMBA_GRAD_GUARD_MAX_ABORT_FRAC": "0.05",
    "NEUROMAMBA_GRAD_GUARD_BAD_EPOCH_PATIENCE": "2",
    "NEUROMAMBA_GRAD_GUARD_FROZEN_FLOOR": "300",
    "NEUROMAMBA_SUBTYPE_COND_LOSS": "0",
    "NEUROMAMBA_SUBJECT_FIRST_SAMPLER": "0",
    "NEUROMAMBA_SELECT_UTILITY": "0",
    "NEUROMAMBA_MIN_EPOCH_UTILITY": "2",
    "NEUROMAMBA_SUB_W_MAX": "1.0",
    "NEUROMAMBA_SAMPLER_QUOTA": "",
    "NEUROMAMBA_NOTYPE_FOG_W": "1",
    "NEUROMAMBA_WARMUP_EPOCHS": "",
    "NEUROMAMBA_SUB_W_RAMP_EPOCHS": "",
    "NEUROMAMBA_EMA_EPOCH_EVAL": "1",
    "NEUROMAMBA_SH_CONTRASTIVE": "1",
    "NEUROMAMBA_SH_CONTRASTIVE_W": "0.15",
    "NEUROMAMBA_SH_CONTRASTIVE_NORM": "1",
    "NEUROMAMBA_CONTRAST_W_EFF_CAP": "5.0",     # v25.57 new lever (storm fix)
    "NEUROMAMBA_TOKEN_POS_WEIGHT_CAP": "5",
    "NEUROMAMBA_TOKEN_POS_WEIGHT_MODE": "ratio",
    "NEUROMAMBA_TOKEN_PW_SOURCE": "patch",      # v25.57 new lever ('frame' = legacy)
    "NEUROMAMBA_TOKEN_ASL": "0",                # v25.59: token-level ASL (Round 28z) -- hashed for identity
    "NEUROMAMBA_RARE_BOOST": "0",               # v25.59: bounded rare-class stall controller -- hashed
    "NEUROMAMBA_WARM_REPLAY_REPEAT": "1",       # v25.59c: SH rehearsal multiplier (Round 28aa) -- hashed
    "NEUROMAMBA_WALK_HARDNEG_W": "1.0",         # v25.59d: Walk context hard-negative weight (Round 28ac) -- hashedd
    "NEUROMAMBA_SH_HARDNEG_W": "1.0",           # v25.59e: SH context hard-negative weight (audit-gated) -- h
    "NEUROMAMBA_SMALL_CALIB_EP_F1_DEPLOY": "1",  # v25.59f: underpowered-calib deploy rule (Round 28ai) -- hashed
"NEUROMAMBA_PRIOR_CLAMP_DELTA": "0.15",     # v25.59g: MLLS prior clamp band (Round 28ak) -- hashed
"NEUROMAMBA_DEPLOY_FARH_SAFETY": "0.75",      # v25.59g: calib->val FARh inflation safety factor -- hashed
    "NEUROMAMBA_SUB_CE_WEIGHTS": "5.0,2.0,5.0,1.0",
    "NEUROMAMBA_USE_AUGMENTATION": "1",
    "NEUROMAMBA_CROP_POS_PROB": "0.2",
    "NEUROMAMBA_SH_OVERSAMPLE": "5.0",
    "NEUROMAMBA_SH_CROP_PROB": "0.40",
    "NEUROMAMBA_WALK_CROP_PROB": "0.25",
    "NEUROMAMBA_EXTERNAL_PRIOR_MATCH": "0",   # v25.58: deploy label-free prior-matched external rows
    "NEUROMAMBA_SH_SUBJECT_CAP": "4.0",
    "NEUROMAMBA_SUB_W_FLOOR": "0.35",
    "NEUROMAMBA_SUB_POS_WEIGHT": "1",
    "NEUROMAMBA_SUB_POS_WEIGHT_CAP": "20",
    "NEUROMAMBA_SUB_PW_SOURCE": "dataset",       # v25.57 new lever ('batch' = legacy)
    "NEUROMAMBA_POLY1_EPS": "1.0",
    "NEUROMAMBA_WARM_REPLAY": "1",
    "NEUROMAMBA_WARM_REPLAY_MAX": "150",
    "NEUROMAMBA_FINETUNE_LR_RESTART": "1",
    "NEUROMAMBA_FINETUNE_LR_SCALE": "0.5",
    "NEUROMAMBA_WARM_EXCLUDE_VAL": "0",          # v43: participant-nested arm -- remove this repeat's validation
    "NEUROMAMBA_AXIS_FLIP_AUG": "1",
    "NEUROMAMBA_PAD_MASK": "0",                  # v25.58: default 0 = TRAIN-PARITY (training never passes
    "NEUROMAMBA_SAMPLER_EXPOSURE": "order",      # v25.57 new lever ('weighted' = true replacement)
}
EVALUATION_REV = "v25.59"  # v25.59 (Round 28p): xuanwu record ids qualified with the

def evaluation_identity(extra_env=None):
    """SHA-256 over everything that changes EVALUATION OUTPUT (metrics, gates,
    thresholds, calibration, bootstrap, post-processing) but NOT the trained
    model. Fold reuse is allowed iff BOTH training_identity and
    evaluation_identity match; if only this differs, the fold is REEVALUATED
    from frozen OOF predictions."""
    import hashlib
    env = {}
    for k in ("NEUROMAMBA_MAP_INCLUDE_THRESH", "NEUROMAMBA_WF1_INCLUDE_THRESH",
              "NEUROMAMBA_FOG_TOLERANCE", "NEUROMAMBA_CONFORMAL_SSBC",
              "NEUROMAMBA_SUBTYPE_TEMPERATURE",  # v25.51.2: eval-affecting gate
              "NEUROMAMBA_MIN_CALIB_SUBJ_POSTPROC", "NEUROMAMBA_EVENT_BUDGET_CONFORMAL",
              "NEUROMAMBA_BOOTSTRAP_N", "NEUROMAMBA_SCORE_ANCHOR",
              "NEUROMAMBA_MLLS_CORRECT", "NEUROMAMBA_EXTERNAL_PRIOR_MATCH",
              "NEUROMAMBA_PER_SEED_VAL",
              "NEUROMAMBA_ABSTAIN_G_MAX", "NEUROMAMBA_ABSTAIN_G_MIN",
              "NEUROMAMBA_GATE_MODE", "NEUROMAMBA_GATE_FLOOR", "NEUROMAMBA_GATE_BINARY_TH",
              "NEUROMAMBA_FOG_PROB_MODE", "NEUROMAMBA_WINDOW_ROLL",
              "NEUROMAMBA_ACC_SCALE",  # v25.45 audit gap: input-scale env was unhashed
              "NEUROMAMBA_ROLL_WINDOW", "NEUROMAMBA_ROLL_STRIDE",
              "NEUROMAMBA_SEL_F1_GRID", "NEUROMAMBA_CONFORMAL_ALPHA",
              "NEUROMAMBA_FALLBACK_Q_MAX", "NEUROMAMBA_EXTERNAL_MIN_EVENT_F1",
              "NEUROMAMBA_EXTERNAL_MAX_FARH", "NEUROMAMBA_TRIGGER_SMOOTH_WINDOW",
              "NEUROMAMBA_MERGE_GAP_N", "NEUROMAMBA_ABSTAIN_QUALITY",
              "NEUROMAMBA_TTA_FLIP", "NEUROMAMBA_JOINT_CERTIFY",
              "NEUROMAMBA_LOGIT_ADJUST_TAU",
              "NEUROMAMBA_GATE_SH_COND", "NEUROMAMBA_GATE_SH_ETA",
              "NEUROMAMBA_PAD_MASK"):
        env[k] = os.environ.get(k, "")
    if extra_env:
        for k, v in extra_env.items():
            env[k] = str(v)
    payload = {"evaluation_rev": EVALUATION_REV, "env": {k: env[k] for k in sorted(env)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def resume_decision(saved_training_identity, saved_evaluation_identity,
                    current_training_identity, current_evaluation_identity,
                    has_checkpoint=True, baseline_available=None):
    """v25.42.6: fail-closed, stage-aware resume decision.

    Modes:
      exact_resume  -- training AND evaluation identity match: reuse the
                       completed fold result (status=REUSED_RESULT).
      evaluate_only -- training unchanged, evaluation code/config changed:
                       reuse frozen weights/OOF arrays, recompute ALL affected
                       evaluation outputs (status=REEVALUATED_FROZEN_MODEL).
      fresh_training-- training identity differs: do NOT reuse anything that
                       depends on the trained model (status=FRESH).

    Default (any uncertainty) is NOT to reuse training or metrics."""
    if not has_checkpoint:
        return "no_checkpoint", "no saved fold/checkpoint"
    if saved_training_identity is None or current_training_identity is None:
        return "fresh_training", "missing training identity (legacy or unset) -- fail closed"
    if saved_training_identity != current_training_identity:
        return "fresh_training", \
            f"training identity mismatch ({saved_training_identity[:12]} vs {current_training_identity[:12]})"
    if saved_evaluation_identity is None or saved_evaluation_identity != current_evaluation_identity:
        return "evaluate_only", \
            "training identity matches; evaluation identity differs -- recompute metrics from frozen predictions"
    if baseline_available is False:
        return "evaluate_only", "baseline comparator artifact missing -- recompute evaluation"
    return "exact_resume", "training + evaluation identity match"

TRAINING_DIAGNOSTIC_ENV = frozenset({
    "NEUROMAMBA_EVAL_TRAJECTORY_EPOCHS",
    "NEUROMAMBA_BOOTSTRAP_N",
    "NEUROMAMBA_PER_SEED_VAL",
    "NEUROMAMBA_SCORE_ANCHOR",
    "NEUROMAMBA_SUBTYPE_FT_MONITOR",
    "NEUROMAMBA_MLLS_CORRECT",
    "NEUROMAMBA_EVENT_BUDGET_CONFORMAL",
})

TRAINING_ARCH_ENV = {
    "NEUROMAMBA_GRU_HIDDEN": "128", "NEUROMAMBA_GRU_LAYERS": "2",
    "NEUROMAMBA_GRU_DROPOUT": "0.2",
    "NEUROMAMBA_DMODEL": "128", "NEUROMAMBA_NBLOCKS": "2",
    "NEUROMAMBA_WINNER_D": "128", "NEUROMAMBA_WINNER_HEADS": "6",
    "NEUROMAMBA_WINNER_DEPTH": "5", "NEUROMAMBA_WINNER_PATCH": "14",
    "NEUROMAMBA_WINNER_DROPOUT": "0.1",
    "NEUROMAMBA_LORA_RANK": "0",
    "NEUROMAMBA_LORA_ALPHA": "1.0",
    "NEUROMAMBA_LORA_DROPOUT": "0.1",
    "NEUROMAMBA_LORA_FREEZE_BACKBONE": "0",
    "NEUROMAMBA_DISABLE_WALKING_GATE": "0",
    "NEUROMAMBA_TRAIN_WINDOW": "",
    "NEUROMAMBA_ROLL_WINDOW": "4000", "NEUROMAMBA_ROLL_STRIDE": "1000",
    "NEUROMAMBA_EMA_DECAY": "0.999",
    "NEUROMAMBA_EARLY_STOP_PATIENCE": "25", "NEUROMAMBA_EARLY_STOP_MIN_EPOCH": "20",
    "NEUROMAMBA_SWA_START": "25",
    "NEUROMAMBA_FOG_WARMUP_EPOCHS": "2",
    "NEUROMAMBA_RAMP_EPOCHS": "10",
    "NEUROMAMBA_WALKING_GATE_LOSS_WEIGHT": "0.3",
    "NEUROMAMBA_SEED_DIVERSITY": "0",
}

def resolved_training_env():
    """Resolved NEUROMAMBA_* config (explicit env wins, else the default)."""
    env = {k: os.environ.get(k, d) for k, d in TRAINING_ENV_DEFAULTS.items()}
    for k, d in TRAINING_ARCH_ENV.items():
        v = os.environ.get(k, d)
        if v:
            env[k] = v
    try:
        _wd = int(float(env.get("NEUROMAMBA_WINNER_D", "128") or 128))
        _wh = int(float(env.get("NEUROMAMBA_WINNER_HEADS", "6") or 6))
        if _wd % _wh != 0:
            _cands = [h for h in range(_wh, 0, -1) if _wd % h == 0]
            _wh_eff = _cands[0] if _cands else 1
        else:
            _wh_eff = _wh
        env["_effective_winner_heads"] = str(_wh_eff)
    except Exception:
        env["_effective_winner_heads"] = "unknown"
    return env

def training_identity(model_type="sequence", lr=3e-4, batch_size=8, epochs=80,
                      seed=42, n_seeds=1, fold_idx=0, family="tdcsfog",
                      use_rare_split=True, warm_family=None, warm_epochs=0,
                      extra_env=None):
    """v25.42.5: SHA-256 identity of EVERYTHING that changes the trained model.
    Checkpoint reuse is allowed iff the identity matches -- unlike CODE_VERSION
    (which bumps on any edit, even diagnostic-only), this survives
    diagnostic/reporting changes but refuses reuse across any training-affecting
    config change. Saved in per-epoch checkpoints and fold records."""
    import hashlib
    env = resolved_training_env()
    for k in TRAINING_DIAGNOSTIC_ENV:
        env.pop(k, None)
    for _k in ("NEUROMAMBA_STORM_LR_RESCUE", "NEUROMAMBA_STORM_RESCUE_MAX",
               "FOGBENCH_TLS", "FOGBENCH_TLS_W", "FOGBENCH_TLS_PARAMS",
               "NEUROMAMBA_NP_CONSTRAINT", "NEUROMAMBA_NP_ALPHA", "NEUROMAMBA_NP_START",
               "NEUROMAMBA_LODO", "NEUROMAMBA_LODO_COHORT_CAP",
               "NEUROMAMBA_LODO_EXCLUDE_FROZEN_VAL",
                "NEUROMAMBA_B4", "NEUROMAMBA_B5",
                "NEUROMAMBA_B3", "NEUROMAMBA_B6"):
        _v = os.environ.get(_k)
        if _v not in (None, "", "0"):
            env[_k] = str(_v)
        if _k == "NEUROMAMBA_B4" and str(_v or "").strip().lower() not in ("", "0"):
            env["NEUROMAMBA_B4_POSWAP_P"] = str(os.environ.get("NEUROMAMBA_B4_POSWAP_P", 0.30))
            env["NEUROMAMBA_B4_RATEJITTER_P"] = str(os.environ.get("NEUROMAMBA_B4_RATEJITTER_P", 0.30))
            env["NEUROMAMBA_B4_STRETCH_LO"] = str(os.environ.get("NEUROMAMBA_B4_STRETCH_LO", 0.8))
            env["NEUROMAMBA_B4_STRETCH_HI"] = str(os.environ.get("NEUROMAMBA_B4_STRETCH_HI", 1.25))
        if _k == "NEUROMAMBA_B5" and str(_v or "").strip().lower() not in ("", "0"):
            env["NEUROMAMBA_B5_ALPHA"] = str(os.environ.get("NEUROMAMBA_B5_ALPHA", 2.0))
            env["NEUROMAMBA_B5_BETA"] = str(os.environ.get("NEUROMAMBA_B5_BETA", 2.0))
        if _k == "NEUROMAMBA_B3" and str(_v or "").strip().lower() not in ("", "0"):
            env["NEUROMAMBA_B3_RATE_NORM"] = str(os.environ.get("NEUROMAMBA_B3_RATE_NORM", "log"))
            env["NEUROMAMBA_B3_PROMPT_OVERRIDE"] = str(os.environ.get("NEUROMAMBA_B3_PROMPT_OVERRIDE", ""))
        if _k == "NEUROMAMBA_B6" and str(_v or "").strip().lower() not in ("", "0"):
            env["NEUROMAMBA_B6_DANN_W"] = str(os.environ.get("NEUROMAMBA_B6_DANN_W", 0.05))
    if extra_env:
        for k, v in extra_env.items():
            env[k] = str(v)
    payload = {
        "model_type": model_type, "lr": float(lr), "batch_size": int(batch_size),
        "epochs": int(epochs), "seed": int(seed), "n_seeds": int(n_seeds),
        "fold_idx": int(fold_idx), "family": family,
        "use_rare_split": bool(use_rare_split),
        "warm_family": warm_family, "warm_epochs": int(warm_epochs or 0),
        "env": {k: env[k] for k in sorted(env)},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def init_subtype_head_biases_from_priors(model, subtype_priors):
    """v25.51.2 (code-review fix): initialize subtype-head biases from class
    priors (logit(pi)). Handles BOTH head contracts:
      subtype_heads -- ModuleList of Linear(.)/Conv1d(. ,1) heads: one bias
                       scalar per head -> fill_.
      subtype_head  -- the winner's Sequential(..., Conv1d(n_sub, 1)): ONE
                       shared bias vector of shape [n_sub] -> set per index k.
    Returns a message string describing what was set; the empty string means
    "nothing init-able" so the caller can print a loud WARNING (the silent
    hasattr no-op was the v25.51.0 bug)."""
    heads = list(getattr(model, "subtype_heads", []) or [])
    mode = "per_head"
    if not heads:
        seq = getattr(model, "subtype_head", None)
        last = seq[-1] if isinstance(seq, torch.nn.Sequential) and len(seq) > 0 else None
        if isinstance(last, torch.nn.Conv1d) and getattr(last, "bias", None) is not None \
                and int(last.bias.numel()) >= 3:
            heads, mode = [last], "shared_conv1d"
    if not heads:
        return ""
    pis = [float(np.clip(p, 1e-4, 1.0 - 1e-4)) for p in list(subtype_priors)[:3]]
    logits = [float(np.log(pi / (1.0 - pi))) for pi in pis]
    with torch.no_grad():
        if mode == "shared_conv1d":
            for k in range(3):
                heads[0].bias.data[k] = logits[k]
        else:
            for k, h in enumerate(heads[:3]):
                b = getattr(h, "bias", None)
                if b is None or int(b.numel()) != 1:
                    return ""
                b.data.fill_(logits[k])
    suffix = " (winner shared Conv1d bias)" if mode == "shared_conv1d" else ""
    return "[" + ", ".join(f"{v:.2f}" for v in logits) + "]" + suffix

def resolve_branch_config():
    if _flag("NEUROMAMBA_FORCE_PLAIN"):
        print("[branch-guard v25] NEUROMAMBA_FORCE_PLAIN=1 -> all optional branches OFF", flush=True)
        return {"use_spectral": False, "use_attention": False}
    cfg = {
        "use_spectral": _flag("NEUROMAMBA_USE_SPECTRAL"),
        "use_attention": _flag("NEUROMAMBA_USE_ATTENTION"),
    }
    active = [k.replace("use_", "") for k, v in cfg.items() if v]
    if active:
        print(f"[branch-guard v25] resolved ACTIVE branches: {active} (from NEUROMAMBA_USE_* env vars).", flush=True)
    else:
        print("[branch-guard v25] resolved plain backbone (no optional branches active).", flush=True)
    return cfg

def fine_tune_subtype_head(model, seqs, device, batch_size=8, epochs=4, lr=1e-4, seed=42):
    """v25.30: post-selection subtype-head fine-tune. Evidence from the v25.29
    log: calib+thresh subtype APs peak mid-training (epoch 5: SH=0.660
    Turn=0.945 Walk=0.698) and collapse by the fog-windowF1-selected
    checkpoint (epoch 40: SH=0.083 Turn=0.267 Walk=0.065) -- selection
    optimizes the binary fog head only, and the shared-backbone subtype head
    is never chosen or polished. This pass freezes the backbone + fog head and
    trains ONLY subtype_head params with the masked per-class ASL (softmax
    mode gets cohort-ratio CE weights computed from `seqs`), so deployed
    subtype probabilities recover without moving the fog operating point.
    `seqs` must be the CALIB split only: the thresh split stays clean as the
    unbiased before/after check. Starts from the already-loaded selected
    variant (raw/ema/swa -- the weights that get deployed). Skipped when the
    splits carry zero observed subtype positives (binary-only cohorts)."""
    n_sub_pos = int(sum(((s.y_subtype > 0.5) & s.subtype_observed).sum() for s in seqs))
    if n_sub_pos == 0:
        print("[subtype-finetune v25.30] no observed subtype positives in the selection "
              "splits -- skipping (binary-only cohort)", flush=True)
        return None
    if _flag("NEUROMAMBA_WINNER_TOKEN_LOSS", "0"):
        print("[subtype-finetune v25.30] token-loss mode active -- skipping (the trained "
              "head is tok_sub, full-res subtype_head receives no gradient)", flush=True)
        return None
    if not any(n.startswith("subtype_head") for n, _ in model.named_parameters()):
        print("[subtype-finetune v25.30] model has no subtype_head params -- skipping", flush=True)
        return None
    saved = {}
    for n, p in model.named_parameters():
        saved[n] = p.requires_grad
        p.requires_grad_(n.startswith("subtype_head"))
    ft_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not ft_params:
        for n, p in model.named_parameters():
            p.requires_grad_(saved[n])
        return None
    n_pos = np.array([sum(int(((s.y_subtype[:, k] > 0.5) & s.subtype_observed[:, k]).sum())
                          for s in seqs) for k in range(3)])
    n_obs = np.array([sum(int(s.subtype_observed[:, k].sum()) for s in seqs) for k in range(3)])
    ce_w = [float(np.clip(n_obs[k] / max(n_pos[k], 1), 1.0, 20.0)) for k in range(3)] + [1.0]
    opt = torch.optim.Adam(ft_params, lr=lr)
    model.train()
    t0 = time.time()
    _ft_mon = _flag("NEUROMAMBA_SUBTYPE_FT_MONITOR", "1")
    _ft_restore = _flag("NEUROMAMBA_SUBTYPE_FT_RESTORE_BEST", "0")
    _best_ft_map, _best_ft_state = -1.0, None
    for ep in range(epochs):
        tot, n_b = 0.0, 0
        for x_np, y_np, m_np, w_np, y_sub_np, y_sub_obs_np, y_walk_np, y_walk_w_np, med_np, subj_ids, src_ids, _ptags_ft, _ in training_batches(
                seqs, batch_size, epoch=ep, use_aug=False):
            x = torch.from_numpy(x_np).to(device)
            y_sub_t = torch.from_numpy(y_sub_np).to(device)
            y_sub_obs_t = torch.from_numpy(y_sub_obs_np.astype(np.float32)).to(device)
            med_t = torch.from_numpy(med_np).to(device) if getattr(model, "use_med_film", False) else None
            _b3_prompt_ft = (torch.from_numpy(_b3_prompt_matrix(src_ids, _ptags_ft)).to(device)
                             if getattr(model, "use_sensor_film", False) else None)
            opt.zero_grad()
            _, subtype_logits, _ = model(x, med=med_t, prompt=_b3_prompt_ft) if med_t is not None else model(x, prompt=_b3_prompt_ft)
            L_out = subtype_logits.shape[1]
            yt = y_sub_t[:, :L_out].reshape(-1, 3)
            mt = y_sub_obs_t[:, :L_out].reshape(-1, 3)
            if float((yt * mt).sum()) <= 0:
                continue
            if getattr(model, "subtype_ce", False):
                sub_loss = subtype_ce_loss(subtype_logits[:, :L_out], y_sub_t[:, :L_out], mt, ce_w)
            else:
                sub_loss = subtype_asl_loss(subtype_logits[:, :L_out].reshape(-1, 3), yt, mt)
            sub_loss.backward()
            opt.step()
            tot += float(sub_loss.detach().cpu())
            n_b += 1
        if n_b:
            print(f"[subtype-finetune v25.30] epoch={ep} subtype_loss={tot / n_b:.4f} "
                  f"(head-only, lr={lr}, {time.time() - t0:.0f}s)", flush=True)
        if _ft_mon:
            model.eval()
            with torch.no_grad():
                _ft_map = calib_subtype_map(model, seqs, device, window_roll=False)
            model.train()
            print(f"[subtype-finetune v25.30] epoch={ep} calibSubtypeMAP={_ft_map:.4f} "
                  f"(in-set monitor)", flush=True)
            if _ft_map > _best_ft_map:
                _best_ft_map = _ft_map
                _best_ft_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if _ft_restore and _best_ft_state is not None:
        model.load_state_dict(_best_ft_state)
        print(f"[subtype-finetune v25.30] restored best-epoch weights "
              f"(calibSubtypeMAP={_best_ft_map:.4f})", flush=True)
    for n, p in model.named_parameters():
        p.requires_grad_(saved[n])
    model.eval()
    return {"epochs": epochs, "n_sub_pos": n_sub_pos}

_GRAD_NORM_HIST = deque(maxlen=200)  # running window of pre-clip total norms

def adaptive_clip_threshold(percentile=90.0, min_clip=1.0, max_clip=50.0, bootstrap_n=20):
    """Clip at the running p-th percentile of observed grad norms; bootstrap
    phase (fewer than bootstrap_n updates) stays at the conservative min."""
    if len(_GRAD_NORM_HIST) < bootstrap_n:
        return min_clip
    _thr = float(np.clip(np.nanpercentile(_GRAD_NORM_HIST, percentile), min_clip, max_clip))
    if not np.isfinite(_thr):  # v25.46 belt-and-braces: never emit a NaN cap
        return min_clip
    return _thr

def record_grad_norm(total_norm):
    if total_norm is None or not np.isfinite(total_norm):
        return False
    _GRAD_NORM_HIST.append(float(total_norm))
    return True

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None

class _DomainDiscriminator(nn.Module):
    def __init__(self, d_in, n_domains=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_domains))

    def forward(self, h, lam=1.0):
        return self.net(_GradReverseFn.apply(h, lam))

_DOMAIN_DISC = None  # lazy singleton (dim depends on the trunk embedding width)
_DOMAIN_DISC_OPT = None  # v25.42: the discriminator's OWN optimizer -- the
_DOMAIN_IDX = {"defog": 0, "tdcsfog": 1, "notype": 2}

def domain_dann_loss(token_embed, src_ids, device, weight, lr=1e-4):
    """Pooled-trunk DANN loss over the batch's source labels; returns None when
    the batch is single-source (nothing to make invariant) or weight <= 0.
    v25.42: the discriminator gets its own AdamW so it is actually trained;
    gradient reversal still only affects the trunk."""
    global _DOMAIN_DISC, _DOMAIN_DISC_OPT
    uniq = set(str(s) for s in src_ids)
    if weight <= 0 or len(uniq) < 2 or token_embed is None:
        return None
    tok_b = token_embed.mean(dim=1)  # [B, d] pooled over tokens
    if _DOMAIN_DISC is None or _DOMAIN_DISC.net[0].in_features != tok_b.shape[-1]:
        _DOMAIN_DISC = _DomainDiscriminator(tok_b.shape[-1]).to(device)
        _DOMAIN_DISC_OPT = torch.optim.AdamW(_DOMAIN_DISC.parameters(), lr=lr,
                                             weight_decay=1e-5)
    dom_ids = torch.tensor([_DOMAIN_IDX.get(str(s), 2) for s in src_ids], device=device)
    return weight * F.cross_entropy(_DOMAIN_DISC(tok_b, lam=1.0), dom_ids)

def domain_dann_zero_grad():
    if _DOMAIN_DISC_OPT is not None:
        _DOMAIN_DISC_OPT.zero_grad()
    if _GATE_DOMAIN_DISC_OPT is not None:
        _GATE_DOMAIN_DISC_OPT.zero_grad()

def domain_dann_step():
    if _DOMAIN_DISC_OPT is not None:
        _DOMAIN_DISC_OPT.step()
    if _GATE_DOMAIN_DISC_OPT is not None:
        _GATE_DOMAIN_DISC_OPT.step()

def reset_domain_dann():
    """Per-seed/fold reset: the discriminator and its optimizer must NOT leak
    across seeds (v25.42 reproducibility fix)."""
    global _DOMAIN_DISC, _DOMAIN_DISC_OPT
    _DOMAIN_DISC, _DOMAIN_DISC_OPT = None, None
    reset_gate_domain_dann()  # v29ae (B6): same per-seed isolation

_GATE_DOMAIN_DISC = None
_GATE_DOMAIN_DISC_OPT = None
_GATE_DOMAIN_IDX = {s: i for i, s in enumerate(sorted(VALID_SOURCE_TYPES))}

def domain_gate_dann_loss(gate_feat, src_ids, device, weight, lr=1e-4):
    """B6: pooled-gate-feature DANN loss; returns None on single-source batches
    (nothing to make invariant) or when the model emitted no gate features."""
    global _GATE_DOMAIN_DISC, _GATE_DOMAIN_DISC_OPT
    uniq = set(str(s) for s in src_ids)
    if weight <= 0 or len(uniq) < 2 or gate_feat is None:
        return None
    feat_b = gate_feat.mean(dim=-1)  # [B, d] pooled over time (trunk-DANN pattern)
    if _GATE_DOMAIN_DISC is None or _GATE_DOMAIN_DISC.net[0].in_features != feat_b.shape[-1]:
        _GATE_DOMAIN_DISC = _DomainDiscriminator(feat_b.shape[-1],
                                                 n_domains=len(_GATE_DOMAIN_IDX)).to(device)
        _GATE_DOMAIN_DISC_OPT = torch.optim.AdamW(_GATE_DOMAIN_DISC.parameters(), lr=lr,
                                                  weight_decay=1e-5)
    dom_ids = torch.tensor([_GATE_DOMAIN_IDX.get(str(s), 0) for s in src_ids], device=device)
    return weight * F.cross_entropy(_GATE_DOMAIN_DISC(feat_b, lam=1.0), dom_ids)

def reset_gate_domain_dann():
    global _GATE_DOMAIN_DISC, _GATE_DOMAIN_DISC_OPT
    _GATE_DOMAIN_DISC, _GATE_DOMAIN_DISC_OPT = None, None

def ssbc_correct_alpha(alpha, n_calib, confidence=0.90):
    if n_calib <= 0:
        return float(np.clip(alpha, 0.0, 0.5))
    from scipy.stats import beta
    k = int(np.clip(np.ceil((n_calib + 1) * (1.0 - alpha)), 1, n_calib))
    corrected_level = float(beta.ppf(1.0 - confidence, k, n_calib - k + 1))
    return float(np.clip(1.0 - corrected_level, alpha, 0.5))

def log_effective_minority_emphasis():
    """v25.58 (Round 28s): the project's own A2b TODO -- measure the ACTIVE
    rare-class (SH) emphasis stack at startup. Honest accounting (verified
    against the code):
      - sampler 'oversample' is EXPOSURE-INERT in the default 'order' mode
        (only reorders a full permutation -- every sequence yields once), so
        it does NOT multiply exposure; only SAMPLER_EXPOSURE=weighted counts.
      - SH_CROP_PROB + WALK_CROP_PROB are the real exposure levers.
      - the winner token-BCE pos_weight (patch-pooled ratio, cap 5) is the
        real loss lever; the ASL cap-20 applies only to the non-token path.
      - the SH-contrastive term counts only when enabled.
    Lin & Grewal 2026: safe zone ~2.3x total, danger ~60x (stacked
    corrections collapsed precision to 33%). WARN at >=5x, REFUSE at >=20x.
    """
    crop = float(os.environ.get("NEUROMAMBA_SH_CROP_PROB", "0.40"))
    walk_crop = float(os.environ.get("NEUROMAMBA_WALK_CROP_PROB", "0.25"))
    exposure = 1.0 + crop + walk_crop
    mode = os.environ.get("NEUROMAMBA_SAMPLER_EXPOSURE", "order").strip().lower()
    over = float(os.environ.get("NEUROMAMBA_SH_OVERSAMPLE", "5.0"))
    _inert = mode not in ("weighted", "replacement", "true")
    if not _inert:
        exposure *= over
    if _inert and over > 1.0 + 1e-9:
        print(f"[emphasis-audit v25.59b] WARNING: NEUROMAMBA_SH_OVERSAMPLE={over:g} is "
              f"INERT under SAMPLER_EXPOSURE='{mode}' (order mode only reorders -- set "
              f"NEUROMAMBA_SAMPLER_EXPOSURE=weighted for true oversampling)", flush=True)
    loss_mult = float(os.environ.get("NEUROMAMBA_TOKEN_POS_WEIGHT_CAP", "5"))
    if _flag("NEUROMAMBA_SH_CONTRASTIVE", "1"):
        cw = float(os.environ.get("NEUROMAMBA_SH_CONTRASTIVE_W", "0.15"))
        cc = float(os.environ.get("NEUROMAMBA_CONTRAST_W_EFF_CAP", "5.0"))
        loss_mult *= (1.0 + min(cw * cc, cc))
    sub_w = float(os.environ.get("NEUROMAMBA_SUB_W_MAX", "1.0"))
    total = exposure * loss_mult * sub_w
    level = "SAFE" if total < 5.0 else ("WARN" if total < 20.0 else "REFUSE")
    _over_txt = (f" x oversample {over:.1f}" if not _inert
                 else ", oversample INERT (order mode)")
    print(f"[emphasis-audit v25.58] effective SH emphasis ~{total:.1f}x "
          f"(exposure {exposure:.2f} [crop {crop:.2f}+walk {walk_crop:.2f}{_over_txt}], "
          f"loss x{loss_mult:.2f}, sub_w {sub_w:.2f}) -> {level} "
          f"(Lin&Grewal safe ~2.3x, danger ~60x)", flush=True)
    if level == "REFUSE":
        raise RuntimeError("effective minority emphasis > 20x -- reduce stacked "
                           "rare-class levers before training (see emphasis-audit)")
    return total, level

def train_one_config(train_seqs, calib_seqs, thresh_seqs, device="cuda", epochs=80, lr=3e-4,
                      batch_size=8, seed=42, n_train_subjects=None, external_seqs=None,
                      notype_seqs=None, pseudo_seqs=None, d_model=128, n_blocks=2,
                      fog_warmup_epochs=None, ramp_epochs=10, model_type=None,
                      ckpt_tag=None, resume=False, warm_seqs=None, warm_epochs=0,
                      pretrained_state=None, trajectory_val_seqs=None,
                      id_use_rare_split=True):
    """v25.2 training loop. Key changes vs v25:
      - default model = ConvBiGRU (sequence context + full-res heads) --
        Kaggle top-5 all used recurrent models; SH/Walking need multi-second
        context that the v25 4x-nearest-upsampled ResNet never had.
      - walking-gate trained on an activity-BAND target with an ignored
        ambiguous band (v25's median threshold made the gate random).
      - gate applied BEFORE the clinical threshold search in per-epoch eval.
      - per-epoch checkpointing + resume (NEUROMAMBA_CKPT_DIR).
      - notype Event labels now reach the fog head (v25 dropped them).
    """
    if fog_warmup_epochs is None:
        fog_warmup_epochs = int(os.environ.get("NEUROMAMBA_FOG_WARMUP_EPOCHS", 2))
    _EMPHASIS_STATIC_TOTAL, _EMPHASIS_STATIC_LEVEL = log_effective_minority_emphasis()  # v25.58 (Round 28s): A2b startup audit
    global TRAIN_WINDOW  # v25.20-C: winner path defaults to the 1st-place 120 s context
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"[device v25.42.6] device={device} requested but torch.cuda.is_available()=False "
            f"-- refusing to run training on CPU behind a cuda request. Fix the CUDA env "
            f"or pass --device cpu explicitly.")
    if device.startswith("cuda"):
        print(f"[device v25.42.6] training on {device} "
              f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '?'})", flush=True)
    _tag_parts = (ckpt_tag or "run").split("_fold")
    _tid_family = _tag_parts[0]
    _tid_fold = 0
    if len(_tag_parts) > 1:
        try:
            _tid_fold = int(_tag_parts[1].split("_")[0]) - 1
        except ValueError:
            _tid_fold = 0
    _tid = training_identity(
        model_type=model_type or get("NEUROMAMBA_MODEL_TYPE", "sequence"),
        lr=lr, batch_size=batch_size, epochs=epochs, seed=seed,
        n_seeds=1, fold_idx=_tid_fold, family=_tid_family,
        use_rare_split=id_use_rare_split,
        warm_family=get("NEUROMAMBA_WARM_START_FAMILY", None) or None,
        warm_epochs=warm_epochs)
    in_ch = train_seqs[0].signal.shape[1]
    assert in_ch == N_CHANNELS, f"v25 requires {N_CHANNELS} channels, found {in_ch}"
    assert all(s.signal.shape[1] == N_CHANNELS for s in list(train_seqs) + list(calib_seqs) + list(thresh_seqs)), \
        "mixed feature contracts across splits"
    assert train_seqs[0].y_subtype.shape[1] == 3

    if model_type is None:
        model_type = get("NEUROMAMBA_MODEL_TYPE", "sequence")  # v25.2: default winner-style sequence model
    if model_type in ("sequence", "neuromamba") and "NEUROMAMBA_SUB_HEAD_MODE" not in os.environ:
        os.environ["NEUROMAMBA_SUB_HEAD_MODE"] = "sigmoid"
        print(f"[sub-head v25.24] model_type={model_type} -> sigmoid head mode "
              f"(the 1st-place objective: independent masked sigmoids, no None "
              f"class; softmax left SH AP at 0.000 in every pooled run)", flush=True)
    if model_type == "winner" and "NEUROMAMBA_TRAIN_WINDOW" not in os.environ and TRAIN_WINDOW < 12000:
        TRAIN_WINDOW = 12000
        print(f"[train v25.20] model_type=winner -> TRAIN_WINDOW={TRAIN_WINDOW} "
              f"(the winner's 864-token / ~121 s blocks)", flush=True)
    use_walking_gate = not _flag("NEUROMAMBA_DISABLE_WALKING_GATE", "0")
    window_roll = _flag("NEUROMAMBA_WINDOW_ROLL", "1")  # v25: default ON for eval

    if model_type == "resnet1d":
        model = ResNet1DPerFrameAdapter(in_channels=in_ch, use_walking_gate=use_walking_gate).to(device)
        print("[branch-guard v25.2] model_type=resnet1d -> legacy plain architecture.", flush=True)
    elif model_type == "sequence":
        model = ConvBiGRUModel(
            in_channels=in_ch,
            hidden=int(get("NEUROMAMBA_GRU_HIDDEN", 128)),
            n_gru_layers=int(get("NEUROMAMBA_GRU_LAYERS", 2)),
            dropout=float(get("NEUROMAMBA_GRU_DROPOUT", 0.2)),
            use_walking_gate=use_walking_gate,
        ).to(device)
        print("[branch-guard v25.2] model_type=sequence -> winner-style Conv+BiGRU.", flush=True)
    elif model_type == "winner":
        if "NEUROMAMBA_SUB_HEAD_MODE" not in os.environ:
            os.environ["NEUROMAMBA_SUB_HEAD_MODE"] = "sigmoid"
            print("[winner v25.20] default head mode -> sigmoid (the 1st-place objective: "
                  "independent masked sigmoids, no None class)", flush=True)
        model = TransformerBiLSTMWinner(
            in_channels=in_ch,
            d_model=int(get("NEUROMAMBA_WINNER_D", "128")),
            n_heads=int(get("NEUROMAMBA_WINNER_HEADS", "6")),
            n_layers=int(get("NEUROMAMBA_WINNER_DEPTH", "5")),
            patch=int(get("NEUROMAMBA_WINNER_PATCH", "14")),
            dropout=float(get("NEUROMAMBA_WINNER_DROPOUT", "0.1")),
            use_walking_gate=use_walking_gate,
        ).to(device)
        print("[branch-guard v25.13] model_type=winner -> 1st-place patch-Transformer + 2xBiLSTM "
              "(Urazalinov 0.514 architecture, adapted).", flush=True)
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    print(f"[arch] {json.dumps(model.get_architecture_manifest())}", flush=True)
    _causal_model = _flag("NEUROMAMBA_CAUSAL")
    _causal_preproc = _causal_filter_mode()
    print(f"[causal 28bz] model_causal={_causal_model} preproc_causal={_causal_preproc} "
          f"-> deployable_latency={_causal_model and _causal_preproc}"
          + ("" if (_causal_model and _causal_preproc) else
             "   [non-causal: set NEUROMAMBA_CAUSAL=1 AND NEUROMAMBA_CAUSAL_FILTER=1 "
             "for a deployable onset latency]"), flush=True)
    if pretrained_state is not None and model_type == "sequence":
        _load_pretrained_stem(model, pretrained_state, model_type)
    if model_type == "winner" and int(get("NEUROMAMBA_LORA_RANK", "0")) > 0 \
            and _flag("NEUROMAMBA_LORA_FREEZE_BACKBONE", "0"):
        _lora_trainable = freeze_backbone_except_lora(model)
        print(f"[lora v25.55] backbone FROZEN except {len(_lora_trainable)} LoRA/head "
              f"param(s): {[n for n, _ in _lora_trainable[:8]]}"
              f"{' ...' if len(_lora_trainable) > 8 else ''}", flush=True)

    opt2 = _flag("NEUROMAMBA_OPT_2NDPLACE", "0")  # v25.12b
    if opt2:
        print("[opt v25.12b] 2nd-place optimizer alignment: AdamW weight_decay=0.05 "
              "(their wd; lr via --lr, e.g. 1e-3)", flush=True)
    _b2 = float(get("NEUROMAMBA_ADAM_BETA2", "0.999"))
    if abs(_b2 - 0.999) > 1e-9:
        print(f"[opt v25.38] AdamW beta2={_b2:.4f} (winner parity: 0.98)", flush=True)
    if opt2:
        _wd_val = float(get("NEUROMAMBA_OPT_WD", str(min(0.05, 50.0 * lr))))
        _nod_kw = ("bias", "LayerNorm.bias", "LayerNorm.weight",
                   "norm", ".bn.")  # BatchNorm/GroupNorm/norm layers wd=0
        _head_wd0 = _flag("NEUROMAMBA_OPT_HEAD_WD0", "1")  # v25.44b: default ON
        _g_nod, _g_dec, _g_head = [], [], []
        for _n, _p in model.named_parameters():
            if not _p.requires_grad:
                continue
            if any(k in _n for k in _nod_kw):
                _g_nod.append(_p)
            elif _head_wd0 and ("subtype" in _n or "fog_head" in _n or "tok_sub" in _n
                                or "walking_gate" in _n
                                or "hazard_head" in _n):  # v25.45: gate is an aux head, not backbone; v28bp: hazard too
                _g_head.append(_p)
            else:
                _g_dec.append(_p)
        _groups = [{"params": _g_nod, "weight_decay": 0.0},
                   {"params": _g_dec, "weight_decay": _wd_val}]
        if _g_head:
            _groups.append({"params": _g_head, "weight_decay": 0.0})
            print(f"[opt v25.44] subtype-head wd=0 exemption ({len(_g_head)} params)", flush=True)
        print(f"[opt v25.44] 2nd-place grouped wd: {len(_g_nod)} no-decay, "
              f"{len(_g_dec)} decay(wd={_wd_val})", flush=True)
        opt = torch.optim.AdamW(_groups, lr=lr, betas=(0.9, _b2))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, _b2),
                                weight_decay=1e-5)
    warmup_epochs = int(get("NEUROMAMBA_WARMUP_EPOCHS",
                            str(max(3, min(int(0.05 * epochs), 5)))))
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        progress = min((ep - warmup_epochs) / max(epochs - warmup_epochs, 1), 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    lr_restart = _flag("NEUROMAMBA_FINETUNE_LR_RESTART", "1")
    finetune_restarted = False
    replay = _flag("NEUROMAMBA_WARM_REPLAY", "1")
    _replay_seqs, _replay_built = [], False

    total_pos = sum(int((s.labels * s.train_mask.astype(bool)).sum()) for s in train_seqs)
    total_neg = sum(int(s.train_mask.sum()) for s in train_seqs) - total_pos
    print(f"[loss-weight v25] fog pos={total_pos} neg={total_neg} "
          f"prevalence={total_pos / max(total_pos + total_neg, 1):.4f} (Poly-1 handles imbalance; no stacked pos_weight)", flush=True)

    subtype_pos = np.zeros(3, dtype=np.float64)
    subtype_neg = np.zeros(3, dtype=np.float64)
    for s in train_seqs:
        valid = s.train_mask.astype(bool)
        for k in range(3):
            obs_k = s.subtype_observed[:, k] & valid
            yy = s.y_subtype[obs_k, k]
            subtype_pos[k] += yy.sum()
            subtype_neg[k] += len(yy) - yy.sum()
    print(f"[loss-weight v25] subtype_pos={subtype_pos.astype(int).tolist()} "
          f"subtype_neg={subtype_neg.astype(int).tolist()} (ASL gamma_neg={_SUBTYPE_ASL_GAMMA_NEG}, per-class observed mask)", flush=True)
    subtype_prior_train = [float(subtype_pos[k] / max(subtype_pos[k] + subtype_neg[k], 1.0))
                           for k in range(3)]
    globals()["_DATASET_SUB_PW"] = np.clip(
        subtype_neg / np.maximum(subtype_pos, 1.0), 1.0,
        float(get("NEUROMAMBA_SUB_POS_WEIGHT_CAP", "20")))
    print(f"[loss-weight v25.44] subtype train priors (obs) = "
          f"{[round(p, 5) for p in subtype_prior_train]} (SH/Turn/Walk)", flush=True)
    if _flag("NEUROMAMBA_HEAD_PRIOR_INIT", "0"):
        _hp_msg = init_subtype_head_biases_from_priors(model, subtype_prior_train)
        if _hp_msg:
            print(f"[head-prior v25.51.2] subtype head biases initialized from priors "
                  f"{_hp_msg}", flush=True)
        else:
            print("[head-prior v25.51.2] WARNING: model exposes no init-able subtype "
                  "head -- NEUROMAMBA_HEAD_PRIOR_INIT ignored", flush=True)

    token_pw_cap = float(get("NEUROMAMBA_TOKEN_POS_WEIGHT_CAP", "5"))
    token_pw_mode = get("NEUROMAMBA_TOKEN_POS_WEIGHT_MODE", "ratio")
    _tpw_source = get("NEUROMAMBA_TOKEN_PW_SOURCE", "patch")
    if token_pw_cap > 0 and float(subtype_pos.sum()) > 0:
        if _tpw_source == "patch":
            _tpatch = int(float(get("NEUROMAMBA_WINNER_PATCH", "14")))
            _tpos = np.zeros(3, dtype=np.float64)
            _tneg = np.zeros(3, dtype=np.float64)
            for _s in train_seqs:
                _y = np.asarray(_s.y_subtype, dtype=np.float32)
                if _y.ndim != 2 or len(_y) == 0:
                    continue
                _ntok = int(np.ceil(len(_y) / _tpatch))
                _padn = _ntok * _tpatch - len(_y)
                if _padn:
                    _y = np.pad(_y, ((0, _padn), (0, 0)))
                _pool = _y.reshape(_ntok, _tpatch, 3).max(axis=1)
                _tpos += (_pool > 0.5).sum(axis=0)
                _tneg += (_pool <= 0.5).sum(axis=0)
            token_ratio = _tneg / np.maximum(_tpos, 1.0)
            print(f"[loss-weight v25.57] token pw source=patch (pooled): "
                  f"pos={_tpos.astype(int).tolist()} neg={_tneg.astype(int).tolist()} "
                  f"-- legacy frame counts over-weighted rare classes", flush=True)
        else:
            token_ratio = subtype_neg / np.maximum(subtype_pos, 1.0)
        if token_pw_mode == "rare":
            token_pw = np.clip(np.where(token_ratio >= 10.0, token_ratio, 1.0), 1.0, token_pw_cap)
        else:
            token_pw = np.clip(token_ratio, 1.0, token_pw_cap)
        if _flag("NEUROMAMBA_TURN_PLAIN", "0"):
            token_pw = token_pw.copy()
            token_pw[1] = 1.0
            print("[loss-weight v25.59c] TURN_PLAIN=1 -> token pos_weight Turn forced to 1.0", flush=True)
        token_pw_t = torch.tensor(token_pw, dtype=torch.float32, device=device).reshape(1, 1, 3)
        print(f"[loss-weight v25.33] token-BCE pos_weight (SH/Turn/Walk) = {token_pw.tolist()} "
              f"(cap={token_pw_cap:g}, mode={token_pw_mode})", flush=True)
    else:
        token_pw_t = None
    _token_asl = _flag("NEUROMAMBA_TOKEN_ASL", "0")
    if _token_asl:
        print("[loss v25.59] TOKEN_ASL=1 -> subtype token loss = per-class "
              "Asymmetric Loss (gamma_neg SH 1.5 / Turn 2.0 / Walk 2.0, clip 0.05) "
              "on the patch-pooled targets", flush=True)
    _whw_banner = float(os.environ.get("NEUROMAMBA_WALK_HARDNEG_W", "1.0"))
    _shw_banner = float(os.environ.get("NEUROMAMBA_SH_HARDNEG_W", "1.0"))
    if _whw_banner > 1.0 or _shw_banner > 1.0:
        print(f"[loss v25.59f] CONTEXT HARD NEGATIVES: WALK_HARDNEG_W={_whw_banner:g} "
              f"(Walk-channel negatives x{_whw_banner:g} on FoG-not-Walk cells) "
              f"SH_HARDNEG_W={_shw_banner:g} (SH-channel negatives x{_shw_banner:g} on "
              f"FoG-not-SH cells) -- per-cell loss weighting active", flush=True)
    _rare_boost = _flag("NEUROMAMBA_RARE_BOOST", "0")
    _base_token_pw = token_pw_t.clone() if token_pw_t is not None else None
    _pc_map_history = []  # per-epoch (SH, Turn, Walk) calib APs for the controller

    ce_weights = torch.tensor([float(v) for v in get("NEUROMAMBA_SUB_CE_WEIGHTS", _SUB_CE_WEIGHTS_DEFAULT).split(",")],
                              dtype=torch.float32, device=device)
    print(f"[loss-weight v25.3] subtype CE weights (SH/Turn/Walk/None) = {ce_weights.tolist()}", flush=True)

    es_min, es_pat = EARLY_STOP_MIN_EPOCH, EARLY_STOP_PATIENCE

    use_aug = _flag("NEUROMAMBA_USE_AUGMENTATION", "1")
    select_by_wf = _flag("NEUROMAMBA_SELECT_BY_WINDOW_F1", "1")  # v25.41: windowF1 default (P2)
    if select_by_wf:
        print("[train v25.11c] model selection + early stopping use calib WINDOW-F1 "
              "(Lin & Grewal 2026: F1-based stopping +47% over AUC-based)", flush=True)
    partner_pool = build_partner_pool(train_seqs, seed=seed) if use_aug else None
    sh_pool = build_sh_partner_pool(train_seqs, seed=seed) if use_aug else None
    if use_aug:
        print(f"[augment v25] p_augment={_p_augment_default():.2f} p_each={_p_each_default():.2f} "
              f"crop_pos_prob={float(get('NEUROMAMBA_CROP_POS_PROB', 0.2)):.2f}", flush=True)

    if notype_seqs:
        notype_seqs, _dropped_held, _n_keep = filter_notype_for_fold(
            notype_seqs, list(calib_seqs) + list(thresh_seqs or []), tag="calib/thresh")
    combined_train = list(train_seqs) + (list(notype_seqs) if notype_seqs else [])
    if pseudo_seqs:
        combined_train = combined_train + list(pseudo_seqs)
        n_psh = int(sum((s.y_subtype[:, 0] > 0.5).sum() for s in pseudo_seqs))
        n_pturn = int(sum((s.y_subtype[:, 1] > 0.5).sum() for s in pseudo_seqs))
        n_pwalk = int(sum((s.y_subtype[:, 2] > 0.5).sum() for s in pseudo_seqs))
        print(f"[pseudo v25.12] {len(pseudo_seqs)} notype_pseudo sequences in subtype training "
              f"(pseudo SH={n_psh} Turn={n_pturn} Walk={n_pwalk} frames)", flush=True)
    if notype_seqs:
        print(f"[train v25] notype sequences: {len(notype_seqs)} (Event-target only, subtypes masked)", flush=True)
    warm_seqs = list(warm_seqs) if warm_seqs is not None else None
    if warm_seqs:
        warm_sh = sum(int((s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).sum()) for s in warm_seqs)
        print(f"[warm-start v25.9] pretraining {warm_epochs} epochs on {len(warm_seqs)} transfer sequences "
              f"({warm_sh} SH frames) before fine-tuning", flush=True)

    epf1_history, map_history, smoothed_history = [], [], []
    trigger_history = []  # v25.20: median-smoothed per-epoch clinical trigger
    wf_history = []  # v25.11c: calib window-F1 history (F1-based selection)
    sel_hist = []  # v25.41: the ACTIVE selection signal per epoch (P2 floor)
    best_wf_sofar = 0.0  # v25.41: running best of the stable window-F1 signal
    _GRAD_NORM_HIST.clear()
    reset_domain_dann()
    global _CONTRAST_EMA
    _CONTRAST_EMA = None
    swa_weights, swa_count, ema_weights = None, 0, None
    swa_best_score, swa_folds = -1.0, 0  # v25.31: ASWA fold-gating state
    best_state, best_score = None, -1.0
    sel_best_epoch = -1  # v25.43.0 provenance: which epoch produced best_state (was unrecorded)
    pareto_on = _flag("NEUROMAMBA_PARETO_SELECT", "0")
    pareto_alpha = float(get("NEUROMAMBA_PARETO_ALPHA", "0.95"))
    pareto_min_sh = float(get("NEUROMAMBA_PARETO_MIN_SH_AP", "0.05"))  # v25.45: 0 never excludes an SH-collapsed epoch
    pareto_state, pareto_best_map, pareto_epoch, pareto_wf = None, -1.0, -1, 0.0
    _grad_guard = None
    _cond_class_w = None
    _sampler_cls = SubtypeAwareSampler
    if _flag("NEUROMAMBA_GRAD_GUARD", "1"):
        if _patches.GradientGuard is None:
            raise RuntimeError("NEUROMAMBA_GRAD_GUARD=1 requires neuromamba_v25_patches.py")
        _grad_guard = _patches.GradientGuard(
            hard_ceiling=float(get("NEUROMAMBA_GRAD_GUARD_CEILING", "1000")),
            dynamic_cap=float(_gdc) if (_gdc := get("NEUROMAMBA_GRAD_GUARD_DYN_CAP", "").strip()) else None)
        _gdc_txt = f"{_grad_guard.dynamic_cap:.3e}" if _grad_guard.dynamic_cap is not None else "unbounded"
        print(f"[grad-guard v25.52] ENABLED (hard ceiling "
              f"{_grad_guard.hard_ceiling:.0f}, DYN_CAP={_gdc_txt}, "
              f"dynamic = max(ceiling, 3x p99 of recent finite norms); "
              f"aborting batches restores last-finite state)", flush=True)
    if _flag("NEUROMAMBA_SUBTYPE_COND_LOSS", "0"):
        _cond_class_w = _patches.event_count_class_weights(train_seqs).to(device)
        print(f"[cond-loss v25.52] ENABLED -- event-count class weights "
              f"{[round(float(w), 3) for w in _cond_class_w]} (SH/Turn/Walk); "
              f"subtype BCE restricted to FoG-positive observed frames", flush=True)
    if _flag("NEUROMAMBA_SUBJECT_FIRST_SAMPLER", "0"):
        _sampler_cls = _patches.SubjectFirstEventSampler
        print("[sampler v25.52] SubjectFirstEventSampler ENABLED "
              "(subject-uniform rare-class exposure)", flush=True)
        _sqe = get("NEUROMAMBA_SAMPLER_QUOTA", "")
        if _sqe.strip():
            try:
                _qv = [float(v) for v in _sqe.split(",")]
                if len(_qv) != 5 or abs(sum(_qv) - 1.0) > 1e-6:
                    raise ValueError("need 5 comma-separated weights summing to 1.0")
                _qmap = dict(zip(("SH", "Walk", "Turn", "hard_negative", "background"),
                                 _qv))
                def _sampler_cls(seqs, batch_size, seed=0, _q=_qmap):
                    return _patches.SubjectFirstEventSampler(
                        seqs, batch_size, seed=seed, quota=_q)
                print(f"[sampler v25.53] quota override applied: {_qmap}", flush=True)
            except Exception as _sqe_exc:
                raise RuntimeError(
                    f"NEUROMAMBA_SAMPLER_QUOTA='{_sqe}' invalid: {_sqe_exc}") from _sqe_exc
    traj_epochs = [int(x) for x in get("NEUROMAMBA_EVAL_TRAJECTORY_EPOCHS", "").split(",") if x.strip()]
    traj_states = {}
    _tv_traj = None
    last_epoch_run = -1
    wg_loss_weight = float(get("NEUROMAMBA_WALKING_GATE_LOSS_WEIGHT", 0.3))
    start_epoch = 0

    ckpt_dir = Path(get("NEUROMAMBA_CKPT_DIR", "neuromamba_v25_ckpt"))
    ckpt_path = None
    _train_complete = False
    if ckpt_tag:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{ckpt_tag}.pt"
        if resume and ckpt_path.is_file():
            try:
                cp = torch.load(ckpt_path, map_location=device, weights_only=False)
                _cp_tid = cp.get("training_identity")
                _load_ok = True
                _refuse_reason = None
                if _cp_tid and _cp_tid != _tid:
                    _load_ok = False
                    _refuse_reason = (f"checkpoint training identity {_cp_tid[:12]} != "
                                      f"current {_tid[:12]} (training config changed)")
                elif not _cp_tid and cp.get("code_version") not in TRAINING_COMPAT_VERSIONS \
                        and cp.get("code_version") != CODE_VERSION:
                    _load_ok = False
                    _refuse_reason = (f"legacy checkpoint version {cp.get('code_version')} "
                                      f"not training-compatible (no training identity)")
                if _load_ok and cp.get("arch_signature") is not None:
                    _sig = sorted((k, tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items())
                    if _sig != cp["arch_signature"]:
                        _load_ok = False
                        _refuse_reason = "arch signature mismatch (different architecture)"
                if not _load_ok:
                    print(f"[resume v25.42.6] WARNING: {_refuse_reason} "
                          f"-- refusing to load; starting from scratch", flush=True)
                else:
                    model.load_state_dict(cp["model_state"])
                    opt.load_state_dict(cp["opt_state"])
                    sched.load_state_dict(cp["sched_state"])
                    _train_complete = bool(cp.get("train_complete", False))
                    if not _train_complete:
                        try:
                            if cp.get("np_rng_state") is not None:
                                np.random.set_state(cp["np_rng_state"])
                            if cp.get("torch_rng_state") is not None:
                                _trs = cp["torch_rng_state"]
                                if isinstance(_trs, torch.Tensor) and _trs.dtype == torch.uint8:
                                    torch.random.set_rng_state(_trs.cpu())
                            if cp.get("py_rng_state") is not None:
                                _random.setstate(pickle.loads(cp["py_rng_state"]))
                            if device.startswith("cuda") and cp.get("cuda_rng_state"):
                                torch.cuda.set_rng_state_all(cp["cuda_rng_state"])
                        except Exception as _e:
                            print(f"[resume v25.42.6] WARNING: RNG restore failed ({_e}) -- "
                                  f"continuation will differ from a bitwise replay", flush=True)
                    start_epoch = cp["epoch"] + 1
                    epf1_history = cp.get("epf1_history", [])
                    map_history = cp.get("map_history", [])
                    smoothed_history = cp.get("smoothed_history", [])
                    wf_history = cp.get("wf_history", [])
                    sel_hist = cp.get("sel_hist", [])
                    best_wf_sofar = cp.get("best_wf_sofar", 0.0)
                    best_state = cp.get("best_state")
                    best_score = cp.get("best_score", -1.0)
                    sel_best_epoch = cp.get("best_epoch", -1)  # v25.43.0 provenance
                    ema_weights = cp.get("ema_weights")
                    swa_weights, swa_count = cp.get("swa_weights"), cp.get("swa_count", 0)
                    swa_best_score = cp.get("swa_best_score", -1.0)
                    swa_folds = cp.get("swa_folds", 0)
                    if _train_complete:
                        start_epoch = epochs
                        last_epoch_run = max(int(cp.get("epoch", es_min)), es_min)
                        print(f"[resume v25.42.5] seed {ckpt_tag} already COMPLETE "
                              f"(identity {_cp_tid[:12]}) -- skipping training; "
                              f"re-running selection/calibration on the saved best state",
                              flush=True)
                    else:
                        print(f"[resume v25.5] loaded {ckpt_path} -> continuing from epoch {start_epoch} "
                              f"(best smoothed={best_score:.4f})", flush=True)
            except Exception as e:
                print(f"[resume v25.5] WARNING: failed to load {ckpt_path} ({e}); starting from scratch", flush=True)
                start_epoch = 0

    def _save_ckpt(complete=False):
        if ckpt_path is None:
            return
        try:
            _arch_sig = sorted((k, tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items())
            _state_hash = hashlib.sha256(
                b"|".join(k.encode() + b"=" + v.reshape(-1).detach().cpu().numpy().tobytes()
                          for k, v in model.state_dict().items())
            ).hexdigest() if model.state_dict() else "empty"
            _payload = {"code_version": CODE_VERSION, "training_identity": _tid,
                        "train_complete": bool(complete), "epoch": last_epoch_run,
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(), "sched_state": sched.state_dict(),
                        "scaler_state": None,  # no AMP GradScaler in this pipeline
                        "py_rng_state": None,
                        "np_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.random.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                        "arch_signature": _arch_sig, "model_state_sha256": _state_hash,
                        "epf1_history": epf1_history, "map_history": map_history,
                        "smoothed_history": smoothed_history, "wf_history": wf_history, "best_state": best_state,
                        "sel_hist": sel_hist, "best_wf_sofar": best_wf_sofar,  # v25.41 P2 floor state
                        "best_score": best_score, "best_epoch": sel_best_epoch, "ema_weights": ema_weights,
                        "swa_weights": swa_weights, "swa_count": swa_count,
                        "swa_best_score": swa_best_score, "swa_folds": swa_folds,
                        "model_type": model_type}
            try:
                _payload["py_rng_state"] = pickle.dumps(_random.getstate())
            except Exception:
                pass
            _tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save(_payload, _tmp)
            _tmp.replace(ckpt_path)  # atomic rename: no torn checkpoints
        except Exception as e:
            print(f"[ckpt v25.5] WARNING: save failed: {e}", flush=True)

    if _flag("NEUROMAMBA_TF32", "1") and device.startswith("cuda"):  # v25.57: _flag parse
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    map_include_thresh = _flag("NEUROMAMBA_MAP_INCLUDE_THRESH", "0")  # v25.57: _flag parse (selection-affecting gate)
    if map_include_thresh:
        print("[train v25.1] calibSubtypeMAP selection metric uses calib + thresh "
              "(NEUROMAMBA_MAP_INCLUDE_THRESH=1): ~2x more SH positives, less noise", flush=True)
    map_extra = thresh_seqs if map_include_thresh else None
    _nan_aborted = False  # v25.46: non-finite loss aborts the fold immediately
    _anchor_ref = None  # v25.49 score-anchor: frozen THRESH reference batch

    for epoch in range(start_epoch, epochs):
        t_ep0 = time.time()
        last_epoch_run = epoch
        model.train()
        fog_w, sub_w = loss_weight_schedule(epoch, epochs, fog_warmup_epochs=fog_warmup_epochs, ramp_epochs=ramp_epochs)
        term_vals = defaultdict(float)
        n_batches = 0
        in_warm = warm_seqs is not None and epoch < warm_epochs
        if lr_restart and not finetune_restarted and warm_seqs is not None and epoch >= warm_epochs:
            ft_lr_scale = float(os.environ.get("NEUROMAMBA_FINETUNE_LR_SCALE", 0.5))
            ft_lr = lr * ft_lr_scale
            for g in opt.param_groups:
                g["lr"] = ft_lr
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            finetune_restarted = True
            print(f"[lr v25.19] fine-tune phase: LR restarted to {ft_lr} "
                  f"({ft_lr_scale}x base, schedule re-anchored at epoch {epoch})", flush=True)
        if (not in_warm) and replay and warm_seqs is not None and not _replay_built:
            _replay_seqs = [s for s in warm_seqs
                            if (s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).sum() > 0]
            max_r = int(get("NEUROMAMBA_WARM_REPLAY_MAX", "150"))
            if len(_replay_seqs) > max_r:
                rng_r = np.random.RandomState(seed + 1234)
                keep = rng_r.choice(len(_replay_seqs), max_r, replace=False)
                _replay_seqs = [_replay_seqs[i] for i in keep]
            _replay_built = True
            print(f"[replay v25.18a] fine-tune with SH rehearsal: {len(_replay_seqs)} "
                  f"SH-positive warm sequences retained in the batch stream", flush=True)
        if _rare_boost and token_pw_t is not None and _base_token_pw is not None \
                and not in_warm and epoch >= 15 and epoch % 5 == 0 and _pc_map_history:
            _sh_ap, _turn_ap, _walk_ap = _pc_map_history[-1]
            _max_boost = min(2.0, 15.0 / max(_EMPHASIS_STATIC_TOTAL, 1e-6))
            _new_pw = token_pw_t.clone()
            _changed = []
            for _k, _ap in ((0, _sh_ap), (2, _walk_ap)):
                if float(_ap) < 0.05:
                    _boosted = min(float(_new_pw[0, 0, _k]) * 1.5,
                                   float(_base_token_pw[0, 0, _k]) * _max_boost)
                    if _boosted > float(_new_pw[0, 0, _k]) + 1e-6:
                        _new_pw[0, 0, _k] = _boosted
                        _changed.append(f"{SUBTYPE_NAMES[_k]} pw->{_boosted:.2f} (calib AP {_ap:.4f})")
            if _changed:
                token_pw_t = _new_pw
                print(f"[rare-boost v25.59] epoch {epoch}: {', '.join(_changed)} "
                      f"(bounded x{_max_boost:.2f} of base; static emphasis "
                      f"{_EMPHASIS_STATIC_TOTAL:.1f}x)", flush=True)
        cur_train = warm_seqs if in_warm else combined_train
        if (not in_warm) and replay and _replay_seqs:
            cur_train = list(combined_train) + _replay_seqs
            _rep_repeat = max(1, int(get("NEUROMAMBA_WARM_REPLAY_REPEAT", "1")))
            if _rep_repeat > 1:
                cur_train = list(combined_train) + _replay_seqs * _rep_repeat
                print(f"[replay v25.59c] SH rehearsal repeated x{_rep_repeat} "
                      f"({len(_replay_seqs)} x {_rep_repeat} = {len(_replay_seqs) * _rep_repeat} "
                      f"replay entries; SH share of stream {100 * len(_replay_seqs) * _rep_repeat / max(len(cur_train), 1):.0f}%)",
                      flush=True)
        _grad_max = 0.0
        _max_norm_src = []  # v25.58 (Round 28r): storm-source telemetry
        _n_clip = 0
        for x_np, y_np, m_np, w_np, y_sub_np, y_sub_obs_np, y_walk_np, y_walk_w_np, med_np, subj_ids, src_ids, pos_tags, _ in training_batches(
                cur_train, batch_size, epoch=epoch, use_aug=use_aug,
                sampler_cls=_sampler_cls,
                partner_pool=partner_pool, sh_pool=sh_pool):
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            m = torch.from_numpy(m_np).to(device)
            w = torch.from_numpy(w_np).to(device)
            _ntfw = float(get("NEUROMAMBA_NOTYPE_FOG_W", "1"))
            _ntfw_applies = (abs(_ntfw - 1.0) > 1e-12 and len(src_ids) > 0
                             and w_np.size > 0 and float(np.max(w_np)) > 0.0)
            if _ntfw_applies:
                _src_arr = np.asarray(src_ids)
                _nt_mask = (_src_arr == "notype")
                if bool(_nt_mask.any()):
                    _nt_scale = np.ones((len(_src_arr), w_np.shape[1]), dtype=np.float32)
                    _nt_scale[_nt_mask, :] = _ntfw
                    _w_eff = np.clip(w_np.astype(np.float32) * _nt_scale, 0.0, 100.0)
                    w = torch.from_numpy(_w_eff).to(device)
                    term_vals["notype_fog_w"] += int(_nt_mask.sum())
                    if not globals().get("_nm25_ntfw_logged", False):
                        globals()["_nm25_ntfw_logged"] = True
                        print(f"[event-aux v25.54.4] ACTIVE: notype fog-weight={_ntfw} "
                              f"({int(_nt_mask.sum())} notype crop(s) in this batch; "
                              f"line prints ONCE per process)", flush=True)
            y_sub_t = torch.from_numpy(y_sub_np).to(device)
            y_sub_obs_t = torch.from_numpy(y_sub_obs_np.astype(np.float32)).to(device)
            y_walk_t = torch.from_numpy(y_walk_np).to(device)
            y_walk_w_t = torch.from_numpy(y_walk_w_np).to(device)
            med_t = torch.from_numpy(med_np).to(device) if getattr(model, "use_med_film", False) else None
            _b3_prompt_t = (torch.from_numpy(_b3_prompt_matrix(src_ids, pos_tags)).to(device)
                            if getattr(model, "use_sensor_film", False) else None)
            if _b3_prompt_t is not None and not globals().get("_nm29ae_b3_logged", False):
                globals()["_nm29ae_b3_logged"] = True
                print(f"[b3 v29ae] training prompts active (first batch: "
                      f"{_b3_prompt_t.shape[0]}x{_b3_prompt_t.shape[1]}; prints ONCE per process)",
                      flush=True)

            opt.zero_grad()
            domain_dann_zero_grad()  # v25.42: discriminator's own grad buffer
            fog_logits, subtype_logits, aux = (model(x, med=med_t, prompt=_b3_prompt_t)
                                               if med_t is not None
                                               else model(x, prompt=_b3_prompt_t))

            fog_loss, tv = compute_loss(fog_logits, y, m, w=w)
            for k, v in tv.items():
                term_vals[k] += v
            _ep_w = float(get("NEUROMAMBA_EP_LOSS_W", "0"))
            if _ep_w > 0 and epoch >= int(get("NEUROMAMBA_EP_LOSS_START", "3")):
                _ep_l, _ep_tv = episode_soft_f1_loss(fog_logits, y, m)
                for _ek, _ev in _ep_tv.items():
                    term_vals[_ek] += _ev
                fog_loss = (1.0 - _ep_w) * fog_loss + _ep_w * _ep_l

            _sa_w = float(get("NEUROMAMBA_SCORE_ANCHOR_W", "0"))
            if _sa_w > 0 and epoch >= int(get("NEUROMAMBA_SCORE_ANCHOR_START", "3")) and n_batches == 0:
                if _anchor_ref is None:
                    _aw = TRAIN_WINDOW
                    _xs, _ys, _ms = [], [], []
                    for _seq in (thresh_seqs or [])[:4]:
                        L = len(_seq.labels)
                        if L <= 0:
                            continue
                        _sl = np.asarray(_seq.signal, dtype=np.float32)
                        _yl = np.asarray(_seq.labels, dtype=np.float32)
                        _ml = np.asarray(getattr(_seq, "observed", None), dtype=np.float32) \
                            if getattr(_seq, "observed", None) is not None else np.ones(L, np.float32)
                        for _st in range(0, max(L - _aw, 1), _aw):
                            _x = np.zeros((_aw, N_CHANNELS), np.float32)
                            _y = np.zeros(_aw, np.float32); _m = np.zeros(_aw, np.float32)
                            _n = min(_aw, L - _st)
                            _x[:_n] = _sl[_st:_st + _n]
                            _y[:_n] = _yl[_st:_st + _n]
                            _m[:_n] = _ml[_st:_st + _n]
                            _xs.append(_x); _ys.append(_y); _ms.append(_m)
                            if len(_xs) >= 8:
                                break
                        if len(_xs) >= 8:
                            break
                    if not _xs:
                        _anchor_ref = ("empty",)
                    else:
                        _anchor_ref = (torch.from_numpy(np.stack(_xs)).to(device),
                                       torch.from_numpy(np.stack(_ys) > 0.5).to(device),
                                       torch.from_numpy(np.stack(_ms) > 0).to(device))
                if _anchor_ref[0] != "empty":
                    _ax, _ay, _am = _anchor_ref
                    _p_ref = torch.sigmoid(model(_ax, med=None)[0])
                    _l_a = torch.zeros((), device=device)
                    _tq = [float(t) for t in get("NEUROMAMBA_SCORE_ANCHOR_TARGETS", "0.05,0.80").split(",")]
                    _neg = _am & (~_ay)
                    _pos = _am & _ay
                    if bool(_neg.any()):
                        _l_a = _l_a + (_p_ref[_neg].mean() - _tq[0]) ** 2
                    if bool(_pos.any()):
                        _l_a = _l_a + (_p_ref[_pos].mean() - _tq[1]) ** 2
                    fog_loss = fog_loss + _sa_w * _l_a
                    term_vals["score_anchor"] += float((_sa_w * _l_a).detach().cpu())

            L_out = subtype_logits.shape[1]
            _batch_defog = bool(src_ids) and all(str(s) == "defog" for s in src_ids)
            _sh_scale = 1.0
            _sh_mode = get("NEUROMAMBA_DEFOG_SH_SCALE_MODE", "term")  # v25.45 P0 fix
            _sh_masked = False
            if _batch_defog and _flag("NEUROMAMBA_MASK_DEFOG_SH", "1"):
                _sh_weak = float(get("NEUROMAMBA_DEFOG_SH_WEAK", "0.05"))
                if _sh_weak <= 0:
                    y_sub_t[:, :, 0] = 0.0
                    y_sub_obs_t[:, :, 0] = 0.0
                    _sh_scale = 0.0
                else:
                    _sh_scale = _sh_weak
                _sh_masked = True
            yt = y_sub_t[:, :L_out].reshape(-1, 3)
            mt = y_sub_obs_t[:, :L_out].reshape(-1, 3)  # per-class mask
            _cond_active = False
            if _flag("NEUROMAMBA_SUBTYPE_COND_LOSS", "0") and \
                    float((yt * mt).sum()) > 0:
                sub_loss, _ct = _patches.conditional_subtype_loss(
                    subtype_logits[:, :L_out], y[:, :L_out],
                    y_sub_t[:, :L_out], y_sub_obs_t[:, :L_out],
                    class_weight=_cond_class_w)
                for _ck, _cv in _ct.items():
                    term_vals[_ck] = term_vals.get(_ck, 0.0) + (
                        float(_cv) if isinstance(_cv, float) else int(_cv))
                loss = fog_w * fog_loss + sub_w * sub_loss
                _cond_active = True
                if not globals().get("_nm25_cond_logged", False):
                    globals()["_nm25_cond_logged"] = True
                    print("[cond-loss v25.53.2] ACTIVE in loss path -- "
                          "conditional_subtype_loss engaged on FoG-positive "
                          "observed frames", flush=True)
            if float((yt * mt).sum()) > 0 and not _cond_active:
                token_mode = ("token_subtype_logits" in aux
                              and _flag("NEUROMAMBA_WINNER_TOKEN_LOSS", "0"))
                if token_mode:
                    tok_logits = aux["token_subtype_logits"]        # [B, n_tok, 4]
                    patch = int(getattr(model, "patch", 14))
                    L_use = min(tok_logits.shape[1] * patch, y_sub_t.shape[1])
                    L_use = (L_use // patch) * patch                # patch-aligned
                    n_tok = L_use // patch
                    tok_logits = tok_logits[:, :n_tok]
                    y_pool, m_pool = [], []
                    for b in range(y_sub_t.shape[0]):
                        yb = y_sub_t[b, :L_use].reshape(n_tok, patch, 3).max(dim=1).values
                        mb = y_sub_obs_t[b, :L_use].reshape(n_tok, patch, 3).max(dim=1).values
                        y_pool.append(yb)
                        m_pool.append(mb)
                    y_pool = torch.stack(y_pool, dim=0).to(y_sub_t.device)
                    m_pool = torch.stack(m_pool, dim=0).to(y_sub_t.device).float()
                    if getattr(model, "subtype_ce", True):
                        _ce_b = ce_weights.clone() if _sh_scale < 1.0 else ce_weights
                        if _sh_scale < 1.0:
                            _ce_b[0] = _ce_b[0] * _sh_scale
                        sub_loss = subtype_ce_loss(tok_logits, y_pool, m_pool, _ce_b)
                        term_vals["subtype_ce_tok"] += float(sub_loss.detach().cpu())
                    else:
                        _pw_vec = token_pw_t.reshape(-1)
                        if _token_asl:
                            sub_loss = subtype_token_asl_loss(tok_logits, y_pool, m_pool,
                                                              pos_weight=_pw_vec)
                        elif _sh_scale < 1.0 and _sh_scale > 0.0 and _sh_mode == "term":
                            _l_sh = subtype_token_bce_loss(tok_logits[..., :1], y_pool[..., :1], m_pool[..., :1],
                                                           pos_weight=_pw_vec[:1])
                            _l_rest = subtype_token_bce_loss(tok_logits[..., 1:], y_pool[..., 1:], m_pool[..., 1:],
                                                             pos_weight=_pw_vec[1:])
                            sub_loss = _sh_scale * _l_sh + _l_rest
                        else:
                            _pw_b = _pw_vec.clone() if (_sh_scale < 1.0 and _sh_mode == "weight") else _pw_vec
                            if _sh_scale < 1.0 and _sh_mode == "weight":
                                _pw_b[0] = _pw_b[0] * _sh_scale
                            sub_loss = subtype_token_bce_loss(tok_logits, y_pool, m_pool,
                                                              pos_weight=_pw_b)
                        term_vals["subtype_bce_tok"] += float(sub_loss.detach().cpu())
                    use_sh_contrast = _flag("NEUROMAMBA_SH_CONTRASTIVE", "1")
                    if use_sh_contrast and "token_embed" in aux:
                        tok_emb = aux["token_embed"][:, :n_tok].reshape(-1, aux["token_embed"].shape[-1])
                        y_flat = y_pool.reshape(-1, 3)
                        m_flat = m_pool.reshape(-1, 3)
                        contrast_loss = sh_contrastive_aux_loss(tok_emb, y_flat, m_flat)
                        term_vals["sh_contrastive"] += float(contrast_loss.detach().cpu())
                        contrast_w = float(get("NEUROMAMBA_SH_CONTRASTIVE_W", 0.15))
                        if _flag("NEUROMAMBA_SH_CONTRASTIVE_NORM", "1"):
                            _mag = float(contrast_loss.detach().cpu())
                            _CONTRAST_EMA = _mag if _CONTRAST_EMA is None \
                                else 0.99 * _CONTRAST_EMA + 0.01 * _mag
                            w_eff = contrast_w / max(_CONTRAST_EMA, 1e-3)
                            _w_eff_cap = float(get("NEUROMAMBA_CONTRAST_W_EFF_CAP", "5.0"))
                            if _w_eff_cap > 0:
                                w_eff = min(w_eff, _w_eff_cap)
                        else:
                            w_eff = contrast_w
                        sub_loss = sub_loss + w_eff * contrast_loss
                elif getattr(model, "subtype_ce", False):
                    _ce_b = ce_weights.clone() if _sh_scale < 1.0 else ce_weights
                    if _sh_scale < 1.0:
                        _ce_b[0] = max(_ce_b[0] * _sh_scale, 1.0)
                    sub_loss = subtype_ce_loss(subtype_logits[:, :L_out], y_sub_t[:, :L_out], mt, _ce_b)
                    term_vals["subtype_ce"] += float(sub_loss.detach().cpu())
                else:
                    _asl_pw = None
                    if _sh_scale < 1.0 and _sh_scale > 0.0 and _sh_mode == "term":
                        _cap = float(get("NEUROMAMBA_SUB_POS_WEIGHT_CAP", "20"))
                        _pw0 = min(max(float(mt[:, 0].sum().item()) / max(float((yt[:, 0] * mt[:, 0]).sum().item()), 1.0), 1.0), _cap)
                        _l_sh = subtype_asl_loss(subtype_logits[:, :L_out].reshape(-1, 3)[:, :1], yt[:, :1], mt[:, :1],
                                                  pos_weights=[_pw0])
                        _l_rest = subtype_asl_loss(subtype_logits[:, :L_out].reshape(-1, 3)[:, 1:], yt[:, 1:], mt[:, 1:],
                                                   class_offset=1)
                        sub_loss = _sh_scale * _l_sh + _l_rest
                        term_vals["subtype_asl"] += float(sub_loss.detach().cpu())
                    else:
                        if _sh_scale < 1.0 and _sh_mode == "weight":
                            _cap = float(get("NEUROMAMBA_SUB_POS_WEIGHT_CAP", "20"))
                            _pw = []
                            for _k2 in range(3):
                                _y_k2 = yt[:, _k2]
                                _m_k2 = mt[:, _k2]
                                _np2 = float((_y_k2 * _m_k2).sum().item())
                                _nv2 = float(_m_k2.sum().item())
                                _pw.append(1.0 if _np2 <= 0 else min(max(_nv2 / max(_np2, 1.0), 1.0), _cap))
                            _pw[0] = _pw[0] * _sh_scale
                            _asl_pw = _pw
                        sub_loss = subtype_asl_loss(subtype_logits[:, :L_out].reshape(-1, 3), yt, mt,
                                                     pos_weights=_asl_pw)
                        term_vals["subtype_asl"] += float(sub_loss.detach().cpu())
                loss = fog_w * fog_loss + sub_w * sub_loss
            else:
                loss = fog_w * fog_loss

            if "walking_prob" in aux and use_walking_gate:
                _wg_m = m[:, :L_out].reshape(-1)
                if _flag("NEUROMAMBA_GATE_MASK_SH", "0"):
                    _wg_m = _wg_m * (y_sub_t[:, :L_out, 0].reshape(-1) < 0.5).float()
                wg_loss = walking_gate_loss(aux["walking_prob"][:, :L_out].reshape(-1),
                                            y_walk_t[:, :L_out].reshape(-1),
                                            _wg_m,
                                            w=y_walk_w_t[:, :L_out].reshape(-1))
                term_vals["walking_gate"] += float(wg_loss.detach().cpu())
                loss = loss + wg_loss_weight * wg_loss

            if getattr(model, "use_hazard_head", False) and aux.get("hazard_logit") is not None:
                _hz = aux["hazard_logit"]
                _Lh = min(_hz.shape[1], y.shape[1])
                _hz_b = _hz[:, :_Lh]                    # [B, T] per-frame hazard logits
                _m_h = m[:, :_Lh]                       # observed-frame mask (the fog-loss mask)
                _lam_tls = float(get("FOGBENCH_TLS_W", "0.3"))
                _lam_np = float(get("NEUROMAMBA_NP_CONSTRAINT", "0"))
                if not globals().get("_nm25_hazard_banner", False):
                    globals()["_nm25_hazard_banner"] = True
                    _hz_tp0 = str(get("FOGBENCH_TLS_PARAMS", "2.0,0.5,0.3")).split(",")
                    print(f"[hazard-arm v28bp] ACTIVE: TLS lam={_lam_tls} "
                          f"(h={_hz_tp0[0]}s tau={_hz_tp0[1]}s tail={_hz_tp0[2]}), "
                          f"NP lam={_lam_np} alpha={float(get('NEUROMAMBA_NP_ALPHA', '0.05'))} "
                          f"start={int(get('NEUROMAMBA_NP_START', '3'))}", flush=True)
                    if _flag("NEUROMAMBA_PCGRAD", "0"):
                        print("[hazard-arm v28bp] WARNING: NEUROMAMBA_PCGRAD=1 -- the per-task "
                              "PCGrad merge does not include the hazard task; hazard terms only "
                              "receive gradient via the combined-backward fallback", flush=True)
                if _lam_tls > 0:
                    _tls = _load_tls_module()
                    _hz_tp = str(get("FOGBENCH_TLS_PARAMS", "2.0,0.5,0.3")).split(",")
                    _horizon_s, _tau_s, _tail_w = float(_hz_tp[0]), float(_hz_tp[1]), float(_hz_tp[2])
                    _y_np = y[:, :_Lh].detach().cpu().numpy().astype(np.float32)
                    _B_h, _T_h = _y_np.shape
                    _soft_np = np.zeros((_B_h, _T_h), dtype=np.float32)
                    _wtls_np = np.ones((_B_h, _T_h), dtype=np.float32)
                    for _bidx in range(_B_h):
                        _row = _y_np[_bidx]
                        _rid = np.zeros(_T_h, dtype=np.int64)  # crop == one record's contiguous span
                        _ev = _tls.events_from_labels(_row, _rid, use_trainer="local")  # fast; verified identical to the trainer's extractor
                        _soft_np[_bidx] = _tls.tls_soft_labels(
                            _row, _rid, horizon_s=_horizon_s, tau_s=_tau_s,
                            fs=float(CANON_FS), tail_weight=_tail_w, events=_ev)
                        _wtls_np[_bidx] = _tls.tls_class_weights(_row, _soft_np[_bidx])
                    _soft_t = torch.from_numpy(_soft_np).to(device=_hz.device, dtype=_hz.dtype)
                    _wtls_t = torch.from_numpy(_wtls_np).to(device=_hz.device, dtype=_hz.dtype)
                    _mw = _m_h * _wtls_t
                    _bce = F.binary_cross_entropy_with_logits(
                        _hz_b.reshape(-1), _soft_t.reshape(-1), reduction="none").reshape(_hz_b.shape)
                    _haz_tls = (_bce * _mw).sum() / _mw.sum().clamp(min=1.0)
                    loss = loss + _lam_tls * _haz_tls
                    term_vals["hazard_tls"] += float((_lam_tls * _haz_tls).detach().cpu())
                    term_vals["hazard_tls_band_frac"] += float((_wtls_np == 2.0).mean())
                if _lam_np > 0 and epoch >= int(get("NEUROMAMBA_NP_START", "3")):
                    _fp = torch.sigmoid(_hz_b)[y[:, :_Lh] < 0.5]  # hard 0/1 labels; y==0 iff y<0.5
                    if _fp.numel() > 0:
                        _alpha_np = float(get("NEUROMAMBA_NP_ALPHA", "0.05"))
                        _k = max(1, min(int(_alpha_np * _fp.numel()), _fp.numel()))
                        _qhat = torch.topk(_fp, _k).values.mean()  # differentiable top-k quantile
                        _np_pen = torch.relu(_qhat - _alpha_np)
                        loss = loss + _lam_np * _np_pen
                        term_vals["hazard_np"] += float((_lam_np * _np_pen).detach().cpu())
                        term_vals["hazard_np_qhat"] += float(_qhat.detach().cpu())  # raw quantile telemetry

            _dann_w = float(get("NEUROMAMBA_DOMAIN_DANN", "0.0"))
            if _dann_w > 0:
                _dann = domain_dann_loss(aux.get("token_embed"), src_ids, device,
                                         _dann_w, lr=lr)
                if _dann is not None:  # already weight-scaled inside the helper
                    term_vals["domain_dann"] += float(_dann.detach().cpu())
                    loss = loss + _dann

            if getattr(model, "use_gate_dann", False):
                _b6_w = float(get("NEUROMAMBA_B6_DANN_W", "0.05"))
                if _b6_w > 0:
                    _gate_dann = domain_gate_dann_loss(aux.get("gate_feat"), src_ids,
                                                       device, _b6_w, lr=lr)
                    if _gate_dann is not None:
                        if not globals().get("_nm29ae_b6_logged", False):
                            globals()["_nm29ae_b6_logged"] = True
                            print(f"[b6 v29ae] gate DANN firing on a multi-source batch "
                                  f"(w={_b6_w}; prints ONCE per process)", flush=True)
                        term_vals["gate_dann"] += float(_gate_dann.detach().cpu())
                        loss = loss + _gate_dann

            _params_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            if (not bool(torch.isfinite(loss))) or (not _params_finite):
                print(f"[nan-tripwire v25.46] epoch={epoch} step={n_batches}: "
                      f"loss_finite={bool(torch.isfinite(loss))} params_finite={_params_finite} "
                      f"-- ABORTING fold (best checkpoint preserved)", flush=True)
                _nan_aborted = True
                break

            _pcgrad = _flag("NEUROMAMBA_PCGRAD", "0")  # v25.57: _flag parse
            if _pcgrad:
                try:
                    opt.zero_grad()
                    domain_dann_zero_grad()
                    (fog_w * fog_loss).backward()
                    _g_fog = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                              if p.grad is not None and p.requires_grad}
                    opt.zero_grad()
                    _r2 = (model(x, med=med_t, prompt=_b3_prompt_t)
                           if med_t is not None else model(x, prompt=_b3_prompt_t))
                    _L2 = min(_r2[1].shape[1], y_sub_t.shape[1])
                    _sl2 = subtype_asl_loss(_r2[1][:, :_L2].reshape(-1, 3),
                                            y_sub_t[:, :_L2].reshape(-1, 3),
                                            y_sub_obs_t[:, :_L2].reshape(-1, 3),
                                            pos_weights=None)
                    (sub_w * _sl2).backward()
                    _g_sh = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                             if p.grad is not None and p.requires_grad}
                    opt.zero_grad()
                    _g_wg = None
                    if "walking_prob" in aux and use_walking_gate:
                        _r3 = (model(x, med=med_t, prompt=_b3_prompt_t)
                               if med_t is not None else model(x, prompt=_b3_prompt_t))
                        _L3 = min(_r3[2]["walking_prob"].shape[1], y_walk_t.shape[1])
                        _wg_m2 = m[:, :_L3].reshape(-1)
                        if _flag("NEUROMAMBA_GATE_MASK_SH", "0"):
                            _wg_m2 = _wg_m2 * (y_sub_t[:, :_L3, 0].reshape(-1) < 0.5).float()
                        _wg2 = walking_gate_loss(_r3[2]["walking_prob"][:, :_L3].reshape(-1),
                                                 y_walk_t[:, :_L3].reshape(-1), _wg_m2,
                                                 w=y_walk_w_t[:, :_L3].reshape(-1))
                        (wg_loss_weight * _wg2).backward()
                        _g_wg = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                 if p.grad is not None and p.requires_grad}
                        opt.zero_grad()
                    _shared_keys = sorted(set(_g_fog.keys()) & set(_g_sh.keys()))
                    _all_keys = sorted(set(_g_fog.keys()) | set(_g_sh.keys()) |
                                       (set(_g_wg.keys()) if _g_wg is not None else set()))
                    _cos_acc, _n_cf = 0.0, 0
                    for _k in _shared_keys:
                        _gf, _gs = _g_fog[_k], _g_sh[_k]
                        _gf_f, _gs_f = _gf.reshape(-1), _gs.reshape(-1)
                        _dot = float(torch.dot(_gf_f, _gs_f).detach().cpu())
                        _denom = (float(torch.norm(_gf_f).detach().cpu())
                                  * float(torch.norm(_gs_f).detach().cpu()))
                        _cos_acc += (_dot / (_denom + 1e-12)) if _denom > 0 else 0.0
                        if _dot < 0.0:
                            _gf = _gf - (_dot / (_gs_f.dot(_gs_f) + 1e-12)) * _gs
                            _n_cf += 1
                        _g_fog[_k] = _gf + _gs  # SH keeps its gradient; fog' + SH combined
                        if _g_wg is not None and _k in _g_wg:
                            _gw = _g_wg[_k]
                            if float(torch.dot(_gw.reshape(-1), _gs_f).detach().cpu()) < 0.0:
                                _gw = _gw - (torch.dot(_gw.reshape(-1), _gs_f)
                                             / (_gs_f.dot(_gs_f) + 1e-12)) * _gs
                            _g_fog[_k] = _g_fog[_k] + _gw
                    _v_f = torch.cat([_g_fog[_k].reshape(-1) for _k in _shared_keys])
                    _v_s = torch.cat([_g_sh[_k].reshape(-1) for _k in _shared_keys])
                    _nf2 = float(torch.norm(_v_f).detach().cpu())
                    _ns2 = float(torch.norm(_v_s).detach().cpu())
                    _cos_flat = (float(torch.dot(_v_f, _v_s).detach().cpu()) / (_nf2 * _ns2 + 1e-12)
                                 if (_nf2 > 0 and _ns2 > 0) else 0.0)
                    term_vals["pcgrad_cos"] += float(np.clip(_cos_flat, -1.0, 1.0))
                    term_vals["pcgrad_conflict"] += float(_n_cf / max(len(_shared_keys), 1))
                    _final = {}
                    for _k in _all_keys:
                        _g = _g_fog.get(_k)
                        if _k in _shared_keys:
                            _g = _g_fog[_k]  # already fog' + SH (+ WG) merged
                        elif _k in _g_sh:
                            _g = _g_sh[_k].clone()  # SH-private preserved as-is
                        if _g is None and _g_wg is not None and _k in _g_wg:
                            _g = _g_wg[_k].clone()  # gate-private preserved as-is
                        elif _g is not None and _g_wg is not None and _k in _g_wg:
                            _g = _g + _g_wg[_k]
                        if _g is not None:
                            _final[_k] = _g
                    for _n, _p in model.named_parameters():
                        if _n in _final:
                            _p.grad = _final[_n]
                except Exception as _pe:
                    print(f"[pcgrad v25.47b] per-task backward failed "
                          f"({type(_pe).__name__}: {str(_pe)[:110]}) -- "
                          f"falling back to combined backward", flush=True)
                    opt.zero_grad()
                    loss.backward()
            else:
                loss.backward()
            domain_dann_step()
            _clip_mode = get("NEUROMAMBA_GRAD_CLIP_MODE", "adaptive")
            if _clip_mode == "fixed":
                _clip_max = float(get("NEUROMAMBA_GRAD_CLIP", "1.0"))
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_max)
            else:
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                _gn_f = float(_gn) if _gn is not None else 0.0
                if _grad_guard is not None:
                    _rec_cap = float(_grad_guard.dynamic_ceiling())
                else:
                    _rec_cap = 100.0 * float(get("NEUROMAMBA_GRAD_CLIP", "1.0"))
                if _gn_f <= _rec_cap:
                    record_grad_norm(_gn_f)
                elif np.isfinite(_gn_f):
                    record_grad_norm(_rec_cap)
                _clip_max = adaptive_clip_threshold()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_max)
            if _gn is not None and torch.isfinite(_gn):
                _grad_max = max(_grad_max, float(_gn))
                if float(_gn) >= _grad_max or not _max_norm_src:
                    _max_norm_src = [str(s) for s in src_ids]
                if float(_gn) > _clip_max:
                    _n_clip += 1
            if _grad_guard is not None and _grad_guard.check_and_maybe_abort(
                    model, opt, loss, float(_gn) if _gn is not None else 0.0):
                continue
            opt.step()
            ema_weights = update_ema(ema_weights, model, EMA_DECAY)
            n_batches += 1

        if _nan_aborted:
            print(f"[nan-tripwire v25.46] fold aborted at epoch {epoch} after non-finite "
                  f"loss; skipping remaining epochs (best_state kept)", flush=True)
            break

        sched.step()
        _clip_frac = _n_clip / max(n_batches, 1)
        _storm = globals().setdefault("_nm25_storm", {"n": 0, "c30": 0,
                                                      "prev_ab": 0, "prev_tot": 0,
                                                      "bad_ab": 0, "cap_frozen": False})
        if epoch <= 1:
            _storm["n"] = 0  # folds restart epoch numbering -> reset per fold
            _storm["c30"] = 0
            _storm["prev_ab"] = 0
            _storm["prev_tot"] = 0
            _storm["bad_ab"] = 0
            _storm["rescues"] = 0  # v25.60 (Round 28as) storm-rescue counter per fold
            _storm["cap_frozen"] = False
        if _clip_frac > 0.30:
            _storm["c30"] += 1
            if _storm["c30"] == 2:
                print("[quality v25.53.2] FLAG: clip fraction >30% for 2 consecutive "
                      "epochs -- run marked SUSPECT; require trajectory-audit corroboration "
                      "before trusting this fold's selection", flush=True)
        else:
            _storm["c30"] = 0
        if _clip_frac > 0.80:
            _storm["n"] += 1
            print(f"[quality v25.53.2] storm epoch {epoch}: "
                  f"clip={_clip_frac:.0%} ({_n_clip}/{n_batches}) "
                  f"grad_max={_grad_max:.2e} (streak {_storm['n']}/3)", flush=True)
            if _storm["n"] >= 3:
                print("[quality v25.53.2] ABORT: 3 consecutive storm epochs -- "
                      "failing fold; best_state preserved", flush=True)
                _nan_aborted = True
        else:
            _storm["n"] = 0
        _abort_frac_epoch = 0.0
        _ab = 0
        _tot = 0
        if _grad_guard is not None:
            _tot = _grad_guard.n_total - _storm["prev_tot"]
            _ab = _grad_guard.n_aborted - _storm["prev_ab"]
            _storm["prev_tot"] = _grad_guard.n_total
            _storm["prev_ab"] = _grad_guard.n_aborted
            if _tot > 0:
                _abort_frac_epoch = _ab / _tot
        _storm["abort_frac"] = _abort_frac_epoch
        _storm["unhealthy"] = False
        _cap_warmup = int(float(get("NEUROMAMBA_GRAD_GUARD_WARMUP_EPOCHS", "10")))
        if _grad_guard is not None and epoch == _cap_warmup and not _storm["cap_frozen"]:
            _storm["cap_frozen"] = True
            _hist = getattr(_grad_guard, "history", None)
            if _hist:
                _q95 = float(np.percentile(np.asarray(_hist, dtype=np.float64), 95.0))
                _hard = float(get("NEUROMAMBA_GRAD_GUARD_CEILING", "1000"))
                _floor = float(get("NEUROMAMBA_GRAD_GUARD_FROZEN_FLOOR", "300"))
                _frozen = min(_hard, max(3.0 * _q95, _floor))
                _cur_cap = getattr(_grad_guard, "dynamic_cap", None)
                _grad_guard.dynamic_cap = _frozen if _cur_cap is None else min(_cur_cap, _frozen)
                _was_txt = f"{_cur_cap:.3e}" if _cur_cap is not None else "unbounded"
                print(f"[grad-guard v25.54] warmup cap FROZEN: 3xQ95={_q95:.3e} -> "
                      f"cap={_grad_guard.dynamic_cap:.3e} (was {_was_txt})", flush=True)
        _max_ab = float(get("NEUROMAMBA_GRAD_GUARD_MAX_ABORT_FRAC", "0.05"))
        _pat_ab = int(float(get("NEUROMAMBA_GRAD_GUARD_BAD_EPOCH_PATIENCE", "2")))
        if epoch >= _cap_warmup and _abort_frac_epoch > _max_ab:
            _storm["bad_ab"] += 1
            _storm["unhealthy"] = True
            print(f"[quality v25.54] abort-fraction epoch {epoch}: {_abort_frac_epoch:.1%} "
                  f"({_ab}/{_tot}) > {_max_ab:.0%} (streak {_storm['bad_ab']}/{_pat_ab})", flush=True)
            if (_flag("NEUROMAMBA_STORM_LR_RESCUE", "0") and _storm["bad_ab"] == 1
                    and _storm.get("rescues", 0) < int(float(get("NEUROMAMBA_STORM_RESCUE_MAX", "2")))):
                _storm["rescues"] = _storm.get("rescues", 0) + 1
                for _g in opt.param_groups:
                    _g["lr"] = _g["lr"] * 0.5
                opt.state.clear()
                print(f"[storm-rescue v25.60] epoch {epoch}: abort frac {_abort_frac_epoch:.1%} "
                      f"-> LR halved to {float(opt.param_groups[0]['lr']):.3e} + Adam moments reset "
                      f"(rescue {_storm['rescues']}/{int(float(get('NEUROMAMBA_STORM_RESCUE_MAX', '2')))})",
                      flush=True)
            if _storm["bad_ab"] >= _pat_ab:
                print(f"[quality v25.54] ABORT: {_pat_ab} consecutive epochs with "
                      f"abort fraction > {_max_ab:.0%} -- failing fold; best_state preserved", flush=True)
                _nan_aborted = True
        else:
            _storm["bad_ab"] = 0
        if _grad_max > 20.0:
            _src_txt = f" src={sorted(set(_max_norm_src))}" if _max_norm_src else ""
            if _clip_frac > 0.30:
                print(f"[grad v25.42.1] WARNING: max grad norm {_grad_max:.2f} at epoch {epoch} "
                      f"({_n_clip}/{max(n_batches, 1)} batches clipped = {_clip_frac:.0%}) -- "
                      f"possible collapse (check sh_contrastive scale / seed-diversity width){_src_txt}", flush=True)
            else:
                print(f"[grad v25.42.1] info: grad spike {_grad_max:.2f} at epoch {epoch} "
                      f"({_n_clip}/{max(n_batches, 1)} batches clipped = {_clip_frac:.0%}) -- "
                      f"benign outlier, adaptive clip active{_src_txt}", flush=True)
        print(f"[loss] epoch={epoch} fog_w={fog_w:.2f} sub_w={sub_w:.2f} "
              f"{json.dumps({k: v / max(n_batches, 1) for k, v in term_vals.items()})} "
              f"grad_max={_grad_max:.3f} clip={_n_clip}", flush=True)

        if in_warm:
            print(f"[time] warm epoch={epoch} train={time.time() - t_ep0:.1f}s", flush=True)
            _save_ckpt()
            continue

        t_eval0 = time.time()
        not_final = epoch < es_min
        sel_seqs = list(calib_seqs) + (list(map_extra) if map_extra is not None else [])
        n_sh_sel = int(sum((s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).sum() for s in sel_seqs))
        n_turn_sel = int(sum((s.subtype_observed[:, 1] & (s.y_subtype[:, 1] > 0.5)).sum() for s in sel_seqs))
        n_walk_sel = int(sum((s.subtype_observed[:, 2] & (s.y_subtype[:, 2] > 0.5)).sum() for s in sel_seqs))
        use_raw_epoch = _flag("NEUROMAMBA_EPOCH_RAW_TRIGGER", "1")
        ema_epoch_eval = _flag("NEUROMAMBA_EMA_EPOCH_EVAL", "1")
        res = run_inference_cached(model, sel_seqs, device, collect_subtype=True,
                                   collect_gate=use_walking_gate, window_roll=window_roll)
        n_calib_frames = int(sum(len(s.labels) for s in calib_seqs))
        probs, labels, ids = res[0][:n_calib_frames], res[1][:n_calib_frames], res[2][:n_calib_frames]
        gated_flag = use_walking_gate and len(res) >= 5 and res[4] is not None
        if gated_flag:
            gate_w = _gate_weight(res[4][:n_calib_frames])
            probs = probs * gate_w
        sel_wf_pooled = select_by_wf and _flag("NEUROMAMBA_WF1_INCLUDE_THRESH", "0") \
            and map_extra is not None
        if sel_wf_pooled:
            sel_probs, sel_labels, sel_ids = res[0], res[1], res[2]
            if gated_flag:
                _shp_v = np.asarray(res[3]) if (len(res) >= 4 and res[3] is not None) else None
                sel_probs = apply_gate_flat(sel_probs, _gate_weight(res[4]), _shp_v)
        else:
            sel_probs, sel_labels, sel_ids = probs, labels, ids
        smoothed_prev = float(np.mean(epf1_history[-5:])) if epf1_history else 0.0
        light = epoch < EARLY_EPOCHS_LIGHT_EVAL and smoothed_prev < LIGHT_EVAL_EPF1_FLOOR
        _tep = 1.0
        if light:
            thr = 0.05
            pred = causal_event_postprocess(probs, ids, trigger=thr)
            ep = episode_f1(labels, probs, thr, ids, pred_binary=pred)
            farh = false_alarm_events_per_hour(labels, probs, thr, ids, pred_binary=pred)["farh_events_per_hour"]
            result = {"trigger": thr, "sensitivity": float("nan")}
        else:
            _ece_pre = _ece_post = float("nan")
            if int(np.asarray(labels).sum()) >= 20 and not _flag("NEUROMAMBA_EPOCH_T_FIXED", "0"):
                _tep = float(fit_temperature(probs, labels)["temperature"])
            probs_diag = apply_temperature(probs, _tep) if _tep != 1.0 else probs
            _ece_pre = compute_ece(probs, labels)
            _ece_post = compute_ece(probs_diag, labels) if _tep != 1.0 else _ece_pre
            result = find_clinical_threshold_fast(labels, probs_diag, ids, sens_min=CLINICAL_TARGET["sensitivity_min"],
                                                    farh_max=CLINICAL_TARGET["farh_max"])
            pred = causal_event_postprocess(probs_diag, ids, trigger=result["trigger"])
            ep = episode_f1(labels, probs_diag, result["trigger"], ids, pred_binary=pred)
            farh = false_alarm_events_per_hour(labels, probs_diag, result["trigger"], ids, pred_binary=pred)["farh_events_per_hour"]
        epf1_history.append(ep)
        trigger_history.append(result["trigger"])
        smoothed_trigger = smoothed_trigger_value(trigger_history,
                                                  int(get("NEUROMAMBA_TRIGGER_SMOOTH_WINDOW", "5")))
        if use_raw_epoch:
            thr_epoch = float(result["trigger"])
        else:
            thr_epoch = smoothed_trigger
            pred = causal_event_postprocess(probs_diag if _tep != 1.0 else probs, ids, trigger=thr_epoch)
            ep = episode_f1(labels, probs_diag if _tep != 1.0 else probs, thr_epoch, ids, pred_binary=pred)
            farh = false_alarm_events_per_hour(labels, probs_diag if _tep != 1.0 else probs, thr_epoch, ids, pred_binary=pred)["farh_events_per_hour"]

        gate_stats = ""
        if gated_flag:
            raw_probs_epoch = probs / np.maximum(gate_w, 1e-6)
            gate_stats = (f" gateMean={float(np.mean(gate_w)):.2f} "
                          f"gateFlip={gate_flip_fraction(raw_probs_epoch, probs, thr_epoch) * 100.0:.1f}%")

        map_dict = subtype_map_from_probs(res[3], sel_seqs) if (len(res) >= 4 and res[3] is not None) else None
        calib_map = float(map_dict["mAP"]) if map_dict is not None else 0.0
        if _tep != 1.0:
            sel_probs_diag = apply_temperature(sel_probs, _tep)
        else:
            sel_probs_diag = sel_probs
        wf = window_level_f1(sel_labels, sel_probs_diag, sel_ids, threshold=thr_epoch)["window_f1"]
        wf_grid = selection_window_f1_grid(sel_labels, sel_probs, sel_ids)
        wf_sel = wf_grid[0] if wf_grid is not None else wf
        wf_raw, map_dict_raw, calib_map_raw = wf, map_dict, calib_map
        best_state_kind = "raw"
        ema_scored = False
        wf_e = float("nan")
        if ema_epoch_eval and ema_weights is not None and not light:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                model.load_state_dict(ema_weights)
                res_e = run_inference_cached(model, sel_seqs, device, collect_subtype=True,
                                             collect_gate=use_walking_gate, window_roll=window_roll)
                if sel_wf_pooled:
                    sel_probs_e = res_e[0]
                    if gated_flag:
                        _shp_e = np.asarray(res_e[3]) if (len(res_e) >= 4 and res_e[3] is not None) else None
                        sel_probs_e = apply_gate_flat(sel_probs_e, _gate_weight(res_e[4]), _shp_e)
                    sel_labels_e, sel_ids_e = res_e[1], res_e[2]
                else:
                    sel_probs_e = res_e[0][:n_calib_frames]
                    if gated_flag:
                        _shp_e2 = np.asarray(res_e[3])[:n_calib_frames] if (len(res_e) >= 4 and res_e[3] is not None) else None
                        sel_probs_e = apply_gate_flat(sel_probs_e, _gate_weight(res_e[4][:n_calib_frames]), _shp_e2)
                    sel_labels_e, sel_ids_e = res_e[1][:n_calib_frames], res_e[2][:n_calib_frames]
                wf_e = window_level_f1(sel_labels_e, sel_probs_e, sel_ids_e, threshold=thr_epoch)["window_f1"]
                wf_e_grid = selection_window_f1_grid(sel_labels_e, sel_probs_e, sel_ids_e)
                map_dict_e = subtype_map_from_probs(res_e[3], sel_seqs) if (len(res_e) >= 4 and res_e[3] is not None) else None
                calib_map_e = float(map_dict_e["mAP"]) if map_dict_e is not None else 0.0
            model.load_state_dict(raw_sd)
            wf, map_dict, calib_map = wf_e, map_dict_e, calib_map_e
            wf_sel = wf_e_grid[0] if wf_e_grid is not None else wf_e
            best_state_kind = "ema"
            ema_scored = True
        blend_lam = float(get("NEUROMAMBA_SELECT_MAP_BLEND", "0.0"))
        if blend_lam > 0:
            _sh_subj, _sh_events, _sh_per_subj = set(), 0, {}
            for _s in sel_seqs:
                _m = (_s.subtype_observed[:, 0] & (_s.y_subtype[:, 0] > 0.5)).astype(np.int64)
                _n_runs = int(np.sum(np.diff(np.concatenate([[0], _m, [0]])) == 1))
                if _n_runs > 0:
                    _sh_subj.add(_s.subject_id)
                    _sh_events += _n_runs
                    _sh_per_subj[_s.subject_id] = _sh_per_subj.get(_s.subject_id, 0) + _n_runs
            _max_frac = (max(_sh_per_subj.values()) / max(_sh_events, 1)) if _sh_events else 1.0
            if len(_sh_subj) < 4 or _sh_events < 8 or _max_frac > 0.5:
                print(f"[selection v25.58] SH-power gate: {len(_sh_subj)} SH subjects / "
                      f"{_sh_events} SH events / max-subject frac {_max_frac:.2f} -> "
                      f"mAP blend {blend_lam:.2f} FORCED TO 0 (underpowered selection "
                      f"split; blended noise would drag selection)", flush=True)
                blend_lam = 0.0
        min_epoch_map = int(get("NEUROMAMBA_MIN_EPOCH_MAP_SEL", "10"))
        fog_tol = float(get("NEUROMAMBA_FOG_TOLERANCE", "0.85"))
        best_wf_sofar = max(best_wf_sofar, float(wf_sel))
        map_mature = (epoch >= min_epoch_map) and (
            best_wf_sofar <= 1e-9 or float(wf_sel) >= fog_tol * best_wf_sofar)
        _mapv = float(min(max(calib_map, 0.0), 1.0))
        if _flag("NEUROMAMBA_SELECT_UTILITY", "0"):
            _farh_budget = float(CLINICAL_TARGET["farh_max"])
            _sens_raw = float(result.get("sensitivity", float("nan"))) \
                if float(farh) <= _farh_budget else 0.0
            _sens_b = _sens_raw if np.isfinite(_sens_raw) else 0.0
            if epoch < int(get("NEUROMAMBA_MIN_EPOCH_UTILITY", "2")):
                sel_signal = -1.0
                sel_name_epoch = "utility(pre-maturity-floor)"
            else:
                sel_signal = (float(ep) + 0.5 * _mapv
                              + 0.3 * _sens_b + 0.2 * float(wf_sel)) / 2.0
                sel_name_epoch = "utility(conv)"
        elif select_by_wf:
            if epoch < int(get("NEUROMAMBA_MIN_EPOCH_UTILITY", "2")):
                sel_signal = -1.0
                sel_name_epoch = "windowF1(pre-maturity-floor)"
            elif blend_lam > 0 and map_mature:
                sel_signal = (1.0 - blend_lam) * wf_sel + blend_lam * _mapv
                sel_name_epoch = "windowF1+mAP"
            else:
                sel_signal = wf_sel
                sel_name_epoch = "windowF1"
        else:
            if map_mature:
                sel_signal = (1.0 - blend_lam) * wf_sel + blend_lam * _mapv
                sel_name_epoch = "mAP"
            else:
                sel_signal = wf_sel
                sel_name_epoch = "windowF1(pre-mAP-floor)"
        map_history.append(calib_map)
        if map_dict is not None:
            _pc_map_history.append((float(map_dict["StartHesitation"]),
                                    float(map_dict["Turn"]),
                                    float(map_dict["Walking"])))
        wf_history.append(wf_sel)
        sel_hist.append(float(sel_signal))
        smoothed = smoothed_selection_score(sel_hist, window=3)
        smoothed_history.append(smoothed)
        if smoothed > best_score:
            _deg_epoch = (map_dict is not None and
                          float(map_dict["StartHesitation"]) < 1e-4 and
                          float(map_dict["Walking"]) < 1e-4)
            if not _deg_epoch:
                best_score = smoothed
                sel_best_epoch = epoch  # v25.43.0 provenance
                if best_state_kind == "ema":
                    best_state = {k: v.detach().clone() for k, v in ema_weights.items()}
                else:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if pareto_on:
            _sm_map = float(smoothed_selection_score(map_history, window=3))
            _wf_ok = best_wf_sofar <= 1e-9 or float(smoothed) >= pareto_alpha * best_wf_sofar
            _sh_ap = float(map_dict["StartHesitation"]) if map_dict is not None else 0.0
            if _wf_ok and _sh_ap >= pareto_min_sh and _sm_map > pareto_best_map:
                if best_state_kind == "ema":
                    pareto_state = {k: v.detach().clone() for k, v in ema_weights.items()}
                else:
                    pareto_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                pareto_best_map, pareto_epoch, pareto_wf = _sm_map, epoch, float(smoothed)
        if traj_epochs and epoch in traj_epochs:
            traj_states[epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch >= SWA_START and not _storm.get("unhealthy", False):
            if swa_weights is None:
                swa_weights = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
                swa_count = 1
                swa_best_score = float(smoothed) if np.isfinite(smoothed) else -1.0
                swa_folds = 1
            elif np.isfinite(smoothed) and smoothed > swa_best_score:
                sd = model.state_dict()
                for k in swa_weights:
                    swa_weights[k] = (swa_weights[k] * swa_count + sd[k].detach().float()) / (swa_count + 1)
                swa_count += 1
                swa_best_score = float(smoothed)
                swa_folds += 1

        sens = result.get("sensitivity", float("nan"))
        if not np.isfinite(sens):
            sens = 0.0
        per_class = (f" SH={map_dict['StartHesitation']:.3f} Turn={map_dict['Turn']:.3f} Walk={map_dict['Walking']:.3f}"
                     if map_dict is not None else "")
        if use_raw_epoch:
            thr_txt = f"thr={thr_epoch:.3f}(smooth={smoothed_trigger:.3f})"
        else:
            thr_txt = f"thr={smoothed_trigger:.3f}(raw={result['trigger']:.3f})"
        if _tep != 1.0:
            thr_txt += f" Tep={_tep:.2f} ECE={_ece_pre:.3f}->{_ece_post:.3f}"
        ema_txt = f" emaWF1={wf_e:.3f} rawWF1={wf_raw:.3f}" if ema_scored else ""
        sel_txt = f"selWF1@grid={wf_sel:.3f}" if wf_grid is not None else f"selWF1@thr={wf_sel:.3f}"
        if blend_lam > 0:
            sel_txt += f" (blend {blend_lam:.2f})"
        print(f"[eval] epoch={epoch} {thr_txt} "
              f"epF1={ep:.3f} sens={sens:.3f} "
              f"FARh={farh:.2f}/h light={light} rawFogF1@0.05={episode_f1(labels, probs, 0.05, ids):.3f} "
              f"calibSubtypeMAP={calib_map:.4f} windowF1@thr={wf:.3f}{ema_txt} {sel_txt} smoothed={smoothed:.4f} "
              f"sel={sel_name_epoch} gated={gated_flag}{gate_stats} "
              f"shSel={n_sh_sel} turnSel={n_turn_sel} walkSel={n_walk_sel}{per_class}"
              f"{' [NOT FINAL -- before early-stop floor]' if not_final else ''}", flush=True)
        print(f"[time] epoch={epoch} train={t_eval0 - t_ep0:.1f}s eval={time.time() - t_eval0:.1f}s", flush=True)
        _save_ckpt()

        should_stop, best_idx = early_stop_should_stop(smoothed_history, es_pat, es_min)
        if should_stop:
            sel_name = "windowF1" if select_by_wf else "calibSubtypeMAP"
            print(f"[early-stop] epoch={epoch}: no smoothed {sel_name} improvement in {es_pat} epochs "
                  f"(best epoch {best_idx}, smoothed {sel_name}={smoothed_history[best_idx]:.4f}). Stopping.", flush=True)
            break

    if last_epoch_run < es_min - 1:
        print(f"[v25-NOTE] training ended at epoch {last_epoch_run} BEFORE the early-stop floor "
              f"(min_epoch={es_min}). Metrics from this run are NOT FINAL.", flush=True)
    if ckpt_path is not None:
        _save_ckpt(complete=True)

    if swa_weights is not None:
        print(f"[aswa v25.31] swa folded {swa_folds} score-improving checkpoint(s) "
              f"since epoch {SWA_START} (best folded score {swa_best_score:.4f})", flush=True)

    if traj_epochs and trajectory_val_seqs and traj_states:
        try:
            _tv = []
            for _ep in sorted(traj_states):
                model.load_state_dict(traj_states[_ep])
                model.eval()
                res_t = run_inference_cached(model, trajectory_val_seqs, device,
                                             collect_subtype=True,
                                             collect_gate=use_walking_gate,
                                             window_roll=window_roll)
                t_probs = np.asarray(res_t[0], dtype=np.float32)
                t_labels = np.asarray(res_t[1], dtype=np.float32)
                t_ids = np.asarray(res_t[2])
                if use_walking_gate and len(res_t) >= 5 and res_t[4] is not None:
                    _t_shp = np.asarray(res_t[3]) if (len(res_t) >= 4 and res_t[3] is not None) else None
                    t_probs = apply_gate_flat(t_probs, _gate_weight(np.asarray(res_t[4])), _t_shp)
                _sub_map = None
                if len(res_t) >= 4 and res_t[3] is not None:
                    sl, so, sso = concat_labels(trajectory_val_seqs)
                    n_s = min(len(sl), len(np.asarray(res_t[3])))
                    _sub_map = compute_official_like_map(sl[:n_s], np.asarray(res_t[3])[:n_s], sso[:n_s])
                _pred05 = causal_event_postprocess(t_probs, t_ids, trigger=0.05)
                _ev = event_level_f1(t_labels, _pred05, t_ids)  # plain float (v25.42.3 fix)
                _farh = false_alarm_events_per_hour(t_labels, t_probs, 0.05, t_ids,
                                                    pred_binary=_pred05)["farh_events_per_hour"]
                _wf = window_level_f1(t_labels, t_probs, t_ids, threshold=0.05)["window_f1"]
                _tv.append({
                    "epoch": int(_ep),
                    "mAP": float(_sub_map["mAP"]) if _sub_map else float("nan"),
                    "SH": float(_sub_map["StartHesitation"]) if _sub_map else float("nan"),
                    "Turn": float(_sub_map["Turn"]) if _sub_map else float("nan"),
                    "Walk": float(_sub_map["Walking"]) if _sub_map else float("nan"),
                    "windowF1_05": float(_wf),
                    "eventF1_05": float(_ev) if _ev is not None else float("nan"),
                    "farh_05": float(_farh)})
            _tv_traj = _tv
            print(f"[trajectory v25.42.3] VAL eval of raw checkpoints @ "
                  f"{[t['epoch'] for t in _tv]}:", flush=True)
            for _t in _tv:
                print(f"  epoch {_t['epoch']:>3}: mAP={_t['mAP']:.4f} "
                      f"SH={_t['SH']:.3f} Turn={_t['Turn']:.3f} Walk={_t['Walk']:.3f} "
                      f"WF1@0.05={_t['windowF1_05']:.3f} eventF1={_t['eventF1_05']:.3f} "
                      f"FARh={_t['farh_05']:.2f}/h", flush=True)
        except Exception as e:
            print(f"[trajectory v25.42.3] eval failed: {e}", flush=True)
            traceback.print_exc()

    variants = {"raw": best_state}
    if pareto_on and pareto_state is not None:
        variants["pareto"] = pareto_state
        print(f"[pareto v25.42.2] Pareto candidate: epoch={pareto_epoch} "
              f"smoothedWF1={pareto_wf:.4f} smoothedMAP={pareto_best_map:.4f} "
              f"(WF1 floor {pareto_alpha:.2f}x best; SH floor {pareto_min_sh:.2f})", flush=True)
    if ema_weights is not None:
        variants["ema"] = ema_weights
    if swa_weights is not None:
        variants["swa"] = swa_weights
    best_name, best_vscore = "raw", -1.0
    best_trig = None
    for name, state in variants.items():
        if state is None:
            continue
        model.load_state_dict(state)
        if select_by_wf:
            _v = calib_window_f1(model, calib_seqs, device, window_roll=window_roll,
                                 extra_seqs=map_extra, fit_trigger=True)
            if isinstance(_v, tuple):
                vscore, vtrig = _v
            else:
                vscore, vtrig = _v, None
            print(f"[variant] {name} calibWindowF1={vscore:.4f}"
                  f"{f'@thr={vtrig:.3f}' if vtrig is not None else ''}", flush=True)
        else:
            vscore = calib_subtype_map(model, calib_seqs, device, window_roll=window_roll, extra_seqs=map_extra)
            vtrig = None
            print(f"[variant] {name} calibSubtypeMAP={vscore:.4f}", flush=True)
        if vscore > best_vscore:
            best_vscore, best_name, best_trig = vscore, name, vtrig
    if variants.get(best_name) is not None:
        model.load_state_dict(variants[best_name])
    print(f"[variant] selected={best_name} {'calibWindowF1' if select_by_wf else 'calibSubtypeMAP'}={best_vscore:.4f}"
          f"{f'@thr={best_trig:.3f}' if best_trig is not None else ''}", flush=True)

    if _flag("NEUROMAMBA_SUBTYPE_FINETUNE", "0"):
        _chk_extra = thresh_seqs if (thresh_seqs is not None and len(thresh_seqs)) else None
        _chk_name = "THRESH (unbiased -- not in the fine-tune set)" if _chk_extra is not None else "calib (in-set)"
        _before = calib_subtype_map(model, calib_seqs, device, window_roll=window_roll,
                                    extra_seqs=_chk_extra)
        _pre_ft_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ft_seqs = list(calib_seqs)
        _ft = fine_tune_subtype_head(model, ft_seqs, device, batch_size=batch_size,
                                     epochs=int(get("NEUROMAMBA_SUBTYPE_FT_EPOCHS", "4")),
                                     lr=float(get("NEUROMAMBA_SUBTYPE_FT_LR", "1e-4")), seed=seed)
        if _ft is not None:
            _aft = calib_subtype_map(model, calib_seqs, device, window_roll=window_roll,
                                     extra_seqs=_chk_extra)
            _reverted = bool(_chk_extra is not None and np.isfinite(_aft) and np.isfinite(_before)
                             and _aft < _before)
            if _reverted:
                model.load_state_dict(_pre_ft_state)
            del _pre_ft_state
            print(f"[subtype-finetune v25.51.2] start-variant={best_name} "
                  f"calibSubtypeMAP before={_before:.4f} after={_aft:.4f} "
                  f"(checked on {_chk_name}; fog head untouched) "
                  f"{'-> REVERTED (unbiased check regressed; pre-finetune head restored)' if _reverted else '-> KEPT'}",
                  flush=True)

    res_cal = run_inference_cached(model, calib_seqs, device, collect_subtype=True,
                                   collect_gate=use_walking_gate, window_roll=window_roll)
    calib_probs, calib_labels, calib_ids = res_cal[0], res_cal[1], res_cal[2]
    res4 = res_cal
    res_th = run_inference_cached(model, thresh_seqs, device, collect_subtype=True,
                                  collect_gate=use_walking_gate, window_roll=window_roll)
    th_probs, th_labels, th_ids = res_th[0], res_th[1], res_th[2]
    res4_th = res_th

    subtype_temps = [1.0, 1.0, 1.0]
    ece_before, ece_after = float("nan"), float("nan")
    calib_sub = None  # v25.11: (probs, labels, observed) on calib for hetero-blend tuning
    if len(res4) >= 4 and res4[3] is not None and len(res4_th) >= 4 and res4_th[3] is not None:
        calib_sub_probs = np.asarray(res4[3])
        calib_sub_labels, calib_observed, calib_sub_observed = concat_labels(calib_seqs)
        n = min(len(calib_sub_labels), len(calib_sub_probs))
        calib_sub_probs, calib_sub_labels, calib_sub_observed = calib_sub_probs[:n], calib_sub_labels[:n], calib_sub_observed[:n]
        calib_sub = (calib_sub_probs, calib_sub_labels, calib_sub_observed)
        if _flag("NEUROMAMBA_SUBTYPE_TEMPERATURE", "1"):
            subtype_temps = fit_per_subtype_temperature(calib_sub_probs, calib_sub_labels, calib_sub_observed)
            calib_sub_probs_t = apply_temperature_subtype(calib_sub_probs, subtype_temps)
            ece_before = compute_per_subtype_ece(calib_sub_probs, calib_sub_labels, calib_sub_observed).get("max_ece")
            ece_after = compute_per_subtype_ece(calib_sub_probs_t, calib_sub_labels, calib_sub_observed).get("max_ece")
            print(f"[calib] per-subtype T={[round(t, 3) for t in subtype_temps]} "
                  f"ECE max {ece_before:.4f} -> {ece_after:.4f}", flush=True)
        else:
            subtype_temps = [1.0, 1.0, 1.0]
            print("[calib v25.51.2] NEUROMAMBA_SUBTYPE_TEMPERATURE=0 -> per-subtype "
                  "temperatures DISABLED (T=1.0 shipped)", flush=True)

    calib_gated = calib_probs.copy()
    th_gated_raw = th_probs.copy()
    if use_walking_gate:
        calib_gated = _apply_walking_gate_inference(model, calib_seqs, calib_gated, device,
                                                    window_roll=window_roll,
                                                    gate_probs=res_cal[4] if len(res_cal) >= 5 else None)
        th_gated_raw = _apply_walking_gate_inference(model, thresh_seqs, th_gated_raw, device,
                                                     window_roll=window_roll,
                                                     gate_probs=res_th[4] if len(res_th) >= 5 else None)
    temp_res = fit_temperature(calib_gated, calib_labels)
    temperature = temp_res["temperature"]
    calib_probs_t2 = apply_temperature(calib_gated, temperature)
    if _flag("NEUROMAMBA_EPOCH_T_FIXED", "0"):
        print(f"[calib v25.51.2] per-epoch diagnostic T frozen at 1.0 "
              f"(fold-end deployed T={temperature:.3f} fitted normally)", flush=True)
    ece_before_bin = compute_ece(calib_gated, calib_labels)
    ece_after_bin = compute_ece(calib_probs_t2, calib_labels)
    print(f"[calib] binary T(gated)={temperature:.3f} ECE {ece_before_bin:.4f} -> {ece_after_bin:.4f}", flush=True)

    pareto = pareto_frontier_report(calib_labels, calib_probs_t2, calib_ids,
                                     sens_min=CLINICAL_TARGET["sensitivity_min"], farh_max=CLINICAL_TARGET["farh_max"])
    calib_subjects = np.array([s.subject_id for s in calib_seqs for _ in range(len(s.labels))])
    postproc = inner_cv_threshold_search(calib_labels, calib_probs_t2, calib_ids, calib_subjects,
                                          n_folds=3, sens_min=CLINICAL_TARGET["sensitivity_min"],
                                          farh_max=CLINICAL_TARGET["farh_max"])
    if postproc.get("note") and "no config met" in postproc.get("note", ""):
        postproc = tune_postprocessing_params(calib_labels, calib_probs_t2, calib_ids, sens_min=0.50)
    if select_by_wf and best_trig is not None:
        _inner_trig = postproc.get("trigger")
        postproc = deploy_variant_trigger(postproc, best_trig, select_by_wf)
        print(f"[postproc v25.36] deploying the variant-selected trigger {best_trig:.3f} "
              f"(inner-CV was {_inner_trig:.3f}) -- selection and deployment now share "
              f"the same operating point", flush=True)
    _n_calib_subj = len(set(calib_subjects)) if calib_subjects is not None else 0
    _min_calib_postproc = int(get("NEUROMAMBA_MIN_CALIB_SUBJ_POSTPROC", "15"))
    if 0 < _n_calib_subj < _min_calib_postproc:
        _cn0 = int(postproc.get("confirm_n", 3))
        _cd0 = int(postproc.get("cooldown_n", 30))
        postproc["confirm_n"] = max(_cn0, 2)
        postproc["cooldown_n"] = max(_cd0, 30)
        postproc["merge_gap_n"] = max(int(postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT)),
                                      MERGE_GAP_N_DEFAULT)
        _alpha_eff = ssbc_correct_alpha(0.10, _n_calib_subj)
        postproc["ssbc_alpha_effective"] = _alpha_eff
        if (int(postproc["confirm_n"]), int(postproc["cooldown_n"])) != (_cn0, _cd0):
            print(f"[postproc v25.41] calib has {_n_calib_subj} subjects "
                  f"(<{_min_calib_postproc}) -- aggressive postproc overfits; "
                  f"forcing confirm_n={postproc['confirm_n']} "
                  f"cooldown_n={postproc['cooldown_n']} (was {_cn0}/{_cd0}); "
                  f"SSBC-corrected alpha {0.10:.2f} -> {_alpha_eff:.3f} "
                  f"(realized FARh guarantee is wider than nominal)", flush=True)
    _budget_env = get("NEUROMAMBA_EVENT_BUDGET_CONFORMAL", "0")
    if _budget_env not in ("0", ""):
        _budget = float(_budget_env)
        try:
            _bt, _bf = event_budget_conformal_trigger(calib_labels, calib_probs_t2,
                                                      calib_ids, calib_subjects,
                                                      budget_farh=_budget)
            postproc["event_budget_conformal"] = {"budget_farh": _budget,
                                                  "trigger": _bt,
                                                  "frac_within_budget": _bf}
            if _bt is not None:
                print(f"[postproc v25.42] event-budget conformal (report-only): "
                      f"trigger={_bt:.4f} keeps FARh<={_budget:.1f}/h for {_bf:.0%} of "
                      f"calib subjects; deployed trigger unchanged", flush=True)
            else:
                print(f"[postproc v25.42] event-budget conformal (report-only): NO "
                      f"trigger keeps FARh<={_budget:.1f}/h for >=90% of calib subjects",
                      flush=True)
        except Exception as _e:
            print(f"[postproc v25.42] event-budget conformal skipped ({_e})", flush=True)
    if _flag("NEUROMAMBA_DEPLOY_CONFORMAL", "0"):  # v25.57: _flag parse
        _dep_budget = float(get("NEUROMAMBA_DEPLOY_CONFORMAL_BUDGET", "10"))
        try:
            _dt, _df = event_budget_conformal_trigger(calib_labels, calib_probs_t2,
                                                       calib_ids, calib_subjects,
                                                       budget_farh=_dep_budget)
            if _dt is not None:
                postproc = dict(postproc)
                postproc["trigger"] = float(_dt)
                postproc["deployed_via"] = "conformal_far_budget"
                frozen_threshold = float(_dt)
                print(f"[postproc v25.47] DEPLOYING conformal FAR-budget trigger "
                      f"{_dt:.4f} (FARh<={_dep_budget:.0f}/h for {_df:.0%} of calib "
                      f"subjects) -- sensitivity at this locked trigger is the "
                      f"honest clinical number", flush=True)
            else:
                print(f"[postproc v25.47] conformal deploy requested but NO trigger "
                      f"keeps FARh<={_dep_budget:.0f}/h for >=90% of calib subjects -- "
                      f"falling through to the standard fallback", flush=True)
        except Exception as _e:
            print(f"[postproc v25.47] conformal deploy skipped ({_e})", flush=True)
    print(f"[postproc] tuned {json.dumps({k: v for k, v in postproc.items() if k != 'note'})}", flush=True)

    frozen_threshold = postproc.get("trigger", 0.05)
    th_probs_t = apply_temperature(th_gated_raw, temperature)

    _chk = causal_event_postprocess(calib_probs_t2, calib_ids, trigger=frozen_threshold,
                                    sustain=postproc.get("sustain", frozen_threshold * 0.4),
                                    confirm_n=postproc.get("confirm_n", 3),
                                    cooldown_n=postproc.get("cooldown_n", 30),
                                    merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
    _tp = int(((_chk == 1) & (calib_labels == 1)).sum())
    _fn = int(((_chk == 0) & (calib_labels == 1)).sum())
    _sens_c = _safe_sensitivity(_tp, _fn)
    _farh_c = false_alarm_events_per_hour(calib_labels, calib_probs_t2, frozen_threshold, calib_ids,
                                          pred_binary=_chk)["farh_events_per_hour"]
    _conformal_locked = postproc.get("deployed_via") == "conformal_far_budget"
    if not _conformal_locked and not (_sens_c >= CLINICAL_TARGET["sensitivity_min"] and _farh_c <= 15.0):
        _alt = tune_postprocessing_params(calib_labels, calib_probs_t2, calib_ids,
                                          sens_min=CLINICAL_TARGET["sensitivity_min"],
                                          farh_max=CLINICAL_TARGET["farh_max"])
        if _alt.get("sensitivity", 0.0) >= CLINICAL_TARGET["sensitivity_min"] and \
           _alt.get("farh", 1e9) <= CLINICAL_TARGET["farh_max"]:
            print(f"[postproc v25.9] inner-CV trigger failed on full calib "
                  f"(sens={_sens_c:.2f}, FARh={_farh_c:.1f}/h) -- retuned on calib: "
                  f"trigger={_alt['trigger']:.3f} sens={_alt['sensitivity']:.2f} FARh={_alt['farh']:.1f}/h", flush=True)
            postproc = _alt
            frozen_threshold = _alt["trigger"]
        else:
            _scan = pareto_frontier_report(calib_labels, calib_probs_t2, calib_ids, n_grid=12)
            _feas10 = [r for r in (_scan.get("rows") or []) if r["farh"] <= CLINICAL_TARGET["farh_max"]]
            if _feas10:
                _n_calib_deploy = len(set(calib_subjects)) if calib_subjects is not None else 0
                _min_calib_deploy = int(get("NEUROMAMBA_MIN_CALIB_SUBJ_POSTPROC", "15"))
                _fb_early = _best_episode_f1_point(calib_labels, calib_probs_t2, calib_ids)
                _farh_safety = float(get("NEUROMAMBA_DEPLOY_FARH_SAFETY", "0.75"))
                _farh_deploy_cap = max(float(CLINICAL_TARGET["farh_max"]) * _farh_safety, 1.0)
                _fb_feasible = (_fb_early is not None
                                and _fb_early["farh"] <= _farh_deploy_cap)
                _deployed_small_calib = False
                if (_n_calib_deploy < _min_calib_deploy and _fb_feasible
                        and _flag("NEUROMAMBA_SMALL_CALIB_EP_F1_DEPLOY", "1")):
                    postproc = dict(_fb_early)
                    frozen_threshold = float(postproc["trigger"])
                    _deployed_small_calib = True
                    print(f"[postproc v25.59f/g] UNDERPOWERED CALIB ({_n_calib_deploy} subjects "
                          f"< {_min_calib_deploy}) -- deploying the FAR-feasible max-epF1 point "
                          f"(farh safety x{_farh_safety:g}: deploy cap {_farh_deploy_cap:.1f}/h on "
                          f"calib, ~x2.3 calib->val inflation measured): "
                          f"trigger={_fb_early['trigger']:.3f} epF1={_fb_early['ep_f1']:.3f} "
                          f"sens={_fb_early['sensitivity']:.2f} FARh={_fb_early['farh']:.1f}/h "
                          f"[clinical_target_unmet=True, small_calib_epf1_deploy=True]", flush=True)
                    _p10 = max(_feas10, key=lambda r: r["sensitivity"])
                    postproc["max_sens_tail_diagnostic_DO_NOT_DEPLOY"] = {
                        "trigger": float(_p10["trigger"]), "sensitivity": float(_p10["sensitivity"]),
                        "farh": float(_p10["farh"]),
                        "note": "the extreme-tail max-sens point -- NOT deployed (small calib)"}
                else:
                    _p10 = max(_feas10, key=lambda r: r["sensitivity"])
                    print(f"[postproc v25.46] clinical target unmet -- deploying CONSTRAINED point "
                          f"(max sens @ FARh<={CLINICAL_TARGET['farh_max']:.0f}): "
                          f"trigger={_p10['trigger']:.3f} sens={_p10['sensitivity']:.2f} "
                          f"FARh={_p10['farh']:.1f}/h [clinical_target_unmet=True]", flush=True)
                    postproc.update({"trigger": _p10["trigger"]})
                    frozen_threshold = _p10["trigger"]
                if _deployed_small_calib:
                    postproc["max_epf1_deployed_via_small_calib"] = {
                        "trigger": float(_fb_early["trigger"]), "ep_f1": float(_fb_early["ep_f1"]),
                        "sensitivity": float(_fb_early["sensitivity"]), "farh": float(_fb_early["farh"])}
                else:
                    _fb = _fb_early if _fb_early is not None \
                        else _best_episode_f1_point(calib_labels, calib_probs_t2, calib_ids)
                    if _fb is not None:
                        postproc["max_epf1_diagnostic_DO_NOT_DEPLOY"] = {
                            "trigger": float(_fb["trigger"]), "ep_f1": float(_fb["ep_f1"]),
                            "sensitivity": float(_fb["sensitivity"]), "farh": float(_fb["farh"]),
                            "note": "diagnostic only -- NOT the deployed operating point"}
                        print(f"[postproc v25.51.3] report-only max-epF1 diagnostic "
                              f"(NOT deployed): trigger={_fb['trigger']:.3f} epF1={_fb['ep_f1']:.3f} "
                              f"sens={_fb['sensitivity']:.2f} FARh={_fb['farh']:.1f}/h", flush=True)
            else:
                print(f"[postproc v25.46] NO calib point meets FARh<={CLINICAL_TARGET['farh_max']:.0f} "
                      f"alone -- detector is not deployable on this fold "
                      f"[clinical_target_unmet=True]", flush=True)
                _fb = _best_episode_f1_point(calib_labels, calib_probs_t2, calib_ids)
                if _fb is not None:
                    print(f"[postproc v25.25] clinical target unmet (sens={_sens_c:.2f}, "
                          f"FARh={_farh_c:.1f}/h) -- deploying max-episode-F1 point on full "
                          f"calib: trigger={_fb['trigger']:.3f} epF1={_fb['ep_f1']:.3f} "
                          f"sens={_fb['sensitivity']:.2f} FARh={_fb['farh']:.1f}/h", flush=True)
                    postproc = dict(_fb)
                    frozen_threshold = _fb["trigger"]

    _chk_f = causal_event_postprocess(calib_probs_t2, calib_ids, trigger=frozen_threshold,
                                      sustain=postproc.get("sustain", frozen_threshold * 0.4),
                                      confirm_n=postproc.get("confirm_n", 3),
                                      cooldown_n=postproc.get("cooldown_n", 30),
                                      merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
    _tp_f = int(((_chk_f == 1) & (calib_labels == 1)).sum())
    _fn_f = int(((_chk_f == 0) & (calib_labels == 1)).sum())
    _sens_f = _safe_sensitivity(_tp_f, _fn_f)
    _farh_f = false_alarm_events_per_hour(calib_labels, calib_probs_t2, frozen_threshold, calib_ids,
                                          pred_binary=_chk_f)["farh_events_per_hour"]
    postproc["sensitivity"] = float(_sens_f)
    postproc["farh"] = float(_farh_f)
    print(f"[postproc v25.27] full-calib verified: trigger={frozen_threshold:.3f} "
          f"sens={_sens_f:.2f} FARh={_farh_f:.1f}/h", flush=True)

    joint_cert = None
    if _flag("NEUROMAMBA_JOINT_CERTIFY", "1"):
        try:
            joint_cert = joint_conformal_operating_points(
                calib_labels, calib_probs_t2, calib_ids, calib_subjects,
                sens_target=CLINICAL_TARGET["sensitivity_min"],
                farh_budget=CLINICAL_TARGET["farh_max"],
                alpha=float(get("NEUROMAMBA_CONFORMAL_ALPHA", "0.10")),
                confirm_n=int(postproc.get("confirm_n", 3)),
                cooldown_n=int(postproc.get("cooldown_n", 30)))
        except Exception as _jc:
            print(f"[joint-certify v25.44] failed (non-fatal): {_jc}", flush=True)
            traceback.print_exc()

    th_probs_gated = th_probs_t
    th_probs_gated, abst_frac, abst_pos_frac = _apply_abstention(th_probs_gated, thresh_seqs, labels=th_labels)

    pred = causal_event_postprocess(th_probs_gated, th_ids, trigger=frozen_threshold,
                                     sustain=postproc.get("sustain", frozen_threshold * 0.4),
                                     confirm_n=postproc.get("confirm_n", 3), cooldown_n=postproc.get("cooldown_n", 30),
                                     merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))

    final_ep = episode_f1(th_labels, th_probs_gated, frozen_threshold, th_ids, pred_binary=pred)
    final_event_f1 = event_level_f1(th_labels, pred, th_ids)
    final_farh = false_alarm_events_per_hour(th_labels, th_probs_gated, frozen_threshold, th_ids, pred_binary=pred)["farh_events_per_hour"]
    calib_split_report = report_calibration_per_split(calib_gated, calib_labels, th_gated_raw, th_labels, temperature)
    clin_ops = _clinical_op_points(pareto, th_labels, th_probs_gated, th_ids,
                                   threshold=frozen_threshold)
    clin_ops["coverage_farh"] = coverage_farh_curve(th_labels, th_probs_gated, th_ids,
                                                    frozen_threshold, postproc)
    print(f"[thresh-split] thr={frozen_threshold:.3f} frameF1={final_ep:.3f} eventF1={final_event_f1:.3f} FARh={final_farh:.2f}/h "
          f"walking_gate={'ON' if use_walking_gate else 'OFF'} (scored on the THRESH split, NOT val -- "
          f"the fold [VAL] line is the reportable operating point)", flush=True)
    if final_event_f1 == 0.0 and final_farh == 0.0:
        n_th = sum(len(s.labels) for s in thresh_seqs)
        print(f"[thresh-split v25.26] WARNING: the frozen threshold {frozen_threshold:.3f} fires nothing on "
              f"the THRESH split ({n_th} frames) -- with tiny calib/thresh splits the per-split "
              f"probability scale differs, so thresh-split diagnostics are unreliable. The fold "
              f"[VAL] line is the reportable operating point.", flush=True)
        try:
            cq = np.quantile(np.asarray(calib_gated), [0.5, 0.9])
            tq = np.quantile(np.asarray(th_gated_raw), [0.5, 0.9])
            print(f"[thresh-split v25.30] scale shift: calib p50/p90={cq[0]:.4f}/{cq[1]:.4f} "
                  f"vs thresh p50/p90={tq[0]:.4f}/{tq[1]:.4f} "
                  f"(thresh/calib p50 ratio={tq[0] / max(cq[0], 1e-9):.2f})", flush=True)
        except Exception as _e:
            print(f"[thresh-split v25.30] scale-shift diagnostic failed: {_e}", flush=True)
    print(f"[thresh-split v25.11] window-F1@frozenThr={clin_ops['window_f1']:.3f} (contest ceiling 0.73-0.81) | "
          f"best sens@FARh<=10: {clin_ops['sens_at_farh10']:.2f} | "
          f"best FARh@sens>=0.85: {clin_ops['farh_at_sens85']:.1f}/h", flush=True)
    print(f"[thresh-split v25.15] coverage->FARh curve: "
          + " | ".join(f"cov {r['coverage']:.2f}: sens {r['sensitivity']:.2f}, FARh {r['farh']:.1f}/h"
                       for r in clin_ops["coverage_farh"]), flush=True)

    subprobs_final, sublabels_final, subobserved_final = None, None, None
    res_final = res4_th  # v25.1-speed: identical pass (model unchanged since res4_th)
    per_subtype_thresholds = None
    kaggle_map = None
    if len(res_final) >= 4 and res_final[3] is not None:
        subprobs_final = np.asarray(res_final[3])
        sublabels_final, subobserved_final, sub_obs_final = concat_labels(thresh_seqs)
        n = min(len(sublabels_final), len(subprobs_final))
        subprobs_final, sublabels_final, subobserved_final = subprobs_final[:n], sublabels_final[:n], sub_obs_final[:n]
        subprobs_t = apply_temperature_subtype(subprobs_final, subtype_temps)
        kaggle_map = compute_kaggle_map(subprobs_t, sublabels_final, subobserved_final)
        per_subtype_thresholds = per_subtype_threshold_search(sublabels_final, subprobs_t, th_ids[:n])
        print(f"[internal] subtype mAP={json.dumps(kaggle_map, default=str)} "
              f"[scored on the selection split; use the fold [VAL] mAP for the honest number]", flush=True)

    tripod = build_tripod_report(th_labels, pred, th_ids, probs=th_probs_gated,
                                  subject_ids=np.array([s.subject_id for s in thresh_seqs for _ in range(len(s.labels))]))

    _fog_prior_train = float(sum(int(((s.labels > 0.5) & (np.asarray(s.observed) > 0.5)).sum())
                                 for s in train_seqs))
    _fog_prior_train = _fog_prior_train / max(1, sum(int((np.asarray(s.observed) > 0.5).sum())
                                                     for s in train_seqs))
    external_report = None
    if external_seqs:
        external_report = _evaluate_external(
            [model], external_seqs, device,
            temperature=temperature, frozen_threshold=frozen_threshold,
            postproc=postproc, use_walking_gate=use_walking_gate,
            window_roll=window_roll, seed=seed,
            train_subjects=set(s.subject_id for s in list(train_seqs) + list(calib_seqs) + list(thresh_seqs)),
            source_fog_prior=_fog_prior_train,
            score_anchor_ref=(quantile_summary(calib_probs_t2)
                              if _flag("NEUROMAMBA_SCORE_ANCHOR", "0") else None))

    results = {
        "frozen_threshold": frozen_threshold, "temperature": temperature, "subtype_temps": subtype_temps,
        "postproc": postproc, "per_subtype_thresholds": per_subtype_thresholds,
        "pareto_frontier": pareto, "model_type": model_type,
        "final_episode_f1": final_ep, "final_event_f1": final_event_f1, "final_farh": final_farh,
        "ece_before": ece_before_bin, "ece_after": ece_after_bin, "ece_calib_vs_thresh": calib_split_report,
        "tripod_internal": tripod, "tripod_external_validation": external_report,
        "best_variant": best_name, "best_calib_subtype_map": best_vscore,
        "best_epoch": int(pareto_epoch if (best_name == "pareto" and pareto_epoch >= 0)
                          else sel_best_epoch),
        "pareto_epoch": int(pareto_epoch),
        "architecture": model.get_architecture_manifest(), "kaggle_map": kaggle_map,
        "subprobs": subprobs_final, "sublabels": sublabels_final, "subobserved": subobserved_final,
        "calib_sub": calib_sub,
        "abstention_frac": abst_frac, "abstention_pos_frac": abst_pos_frac,
        "clinical_op_points": clin_ops,
        "subtype_prior_train": subtype_prior_train,  # v25.44: logit-adjust priors
        "logit_adjust_tau": float(get("NEUROMAMBA_LOGIT_ADJUST_TAU", "0")),
        "joint_certified_ops": joint_cert,  # v25.44: certified (sens, FARh) envelope
        "sensitivity": final_event_f1,
        "use_walking_gate": use_walking_gate, "window_roll": window_roll,
        "training_completed_past_early_stop_floor": bool(last_epoch_run >= es_min - 1),
        "seed_finalized": bool(last_epoch_run >= es_min - 1) or bool(_train_complete)
                          or bool(epochs < es_min),
        "seed_finalized_epoch": int(last_epoch_run),
        "fog_prior_train": _fog_prior_train,
        "score_anchor_ref_q": quantile_summary(calib_probs_t2),
    }
    if _tv_traj is not None:
        results["trajectory_eval"] = _tv_traj
    _c_alpha = float(get("NEUROMAMBA_CONFORMAL_ALPHA", "0.10"))
    _ct, _n_neg = conformal_trigger(calib_probs_t2, calib_labels, calib_ids, alpha=_c_alpha)
    if _ct is not None:
        results["conformal_trigger"] = _ct
        results["conformal_alpha"] = _c_alpha
        _pred_c = causal_event_postprocess(th_probs_t, th_ids, trigger=_ct)
        _farh_c = false_alarm_events_per_hour(th_labels, th_probs_t, _ct, th_ids,
                                              pred_binary=_pred_c)["farh_events_per_hour"]
        _ep_c = event_level_metrics(th_labels, _pred_c, th_ids)
        print(f"[conformal v25.31] alpha={_c_alpha:.2f} trigger={_ct:.3f} "
              f"(fit on {_n_neg} calib neg-windows) -> THRESH alarm FARh={_farh_c:.2f}/h "
              f"sens={_ep_c['sensitivity']:.2f} (guarantee: neg-window alarm rate "
              f"<= {_c_alpha:.2f} + 1/{_n_neg + 1})", flush=True)
    return model, results

def _gate_mode():
    """v25.16g: walking-gate application mode.
    'multiply' (default): prob *= floor + (1-floor)*walking_prob -- smooth
        attenuation (the historical behavior; distorts probability scale on
        shifted cohorts, seen as external ECE 0.5-0.9).
    'binary': prob kept intact when walking_prob >= NEUROMAMBA_GATE_BINARY_TH,
        else 0 -- the FM-FoG / LIFT-PD style activity trigger. Preserves the
        calibration of the kept frames (the model's probability stays a
        probability), at the cost of hard suppression during gait-initiation
        transitions (a StartHesitation risk)."""
    mode = os.environ.get("NEUROMAMBA_GATE_MODE", "multiply").strip().lower()
    if mode not in ("multiply", "binary"):
        raise ValueError(f"NEUROMAMBA_GATE_MODE must be multiply or binary, got {mode}")
    return mode

_GATE_CFG_ECHOED = False  # v25.57 (C5): one-time EFFECTIVE gate-config echo latch

def _gate_weight(wp):
    """v25.9: walking-gate attenuation floor (env NEUROMAMBA_GATE_FLOOR, default
    0.1). Lower floor = stronger suppression of non-walking false alarms (the
    main FARh lever when the clinical target is missed)."""
    floor = float(os.environ.get("NEUROMAMBA_GATE_FLOOR", "0.1"))
    return floor + (1.0 - floor) * wp

def apply_gate_flat(probs, gate_w, sh_probs=None):
    """v25.47 (review 3, solution 4): ONE gate-application point, mode-aware and
    SH-conditional. With NEUROMAMBA_GATE_SH_COND=1, frames the SH head scores
    >= NEUROMAMBA_GATE_SH_ETA (default 0.2) are EXEMPT from the walking gate --
    a global walking prior is anti-StartHesitation by construction (SH is
    failure to initiate stepping). Identical to plain multiply otherwise."""
    gw = np.asarray(gate_w, dtype=np.float64)
    gated = probs.copy()
    if sh_probs is not None and _flag("NEUROMAMBA_GATE_SH_COND", "0"):
        shp = np.asarray(sh_probs, dtype=np.float64)
        if shp.ndim == 2:
            shp = shp[:, 0]
        elif shp.ndim != 1:
            shp = shp.reshape(-1)
        eta = float(os.environ.get("NEUROMAMBA_GATE_SH_ETA", "0.2"))
        ex = shp >= eta
        n = min(len(gated), len(ex))
        mask = ~ex[:n]
        gated[:n][mask] = gated[:n][mask] * gw[:n][mask]
    else:
        gated *= gw
    return gated

def _apply_walking_gate_inference(model, seqs, probs, device, window_roll=False, gate_probs=None,
                                sh_probs=None):
    """v25-FIX-6: apply the walking gate to FoG probs to suppress FARh during
    sitting/standing/voluntary stops. The gate is jointly trained with the
    main model. v25.1-speed: pass gate_probs collected from the SAME inference
    pass to skip re-running the model (was a full extra pass per split). The
    fallback keeps the original behavior (full-sequence gate pass).
    v25.16g: NEUROMAMBA_GATE_MODE=binary switches to a hard activity trigger
    (kept probs stay calibrated).
    v25.57 (C5): one-time EFFECTIVE-config echo -- the project's P2 lesson is
    that a banner that can lie is worse than no banner; this prints what the
    gate will ACTUALLY do on the first application (floor/SH_COND/eta)."""
    global _GATE_CFG_ECHOED
    if not _GATE_CFG_ECHOED:
        _GATE_CFG_ECHOED = True
        try:
            print(f"[gate v25.57] EFFECTIVE: mode={_gate_mode()} "
                  f"floor={float(os.environ.get('NEUROMAMBA_GATE_FLOOR', '0.1')):.3f} "
                  f"SH_COND={bool(_flag('NEUROMAMBA_GATE_SH_COND', '0'))} "
                  f"eta={float(os.environ.get('NEUROMAMBA_GATE_SH_ETA', '0.2')):.2f} "
                  f"sh_probs={'provided' if sh_probs is not None else 'NOT provided (exemption inert)'}", flush=True)
        except Exception:
            pass
    if gate_probs is None:
        res = run_inference_cached(model, seqs, device, collect_gate=True, window_roll=False)
        gate_probs = res[-1]
    gated = probs.copy()
    mode = _gate_mode()
    bin_th = float(os.environ.get("NEUROMAMBA_GATE_BINARY_TH", "0.5"))
    idx = 0
    for s in seqs:
        L = len(s.labels)
        gp = gate_probs[idx:idx + L]
        if mode == "binary":
            gated[idx:idx + L] = np.where(gp >= bin_th, probs[idx:idx + L], 0.0)
        else:
            gate_w = _gate_weight(gp)
            _shp = None
            if sh_probs is not None:
                _shp = np.asarray(sh_probs)[idx:idx + L]
            gated[idx:idx + L] = apply_gate_flat(gated[idx:idx + L], gate_w, _shp)
        idx += L
    return gated

print("[train] BASELINE_AND_ENSEMBLE_V25_LOADED", flush=True)

def _baseline_predict(model, seqs, device):
    """v25.11: full-length fog + subtype probability inference for the frozen
    ResNet1D baseline (no window-roll; nearest-upsampled logits, matching the
    published 8th-place configuration)."""
    model.eval()
    all_fog, all_sub = [], []
    with torch.inference_mode():
        for s in seqs:
            T_s = len(s.labels)
            x = torch.from_numpy(s.signal[None].transpose(0, 2, 1)).to(device)
            fog_logits, sub_logits, _ = model(x)
            fog_up = F.interpolate(fog_logits.unsqueeze(1), size=T_s, mode="nearest").squeeze(1)
            sub_up = F.interpolate(sub_logits.permute(0, 2, 1), size=T_s, mode="nearest").permute(0, 2, 1)
            all_fog.append(torch.sigmoid(fog_up)[0].cpu().numpy())
            all_sub.append(torch.sigmoid(sub_up)[0].cpu().numpy())
    return np.concatenate(all_fog, axis=0), np.concatenate(all_sub, axis=0)

def _tune_hetero_blend(baseline_model, results, calib_seqs, thresh_seqs, device,
                       grid=None):
    """v25.11: inner-fold blend-weight search between the NeuroMamba seed
    ensemble and the frozen ResNet1D baseline. Scored on calib+thresh (the same
    selection splits used for calibSubtypeMAP and threshold tuning) so the outer
    val fold is touched only once, with the chosen weight. Returns w such that
    blended = w * main + (1 - w) * baseline."""
    if grid is None:
        grid = np.round(np.linspace(0.0, 1.0, 11), 2)
    cp, cl, co = results["calib_sub"]
    tp, tl, to = results["subprobs"], results["sublabels"], results["subobserved"]
    main_p = np.concatenate([cp, tp[:len(tl)]], axis=0)
    main_l = np.concatenate([cl, tl], axis=0)
    main_o = np.concatenate([co, to[:len(tl)]], axis=0)
    bf, bp = _baseline_predict(baseline_model, list(calib_seqs) + list(thresh_seqs), device)
    bp = bp[:len(main_p)]
    n = min(len(main_p), len(bp))
    main_p, bp, main_l, main_o = main_p[:n], bp[:n], main_l[:n], main_o[:n]
    best_w, best_map = 1.0, -1.0
    for w in grid:
        m = compute_kaggle_map(w * main_p + (1.0 - w) * bp, main_l, main_o)
        print(f"[hetero-blend v25.11] w={w:.2f} calib+thresh mAP={m['mAP']:.4f}", flush=True)
        if m["mAP"] > best_map:
            best_w, best_map = float(w), m["mAP"]
    print(f"[hetero-blend v25.11] selected w={best_w:.2f} (calib+thresh mAP {best_map:.4f})", flush=True)
    return best_w

def run_resnet1d_baseline_fold(train_seqs_raw, val_seqs_raw, device, epochs, lr, batch_size, fold_idx,
                               repeat=0):
    """Frozen comparator baseline (no augmentation, no ASL, no curriculum).
    v25.42.6: FIXED seed per fold -- the baseline was previously unseeded
    (global RNG), so the comparator varied run to run and machine to machine,
    making gate_2 (beats baseline) non-reproducible across environments.
    v25.42.7 (Phase 1): dedicated baseline seed + FULL RNG reset (python,
    numpy, torch CPU, torch CUDA) before training, and a baseline training
    identity saved alongside the OOF artifact so evaluation-only reruns can
    verify they reuse the same frozen comparator.
    v25.42.8: the seed includes the REPEAT (base + repeat*10000 + fold), so
    repeated-CV baselines are distinct per repeat and provenance is explicit."""
    base_seed = int(os.environ.get("NEUROMAMBA_BASELINE_SEED", "7000"))
    seed = base_seed + repeat * 10000 + fold_idx
    print(f"[baseline v25.42.8] comparator seed={seed} "
          f"(base={base_seed}+repeat*10000+fold); full RNG reset before training",
          flush=True)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _base_tid = training_identity(
        model_type="resnet1d", lr=lr, batch_size=batch_size, epochs=epochs,
        seed=seed, n_seeds=1, fold_idx=fold_idx, family="baseline",
        use_rare_split=True, extra_env={"baseline_seed": str(seed),
                                        "baseline_repeat": str(repeat)})
    in_ch = train_seqs_raw[0].signal.shape[1]
    model = ResNet1DBaseline(in_channels=in_ch, use_walking_gate=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    subtype_pos = np.zeros(3, dtype=np.float64)
    subtype_neg = np.zeros(3, dtype=np.float64)
    for s in train_seqs_raw:
        valid = s.train_mask.astype(bool)
        for k in range(3):
            obs_k = s.subtype_observed[:, k] & valid
            yy = s.y_subtype[obs_k, k]
            subtype_pos[k] += yy.sum()
            subtype_neg[k] += len(yy) - yy.sum()
    spw = torch.tensor(np.clip(subtype_neg / np.maximum(subtype_pos, 1.0), 1.0, 50.0),
                        dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        for x_np, y_np, m_np, w_np, y_sub_np, y_sub_obs_np, y_walk_np, y_walk_w_np, med_np, _, _, _ptags_base, _ in training_batches(train_seqs_raw, batch_size, epoch=epoch,
                                                                                                                                     sampler_cls=BalancedSequenceSampler):
            x = torch.from_numpy(x_np).transpose(1, 2).to(device)
            y = torch.from_numpy(y_np).to(device)
            m = torch.from_numpy(m_np).to(device)
            T_in = y.shape[1]
            fog_logits, subtype_logits, _ = model(x)
            fog_logits = F.interpolate(fog_logits.unsqueeze(1), size=T_in, mode="nearest").squeeze(1)
            fog_loss = (F.binary_cross_entropy_with_logits(fog_logits, y, reduction="none") * m).sum() / m.sum().clamp(min=1.0)
            sub_up = F.interpolate(subtype_logits.permute(0, 2, 1), size=T_in, mode="nearest").permute(0, 2, 1)
            yt = torch.from_numpy(y_sub_np[:, :T_in]).to(device).reshape(-1, 3)
            mt = torch.from_numpy(y_sub_obs_np[:, :T_in].astype(np.float32)).to(device).reshape(-1, 3)
            loss = fog_loss
            if float((yt * mt).sum()) > 0:
                sub_bce = F.binary_cross_entropy_with_logits(sub_up.reshape(-1, 3), yt, reduction="none", pos_weight=spw)
                sub_loss = (sub_bce * mt).sum() / mt.sum().clamp(min=1.0)
                loss = 0.25 * fog_loss + 1.0 * sub_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"[baseline] fold={fold_idx} epoch={epoch}", flush=True)
    model.eval()
    fog_probs, sub_probs = _baseline_predict(model, val_seqs_raw, device)
    sub_labels, sub_observed, sub_sub_obs = concat_labels(val_seqs_raw)
    n = min(len(sub_probs), len(sub_labels))
    kaggle_map = compute_kaggle_map(sub_probs[:n], sub_labels[:n], sub_sub_obs[:n])
    print(f"[baseline] fold={fold_idx} resnet1d kaggle_map={json.dumps(kaggle_map, default=str)}", flush=True)
    return {"fold": fold_idx, "kaggle_map": kaggle_map, "sub_probs": sub_probs[:n],
            "sub_labels": sub_labels[:n], "sub_observed": sub_sub_obs[:n],
            "fog_probs": fog_probs[:n], "model": model,
            "training_identity": _base_tid, "seed": seed}

def _build_notype_pseudo_seqs(models, notype_seqs, device, window_roll=True):
    """v25.12: notype SUBTYPE pseudo-labeling -- the verified 2nd-place recipe
    (TakoiHirokazu public code, mAP 0.451): ensemble-average the subtype heads
    over the notype recordings, then on Event==1 frames take one-hot(argmax)
    as the pseudo subtype (observed=True, label_weight=confidence); Event==0
    frames become HARD zeros (the None class). The fog head still sees the
    original Event targets. Annotated negatives are never flipped."""
    seqs_out = []
    for s in notype_seqs:
        acc = None
        for m in models:
            rv = run_inference_cached(m, [s], device, collect_subtype=True, window_roll=window_roll)
            sub = np.asarray(rv[3])[:len(s.labels)] if rv[3] is not None else None
            if sub is None:
                continue
            acc = sub if acc is None else acc + sub
        if acc is None:
            continue
        probs = acc / max(1, len(models))
        event = s.y_subtype[:, 0] > 0.5
        conf = probs.max(axis=1)
        hard = probs.argmax(axis=1)
        y_sub = np.zeros_like(s.y_subtype)
        y_sub[event, hard[event]] = 1.0
        so = np.ones((len(s.labels), 3), dtype=bool)
        lw = np.where(event, np.clip(conf, 0.1, 1.0), 1.0).astype(np.float32)
        seqs_out.append(FoGSequence(signal=s.signal, y_subtype=y_sub, subject_id=s.subject_id,
                                    record_id=s.record_id, source_type="notype_pseudo",
                                    observed=s.observed, label_weight=lw, subtype_observed=so,
                                    med_state=s.med_state, fs_hz=s.fs_hz,
                                    unit_transform=s.unit_transform, strides=s.strides,
                                    position_tag=getattr(s, "position_tag", "")))
    if seqs_out:
        n_sh = int(sum((q.y_subtype[:, 0] > 0.5).sum() for q in seqs_out))
        n_tu = int(sum((q.y_subtype[:, 1] > 0.5).sum() for q in seqs_out))
        n_wa = int(sum((q.y_subtype[:, 2] > 0.5).sum() for q in seqs_out))
        print(f"[pseudo v25.12] built {len(seqs_out)} notype_pseudo sequences from "
              f"{len(models)} teacher models: SH={n_sh} Turn={n_tu} Walk={n_wa} pseudo frames", flush=True)
    return seqs_out

def _slice_adapt_seqs(seqs, minutes=50.0):
    """v25.14c: take the FIRST `minutes` of labeled data from a cohort (the KU
    Leuven adaptation paradigm: 50 min of target-site annotations). Returns
    sliced FoGSequences (chronological prefix of each recording until the
    minute budget is exhausted)."""
    budget = int(minutes * 60 * 100)  # frames at the 100 Hz canonical rate
    out, used = [], 0
    for s in seqs:
        if used >= budget:
            break
        take = min(len(s.labels), budget - used)
        if take < 100:
            continue
        out.append(FoGSequence(signal=s.signal[:take], y_subtype=s.y_subtype[:take],
                               subject_id=s.subject_id, record_id=s.record_id + "_adapt",
                               source_type=s.source_type, observed=s.observed[:take],
                               label_weight=s.label_weight[:take],
                               subtype_observed=s.subtype_observed[:take],
                               med_state=s.med_state, fs_hz=s.fs_hz,
                               unit_transform=s.unit_transform, strides=s.strides,
                               position_tag=getattr(s, "position_tag", "")))
        used += take
    return out

def _finetune_external(model, adapt_seqs, device, epochs=3, lr=1e-4, seed=0):
    """v25.14c: low-LR fine-tune on a tiny target-site slice (fog head + walking
    gate only -- external cohorts have subtype_observed=zeros so the subtype
    loss contributes nothing). The model passed in is expected to be a CLONE;
    the frozen model used for the primary evaluation must never be mutated."""
    torch.manual_seed(seed)
    _adapt_prefixes = ("fog_head", "tok_fog", "walking_gate", "hazard_head")  # v28bp: hazard head adapts too
    for _n, _p in model.named_parameters():
        _p.requires_grad = any(_n.startswith(pre) for pre in _adapt_prefixes)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("_finetune_external: no head parameters matched "
                           "('fog_head'/'tok_fog'/'walking_gate')")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        model.train()
        for x_np, y_np, m_np, w_np, y_sub_np, y_sub_obs_np, y_walk_np, y_walk_w_np, med_np, _ft_subj, _ft_src, _ft_ptags, _ in training_batches(
                adapt_seqs, batch_size=8, epoch=epoch, use_aug=False):
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            m = torch.from_numpy(m_np).to(device)
            w = torch.from_numpy(w_np).to(device)
            med_t = torch.from_numpy(med_np).to(device) if getattr(model, "use_med_film", False) else None
            _ft_prompt = (torch.from_numpy(_b3_prompt_matrix(_ft_src, _ft_ptags)).to(device)
                          if getattr(model, "use_sensor_film", False) else None)
            fog_logits, subtype_logits, aux = (model(x, med=med_t, prompt=_ft_prompt)
                                               if med_t is not None else model(x, prompt=_ft_prompt))
            if fog_logits.shape[1] != y.shape[1]:
                fog_logits = F.interpolate(fog_logits.unsqueeze(1), size=y.shape[1],
                                           mode="nearest").squeeze(1)
            fog_loss, _ = compute_loss(fog_logits, y, m, w=w)
            loss = fog_loss
            if "walking_prob" in aux:
                L_out = fog_logits.shape[1]
                wg = walking_gate_loss(aux["walking_prob"][:, :L_out].reshape(-1),
                                       torch.from_numpy(y_walk_np[:, :L_out]).to(device).reshape(-1),
                                       m[:, :L_out].reshape(-1),
                                       w=torch.from_numpy(y_walk_w_np[:, :L_out]).to(device).reshape(-1))
                loss = loss + 0.3 * wg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model

def _evaluate_external(models, external_seqs, device, temperature, frozen_threshold,
                       postproc, use_walking_gate=True, window_roll=True, seed=0,
                       train_subjects=None, conformal_trigger=None,
                       source_fog_prior=None, score_anchor_ref=None):
    """v25.15c: FROZEN external evaluation of an ensemble (or single model).
    Fog and gate probabilities are blended by simple averaging across models
    (subtypes are not needed -- external cohorts are binary-only). Includes the
    subject-overlap screen and the OPTIONAL adaptation experiment (a deep copy
    of the LAST model). Must run ONCE per fold on the final blended system."""
    models = list(models) if models else []
    if not models or not external_seqs:
        return None
    ext_min_ef1 = float(get("NEUROMAMBA_EXTERNAL_MIN_EVENT_F1", 0.3))
    ext_max_farh = float(get("NEUROMAMBA_EXTERNAL_MAX_FARH", 30.0))
    p_acc, g_acc = [], []
    r = None
    s_acc = []
    for m in models:
        r = run_inference_cached(m, external_seqs, device, collect_subtype=True,
                                 collect_gate=use_walking_gate, window_roll=window_roll)
        p_acc.append(r[0])
        s_acc.append(r[3] if len(r) >= 4 and r[3] is not None else None)
        g_acc.append(r[4] if len(r) >= 5 and r[4] is not None else None)
    ext_probs = np.mean(np.stack(p_acc, axis=0), axis=0)
    ext_labels, ext_ids = r[1], r[2]
    ext_probs_ungated = ext_probs.copy()
    if use_walking_gate and all(g is not None for g in g_acc):
        gate_blend = np.mean(np.stack(g_acc, axis=0), axis=0)
        _ext_shp = None
        _s_ok = [s for s in s_acc if s is not None]
        if _s_ok and len(_s_ok) == len(s_acc):
            _ext_shp = np.mean(np.stack(_s_ok, axis=0), axis=0)
        ext_probs = _apply_walking_gate_inference(models[0], external_seqs, ext_probs, device,
                                                  window_roll=window_roll, gate_probs=gate_blend,
                                                  sh_probs=_ext_shp)
    ext_probs = apply_temperature(ext_probs, temperature)
    ext_probs_ungated_t = apply_temperature(ext_probs_ungated, temperature)
    ext_probs, ext_abst_frac, ext_abst_pos = _apply_abstention(ext_probs, external_seqs, labels=ext_labels)
    ext_pred = causal_event_postprocess(ext_probs, ext_ids, trigger=frozen_threshold,
                                        sustain=postproc.get("sustain", frozen_threshold * 0.4),
                                        confirm_n=postproc.get("confirm_n", 3),
                                        cooldown_n=postproc.get("cooldown_n", 30),
                                        merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
    ext_sources = np.array([s.source_type for s in external_seqs for _ in range(len(s.labels))])
    ext_subjects = np.array([s.subject_id for s in external_seqs for _ in range(len(s.labels))])
    if train_subjects is None:
        train_subjects = set()
    ext_subjects_set = set(s.subject_id for s in external_seqs)
    overlap = train_subjects & ext_subjects_set
    print(f"[external v25.11] subject-overlap screen: {len(overlap)}/{len(ext_subjects_set)} "
          f"external subjects also in train (expected 0)", flush=True)
    cohorts = {}
    for src in np.unique(ext_sources):
        m = ext_sources == src
        ef1 = event_level_f1(ext_labels[m], ext_pred[m], ext_ids[m])
        farh = false_alarm_events_per_hour(ext_labels[m], ext_probs[m], frozen_threshold, ext_ids[m],
                                           pred_binary=ext_pred[m])["farh_events_per_hour"]
        _pred_u = causal_event_postprocess(
            ext_probs_ungated_t[m], ext_ids[m], trigger=frozen_threshold,
            sustain=postproc.get("sustain", frozen_threshold * 0.4),
            confirm_n=postproc.get("confirm_n", 3), cooldown_n=postproc.get("cooldown_n", 30),
            merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
        _ef1_u = float(event_level_f1(ext_labels[m], _pred_u, ext_ids[m]))
        _farh_u = float(false_alarm_events_per_hour(
            ext_labels[m], ext_probs_ungated_t[m], frozen_threshold, ext_ids[m],
            pred_binary=_pred_u)["farh_events_per_hour"])
        _pass_g = bool(ef1 >= ext_min_ef1 and farh <= ext_max_farh)
        _pass_u = bool(_ef1_u >= ext_min_ef1 and _farh_u <= ext_max_farh)
        cohorts[str(src)] = {
            "event_f1": float(ef1),
            "episode_f1_frame": episode_f1(ext_labels[m], ext_probs[m], frozen_threshold, ext_ids[m], pred_binary=ext_pred[m]),
            "farh": float(farh),
            "event_f1_ungated": _ef1_u,
            "farh_ungated": _farh_u,
            "ece": compute_ece(ext_probs[m], ext_labels[m]),
            "ece_ungated": compute_ece(ext_probs_ungated[m], ext_labels[m]),
            "window_f1": window_level_f1(ext_labels[m], ext_probs[m], ext_ids[m],
                                         threshold=frozen_threshold)["window_f1"],
            "window_metrics": window_level_metrics(ext_labels[m], ext_probs[m], ext_ids[m],
                                                   threshold=frozen_threshold),
            "event_metrics": event_level_metrics(ext_labels[m], ext_pred[m], ext_ids[m]),
            "window_metrics_ungated": window_level_metrics(ext_labels[m], ext_probs_ungated_t[m], ext_ids[m],
                                                           threshold=frozen_threshold),
            "event_metrics_ungated": event_level_metrics(ext_labels[m], _pred_u, ext_ids[m]),
            "n_sequences": int(np.unique(ext_ids[m]).size),
            "n_subjects": int(np.unique(ext_subjects[m]).size),
            "n_frames": int(m.sum()),
            "icc": icc_report(ext_labels[m], ext_pred[m], ext_ids[m]),
            "per_subject_farh": per_subject_farh_distribution(ext_labels[m], ext_pred[m], ext_subjects[m]),
            "passed": _pass_g or _pass_u,
            "pass_mode": "gated" if _pass_g else ("ungated" if _pass_u else "none"),
        }
        print(f"[external v25.54] {src}: gated eF1={ef1:.3f} FARh={farh:.1f}/h | "
              f"ungated eF1={_ef1_u:.3f} FARh={_farh_u:.1f}/h | "
              f"pass={cohorts[str(src)]['passed']} mode={cohorts[str(src)]['pass_mode']}", flush=True)
    n_passed = sum(1 for c in cohorts.values() if c["passed"])
    if "fogstar" in cohorts and _FOGSTAR_SUBJECT_META:
        m = ext_sources == "fogstar"
        subj_f = ext_subjects[m]
        meta_ids = {f"fogstar_{k}": v for k, v in _FOGSTAR_SUBJECT_META.items()}
        known = np.array([s in meta_ids for s in subj_f])
        if known.sum() >= 100:
            meta = [meta_ids[s] for s in subj_f[known]]
            sex = np.array([0 if d["gender"] == "M" else 1 for d in meta], dtype=np.int64)
            age = np.array([d["age"] for d in meta], dtype=np.float64)
            hy = np.array([d["h_y"] for d in meta], dtype=np.float64)
            fair = {"gender_M_F": fairness_report(ext_pred[m][known], ext_labels[m][known], sex)}
            if np.nanmedian(age) > 0:
                fair["age_median_split"] = fairness_report(
                    ext_pred[m][known], ext_labels[m][known], (age > np.nanmedian(age)).astype(np.int64))
            if np.isfinite(np.nanmedian(hy)):
                fair["h_y_median_split"] = fairness_report(
                    ext_pred[m][known], ext_labels[m][known], (hy > np.nanmedian(hy)).astype(np.int64))
            cohorts["fogstar"]["fairness"] = fair
            print(f"[fairness v25.16] fogstar gender DPR={fair['gender_M_F']['dpr']:.3f} "
                  f"EOR={fair['gender_M_F']['eor']:.3f} (0.8 four-fifths threshold; "
                  f"all published models fail it -- report, don't assert)", flush=True)
    external_report = {
        "cohorts": cohorts,
        "event_f1": float(event_level_f1(ext_labels, ext_pred, ext_ids)),
        "farh": false_alarm_events_per_hour(ext_labels, ext_probs, frozen_threshold, ext_ids, pred_binary=ext_pred),
        "window_f1": window_level_f1(ext_labels, ext_probs, ext_ids,
                                     threshold=frozen_threshold)["window_f1"],
        "n_sequences": len(external_seqs),
        "n_subjects": int(np.unique(ext_subjects).size),
        "passed": bool(n_passed >= 2),
        "criteria": {"min_event_f1": ext_min_ef1, "max_farh": ext_max_farh},
        "abstention_frac": ext_abst_frac, "abstention_pos_frac": ext_abst_pos,
        "train_subject_overlap": int(len(overlap)),
        "evaluation_rev": EVALUATION_REV,
    }
    print(f"[external] criteria={external_report['criteria']} passed={external_report['passed']}", flush=True)
    for src, c in cohorts.items():
        m = ext_sources == src
        icc_s = c.get("icc") or {}
        icc_txt = (f"ICC(%TF)={icc_s.get('icc_percent_time_frozen', 0):.3f} "
                   f"ICC(#ep)={icc_s.get('icc_n_episodes', 0):.3f} "
                   f"ICC(dur)={icc_s.get('icc_duration_s', 0):.3f}"
                   if icc_s.get("icc_percent_time_frozen") is not None else "ICC=n/a")
        print(f"[external] {src}: eventF1={c['event_f1']:.3f} FARh={c['farh']:.2f}/h "
              f"ECE={c['ece']:.4f} ECE_ungated={c['ece_ungated']:.4f} n_subj={c['n_subjects']} "
              f"per-subject FARh p95={c['per_subject_farh']['p95']:.2f} "
              f"{icc_txt} passed={c['passed']}", flush=True)
        wmu = c.get("window_metrics_ungated") or {}
        emu = c.get("event_metrics_ungated") or {}
        if c.get("window_metrics") and wmu:
            print(f"[external v25.27] {src} gate cost: windowF1 {c['window_metrics']['f1']:.3f} -> "
                  f"ungated {wmu.get('f1', float('nan')):.3f}; eventF1 {c['event_f1']:.3f} -> "
                  f"ungated {emu.get('f1', float('nan')):.3f}; "
                  f"AUC {c['window_metrics'].get('auc', float('nan')):.3f} -> "
                  f"ungated {wmu.get('auc', float('nan')):.3f}", flush=True)
        if conformal_trigger is not None:
            try:
                _pc = causal_event_postprocess(ext_probs[m], ext_ids[m], trigger=conformal_trigger)
                _fc = false_alarm_events_per_hour(ext_labels[m], ext_probs[m], conformal_trigger,
                                                  ext_ids[m], pred_binary=_pc)["farh_events_per_hour"]
                _ec = event_level_metrics(ext_labels[m], _pc, ext_ids[m])
                c["conformal_farh"] = float(_fc)
                c["conformal_sens"] = float(_ec["sensitivity"])
                print(f"[external conformal v25.31] {src}: FARh={_fc:.2f}/h "
                      f"sens={_ec['sensitivity']:.2f} eventF1={_ec['f1']:.2f} "
                      f"at conformal trigger={conformal_trigger:.3f}", flush=True)
            except Exception as _e:
                print(f"[external conformal v25.31] {src} failed: {_e}", flush=True)
        if source_fog_prior is not None and _flag("NEUROMAMBA_MLLS_CORRECT", "0"):
            try:
                _q_raw = estimate_binary_prior_em(ext_probs[m], source_fog_prior)
                _pc_delta = float(get("NEUROMAMBA_PRIOR_CLAMP_DELTA", "0.15"))
                _q = float(np.clip(_q_raw, source_fog_prior - _pc_delta,
                                   source_fog_prior + _pc_delta))
                if abs(_q - _q_raw) > 1e-6:
                    print(f"[external mlls v25.59g] {src}: EM prior {_q_raw:.4f} CLAMPED to "
                          f"{_q:.4f} (source {source_fog_prior:.4f} +/- {_pc_delta:.2f})", flush=True)
                _pc = correct_probs_binary(ext_probs[m], source_fog_prior, _q)
                _pred_mlls = causal_event_postprocess(_pc, ext_ids[m], trigger=frozen_threshold)
                _farh_m = false_alarm_events_per_hour(ext_labels[m], _pc, frozen_threshold,
                                                      ext_ids[m], pred_binary=_pred_mlls)["farh_events_per_hour"]
                _em_m = event_level_metrics(ext_labels[m], _pred_mlls, ext_ids[m])
                c["mlls_target_prior"] = float(_q)
                c["mlls_farh"] = float(_farh_m)
                c["mlls_sens"] = float(_em_m["sensitivity"])
                print(f"[external mlls v25.31] {src}: est. fog prior={_q:.4f} "
                      f"(source {source_fog_prior:.4f}) -> FARh={_farh_m:.2f}/h "
                      f"sens={_em_m['sensitivity']:.2f} eventF1={_em_m['f1']:.2f} "
                      f"at frozen thr={frozen_threshold:.3f}", flush=True)
                if _flag("NEUROMAMBA_EXTERNAL_PRIOR_MATCH", "0"):
                    c["event_f1_frozen"] = c.get("event_f1")
                    c["farh_frozen"] = c.get("farh")
                    c["event_f1"] = float(_em_m["f1"]) if np.isfinite(_em_m["f1"]) else float("nan")
                    c["farh"] = float(_farh_m)
                    c["ece"] = float(compute_ece(_pc, ext_labels[m]))
                    c["passed"] = bool(np.isfinite(_em_m["f1"]) and _em_m["f1"] >= ext_min_ef1
                                       and np.isfinite(_farh_m) and _farh_m <= ext_max_farh)
                    c["pass_mode"] = "prior_matched"
                    print(f"[external v25.58] {src}: PRIOR-MATCHED DEPLOYED "
                          f"(label-free) -> eF1={c['event_f1']:.3f} FARh={c['farh']:.2f}/h "
                          f"pass={c['passed']} mode={c['pass_mode']}", flush=True)
            except Exception as _e:
                print(f"[external mlls v25.31] {src} failed: {_e}", flush=True)
        if score_anchor_ref is not None:
            try:
                _pa, _ok = anchor_scores_to_reference(ext_probs[m], score_anchor_ref)
                if _ok:
                    _c_seqs = [s for s in external_seqs if s.source_type == src]
                    _pa, _, _ = _apply_abstention(_pa, _c_seqs, labels=ext_labels[m])
                    _pred_a = causal_event_postprocess(
                        _pa, ext_ids[m], trigger=frozen_threshold,
                        sustain=postproc.get("sustain", frozen_threshold * 0.4),
                        confirm_n=postproc.get("confirm_n", 3),
                        cooldown_n=postproc.get("cooldown_n", 30),
                        merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
                    _farh_a = false_alarm_events_per_hour(ext_labels[m], _pa, frozen_threshold,
                                                          ext_ids[m], pred_binary=_pred_a)["farh_events_per_hour"]
                    _ef1_a = event_level_f1(ext_labels[m], _pred_a, ext_ids[m])
                    _em_a = event_level_metrics(ext_labels[m], _pred_a, ext_ids[m])
                    c["score_anchored_event_f1"] = float(_ef1_a)
                    c["score_anchored_sens"] = float(_em_a["sensitivity"])
                    c["score_anchored_farh"] = float(_farh_a)
                    _acc = globals().setdefault("_nm25_anchor_acc", {})
                    _acc[src] = (c, float(_ef1_a))
                    if globals().get("_nm25_anchor_void", False):
                        c["score_anchored_event_metrics_void"] = True
                    elif len(_acc) >= 2:
                        _avals = [v[1] for v in _acc.values()]
                        if max(_avals) - min(_avals) < 0.02:
                            globals()["_nm25_anchor_void"] = True
                            for _sc, (_cc, _) in _acc.items():
                                _cc["score_anchored_event_metrics_void"] = True
                            print("[anchor-guard v25.53.2] DEGENERACY (event-metric stage): "
                                  f"anchored eventF1 identical across cohorts "
                                  f"{[round(v, 4) for v in _avals]} -> ALL anchored EVENT "
                                  "metrics marked score_anchored_event_metrics_void=True. "
                                  "Native-scale frozen-protocol rows remain the ONLY gate "
                                  "input; anchored probability-scale fields kept as "
                                  "diagnostic only.", flush=True)
                    c["score_anchored_window_f1"] = window_level_f1(
                        ext_labels[m], _pa, ext_ids[m], threshold=frozen_threshold)["window_f1"]
                    c["score_anchored_ece"] = compute_ece(_pa, ext_labels[m])
                    print(f"[external score-anchor v25.37] {src}: eventF1 {c['event_f1']:.3f} -> "
                          f"{_ef1_a:.3f}, sens {c['event_metrics']['sensitivity']:.2f} -> "
                          f"{_em_a['sensitivity']:.2f}, FARh {c['farh']:.2f} -> {_farh_a:.2f}/h, "
                          f"winF1 {c['window_f1']:.3f} -> {c['score_anchored_window_f1']:.3f}, "
                          f"ECE {c['ece']:.4f} -> {c['score_anchored_ece']:.4f} "
                          f"(unsupervised; frozen trigger unchanged)", flush=True)
            except Exception as _e:
                print(f"[external score-anchor v25.37] {src} failed: {_e}", flush=True)
        if source_fog_prior is not None and _flag("NEUROMAMBA_DWLR_WNET", "0"):
            try:
                uids = np.unique(ext_ids[m])
                seq_means = np.array([float(ext_probs[m][ext_ids[m] == u].mean()) for u in uids])
                pseudo = (seq_means > 0.5).astype(np.int64)
                if len(seq_means) >= 8 and 0.0 < pseudo.mean() < 1.0:
                    w_np, q_w = wnet_weighted_prior(seq_means, pseudo, source_fog_prior)
                    c["dwlr_weighted_prior"] = q_w
                    c["dwlr_weights_mean"] = float(np.mean(w_np))
                    _em_prior = c.get("mlls_target_prior", float("nan"))
                    print(f"[external dwlr v25.32] {src}: WNet weighted prior={q_w:.4f} "
                          f"(EM={_em_prior:.4f}), weights mean={np.mean(w_np):.2f} "
                          f"(sum={w_np.sum():.0f}, n_seq={len(seq_means)})", flush=True)
            except Exception as _e:
                print(f"[external dwlr v25.32] {src} failed: {_e}", flush=True)

    if _flag("NEUROMAMBA_TTA_ENTROPY", "0"):
        for src in np.unique(ext_sources):
            c_seqs = [s for s in external_seqs if s.source_type == src]
            try:
                _mean_p = float(np.mean(ext_probs[ext_sources == src]))
                if _mean_p < 0.02 or _mean_p > 0.98:
                    print(f"[external tta-entropy v25.32] {src} skipped (homogeneous "
                          f"stream, mean p={_mean_p:.3f})", flush=True)
                    continue
                m_adapt = entropy_min_tta(models[-1], c_seqs, device,
                                          steps=int(get("NEUROMAMBA_TTA_STEPS", "40")),
                                          activity_gate=_flag("NEUROMAMBA_TTA_ACTIVITY_GATE", "0"),
                                          activity_frac=float(get("NEUROMAMBA_TTA_ACTIVITY_FRAC", "0.5")))
                rv = run_inference_cached(m_adapt, c_seqs, device,
                                          collect_gate=use_walking_gate, window_roll=window_roll)
                p_a = rv[0]
                if use_walking_gate and len(rv) >= 5 and rv[4] is not None:
                    p_a = _apply_walking_gate_inference(m_adapt, c_seqs, p_a, device,
                                                        window_roll=window_roll, gate_probs=rv[4])
                p_a = apply_temperature(p_a, temperature)
                l_a, id_a = rv[1], rv[2]
                pred_a = causal_event_postprocess(p_a, id_a, trigger=frozen_threshold,
                                                  sustain=postproc.get("sustain", frozen_threshold * 0.4),
                                                  confirm_n=postproc.get("confirm_n", 3),
                                                  cooldown_n=postproc.get("cooldown_n", 30),
                                                  merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
                farh_a = false_alarm_events_per_hour(l_a, p_a, frozen_threshold, id_a,
                                                     pred_binary=pred_a)["farh_events_per_hour"]
                em_a = event_level_metrics(l_a, pred_a, id_a)
                cohorts[src]["tta_entropy_farh"] = float(farh_a)
                cohorts[src]["tta_entropy_sens"] = float(em_a["sensitivity"])
                print(f"[external tta-entropy v25.32] {src}: FARh={farh_a:.2f}/h "
                      f"sens={em_a['sensitivity']:.2f} eventF1={em_a['f1']:.2f} "
                      f"(frozen FARh={cohorts[src]['farh']:.2f}/h)", flush=True)
            except Exception as _e:
                print(f"[external tta-entropy v25.32] {src} failed: {_e}", flush=True)

    if _flag("NEUROMAMBA_EXTERNAL_ADAPT", "0"):
        adapt_min = float(get("NEUROMAMBA_ADAPT_MINUTES", "50.0"))
        adapt_epochs = int(get("NEUROMAMBA_ADAPT_EPOCHS", "3"))
        adapt_lr = float(get("NEUROMAMBA_ADAPT_LR", "1e-4"))
        adapted_cohorts = {}
        for src in np.unique(ext_sources):
            c_seqs = [s for s in external_seqs if s.source_type == src]
            adapt_seqs = _slice_adapt_seqs(c_seqs, minutes=adapt_min)
            if not adapt_seqs:
                continue
            m_adapt = copy.deepcopy(models[-1])
            _finetune_external(m_adapt, adapt_seqs, device, epochs=adapt_epochs,
                               lr=adapt_lr, seed=seed)
            res_a = run_inference_cached(m_adapt, c_seqs, device, collect_gate=use_walking_gate,
                                         window_roll=window_roll)
            pa = res_a[0]
            if use_walking_gate:
                pa = _apply_walking_gate_inference(m_adapt, c_seqs, pa, device,
                                                   window_roll=window_roll,
                                                   gate_probs=res_a[4] if len(res_a) >= 5 else None)
            pa = apply_temperature(pa, temperature)
            pa, _, _ = _apply_abstention(pa, c_seqs, labels=res_a[1])
            pred_a = causal_event_postprocess(pa, res_a[2], trigger=frozen_threshold,
                                              sustain=postproc.get("sustain", frozen_threshold * 0.4),
                                              confirm_n=postproc.get("confirm_n", 3),
                                              cooldown_n=postproc.get("cooldown_n", 30),
                                              merge_gap_n=postproc.get("merge_gap_n", MERGE_GAP_N_DEFAULT))
            adapted_cohorts[str(src)] = {
                "event_f1": float(event_level_f1(res_a[1], pred_a, res_a[2])),
                "farh": false_alarm_events_per_hour(res_a[1], pa, frozen_threshold, res_a[2],
                                                    pred_binary=pred_a)["farh_events_per_hour"],
                "window_f1": window_level_f1(res_a[1], pa, res_a[2],
                                             threshold=frozen_threshold)["window_f1"],
                "adapt_minutes": adapt_min, "adapt_epochs": adapt_epochs,
            }
            print(f"[external-adapt v25.14c] {src}: {adapt_min:.0f} min fine-tune -> "
                  f"eventF1={adapted_cohorts[str(src)]['event_f1']:.3f} "
                  f"(frozen {cohorts[str(src)]['event_f1']:.3f}), "
                  f"windowF1={adapted_cohorts[str(src)]['window_f1']:.3f} "
                  f"(frozen {cohorts[str(src)].get('window_f1', float('nan')):.3f})", flush=True)
        if adapted_cohorts:
            external_report["adapted"] = {
                "cohorts": adapted_cohorts,
                "note": "ADAPTATION -- per-cohort fine-tune on the first N minutes of target-site "
                        "labels; NOT frozen external validation (KU Leuven paradigm)",
            }
    return external_report

def run_seed_ensemble(train_seqs, calib_seqs, thresh_seqs, device, epochs, lr, batch_size, base_seed,
                       n_seeds=4, external_seqs=None, notype_seqs=None, model_type=None,
                       fold_tag=None, resume=False, warm_seqs=None, warm_epochs=0,
                       pretrained_state=None, trajectory_val_seqs=None,
                       id_use_rare_split=True):
    """v25: 4-seed ensemble (1st-place approach). Each seed trains independently;
    final predictions are simple-averaged. Blend weights are NOT tuned on the
    outer fold (no leakage). v25.2: returns the LAST seed's model+results so
    the caller can tune thresholds without a 5th full training (was wasted ~20%
    of total compute)."""
    all_sub_probs = []
    sub_labels_ref, sub_observed_ref = None, None
    models, all_results = [], []
    seed_diverse = _flag("NEUROMAMBA_SEED_DIVERSITY", "0")
    diversity_cfgs = [("128", "2"), ("160", "2"), ("128", "3"), ("96", "3"),
                      ("160", "3"), ("96", "2")]
    _saved_hid = os.environ.get("NEUROMAMBA_GRU_HIDDEN")
    _saved_nblk = os.environ.get("NEUROMAMBA_NBLOCKS")
    for s_idx in range(n_seeds):
        seed = base_seed + s_idx * 1000
        print(f"\n[seed-ensemble] training seed {s_idx + 1}/{n_seeds} (seed={seed})", flush=True)
        try:
            if seed_diverse:
                hid, nblk = diversity_cfgs[s_idx % len(diversity_cfgs)]
                os.environ["NEUROMAMBA_GRU_HIDDEN"] = hid
                os.environ["NEUROMAMBA_NBLOCKS"] = nblk
                print(f"[seed-diversity v25.20] seed {s_idx + 1}: hidden={hid} n_blocks={nblk} "
                      f"(model_type={model_type})", flush=True)
            model, results = train_one_config(train_seqs, calib_seqs, thresh_seqs, device=device, epochs=epochs,
                                               lr=lr, batch_size=batch_size, seed=seed, notype_seqs=notype_seqs,
                                               external_seqs=None, model_type=model_type,
                                               ckpt_tag=f"{fold_tag}_seed{s_idx + 1}" if fold_tag else None,
                                               resume=resume, warm_seqs=warm_seqs, warm_epochs=warm_epochs,
                                               pretrained_state=pretrained_state,
                                               trajectory_val_seqs=trajectory_val_seqs,
                                               id_use_rare_split=id_use_rare_split)
            _finalized = bool(results.get("seed_finalized", False))
            if not _finalized:
                _fin_epoch = results.get("seed_finalized_epoch", -1)
                print(f"[seed-finalize v25.42.7] EXCLUDING seed {s_idx + 1} from the "
                      f"ensemble: NOT FINAL (epoch {_fin_epoch} < floor {EARLY_STOP_MIN_EPOCH}) "
                      f"-- blend will use {s_idx} of {n_seeds} seeds", flush=True)
            else:
                models.append(model)
                all_results.append(results)
                if results.get("subprobs") is not None:
                    all_sub_probs.append(results["subprobs"])
                    sub_labels_ref = results["sublabels"]
                    sub_observed_ref = results["subobserved"]
        except Exception as e:
            print(f"[seed-ensemble] seed {s_idx + 1} FAILED: {e}", flush=True)
            traceback.print_exc()
    if seed_diverse:
        if _saved_hid is None:
            os.environ.pop("NEUROMAMBA_GRU_HIDDEN", None)
        else:
            os.environ["NEUROMAMBA_GRU_HIDDEN"] = _saved_hid
        if _saved_nblk is None:
            os.environ.pop("NEUROMAMBA_NBLOCKS", None)
        else:
            os.environ["NEUROMAMBA_NBLOCKS"] = _saved_nblk
    if len(all_sub_probs) >= 2:
        avg_probs = np.mean(np.stack(all_sub_probs, axis=0), axis=0)
        blend_map = compute_kaggle_map(avg_probs, sub_labels_ref, sub_observed_ref) if sub_labels_ref is not None else None
        if blend_map is not None:
            print(f"[seed-ensemble] blended mAP ({len(all_sub_probs)} seeds) = {blend_map['mAP']:.4f}", flush=True)
        _trajs = [r.get("trajectory_eval") for r in all_results if r.get("trajectory_eval")]
        if _trajs:
            _keys = ["mAP", "SH", "Turn", "Walk", "windowF1_05", "eventF1_05", "farh_05"]
            _agg = []
            for i, _t in enumerate(_trajs[0]):
                _row = {"epoch": _t["epoch"]}
                for _k in _keys:
                    _vals = [tr[i][_k] for tr in _trajs if i < len(tr) and np.isfinite(tr[i][_k])]
                    _row[_k] = float(np.mean(_vals)) if _vals else float("nan")
                _agg.append(_row)
            all_results[-1]["trajectory_eval"] = _agg
            print(f"[trajectory v25.42.3] seed-averaged VAL trajectory "
                  f"({len(_trajs)} seeds): "
                  f"mAP={[round(t['mAP'], 3) for t in _agg]} "
                  f"SH={[round(t['SH'], 3) for t in _agg]} "
                  f"Walk={[round(t['Walk'], 3) for t in _agg]}", flush=True)
        return avg_probs, blend_map, sub_labels_ref, sub_observed_ref, models, all_results
    return None, None, sub_labels_ref, sub_observed_ref, models, all_results

print("[kaggle_eval] KAGGLE_EVAL_V25_LOADED", flush=True)

KAGGLE_LEADERBOARD = {
    "1st place (private LB, Transformer+2BiLSTM)": 0.514, "1st place (public LB)": 0.527,
    "2nd place (GRU, separate tdcs/defog)": 0.451,
    "3rd place (Transformer+GRU, pooled)": 0.436,
    "4th place (MultiResidualBiGRU, best-val-loss)": 0.417,
    "5th place (WaveNet+GRU, binary focal)": 0.390,
    "8th place (private LB, 1D-ResNet, 6-block)": 0.356, "8th place (public LB)": 0.357,
    "Competition median": 0.220,
}
SUBTYPE_LEADERBOARD = {
    "StartHesitation": {"1st place F1": 0.459, "4th place F1": 0.495, "8th place range": (0.010, 0.063)},
    "Turn": {"1st place F1": 0.814, "4th place F1": 0.754, "8th place range": (0.420, 0.682)},
    "Walking": {"1st place F1": 0.079, "2nd place F1": 0.407, "8th place range": (0.023, 0.252)},
}

def print_leaderboard_comparison(mean_map=None, sub_maps=None):
    print("-" * 70)
    print("Kaggle Competition -- Leaderboard Comparison (private LB mAP)")
    print("-" * 70)
    for name, val in KAGGLE_LEADERBOARD.items():
        marker = "(SURPASSED)" if (mean_map is not None and mean_map > val) else ""
        level = "(>90% level)" if (mean_map is not None and mean_map > val * 0.9) else ""
        print(f"{name:50s} {val:8.3f} {marker}{level}")
    label = "NeuroMamba v25 CV mean"
    if mean_map is not None:
        print(f"{label:50s} {mean_map:8.4f} -- OUR RESULT")
    else:
        print(f"{label:50s} NA")
    if sub_maps:
        print("-" * 70)
        print("Per-subtype F1 reference (Salomon Nature Commun. 2024 Table 1):")
        for name in SUBTYPE_NAMES:
            refs = SUBTYPE_LEADERBOARD.get(name, {})
            ours = sub_maps.get(name, None)
            ours_str = f"{ours:.3f}" if ours is not None else "NA"
            print(f"  {name:20s} ours_F1={ours_str}  refs={refs}")

def per_source_metrics(labels, probs, ids, pred, sources):
    out = {}
    for src in np.unique(sources):
        m = sources == src
        if m.sum() == 0:
            continue
        out[src] = {"episode_f1_frame": episode_f1(labels[m], probs[m], 0.5, ids[m], pred_binary=pred[m]),
                     "event_f1": event_level_f1(labels[m], pred[m], ids[m]),
                     "farh": false_alarm_events_per_hour(labels[m], probs[m], 0.5, ids[m], pred_binary=pred[m])}
    return out

def run_fold(train_seqs_raw, val_seqs_raw, device, epochs, lr, batch_size, seed, fold_idx,
             external_seqs=None, notype_seqs=None, run_baseline=False, model_type=None,
             use_rare_split=True, n_seeds=1, fold_tag=None, resume=False,
             warm_seqs=None, warm_epochs=0, training_identity=None, p1_manifest=None):
    """v25: uses rare_subtype_aware_split by default (1st-place trick).
    v25.2: fold_tag/resume wire per-epoch checkpoints (no wasted GPU hours).
    v25.20: inner calib/thresh fractions are env-tunable
    (NEUROMAMBA_CALIB_FRAC / NEUROMAMBA_THRESH_FRAC) -- the contest-mimic
    track uses 0.05/0.05 so the model trains on ~85-90% of subjects (the
    1st place's data efficiency) instead of ~60%."""
    calib_frac = float(get("NEUROMAMBA_CALIB_FRAC", "0.15"))
    thresh_frac = float(get("NEUROMAMBA_THRESH_FRAC", "0.15"))
    diag = {}  # v25.42.10: split-power diagnostic (held_sh_*) -- attached to
    _lodo_held = lodo_parse_env()
    try:
        if _lodo_held:
            _tlv = [s for s in train_seqs_raw if s.source_type in _LODO_TLVMC_SOURCES]
            _ext = [s for s in train_seqs_raw if s.source_type not in _LODO_TLVMC_SOURCES]
            if not _tlv:
                raise RuntimeError("[lodo v25.61] no TLVMC sequences in the train "
                                   "pool -- calib/thresh must be TLVMC-only")
            f_train, f_calib, f_thresh, diag = three_way_subject_split(
                _tlv, seed=seed + fold_idx,
                calib_frac=calib_frac, thresh_frac=thresh_frac)
            diag = dict(diag)
            diag["split_method"] = "lodo_tlvmc_calib"
            _cap = float(get("NEUROMAMBA_LODO_COHORT_CAP", "0"))
            _n_ext_before = len(_ext)
            _ext = lodo_cohort_cap_subsample(_ext, _cap, seed + fold_idx)
            f_train = f_train + _ext
            print(f"[lodo v25.61] inner split: calib={len(f_calib)} / "
                  f"thresh={len(f_thresh)} (TLVMC-only), train = TLVMC "
                  f"{len(f_train) - len(_ext)} + external {len(_ext)}"
                  f" (pool had {_n_ext_before}; cap={'off' if _cap <= 0 else _cap})",
                  flush=True)
        elif use_rare_split:
            f_train, f_calib, f_thresh, diag = rare_subtype_aware_split(
                train_seqs_raw, seed=seed + fold_idx,
                calib_frac=calib_frac, thresh_frac=thresh_frac)
        else:
            f_train, f_calib, f_thresh, diag = three_way_subject_split(
                train_seqs_raw, seed=seed + fold_idx,
                calib_frac=calib_frac, thresh_frac=thresh_frac)
    except Exception as e:
        if _lodo_held:
            raise
        print(f" [WARN] split failed ({e}); using 70/15/15 fallback", flush=True)
        traceback.print_exc()
        rng = np.random.RandomState(seed + fold_idx)
        subs = sorted(set(s.subject_id for s in train_seqs_raw))
        rng.shuffle(subs)
        n = len(subs)
        thr = set(subs[:max(1, int(n * 0.15))])
        cal = set(subs[max(1, int(n * 0.15)):max(2, int(n * 0.30))])
        f_train = [s for s in train_seqs_raw if s.subject_id not in (thr | cal)]
        f_calib = [s for s in train_seqs_raw if s.subject_id in cal]
        f_thresh = [s for s in train_seqs_raw if s.subject_id in thr]

    n_train_subjects = len(set(s.subject_id for s in f_train))
    t0 = time.time()
    seed_models, seed_results = None, None
    _traj_val = val_seqs_raw if get("NEUROMAMBA_EVAL_TRAJECTORY_EPOCHS", "").strip() else None
    pretrained_state = None
    if _flag("NEUROMAMBA_SSL_PRETRAIN", "0"):
        ssl_seqs = list(f_train) + (list(notype_seqs) if notype_seqs else [])
        pretrained_state = pretrain_ssl_encoder(
            ssl_seqs, device,
            epochs=int(get("NEUROMAMBA_SSL_EPOCHS", "10")),
            lr=float(get("NEUROMAMBA_SSL_LR", "1e-3")),
            batch_size=16,
            window=int(get("NEUROMAMBA_SSL_WINDOW", str(TRAIN_WINDOW))),
            mask_frac=float(get("NEUROMAMBA_SSL_MASK_FRAC", "0.2")),
            seed=seed + fold_idx)
    if n_seeds > 1:
        avg_probs, blend_map, sub_labels_ref, sub_observed_ref, seed_models, seed_results = run_seed_ensemble(
            f_train, f_calib, f_thresh, device, epochs, lr, batch_size, seed,
            n_seeds=n_seeds, external_seqs=external_seqs, notype_seqs=notype_seqs, model_type=model_type,
            fold_tag=fold_tag, resume=resume, warm_seqs=warm_seqs, warm_epochs=warm_epochs,
            pretrained_state=pretrained_state, trajectory_val_seqs=_traj_val,
            id_use_rare_split=use_rare_split)
        if not seed_models or seed_models[-1] is None:
            raise RuntimeError("seed ensemble produced no trained model (all seeds failed)")
        model, results = seed_models[-1], seed_results[-1]
        if model is None or results is None:
            raise RuntimeError("seed ensemble produced no trained model")
        if avg_probs is not None:
            results["subprobs"] = avg_probs
            results["kaggle_map"] = blend_map
    else:
        model, results = train_one_config(f_train, f_calib, f_thresh, device=device, epochs=epochs, lr=lr,
                                           batch_size=batch_size, seed=seed + fold_idx, n_train_subjects=n_train_subjects,
                                           external_seqs=external_seqs, notype_seqs=notype_seqs, model_type=model_type,
                                           ckpt_tag=fold_tag, resume=resume,
                                           warm_seqs=warm_seqs, warm_epochs=warm_epochs,
                                           pretrained_state=pretrained_state,
                                           trajectory_val_seqs=_traj_val,
                                           id_use_rare_split=use_rare_split)

    pseudo_used = 0
    if _flag("NEUROMAMBA_NOTYPE_PSEUDO", "0") and notype_seqs:
        n_rounds = int(get("NEUROMAMBA_NOTYPE_PSEUDO_ROUNDS", "1"))
        teachers = list(seed_models) if seed_models else [model]
        m_r, r_r = None, None
        for rnd in range(n_rounds):
            pseudo_seqs = _build_notype_pseudo_seqs(teachers, notype_seqs, device,
                                                    window_roll=_flag("NEUROMAMBA_WINDOW_ROLL", "1"))
            if not pseudo_seqs:
                print("[pseudo v25.12] no pseudo sequences built -- skipping pseudo rounds", flush=True)
                break
            m_r, r_r = train_one_config(f_train, f_calib, f_thresh, device=device, epochs=epochs, lr=lr,
                                        batch_size=batch_size, seed=seed + 770 + rnd,
                                        n_train_subjects=n_train_subjects, external_seqs=None,
                                        notype_seqs=notype_seqs, pseudo_seqs=pseudo_seqs,
                                        model_type=model_type,
                                        ckpt_tag=(f"{fold_tag}_pseudo{rnd + 1}" if fold_tag else None),
                                        resume=resume, warm_seqs=warm_seqs, warm_epochs=warm_epochs,
                                        pretrained_state=pretrained_state)
            teachers = [m_r]
        if m_r is not None:
            base_v = max([r.get("best_calib_subtype_map", -1.0) for r in (seed_results if seed_results else [results])])
            r_v = r_r.get("best_calib_subtype_map", -1.0)
            if r_v >= base_v:
                model, results = m_r, r_r
                seed_models = [m_r]
                pseudo_used = 1
                print(f"[pseudo v25.12] ADOPTED pseudo-round model (calibSubtypeMAP {r_v:.4f} "
                      f">= seed-ensemble best {base_v:.4f})", flush=True)
            else:
                print(f"[pseudo v25.12] kept seed-ensemble (pseudo {r_v:.4f} < {base_v:.4f})", flush=True)

    if external_seqs and not results.get("tripod_external_validation"):
        ens_models = list(seed_models) if seed_models else [model]
        results["tripod_external_validation"] = _evaluate_external(
            ens_models, external_seqs, device,
            temperature=results["temperature"], frozen_threshold=results["frozen_threshold"],
            postproc=results["postproc"], use_walking_gate=results.get("use_walking_gate", True),
            window_roll=_flag("NEUROMAMBA_WINDOW_ROLL", "1"), seed=seed,
            train_subjects=set(s.subject_id for s in list(f_train) + list(f_calib) + list(f_thresh)),
            conformal_trigger=results.get("conformal_trigger"),
            source_fog_prior=results.get("fog_prior_train"),
            score_anchor_ref=(results.get("score_anchor_ref_q")
                              if _flag("NEUROMAMBA_SCORE_ANCHOR", "0") else None))

    train_time = time.time() - t0

    frozen = results["frozen_threshold"]
    pp = results["postproc"]
    use_wg_val = results.get("use_walking_gate", True)
    eval_models = seed_models if seed_models else [model]
    probs_acc, sub_acc, gate_acc = [], [], []
    _haz_stream_on = (_flag("NEUROMAMBA_HAZARD_STREAM", "0")
                      and bool(getattr(model, "use_hazard_head", False)))
    _haz_acc = []
    for m in eval_models:
        rv = run_inference_cached(m, val_seqs_raw, device, collect_subtype=True,
                                  collect_gate=use_wg_val,
                                  collect_hazard=_haz_stream_on,
                                  window_roll=_flag("NEUROMAMBA_WINDOW_ROLL", "1"))
        probs_acc.append(rv[0])
        sub_acc.append(rv[3])
        gate_acc.append(rv[4] if len(rv) >= 5 else None)
        if _haz_stream_on:
            _haz_acc.append(rv[5] if len(rv) >= 6 else None)
    probs = np.mean(np.stack(probs_acc, axis=0), axis=0)
    sub_blend = np.mean(np.stack(sub_acc, axis=0), axis=0) if sub_acc and sub_acc[0] is not None else None
    gate_blend = np.mean(np.stack(gate_acc, axis=0), axis=0) if use_wg_val and gate_acc and gate_acc[0] is not None else None
    haz_blend = (np.mean(np.stack(_haz_acc, axis=0), axis=0)
                 if _haz_acc and _haz_acc[0] is not None else None)  # v28bp seed-ensembled p_haz
    labels, ids = rv[1], rv[2]
    res4 = (probs, labels, ids, sub_blend, gate_blend)
    hetero_w, baseline_result = 1.0, None
    if run_baseline:
        try:
            _rep_of_fold = 0
            if fold_tag and "_r" in fold_tag:
                try:
                    _rep_of_fold = int(fold_tag.split("_r")[1].split("_")[0]) - 1
                except ValueError:
                    _rep_of_fold = 0
            baseline_result = run_resnet1d_baseline_fold(f_train, val_seqs_raw, device,
                                                         min(epochs, 40), lr, batch_size, fold_idx,
                                                         repeat=_rep_of_fold)
        except Exception as e:
            print(f"[BASELINE FOLD {fold_idx + 1}] FAILED: {e}", flush=True)
            traceback.print_exc()
        if (_flag("NEUROMAMBA_HETERO_BLEND", "0") and baseline_result is not None
                and baseline_result.get("model") is not None
                and results.get("calib_sub") is not None and sub_blend is not None):
            hetero_w = _tune_hetero_blend(baseline_result["model"], results, f_calib, f_thresh, device)
            if hetero_w < 1.0 - 1e-6:
                probs = hetero_w * probs + (1.0 - hetero_w) * baseline_result["fog_probs"]
                sub_blend = hetero_w * sub_blend + (1.0 - hetero_w) * baseline_result["sub_probs"]
                res4 = (probs, labels, ids, sub_blend, gate_blend)
    if use_wg_val and gate_blend is not None:
        _val_shp = sub_blend if (sub_blend is not None and _flag("NEUROMAMBA_GATE_SH_COND", "0")) else None
        probs = _apply_walking_gate_inference(model, val_seqs_raw, probs, device,
                                               window_roll=_flag("NEUROMAMBA_WINDOW_ROLL", "1"),
                                               gate_probs=gate_blend,
                                               sh_probs=_val_shp)
    probs = apply_temperature(probs, results["temperature"])
    if _haz_stream_on and haz_blend is not None:
        if use_wg_val and gate_blend is not None:
            _haz_shp = sub_blend if (sub_blend is not None and _flag("NEUROMAMBA_GATE_SH_COND", "0")) else None
            haz_blend = _apply_walking_gate_inference(model, val_seqs_raw, haz_blend, device,
                                                      window_roll=_flag("NEUROMAMBA_WINDOW_ROLL", "1"),
                                                      gate_probs=gate_blend,
                                                      sh_probs=_haz_shp)
        results["hazard_probs"] = np.asarray(haz_blend, dtype=np.float32)
        if not globals().get("_nm25_hazstream_banner", False):
            globals()["_nm25_hazstream_banner"] = True
            print("[hazard-stream v28bp] ACTIVE: saved OOF fog_probs carry the "
                  "walking-gated, seed-ensembled hazard probability "
                  "(NEUROMAMBA_HAZARD_STREAM=1, temperature=1)", flush=True)
    probs, abst_frac_val, abst_pos_val = _apply_abstention(probs, val_seqs_raw, labels=labels)
    pred = causal_event_postprocess(probs, ids, trigger=frozen,
                                     sustain=pp.get("sustain", frozen * 0.4),
                                     confirm_n=pp.get("confirm_n", 3), cooldown_n=pp.get("cooldown_n", 30),
                                     merge_gap_n=pp.get("merge_gap_n", MERGE_GAP_N_DEFAULT))

    sources = np.array([s.source_type for s in val_seqs_raw for _ in range(len(s.labels))])
    subjects = np.array([s.subject_id for s in val_seqs_raw for _ in range(len(s.labels))])

    sub_probs, sub_labels, sub_observed, kaggle_map = None, None, None, None
    seed_val_maps = None  # v25.40: per-seed val mAP (between-seed variance)
    if len(res4) >= 4 and res4[3] is not None:
        sub_probs = np.asarray(res4[3])
        sub_labels, sub_observed, sub_sub_obs = concat_labels(val_seqs_raw)
        n = min(len(sub_labels), len(sub_probs))
        sub_probs, sub_labels, sub_observed = sub_probs[:n], sub_labels[:n], sub_sub_obs[:n]
        sub_probs_t = apply_temperature_subtype(sub_probs, results.get("subtype_temps", [1.0, 1.0, 1.0]))
        _tau_la = float(os.environ.get("NEUROMAMBA_LOGIT_ADJUST_TAU", "0"))
        _priors_la = results.get("subtype_prior_train")
        if _tau_la > 0 and _priors_la:
            sub_probs_t = apply_logit_adjust_subtype(sub_probs_t, _priors_la, _tau_la)
            print(f"[logit-adjust v25.44] VAL subtype scores adjusted: tau={_tau_la} "
                  f"train_priors={[round(p, 5) for p in _priors_la]} "
                  f"(post-hoc, no eval labels used)", flush=True)
        official_map = compute_official_like_map(sub_labels, sub_probs_t, sub_observed)
        kaggle_map = {"mAP": official_map["mAP"], "comparable": official_map["comparable"],
                      **{k: official_map[k] for k in SUBTYPE_NAMES}}
        if _flag("NEUROMAMBA_PER_SEED_VAL", "1") and len(sub_acc) > 1 and sub_acc[0] is not None:
            seed_val_maps = []
            _temps = results.get("subtype_temps", [1.0, 1.0, 1.0])
            for _sp in sub_acc:
                _sp = np.asarray(_sp)[:n]
                _t = apply_temperature_subtype(_sp, _temps)
                seed_val_maps.append(float(compute_official_like_map(sub_labels, _t, sub_observed)["mAP"]))
            print(f"[seed-val v25.40] per-seed val mAP {[round(x, 4) for x in seed_val_maps]} "
                  f"mean={float(np.mean(seed_val_maps)):.4f} sd={float(np.std(seed_val_maps)):.4f}", flush=True)

    metrics = {
        "episode_f1_frame": episode_f1(labels, probs, frozen, ids, pred_binary=pred),
        "event_f1": event_level_f1(labels, pred, ids),
        "farh": false_alarm_events_per_hour(labels, probs, frozen, ids, pred_binary=pred),
        "ece": compute_ece(probs, labels),
        "brier": brier_score(probs, labels),
        "reliability_diagram_slope": calibration_slope_intercept(probs, labels)[0],
        "reliability_diagram_intercept": calibration_slope_intercept(probs, labels)[1],
        "logistic_calib_slope": logistic_calibration_slope_intercept(probs, labels)[0],
        "logistic_calib_intercept": logistic_calibration_slope_intercept(probs, labels)[1],
        "window_f1": window_level_f1(labels, probs, ids, threshold=frozen)["window_f1"],
        "window_metrics": window_level_metrics(labels, probs, ids, threshold=frozen),
        "event_metrics": event_level_metrics(labels, pred, ids),
        "icc": icc_report(labels, pred, ids),
        "per_source": per_source_metrics(labels, probs, ids, pred, sources),
        "per_subject_farh": per_subject_farh_distribution(labels, pred, subjects),
        "decision_curve": decision_curve_analysis(labels, pred, ids),
        "cueing_latency": onset_latency_report(labels, pred, ids),
        "kaggle_map": kaggle_map,
        "seed_val_maps": seed_val_maps,  # v25.40: per-seed val mAP (between-seed variance)
        "abstention_frac": abst_frac_val, "abstention_pos_frac": abst_pos_val,
        "per_medication": medication_subgroup_report(
            labels, pred, probs, ids,
            np.array([s.med_state for s in val_seqs_raw for _ in range(len(s.labels))],
                     dtype=np.float32), threshold=frozen),
    }
    sh_event_recall, sh_subject_coverage = float("nan"), float("nan")
    if sub_probs is not None and sub_labels is not None:
        try:
            _sh_p = np.asarray(sub_probs)[:, 0]
            _sh_l = np.asarray(sub_labels)[:, 0]
            _sh_o = np.asarray(sub_observed)[:, 0] if sub_observed is not None else None
            _sh_pos = (_sh_l > 0.5)
            _sh_pred = (_sh_p > 0.5)
            if _sh_o is not None:
                _sh_pos = _sh_pos & _sh_o
            _ev_ids = np.asarray(ids)
            _runs, _detected = [], 0
            _subj_pos = set(); _subj_cov = set()
            for _uid in np.unique(_ev_ids):
                _m = _ev_ids == _uid
                _y = _sh_pos[_m]; _pr = _sh_pred[_m]
                _start = None
                for _t in range(len(_y) + 1):
                    _on = _t < len(_y) and _y[_t]
                    if _on and _start is None:
                        _start = _t
                    elif not _on and _start is not None:
                        _runs.append((_start, _t - 1))
                        _start = None
                for (_s, _e) in _runs:
                    if _pr[_s:_e + 1].any():
                        _detected += 1
                    _subj_pos.add(_uid)
                    if _pr[_s:_e + 1].any():
                        _subj_cov.add(_uid)
            if _runs:
                sh_event_recall = float(_detected) / len(_runs)
            if _subj_pos:
                sh_subject_coverage = float(len(_subj_cov)) / len(_subj_pos)
        except Exception as _e:
            print(f"[SH-metrics v25.42.10] failed: {_e}", flush=True)
    metrics["sh_event_recall"] = sh_event_recall
    metrics["sh_subject_coverage"] = sh_subject_coverage
    metrics["sh_positive_subjects_val"] = len(set(s.subject_id for s in val_seqs_raw
                                                  if int((s.subtype_observed[:, 0] &
                                                          (s.y_subtype[:, 0] > 0.5)).sum()) > 0))
    km = metrics.get("kaggle_map")
    icc = metrics.get("icc") or {}
    if icc.get("icc_percent_time_frozen") is not None:
        print(f"  [icc v25.21] %TF ICC={icc['icc_percent_time_frozen']:.3f} "
              f"#episodes ICC={icc.get('icc_n_episodes')} "
              f"duration ICC={icc.get('icc_duration_s')} (n_rec={icc.get('n_recordings')}) "
              f"[contest refs: %TF 0.877-0.949, #episodes <0.6]", flush=True)
    print(f"[VAL] frameF1={metrics['episode_f1_frame']:.3f} eventF1={metrics['event_f1']:.3f} "
          f"FARh={metrics['farh']['farh_events_per_hour']:.2f}/h ECE={metrics['ece']:.4f} "
          f"windowF1={metrics['window_f1']:.3f} (contest ceiling 0.73-0.81) train={train_time:.0f}s", flush=True)
    _traj = results.get("trajectory_eval") if results else None
    if _traj:
        print(f"[trajectory v25.42.3] VAL trajectory (seed-averaged, raw checkpoints):", flush=True)
        for _t in _traj:
            print(f"  epoch {_t['epoch']:>3}: mAP={_t['mAP']:.4f} SH={_t['SH']:.3f} "
                  f"Turn={_t['Turn']:.3f} Walk={_t['Walk']:.3f} "
                  f"WF1@0.05={_t['windowF1_05']:.3f} eventF1={_t['eventF1_05']:.3f} "
                  f"FARh={_t['farh_05']:.2f}/h", flush=True)
    cl = metrics["cueing_latency"]
    cl_dep = cl.get("deployable", True)
    cl_dep_note = "" if cl_dep else "  [NON-DEPLOYABLE: negative latency is an artifact, not a real cueing latency]"
    print(f"  cueing: detected {cl['detected_frac'] * 100:.0f}% of {cl['n_events']} events, "
          f"median onset->alarm {cl['median_latency_s']:.2f}s, within-1s {cl['within_1s_frac'] * 100:.0f}% "
          f"[v25.42 caveat: bidirectional inference -- negative latency is NOT a "
          f"causal early-warning claim; use the causal/window-roll path for that]"
          + cl_dep_note, flush=True)
    if km:
        print(f"  mAP(internal)={km['mAP']:.4f}  SH={km.get('StartHesitation', 0):.3f} "
              f"Turn={km.get('Turn', 0):.3f} Walk={km.get('Walking', 0):.3f}", flush=True)
    try:
        if (baseline_result is not None and baseline_result.get("model") is not None
                and (results or {}).get("subprobs") is not None
                and (results or {}).get("sublabels") is not None
                and f_thresh):
            _m_sel = apply_temperature_subtype(
                np.asarray(results["subprobs"], dtype=np.float64),
                results.get("subtype_temps", [1.0, 1.0, 1.0]))
            _lab_sel = np.asarray(results["sublabels"], dtype=np.float64)
            _obs_sel = np.asarray(results["subobserved"], dtype=bool)
            _bp_fog, _bp_sub = _baseline_predict(baseline_result["model"], f_thresh, device)
            _b_sel = np.asarray(_bp_sub, dtype=np.float64)
            _n_sel = min(len(_lab_sel), len(_m_sel), len(_b_sel))
            _m_sel, _lab_sel, _obs_sel, _b_sel = (_m_sel[:_n_sel], _lab_sel[:_n_sel],
                                                  _obs_sel[:_n_sel], _b_sel[:_n_sel])
            _map_main = compute_kaggle_map(_m_sel, _lab_sel, _obs_sel)
            _map_base = compute_kaggle_map(_b_sel, _lab_sel, _obs_sel)
            _blend35 = compute_kaggle_map(0.65 * _m_sel + 0.35 * _b_sel, _lab_sel, _obs_sel)
            _blend50 = compute_kaggle_map(0.50 * _m_sel + 0.50 * _b_sel, _lab_sel, _obs_sel)
            _src_sel = np.array([s.source_type for s in f_thresh for _ in range(len(s.labels))])[:_n_sel]
            _cohorts = {}
            for _s in np.unique(_src_sel):
                _cm = _src_sel == _s
                _cohorts[str(_s)] = compute_kaggle_map(_m_sel[_cm], _lab_sel[_cm], _obs_sel[_cm])
            metrics["selection_split_ensemble"] = {
                "main": _map_main, "baseline": _map_base,
                "blend_0.35": _blend35, "blend_0.5": _blend50,
                "per_cohort_main": _cohorts, "n_frames": int(_n_sel),
                "note": "report-only: never selects, deploys, or gates"}
            print(f"  [sel-ensemble v25.60] Selection-split mAP: main={_map_main['mAP']:.4f} "
                  f"baseline={_map_base['mAP']:.4f} blend0.35={_blend35['mAP']:.4f} "
                  f"blend0.5={_blend50['mAP']:.4f} "
                  f"(report-only -- never selects or deploys)", flush=True)
            print(f"  [sel-ensemble v25.60] per-cohort (main): "
                  + " | ".join(f"{_s}: mAP={_c['mAP']:.4f} (SH={_c['StartHesitation']:.3f} "
                               f"Turn={_c['Turn']:.3f} Walk={_c['Walking']:.3f})"
                               for _s, _c in _cohorts.items()), flush=True)
    except Exception as _se:
        print(f"[sel-ensemble v25.60] WARNING: selection-split ensemble report failed: {_se}",
              flush=True)
    _psf = metrics.get("per_subject_farh")
    if isinstance(_psf, dict) and _psf.get("per_subject"):
        _fv = np.asarray(list(_psf["per_subject"].values()), dtype=np.float64)
        if len(_fv):
            _q = np.percentile(_fv, [25, 50, 75, 90, 95])
            print(f"  [FARh envelope v25.42.2] per-subject FARh/h: "
                  f"Q25={_q[0]:.2f} med={_q[1]:.2f} Q75={_q[2]:.2f} "
                  f"Q90={_q[3]:.2f} Q95={_q[4]:.2f} "
                  f"(mean={float(np.mean(_fv)):.2f}, n_subj={len(_fv)})", flush=True)
    for med_name, med_r in (metrics.get("per_medication") or {}).items():
        print(f"  [med-subgroup v25.15] {med_name}: eventF1={med_r['event_f1']:.3f} "
              f"FARh={med_r['farh']:.2f}/h windowF1={med_r['window_f1']:.3f} "
              f"(n_frames={med_r['n_frames']})", flush=True)

    ens_pack = (sub_probs, sub_labels, sub_observed) if sub_probs is not None else (None, None, None)
    try:
        results["eval_inputs"] = {
            "fog_probs": np.asarray(probs, dtype=np.float32),
            "fog_labels": np.asarray(labels, dtype=np.float32),
            "fog_ids": np.asarray(ids),
            "fog_subjects": np.asarray(subjects),
            "fog_sources": np.asarray(sources),
            "fog_med": np.asarray([s.med_state for s in val_seqs_raw for _ in range(len(s.labels))],
                                  dtype=np.float32),
            "frozen_threshold": float(frozen),
            "temperature": float(results.get("temperature", 1.0)),
            "subtype_temps": list(results.get("subtype_temps", [1.0, 1.0, 1.0])),
            "use_walking_gate": bool(use_wg_val),
            "postproc": dict(pp) if isinstance(pp, dict) else {},
            "seed_sub_probs": [np.asarray(x, dtype=np.float32) for x in sub_acc if x is not None],
        }
        if _haz_stream_on and results.get("hazard_probs") is not None:
            results["eval_inputs"]["fog_probs"] = np.asarray(results.pop("hazard_probs"), dtype=np.float32)
            results["eval_inputs"]["hazard_stream"] = 1
    except Exception as e:
        print(f"[eval-inputs v25.42.6] WARNING: could not snapshot eval inputs: {e}", flush=True)
    _fin_seeds = 0
    _excl_seeds = 0
    _seed_fin_epochs = []
    _sel_best_epoch = -1  # v25.46 FIX: sel_best_epoch is train_one_config-local;
    for _r in (seed_results if seed_results else [results]):
        if _r.get("seed_finalized"):
            _fin_seeds += 1
        else:
            _excl_seeds += 1
        _seed_fin_epochs.append(int(_r.get("seed_finalized_epoch", -1)))
        _sel_best_epoch = int(_r.get("best_epoch", _r.get("sel_best_epoch", -1)))
    fold_record = {"code_version": CODE_VERSION, "training_identity": training_identity,
               "sel_best_epoch": _sel_best_epoch,
                    "evaluation_identity": evaluation_identity(),
                    "protocol_hash": (p1_manifest or {}).get("protocol_hash"),
                    "split_mode": (p1_manifest or {}).get("split_mode"),
                    "repeat": (p1_manifest or {}).get("repeat"),
                    "split_seed": (p1_manifest or {}).get("split_seed"),
                    "fold": fold_idx + 1, "train_time_s": train_time,
                    "model_type": results.get("model_type"),
                    "use_walking_gate": results.get("use_walking_gate", True),
                    "use_rare_split": use_rare_split,
                    "hetero_blend_weight": hetero_w,
                    "notype_pseudo_used": pseudo_used,
                    "n_seeds_finalized": _fin_seeds,
                    "n_seeds_excluded_not_final": _excl_seeds,
                    "seed_finalized_epochs": _seed_fin_epochs,
                    "main_map": float(kaggle_map["mAP"]) if kaggle_map else None,
                    "baseline_map": float((baseline_result or {}).get("kaggle_map", {}).get("mAP"))
                                    if baseline_result else None,
                    "held_sh_positive_subjects": (diag or {}).get("held_sh_positive_subjects"),
                    "held_sh_events": (diag or {}).get("held_sh_events"),
                    "held_sh_max_subject_event_fraction": (diag or {}).get("held_sh_max_subject_event_fraction"),
                    "clinical_op_points": (results or {}).get("clinical_op_points"),
                    "subtype_prior_train": (results or {}).get("subtype_prior_train"),
                    "logit_adjust_tau": (results or {}).get("logit_adjust_tau"),
                    "joint_certified_ops": (results or {}).get("joint_certified_ops"),
                    **{k: v for k, v in metrics.items() if k != "decision_curve"},
                    "frozen_threshold": results.get("frozen_threshold", 0.5),
                    "temperature": results.get("temperature", 1.0),
                    "tripod_external_validation": results.get("tripod_external_validation")}
    if fold_record.get("main_map") is not None and fold_record.get("baseline_map") is not None:
        fold_record["main_minus_baseline_map"] = float(fold_record["main_map"] - fold_record["baseline_map"])
    else:
        fold_record["main_minus_baseline_map"] = None
    return model, results, metrics, train_time, ens_pack, baseline_result, fold_record

def sh_balanced_group_folds(seqs, n_splits, seed=42):
    """v25.6: subject-disjoint outer CV folds that BALANCE per-fold
    StartHesitation totals. The v25 GroupKFold ignored subtype composition:
    some val folds ended nearly SH-free (fold 2 val SH AP 0.02 while thresh
    was 0.14-0.23 -- fold-composition noise, not model failure). Greedy:
    SH-heavy subjects assigned to the SH-lightest fold first; SH-free subjects
    round-robin for count balance with defog/tdcsfog parity tie-breaks."""
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    subj_sh = defaultdict(int)
    subj_src = {}
    for s in seqs:
        subj_sh[s.subject_id] += int((s.subtype_observed[:, 0] & (s.y_subtype[:, 0] > 0.5)).sum())
        subj_src.setdefault(s.subject_id, s.source_type)
    subs = sorted(set(s.subject_id for s in seqs))
    heavy = sorted([sid for sid in subs if subj_sh[sid] > 0],
                   key=lambda sid: (-subj_sh[sid], rng.random()))
    light = [sid for sid in subs if subj_sh[sid] == 0]
    rng.shuffle(light)
    fold_sh = np.zeros(n_splits, dtype=np.float64)
    fold_n = np.zeros(n_splits, dtype=np.int64)
    fold_src = [defaultdict(int) for _ in range(n_splits)]
    fold_subs = [[] for _ in range(n_splits)]
    for sid in heavy:
        k = int(np.argmin(fold_sh))
        fold_sh[k] += subj_sh[sid]; fold_n[k] += 1
        fold_src[k][subj_src[sid]] += 1
        fold_subs[k].append(sid)
    for sid in light:
        cand = np.argsort(fold_n)[:2]
        k = min(cand, key=lambda kk: (fold_n[kk], fold_src[kk].get(subj_src[sid], 0)))
        fold_n[k] += 1
        fold_src[k][subj_src[sid]] += 1
        fold_subs[k].append(sid)
    folds = []
    sh_totals = []
    for k in range(n_splits):
        va_subs = set(fold_subs[k])
        tr_idx = [i for i, s in enumerate(seqs) if s.subject_id not in va_subs]
        va_idx = [i for i, s in enumerate(seqs) if s.subject_id in va_subs]
        folds.append((tr_idx, va_idx))
        sh_totals.append(int(sum(subj_sh[sid] for sid in fold_subs[k])))
    total_sh = max(1, int(sum(subj_sh.values())))
    fracs = [round(100 * t / total_sh) for t in sh_totals]
    note = ""
    if max(sh_totals) > 2 * min(sh_totals) + 1:
        note = (" (subject-disjoint folds cannot balance further: a single subject "
                f"holds {max(sh_totals)} of {total_sh} SH frames -- report pooled OOF + bootstrap CI)")
    print(f"[outer-fold v25.6] SH-balanced subject folds: per-fold val SH = {sh_totals} "
          f"of {total_sh} total ({fracs}% per fold){note}", flush=True)
    return folds

_LODO_KNOWN_HELD = ("daphnet", "xuanwu", "weargait", "fogstar")
_LODO_TLVMC_SOURCES = ("defog", "tdcsfog")

def lodo_parse_env():
    """NEUROMAMBA_LODO env -> sorted list of held-out cohort names ([] = off)."""
    _raw = str(os.environ.get("NEUROMAMBA_LODO", "")).strip().lower()
    if not _raw:
        return []
    _out = [c.strip() for c in _raw.split(",") if c.strip()]
    _bad = [c for c in _out if c not in _LODO_KNOWN_HELD]
    if _bad:
        raise ValueError(
            f"[lodo v25.61] unknown held-out cohort(s) {_bad}; screen supports "
            f"{list(_LODO_KNOWN_HELD)} (defog/tdcsfog-held folds are a later phase)")
    if len(set(_out)) != len(_out):
        raise ValueError(f"[lodo v25.61] duplicate cohorts in {_out}")
    return _out

def lodo_namespace_guard(seqs):
    """Subject-id namespaces must be cohort-disjoint ACROSS THE LODO BOUNDARY:
    (a) the TLVMC block (defog+tdcsfog as one corpus; their mutual id reuse is
    a TLVMC-internal matter that predates LODO -- the per-fold leak guards own
    it) vs each external cohort, and (b) external vs external. Daphnet ids are
    'S01'-style short tokens; a collision would silently MERGE cohorts in
    every subject-grouped bootstrap and defeat the leak assert. Checked
    invariant, hard error (never auto-rename -- cached artifacts keep ids)."""
    tlvmc_subs = set(s.subject_id for s in seqs
                     if s.source_type in _LODO_TLVMC_SOURCES)
    by_ext = {}
    for s in seqs:
        if s.source_type in _LODO_KNOWN_HELD:
            by_ext.setdefault(s.source_type, set()).add(s.subject_id)
    for src in sorted(by_ext):
        inter = tlvmc_subs & by_ext[src]
        if inter:
            raise RuntimeError(
                f"[lodo v25.61] subject-id namespace collision between TLVMC "
                f"and {src}: {sorted(inter)[:8]} -- cannot build a "
                f"cohort-disjoint fold")
    ext_srcs = sorted(by_ext)
    for i, a in enumerate(ext_srcs):
        for b in ext_srcs[i + 1:]:
            inter = by_ext[a] & by_ext[b]
            if inter:
                raise RuntimeError(
                    f"[lodo v25.61] subject-id namespace collision between "
                    f"{a} and {b}: {sorted(inter)[:8]} -- cannot build a "
                    f"cohort-disjoint fold")

def lodo_fold_indices(pool, held_sources):
    """ONE LODO fold over `pool` (= train cohorts + held cohorts): val = every
    held-cohort sequence, train = the rest. Cohort-disjoint by construction;
    subject-disjointness is asserted (a violation means the namespace guard
    was bypassed). Returns (tr_idx, va_idx) into `pool`."""
    held = set(held_sources)
    va_idx = [i for i, s in enumerate(pool) if s.source_type in held]
    tr_idx = [i for i, s in enumerate(pool) if s.source_type not in held]
    if not va_idx:
        raise RuntimeError(f"[lodo v25.61] no held-out sequences for {sorted(held)}")
    if not tr_idx:
        raise RuntimeError("[lodo v25.61] empty train pool -- nothing to train on")
    leak = set(pool[i].subject_id for i in tr_idx) & set(pool[i].subject_id for i in va_idx)
    if leak:
        raise RuntimeError(f"[lodo v25.61] subject leakage across the LODO boundary: {sorted(leak)[:8]}")
    print(f"[lodo v25.61] fold: train={len(tr_idx)} seqs "
          f"({len(set(pool[i].subject_id for i in tr_idx))} subjects) / "
          f"val={len(va_idx)} seqs "
          f"({len(set(pool[i].subject_id for i in va_idx))} subjects), held={sorted(held)}",
          flush=True)
    return (tr_idx, va_idx)

def lodo_cohort_cap_subsample(ext_seqs, cap, seed):
    """Per-cohort exposure cap (spec 2.3.5: 'do not let the ~40% xuanwu fog
    prior dominate sampling'). In the default 'order' exposure mode every
    sequence appears exactly once per epoch, so a cohort's gradient share IS
    its sequence-count share -- the cap therefore trims the largest external
    cohorts DOWN to `cap` of the external pool (round-robin per subject, so
    every subject stays represented). TLVMC is never touched (it is not part
    of ext_seqs). cap<=0 disables (default off; the launcher pre-registers
    the value). Deterministic in seed."""
    if cap <= 0 or not ext_seqs:
        return ext_seqs
    by_src = {}
    for s in ext_seqs:
        by_src.setdefault(s.source_type, []).append(s)
    n_total = len(ext_seqs)
    out = []
    for src in sorted(by_src):
        seqs = by_src[src]
        max_n = max(1, int(round(cap * n_total)))
        if len(seqs) <= max_n:
            out.extend(seqs)
            continue
        by_sub = {}
        for s in seqs:
            by_sub.setdefault(s.subject_id, []).append(s)
        subs = sorted(by_sub)
        kept = []
        i = 0
        while len(kept) < max_n and any(by_sub[sid] for sid in subs):
            sid = subs[i % len(subs)]
            if by_sub[sid]:
                kept.append(by_sub[sid].pop(0))
            i += 1
        print(f"[lodo v25.61] cohort cap {cap}: {src} trimmed {len(seqs)} -> "
              f"{len(kept)} seqs (subjects kept: {len({s.subject_id for s in kept})})",
              flush=True)
        out.extend(kept)
    return out

def build_single_split_fold(seqs, seed=42, calib_frac=None, thresh_frac=None):
    """v25.20: ONE subject-disjoint train/val split -- the contest-mimic track
    (the winner's fixed-holdout protocol: a small val subject list, everything
    else trains; checkpoint selection uses INNER calib only -- val is scored once, never selected on). The val = calib+thresh of a
    rare-subtype-aware split (SH-balanced); the fractions come from
    NEUROMAMBA_CALIB_FRAC / NEUROMAMBA_THRESH_FRAC (default 0.15/0.15; use
    0.05/0.05 to train on ~85-90% of subjects like the 1st place, whose
    models trained on ~90-100% of data). Returns one (tr_idx, va_idx) fold
    with zero subject overlap."""
    if calib_frac is None:
        calib_frac = float(get("NEUROMAMBA_CALIB_FRAC", "0.15"))
    if thresh_frac is None:
        thresh_frac = float(get("NEUROMAMBA_THRESH_FRAC", "0.15"))
    f_tr, f_ca, f_th, diag = rare_subtype_aware_split(seqs, seed=seed,
                                                      calib_frac=calib_frac,
                                                      thresh_frac=thresh_frac)
    va_subs = set(s.subject_id for s in f_ca) | set(s.subject_id for s in f_th)
    _retry = 0
    while len(va_subs) < 4 and _retry < 10:
        seed = seed + 1
        _retry += 1
        f_tr, f_ca, f_th, diag = rare_subtype_aware_split(seqs, seed=seed,
                                                          calib_frac=calib_frac,
                                                          thresh_frac=thresh_frac)
        va_subs = set(s.subject_id for s in f_ca) | set(s.subject_id for s in f_th)
    if len(va_subs) < 4:
        print(f"[single-split v25.34] WARNING: only {len(va_subs)} val subjects -- "
              f"subject bootstrap CI and gate_2 will be unreliable", flush=True)
    tr_idx = [i for i, s in enumerate(seqs) if s.subject_id not in va_subs]
    va_idx = [i for i, s in enumerate(seqs) if s.subject_id in va_subs]
    leak = set(s.subject_id for s in [seqs[i] for i in tr_idx]) & va_subs
    assert not leak, f"single-split subject leakage: {leak}"
    n_tr_subs = len(set(s.subject_id for s in [seqs[i] for i in tr_idx]))
    print(f"[single-split v25.20] {len(tr_idx)} train seqs ({n_tr_subs} subjects) / "
          f"{len(va_idx)} val seqs ({len(va_subs)} subjects); "
          f"train-subject share = {100.0 * n_tr_subs / max(n_tr_subs + len(va_subs), 1):.1f}%", flush=True)
    return (tr_idx, va_idx)

def filter_notype_for_fold(notype_seqs, held_out_seqs, tag="val", min_keep=0):
    """v25.42.4 leak-guard: drop notype Event-only sequences whose subjects
    appear in the held-out split (val for run_cv_main, calib/thresh for
    train_one_config). Training the fog head on a held-out subject's Event
    labels leaks that subject's gait into selection and the val estimate
    (the v25.42.1 0.2574 tdcsfog number was measured WITHOUT this filter and
    is NOT reportable). Overlap with the TRAIN side is harmless -- the subject
    is already in train with full labels -- so this filters, never asserts.
    Returns (kept_seqs, dropped_subjects_sorted, kept_subjects_count)."""
    if not notype_seqs:
        return notype_seqs, [], 0
    _held = set(s.subject_id for s in held_out_seqs)
    _drop = sorted(set(s.subject_id for s in notype_seqs) & _held)
    if not _drop:
        return notype_seqs, [], len(set(s.subject_id for s in notype_seqs))
    kept = [s for s in notype_seqs if s.subject_id not in _held]
    print(f"[leak-guard v25.42.4] {tag}: DROPPED {len(_drop)} notype subjects "
          f"overlapping the held-out split ({_drop}) -- their Event labels must "
          f"not train the fog head", flush=True)
    print(f"[leak-guard v25.42.4] {tag}: {len(kept)} notype sequences remain "
          f"({len(set(s.subject_id for s in kept))} subjects)", flush=True)
    if min_keep and len(kept) < min_keep:
        print(f"[leak-guard v25.42.4] WARNING: only {len(kept)} notype sequences "
              f"survive filtering (min_keep={min_keep})", flush=True)
    return kept, _drop, len(set(s.subject_id for s in kept))

def family_shared_val_fold(supervised, fam_seqs, seed=42):
    """v25.38 (family-ensemble): winner-style per-domain models on a SHARED
    test set. The 1st place trained a defog model and a tdcsfog model (each
    4-seed ensembled) and averaged their predictions on the SAME private test
    -- per-domain specialization is where his 0.514 came from. With plain
    --families the val split is drawn inside each family (different subjects),
    so per-domain predictions cannot be averaged. This builds the POOLED
    subject-disjoint split once and filters only the TRAIN side to one family;
    the val stays the shared pooled split. Returns (tr_idx, va_idx) into
    `supervised` (family-filtered train, shared val)."""
    fam_id_set = {id(s) for s in fam_seqs}
    tr_p, va_p = build_single_split_fold(supervised, seed=seed)
    tr_f = [i for i in tr_p if id(supervised[i]) in fam_id_set]
    if not tr_f:
        raise RuntimeError(f"family_shared_val_fold: no family subjects in the pooled "
                           f"train split (family has {len(fam_seqs)} seqs)")
    return (tr_f, list(va_p))

def deployment_gap_report(cohort_metrics, lab_cohorts=("daphnet",), free_living_cohorts=("weargait", "fogstar")):
    """v25.22: the Lin & Grewal 2026 deployment-gap quantification, computed on
    OUR frozen external cohorts: window-F1 and AUC on the most lab-like
    cohort(s) vs the most free-living cohort(s), plus the F1 variance ratio
    (their published reference: identical TCN, lab F1 0.9999 vs home 0.55,
    83 % gap, 1299x variance). cohort_metrics: {cohort: window_metrics dict}."""
    lab = [cohort_metrics[c] for c in lab_cohorts if c in cohort_metrics]
    free = [cohort_metrics[c] for c in free_living_cohorts if c in cohort_metrics]
    if not lab or not free:
        return {"available": False, "note": "need >=1 lab-like and >=1 free-living cohort"}
    lab_f1 = [c["f1"] for c in lab]
    free_f1 = [c["f1"] for c in free]
    m_lab, m_free = float(np.mean(lab_f1)), float(np.mean(free_f1))
    v_lab = float(np.var(lab_f1)) if len(lab_f1) > 1 else 0.0
    v_free = float(np.var(free_f1)) if len(free_f1) > 1 else 0.0
    return {"available": True,
            "lab_window_f1": m_lab, "free_living_window_f1": m_free,
            "gap_pct": 100.0 * (m_lab - m_free) / max(abs(m_lab), 1e-9),
            "lab_auc": float(np.mean([c["auc"] for c in lab])),
            "free_living_auc": float(np.mean([c["auc"] for c in free])),
            "variance_ratio": (v_free / max(v_lab, 1e-12)) if v_lab > 0 else float("inf"),
            "reference": "Lin & Grewal 2026 (Sensors 26:1352): lab F1 0.9999 -> "
                         "home 0.55, 83 % gap, 1299x variance"}

def dataset_characteristics(seqs):
    """v25.23: per-source Table-1 statistics (the journal dataset-description
    table): n sequences, n subjects, recording hours, FoG prevalence, sampling
    rate -- the cohort facts reviewers check first (Gregorcic 2025 review
    found no placement/cohort standardization; reporting ours explicitly is
    the fix)."""
    out = {}
    for src in sorted(set(s.source_type for s in seqs)):
        ss = [s for s in seqs if s.source_type == src]
        frames = sum(len(s.labels) for s in ss)
        fog = sum(int(s.labels.sum()) for s in ss)
        fs = float(ss[0].fs_hz) if ss else float("nan")
        out[src] = {"n_sequences": len(ss),
                    "n_subjects": len(set(s.subject_id for s in ss)),
                    "hours": round(frames / max(fs, 1e-6) / 3600.0, 2),
                    "fog_prevalence_pct": round(100.0 * fog / max(frames, 1), 2),
                    "fs_hz": fs}
    return out

def build_journal_report(grand, gate, baseline_ci_diff=None):
    """v25.22: assembles the publication-ready generalization report (markdown
    + structured dict) following the top-journal protocol in
    research/generalization-benchmark.md: internal subject-disjoint CV,
    FROZEN external cohorts (daphnet/xuanwu/weargait), lab-vs-free-living
    deployment gap, 50-min adaptation, clinical operating points, ICC
    agreement vs the human-rater band, calibration, fairness, and the
    honest-claim gate. Clinical/event metrics are the headline; the subtype
    discrimination appears as per-class AP columns in the internal table and
    the pooled leaderboard-style mAP is quarantined in Appendix A
    (kaggle-track, internal diagnostic ONLY -- no leaderboard framing)."""
    fams = grand.get("families", {})
    import statistics as _st
    md = []
    md.append("# NeuroMamba -- Generalization Report (journal protocol)")
    md.append("")
    md.append(f"- Code version: {CODE_VERSION}")
    md.append("- Training data: TLVMc (defog + tdcsfog + notype), lower-back 3-axis accelerometer")
    md.append("- External cohorts (frozen): daphnet, xuanwu, weargait (binary FoG labels)")
    md.append("- Validation: subject-disjoint CV; thresholds fitted on calib splits only, "
              "then FROZEN for all external evaluation")
    md.append("- Event matching: predicted event matches a labeled episode if temporal "
              "IoU >= 0.1; FARh counts predicted events with zero labeled overlap")
    md.append("")

    md.append("## 0. Datasets (Table 1)")
    md.append("")
    dch = grand.get("dataset_characteristics") or {}
    if dch:
        md.append("| Source | Role | n sequences | n subjects | hours | FoG prevalence | fs (Hz) |")
        md.append("|---|---|---|---|---|---|---|")
        for src, d in sorted(dch.items()):
            role = "train" if src in ("defog", "tdcsfog", "notype", "notype_pseudo") else "external (frozen)"
            md.append(f"| {src} | {role} | {d['n_sequences']} | {d['n_subjects']} | "
                      f"{d['hours']} | {d['fog_prevalence_pct']} % | {d['fs_hz']} |")
        md.append("")
    else:
        md.append("(dataset characteristics not available in this report build)")
        md.append("")
    md.append("")

    md.append("## 1. Internal cross-validation (subject-disjoint)")
    md.append("")
    md.append("| Family | Fold | Window F1 | Event F1 | FARh (/h) | ECE | ICC %TF | AP StartHes | AP Turn | AP Walk |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    internal_rows = []
    for fam_name, fs in fams.items():
        for fr in fs.get("fold_results", []):
            if fr.get("status") == "failed":
                continue
            km = fr.get("kaggle_map") or {}
            icc = fr.get("icc") or {}
            farh = (fr.get("farh") or {}).get("farh_events_per_hour", float("nan"))
            md.append(f"| {fam_name} | {fr.get('fold', '?')} | {fr.get('window_f1', float('nan')):.3f} | "
                      f"{fr.get('event_f1', float('nan')):.3f} | {farh:.2f} | {fr.get('ece', float('nan')):.3f} | "
                      f"{icc.get('icc_percent_time_frozen', float('nan')):.3f} | "
                      f"{km.get('StartHesitation', float('nan')):.3f} | {km.get('Turn', float('nan')):.3f} | "
                      f"{km.get('Walking', float('nan')):.3f} |")
            internal_rows.append(fr)
    md.append("")
    if internal_rows:
        wf = [r.get("window_f1") for r in internal_rows if r.get("window_f1") is not None]
        ef = [r.get("event_f1") for r in internal_rows if r.get("event_f1") is not None]
        if wf:
            wf_txt = (f"{_st.mean(wf):.3f} +/- {_st.stdev(wf):.3f}" if len(wf) > 1 else f"{wf[0]:.3f}")
            md.append(f"- Internal window-F1 mean +/- SD (between-fold, subject generalization): **{wf_txt}** "
                      f"(contest same-distribution ceiling 0.73-0.81, Nat Commun 2024)")
        if ef:
            ef_txt = (f"{_st.mean(ef):.3f} +/- {_st.stdev(ef):.3f}" if len(ef) > 1 else f"{ef[0]:.3f}")
            md.append(f"- Internal event-F1 mean +/- SD (between-fold): **{ef_txt}**")
        _seed_sds = []
        for _r in internal_rows:
            _svm = _r.get("seed_val_maps")
            if _svm and len(_svm) > 1 and all(v is not None for v in _svm):
                _seed_sds.append(_st.stdev([float(v) for v in _svm]))
        if _seed_sds:
            md.append(f"- Between-seed SD of the internal per-class discrimination "
                      f"diagnostic (training stochasticity; see Appendix A): "
                      f"{min(_seed_sds):.4f}-{max(_seed_sds):.4f}")
    md.append("")

    md.append("## 2. Frozen external validation (the generalization claim)")
    md.append("")
    md.append("| Cohort | Win sens | Win spec | Win prec | Win F1 | Win AUC | Win PR-AUC | Event sens | Event F1 | Episodes detected | FARh (/h) | FARh p50/p95 | ECE | ICC %TF |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cohort_rows = {}
    for fs in fams.values():
        for fr in fs.get("fold_results", []):
            er = (fr.get("tripod_external_validation") or {}).get("cohorts") or {}
            for src, c in er.items():
                wm = c.get("window_metrics") or {}
                em = c.get("event_metrics") or {}
                icc = c.get("icc") or {}
                row = cohort_rows.setdefault(src, [])
                row.append((wm, em, c, icc))
    gap_input = {}
    for src, rows in sorted(cohort_rows.items()):
        wm_l = [r[0] for r in rows]; em_l = [r[1] for r in rows]
        c_l = [r[2] for r in rows]; icc_l = [r[3] for r in rows]
        def _ms(key, lst, fmt=".3f"):
            vals = [d.get(key) for d in lst if d.get(key) is not None and np.isfinite(d.get(key))]
            if not vals:
                return "n/a"
            return f"{_st.mean(vals):{fmt}} +/- {_st.stdev(vals):{fmt}}" if len(vals) > 1 else f"{vals[0]:{fmt}}"
        win_f1s = [d.get("f1") for d in wm_l if d.get("f1") is not None]
        gap_input[src] = {"f1": float(_st.mean(win_f1s)) if win_f1s else float("nan"),
                          "auc": float(_st.mean([d["auc"] for d in wm_l if d.get("auc") is not None])) if any(d.get("auc") is not None for d in wm_l) else float("nan")}
        farh_p50 = _ms("p50", [ (r[2].get("per_subject_farh") or {}) for r in rows ])
        farh_p95 = _ms("p95", [ (r[2].get("per_subject_farh") or {}) for r in rows ])
        md.append(f"| {src} | {_ms('sensitivity', wm_l)} | {_ms('specificity', wm_l)} | {_ms('precision', wm_l)} | "
                  f"{_ms('f1', wm_l)} | {_ms('auc', wm_l)} | {_ms('pr_auc', wm_l)} | "
                  f"{_ms('sensitivity', em_l)} | {_ms('f1', em_l)} | {_ms('episode_detection_rate', em_l)} | "
                  f"{_ms('farh', c_l)} | {farh_p50}/{farh_p95} | {_ms('ece', c_l)} | {_ms('icc_percent_time_frozen', icc_l)} |")
    md.append("")
    md.append("> Frozen threshold + post-processing fitted on internal calib only; never re-tuned on external data.")
    md.append("")
    md.append("Reference bands (published frozen-transfer): window-F1 0.14-0.63 per cohort "
              "(Yang et al., npj PD 2026); episode detection 88-95 % (Borzi et al. 2025); "
              "ICC(%TF) 0.562 +/- 0.141 external (npj PD 2026); human-rater ICC(%TF) 0.73-0.99.")
    md.append("")
    md.append("**Gate-cost sensitivity (same frozen threshold, gate removed):**")
    def _mmean(lst, key):
        vals = [d.get(key) for d in lst if d and d.get(key) is not None]
        return (f"{_st.mean(vals):.3f}" if vals else "n/a")
    for src, rows in sorted(cohort_rows.items()):
        wms = [r[2].get("window_metrics") or {} for r in rows]
        wmus = [r[2].get("window_metrics_ungated") or {} for r in rows]
        emus = [r[2].get("event_metrics_ungated") or {} for r in rows]
        if any(wmus):
            md.append(f"- {src}: windowF1 {_mmean(wms, 'f1')} (gated) -> {_mmean(wmus, 'f1')} (ungated); "
                      f"eventF1 ungated {_mmean(emus, 'f1')}")
    md.append("")

    dg = deployment_gap_report(gap_input)
    md.append("## 3. Lab vs free-living deployment gap")
    md.append("")
    if dg.get("available"):
        md.append(f"| Lab-like (daphnet) | Free-living (weargait) | F1 gap | AUC lab -> free | Variance ratio |")
        md.append("|---|---|---|---|---|")
        md.append(f"| {dg['lab_window_f1']:.3f} | {dg['free_living_window_f1']:.3f} | "
                  f"{dg['gap_pct']:.1f} % | {dg['lab_auc']:.3f} -> {dg['free_living_auc']:.3f} | "
                  f"{dg['variance_ratio']:.1f}x |")
        md.append("")
        md.append(f"Reference: {dg['reference']}.")
    else:
        md.append(f"{dg.get('note', 'n/a')}")
    md.append("")

    md.append("## 4. Adaptation (50-min target-site fine-tune, KU Leuven paradigm)")
    md.append("")
    md.append("| Cohort | Frozen eventF1 | Adapted eventF1 | delta |")
    md.append("|---|---|---|---|")
    n_adapt = 0
    for fs in fams.values():
        for fr in fs.get("fold_results", []):
            er = (fr.get("tripod_external_validation") or {})
            adapted = (er.get("adapted") or {}).get("cohorts") or {}
            for src, a in adapted.items():
                frozen = (er.get("cohorts") or {}).get(src, {})
                fe = frozen.get("event_f1", float("nan"))
                md.append(f"| {src} | {fe:.3f} | {a.get('event_f1', float('nan')):.3f} | "
                          f"{(a.get('event_f1', 0) or 0) - (fe if np.isfinite(fe) else 0):+.3f} |")
                n_adapt += 1
    if n_adapt == 0:
        md.append("| -- | (not enabled; set NEUROMAMBA_EXTERNAL_ADAPT=1) | | |")
    md.append("")
    md.append("Reference: Yang et al., npj PD 2026 -- external ICC(%TF) 0.562 -> 0.732 with 50 min.")
    md.append("")

    md.append("## 5. Clinical operating points")
    md.append("")
    clin_notes = []
    for fs in fams.values():
        for fr in fs.get("fold_results", []):
            cl = fr.get("cueing_latency") or {}
            med = fr.get("per_medication") or {}
            if cl:
                cl_dep = cl.get("deployable", True)
                clin_notes.append(f"- Fold {fr.get('fold')}: cueing latency median "
                                  f"{cl.get('median_latency_s', float('nan')):.2f} s, "
                                  f"within-1 s {100 * cl.get('within_1s_frac', 0):.0f} %, "
                                  f"detected {100 * cl.get('detected_frac', 0):.0f} % of {cl.get('n_events', 0)} events"
                                  + ("" if cl_dep else "  [NON-DEPLOYABLE: negative latency is an artifact, not a real causal cueing latency]"))
            for mname, mr in med.items():
                clin_notes.append(f"- Fold {fr.get('fold')} [{mname} med]: eventF1 {mr.get('event_f1', float('nan')):.3f}, "
                                  f"FARh {mr.get('farh', float('nan')):.2f}/h, windowF1 {mr.get('window_f1', float('nan')):.3f}")
    if clin_notes:
        md.extend(clin_notes)
    else:
        md.append("- (clinical fold metrics not available)")
    md.append("")
    md.append("Combined target sens >= 0.85 AND FARh <= 10/h: not demonstrated by any published "
              "system; report the coverage->FARh abstention curve instead (see fold JSONs).")
    md.append("")

    md.append("## 5b. Certified joint operating points (distribution-free envelope, v25.44)")
    md.append("")
    cert_rows = []
    n_cert_folds = 0
    for fs in fams.values():
        for fr in fs.get("fold_results", []):
            jc = fr.get("joint_certified_ops") or {}
            rows = jc.get("rows") or []
            if rows:
                _nc = sum(1 for r in rows if r.get("joint_certified"))
                if _nc:
                    n_cert_folds += 1
                best = (jc.get("certified") or [None])[0] or max(
                    rows, key=lambda r: (r.get("sens_lower_bound", 0), -r.get("trigger", 0)), default=None)
                if best is not None:
                    cert_rows.append((fr.get("fold"), jc.get("alpha"), len(rows), _nc, best))
    if cert_rows:
        md.append("| Fold | alpha | grid pts | certified | trigger | sens >= LB (CP) | sens observed | P(FARh > budget) <= UB |")
        md.append("|---|---|---|---|---|---|---|---|")
        for fold_no, alpha_jc, npts, nc, b in cert_rows:
            md.append(f"| {fold_no} | {alpha_jc} | {npts} | {nc} | {b['trigger']:.3f} | "
                      f"{b['sens_lower_bound']:.3f} | {b['sens_point']:.3f} ({b['tp_events']}/{b['n_events']}) | "
                      f"{b['violation_rate_upper_bound']:.2f} "
                      f"({int(b['farh_budget_violations'])}/{int(b['n_subjects'])} calib subjects) |")
        md.append("")
        md.append("Clopper-Pearson exact bounds at Bonferroni delta = alpha/(2m) across the trigger grid and "
                  "both risk families: **P(sens >= LB) >= 1-delta** and **P(subject violates the FARh budget) <= UB** "
                  "hold simultaneously per listed point, marginal over the calibration draw under "
                  "exchangeability (closed-form Learn-then-Test logic; Angelopoulos et al., arXiv:2110.01052; "
                  "Conformal Risk Control, arXiv:2208.02814). Report-only: these envelopes never select or "
                  f"deploy an operating point. {n_cert_folds} fold(s) carry >=1 jointly certified point.")
    else:
        md.append("No trigger on any fold's calibration grid carries the joint guarantee at the target "
                  "(sens >= 0.85 AND subject-level FARh-budget violation rate bound); the honest claim "
                  "remains the observed operating points in Section 5 and the coverage->FARh curve.")
    md.append("")

    md.append("## 6. Statistical rigor and honest-claim gate")
    md.append("")
    md.append(f"- Bootstrap CIs (subject-level): see `bootstrap_ci` per family in the results JSON.")
    md.append(f"- Subject-overlap screen (train  intersect  external): expected 0; reported per fold.")
    if internal_rows:
        _n_ef_total = len(internal_rows)
        _n_ef_ok = sum(1 for r in internal_rows
                       if r.get("event_f1") is not None and np.isfinite(r.get("event_f1")))
        md.append(f"- Event-F1 evaluability (v25.44): {_n_ef_ok}/{_n_ef_total} internal fold(s) "
                  f"evaluable -- degenerate empty folds are reported nan and EXCLUDED from means, "
                  f"never counted as perfect scores.")
    if baseline_ci_diff:
        if baseline_ci_diff.get("status") == "NOT_EVALUATED":
            md.append(f"- vs frozen ResNet1D comparator: **NOT_EVALUATED** "
                      f"({baseline_ci_diff.get('reason', 'artifact unavailable')})")
        else:
            _nvb = baseline_ci_diff.get("n_valid_boot")
            if _nvb is not None and _nvb == 0:
                md.append("- vs frozen ResNet1D comparator: **NON-COMPUTABLE** (0 valid subject-level "
                          "bootstraps -- val split too small); gate_2 treated as failed, not as a tie")
            else:
                md.append(f"- vs frozen ResNet1D comparator: mean delta {baseline_ci_diff.get('diff_mean', float('nan')):.4f} "
                          f"[{baseline_ci_diff.get('ci_lo', float('nan')):.4f}, {baseline_ci_diff.get('ci_hi', float('nan')):.4f}] "
                          f"non-overlapping zero: {baseline_ci_diff.get('non_overlapping_zero', False)}")
    md.append(f"- SOTA gate: {json.dumps(gate, default=str)}")
    md.append("")

    md.append("## 7. Limitations and reporting notes")
    md.append("")
    md.append("- **Fairness (DPR/EOR):** requires demographics metadata; in the 4-dataset "
              "protocol (tlvmc/daphnet/xuanwu/weargait) only medication-state subgroups "
              "are reportable. Add `--external fogstar` to report DPR/EOR by sex/age/H&Y "
              "(Odonga et al. protocol).")
    md.append("- **Figures:** ROC/PR curves per cohort and Bland-Altman plots for %TF are "
              "generated from the per-cohort probability exports (see results JSON); "
              "ICC is reported numerically per the npj PD 2026 convention.")
    md.append("- **Rater ceiling:** human inter-rater ICC(%TF) 0.73-0.99, ICC(#episodes) "
              "0.39-0.95; report model-vs-expert next to expert-vs-expert, never "
              "assert 'exceeds experts' without that comparison.")
    md.append("- **Operating point:** application bar F1 >= 0.75 (alerting) / ~ 0.80 "
              "(real-time cueing) per Lin & Grewal 2026; the combined sens >= 0.85 "
              "AND FARh <= 10/h target has no published precedent and is reported "
              "as the coverage->FARh abstention curve.")
    md.append("- **Walking-gate vs start hesitation (clinical):** the activity gate "
              "attenuates the deployed FoG probability during standing, which can "
              "suppress start-hesitation (a stand->initiation transition) at "
              "inference. Subtype APs (ungated) are unaffected; the clinical "
              "event metrics on TLVMc are the affected axis.")
    md.append("")
    bench = grand.get("benchmark") or {}
    if bench:
        md.append("## 8. Cohort benchmarks (trained on cohort, LOSO -- NOT frozen transfer)")
        md.append("")
        for name, b in bench.items():
            md.append(f"**{name}** -- {b.get('n_folds', 0)} LOSO folds "
                      f"({b.get('trained_on', '')}); "
                      "threshold-free AUC/PR-AUC + F1 at 0.5, comparable to the "
                      "trained-on-cohort literature (DeepFoG sens 0.83/spec 0.88; "
                      "Lin & Grewal window F1 0.9999).")
            if b.get("n_empty_folds"):
                md.append(f"*{b['n_empty_folds']} fold(s) had an all-negative held-out subject "
                          f"({b.get('empty_folds')}) -- annotated empty (AUC/eventF1 nan), excluded "
                          f"from the mean; mean windowF1 over non-empty folds = "
                          f"{b.get('mean_window_f1_nonempty', float('nan')):.3f} "
                          f"(at the calib-derived inner threshold: "
                          f"{b.get('mean_window_f1_inner_nonempty', float('nan')):.3f}), "
                          f"mean AUC = {b.get('mean_auc_nonempty', float('nan')):.3f}.*")
            md.append("")
            md.append("| Fold | Val subject(s) | +win/tot | Win sens | Win spec | Win F1 | Win F1@inner | AUC | PR-AUC | Event F1 |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for fr in b.get("folds", []):
                w = fr.get("window") or {}
                e = fr.get("event") or {}
                if fr.get("empty"):
                    md.append(f"| {fr.get('fold', '?')} | {','.join(fr.get('val_subjects', []))} | "
                              f"{fr.get('n_val_positive_windows', 0)}/{fr.get('n_val_windows', 0)} | "
                              f"-- | -- | -- | -- | nan | nan | nan (empty) |")
                    continue
                wi = fr.get("window_inner") or {}
                md.append(f"| {fr.get('fold', '?')} | {','.join(fr.get('val_subjects', []))} | "
                          f"{fr.get('n_val_positive_windows', 0)}/{fr.get('n_val_windows', 0)} | "
                          f"{w.get('sensitivity', float('nan')):.3f} | "
                          f"{w.get('specificity', float('nan')):.3f} | {w.get('f1', float('nan')):.3f} | "
                          f"{wi.get('f1', float('nan')):.3f} | "
                          f"{w.get('auc', float('nan')):.3f} | {w.get('pr_auc', float('nan')):.3f} | "
                          f"{e.get('f1', float('nan')):.3f} |")
            md.append("")
    md.append("## Appendix A. Kaggle-track -- internal diagnostic ONLY (not part of the journal claim)")
    md.append("")
    md.append("The pooled mean per-subtype AP ('mAP') mimics the contest evaluation "
              "(defog and tdcsfog files scored by their own family models, per-file "
              "union pooling -- no same-subject averaging) and is comparable to the "
              "contest leaderboard (private-LB mAP 0.514, 1st place). It is reported "
              "here solely as an internal diagnostic; the manuscript's primary "
              "discrimination claim is the per-class AP column in Section 1. "
              "**This appendix must not be used as a headline result.**")
    md.append("")
    for fam_name, fs in fams.items():
        ens = fs.get("fold_ensemble_map")
        if not ens:
            continue
        md.append(f"### {fam_name} family -- pooled OOF (kaggle-track)")
        md.append("")
        md.append(f"| Subtype | AP (pooled OOF) |")
        md.append("|---|---|")
        for _k in ("StartHesitation", "Turn", "Walking"):
            _v = ens.get(_k)
            md.append(f"| {_k} | {_v:.4f}" if _v is not None else f"| {_k} | -- |")
        md.append(f"| **mAP** | **{ens.get('mAP', float('nan')):.4f}** |")
        md.append("")
        _ci = fs.get("bootstrap_ci")
        if _ci:
            md.append(f"- Subject-level bootstrap 95% CI: "
                      f"[{_ci.get('ci_lo', float('nan')):.4f}, {_ci.get('ci_hi', float('nan')):.4f}] "
                      f"(unit: {_ci.get('unit', 'subject')}, n={_ci.get('n_boot', '?')})")
        _bd = fs.get("baseline_ci_diff")
        if _bd:
            md.append(f"- vs frozen ResNet1D comparator (kaggle-track): mean delta "
                      f"{_bd.get('diff_mean', float('nan')):.4f} "
                      f"[{_bd.get('ci_lo', float('nan')):.4f}, {_bd.get('ci_hi', float('nan')):.4f}], "
                      f"non-overlapping zero: {_bd.get('non_overlapping_zero', False)}")
        md.append("")
    md.append("*Report generated by neuromamba_v25_sota.py `--journal-report`. "
              "The pooled mAP appears only in Appendix A as an internal "
              "subtype-discrimination diagnostic; no leaderboard comparison is "
              "claimed in the manuscript body.*")
    report = {
        "protocol": {"datasets": ["tlvmc (train)", "daphnet", "xuanwu", "weargait"],
                     "validation": "subject-disjoint CV + frozen external",
                     "threshold_rule": "frozen from calib only",
                     "event_matching": "temporal IoU >= 0.1"},
        "internal_rows": internal_rows,
        "external_cohorts": {src: {"n_folds": len(rows)} for src, rows in cohort_rows.items()},
        "deployment_gap": dg,
        "n_adaptation_cohorts": n_adapt,
        "gate": gate,
    }
    return {"markdown": "\n".join(md), "structured": report}

def run_cohort_loso_benchmark(cohort_seqs, device, epochs=30, lr=3e-4, batch_size=8,
                              seed=42, model_type="neuromamba"):
    """v25.27: train the SAME architecture ON one external cohort with
    leave-one-subject-out CV -- the protocol the Daphnet literature uses for
    its numbers (DeepFoG window sens 0.83/spec 0.88; Lin & Grewal window F1
    0.9999). Labeled 'trained on cohort', NOT frozen transfer -- it exists so
    a reviewer can compare our architecture on the same footing as published
    Daphnet models. Reports window sens/spec/prec/F1 at threshold 0.5 plus
    threshold-free AUC/PR-AUC, and event F1 at 0.5.

    v25.28 (benchmark hygiene, from the first Daphnet LOSO run):
      - binary-only cohorts (subtype_observed=0) FORCE window-F1-based model
        selection (NEUROMAMBA_SELECT_BY_WINDOW_F1=1): the subtype-mAP
        selection metric is structurally dead on such data (subtype_pos=[0,0,0]
        every epoch) and collapsed several LOSO folds.
      - all-negative held-out folds are detected and ANNOTATED (empty=True,
        AUC/PR-AUC nan) instead of silently contributing f1=1.0 / AUC=nan to
        the mean; the summary mean is computed over non-empty folds only and
        n_empty_folds is reported.
      - per-fold diagnostics print the held-out subject ids and their
        positive/negative frame + window counts, so all-negative-fold claims
        are verifiable from the log alone.

    v25.29 (inner operating point, from v25.28's gap: mean AUC 0.873 but
    winF1@0.5 0.149 -- the model ranked FoG well, the fixed 0.5 threshold
    never matched the cohort probability scale):
      - each fold derives an INNER trigger on its (positive-aware) calib
        split by window-F1 and reports winF1/eventF1 at that trigger on the
        held-out subject, next to the fixed-0.5 reference.
      - threshold-free AUC/PR-AUC remain the headline for 1-subject-calib
        folds; mean_window_f1_inner_nonempty is reported in the summary and
        journal section 8."""
    from sklearn.model_selection import GroupKFold
    groups = np.array([s.subject_id for s in cohort_seqs])
    n_subj = len(set(groups))
    kf = GroupKFold(n_splits=min(n_subj, 10))
    folds = list(kf.split(cohort_seqs, groups=groups))
    rows = []
    any_subtype_obs = any(bool(np.any(s.subtype_observed)) for s in cohort_seqs)
    _saved_sel = os.environ.get("NEUROMAMBA_SELECT_BY_WINDOW_F1")
    if not any_subtype_obs:
        os.environ["NEUROMAMBA_SELECT_BY_WINDOW_F1"] = "1"
        print(f"[benchmark v25.28] binary-only cohort -> model selection forced to "
              f"calib WINDOW-F1 (subtype-mAP selection is structurally dead: "
              f"subtype_observed all zeros)", flush=True)
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr = [cohort_seqs[i] for i in tr_idx]
        va = [cohort_seqs[i] for i in va_idx]
        try:
            f_tr, f_ca, f_th, _ = three_way_subject_split(tr, seed=seed + fold_idx)
        except Exception:
            rng = np.random.RandomState(seed + fold_idx)
            subs = list(set(s.subject_id for s in tr))
            rng.shuffle(subs)
            n = len(subs)
            th_s = set(subs[:max(1, n // 6)])
            ca_s = set(subs[max(1, n // 6):max(2, n // 3)])
            f_tr = [s for s in tr if s.subject_id not in (th_s | ca_s)]
            f_ca = [s for s in tr if s.subject_id in ca_s]
            f_th = [s for s in tr if s.subject_id in th_s]
        def _has_pos(seqs):
            return any(int(s.labels.sum()) > 0 for s in seqs)
        if not _has_pos(f_ca) and len(f_ca) > 0:
            pos_tr = [s for s in f_tr if int(s.labels.sum()) > 0]
            neg_ca = [s for s in f_ca if int(s.labels.sum()) == 0]
            if pos_tr and neg_ca:
                swap_in = pos_tr[0]
                swap_out = neg_ca[0]
                f_ca = [s for s in f_ca if s is not swap_out] + [swap_in]
                f_tr = [s for s in f_tr if s is not swap_in] + [swap_out]
                print(f"[benchmark v25.28] fold {fold_idx + 1}: calib had no FoG "
                      f"positives -- swapped subject {swap_in.subject_id} into calib", flush=True)
        model, res = train_one_config(f_tr, f_ca, f_th, device=device, epochs=epochs, lr=lr,
                                      batch_size=batch_size, seed=seed + fold_idx,
                                      model_type=model_type, fog_warmup_epochs=2, ramp_epochs=5)
        rv = run_inference_cached(model, va, device, collect_gate=False, window_roll=False)
        probs, labels, ids = rv[0], rv[1], rv[2]
        va_subjects = sorted(set(s.subject_id for s in va))
        n_pos_frames = int(labels.sum())
        n_win = 0
        n_win_pos = 0
        for uid in np.unique(ids):
            m = ids == uid
            L = int(m.sum())
            for st in range(0, L, 100):
                en = min(st + 100, L)
                if en - st < 50:
                    break
                n_win += 1
                if labels[m][st:en].max() > 0.5:
                    n_win_pos += 1
        wm = window_level_metrics(labels, probs, ids, threshold=0.5)
        em = event_level_metrics(labels, (probs >= 0.5).astype(np.int64), ids)
        empty = n_pos_frames == 0
        inner_trigger = 0.5
        try:
            rv_ca = run_inference_cached(model, f_ca, device, collect_gate=False, window_roll=False)
            p_ca, l_ca, id_ca = rv_ca[0], rv_ca[1], rv_ca[2]
            if int(l_ca.sum()) > 0:
                cand = sorted(set(np.quantile(p_ca, np.linspace(0.05, 0.95, 19)).tolist()) | {0.5})
                best_t, best_wf = 0.5, -1.0
                for t in cand:
                    w_ = window_level_f1(l_ca, p_ca, id_ca, threshold=float(t))["window_f1"]
                    if w_ > best_wf:
                        best_wf, best_t = w_, float(t)
                inner_trigger = best_t
        except Exception:
            inner_trigger = 0.5
        wm_inner = window_level_metrics(labels, probs, ids, threshold=inner_trigger)
        em_inner = event_level_metrics(labels, (probs >= inner_trigger).astype(np.int64), ids)
        row = {"fold": fold_idx + 1,
               "n_val_subjects": len(va_subjects),
               "val_subjects": va_subjects,
               "n_val_frames": int(len(labels)),
               "n_val_positive_frames": n_pos_frames,
               "n_val_windows": n_win, "n_val_positive_windows": n_win_pos,
               "empty": empty,
               "inner_trigger": inner_trigger,
               "window_inner": wm_inner, "event_inner": em_inner,
               "window": wm, "event": em}
        rows.append(row)
        if empty:
            print(f"[benchmark v25.28] fold {fold_idx + 1}: HELD-OUT SUBJECT(S) {va_subjects} "
                  f"HAVE ZERO FoG POSITIVES ({n_win_pos}/{n_win} positive windows) -- fold "
                  f"marked empty; AUC/PR-AUC/eventF1 are nan by design (excluded from the mean)", flush=True)
        else:
            print(f"[benchmark v25.28] {model_type} LOSO fold {fold_idx + 1} "
                  f"(val={va_subjects}, {n_win_pos}/{n_win} positive windows): "
                  f"winF1@0.5={wm['f1']:.3f} winF1@inner={wm_inner['f1']:.3f} "
                  f"(thr={inner_trigger:.3f}) AUC={wm['auc']:.3f} PR-AUC={wm['pr_auc']:.3f} "
                  f"eventF1@0.5={em['f1']:.3f} eventF1@inner={em_inner['f1']:.3f}", flush=True)
    if _saved_sel is None:
        os.environ.pop("NEUROMAMBA_SELECT_BY_WINDOW_F1", None)
    else:
        os.environ["NEUROMAMBA_SELECT_BY_WINDOW_F1"] = _saved_sel
    non_empty = [r for r in rows if not r["empty"]]
    summary = {"trained_on": "cohort (LOSO, NOT frozen transfer)",
               "n_folds": len(rows),
               "n_empty_folds": len(rows) - len(non_empty),
               "empty_folds": [r["fold"] for r in rows if r["empty"]],
               "folds": rows}
    if non_empty:
        wf = [r["window"]["f1"] for r in non_empty]
        auc = [r["window"]["auc"] for r in non_empty]
        ef = [r["event"]["f1"] for r in non_empty]
        wfi = [r["window_inner"]["f1"] for r in non_empty]
        efi = [r["event_inner"]["f1"] for r in non_empty]
        import statistics as _st
        summary["mean_window_f1_nonempty"] = float(_st.mean(wf))
        summary["mean_auc_nonempty"] = float(_st.mean(auc))
        summary["mean_window_f1_inner_nonempty"] = float(_st.mean(wfi))
        summary["mean_event_f1_inner_nonempty"] = float(_st.mean([v for v in efi if np.isfinite(v)])) \
            if any(np.isfinite(v) for v in efi) else float("nan")
        summary["mean_event_f1_nonempty"] = float(_st.mean([v for v in ef if np.isfinite(v)])) \
            if any(np.isfinite(v) for v in ef) else float("nan")
    print(f"[benchmark v25.28] {model_type} LOSO summary: {len(rows)} folds, "
          f"{summary.get('n_empty_folds', 0)} all-negative (excluded), "
          f"mean windowF1(non-empty)={summary.get('mean_window_f1_nonempty', float('nan')):.3f} "
          f"mean windowF1@inner={summary.get('mean_window_f1_inner_nonempty', float('nan')):.3f} "
          f"mean AUC={summary.get('mean_auc_nonempty', float('nan')):.3f}", flush=True)
    return summary

def preflight_audit(args, run_tid):
    """v25.42.6 Tier A: preflight audit -- seconds, NO GPU allocation.

    Verifies: dataset/cache identity, N_CHANNELS contract, split subject
    disjointness (incl. notype), duplicate fingerprints, checkpoint identity
    compatibility, baseline artifact availability. Returns True (safe to run)
    or False (blocking issue -- do NOT allocate GPU time)."""
    problems = []
    checks = []
    print("=" * 70)
    print("TIER A PREFLIGHT AUDIT (no GPU allocation)")
    print("=" * 70)
    try:
        sequences = load_datasets(args.datasets)
        supervised = filter_supervised_sequences(sequences)
        notype_seqs = [s for s in sequences if s.source_type == "notype"]
        checks.append(("dataset load + cache", "OK"))
        print(f"[audit] {len(sequences)} sequences loaded "
              f"(cache hit = canonical sequence identity OK)", flush=True)
    except Exception as e:
        checks.append(("dataset load", f"FAIL {e}"))
        problems.append(f"dataset load: {e}")
        print(f"[audit] FAIL: {e}", flush=True)
        print("[audit] RESULT: BLOCKED (no data) -- fix dataset before allocating GPU", flush=True)
        return False

    _ch = N_CHANNELS
    if _ch != 13:
        problems.append(f"N_CHANNELS contract changed: {_ch} (expected 13)")
        checks.append(("N_CHANNELS contract", f"CHANGED {_ch}"))
    else:
        checks.append(("N_CHANNELS contract", f"{_ch} OK"))

    fam_name = args.families[0] if args.families else "all"
    fam_seqs = [s for s in supervised if s.source_type == fam_name] if fam_name != "all" else supervised
    try:
        partitions = {
            "train": [], "calib": [], "thresh": [], "val": [], "notype_train": []}
        if getattr(args, "family_shared_val", False) and args.n_splits == 1 and fam_name != "all":
            svf = family_shared_val_fold(supervised, fam_seqs, seed=args.seed)
            tr_idx, va_idx = svf
            fam_train = [supervised[i] for i in tr_idx]
            va = [supervised[i] for i in va_idx]
            f_tr, f_ca, f_th, _diag = rare_subtype_aware_split(
                fam_train, seed=args.seed,
                calib_frac=float(get("NEUROMAMBA_CALIB_FRAC", "0.15")),
                thresh_frac=float(get("NEUROMAMBA_THRESH_FRAC", "0.15")))
            partitions = {"train": f_tr, "calib": f_ca, "thresh": f_th,
                          "val": va, "notype_train": []}
            nt_val, _d1, _n1 = filter_notype_for_fold(notype_seqs, va, tag="audit val")
            nt_train, _d2, _n2 = filter_notype_for_fold(
                nt_val, list(f_ca) + list(f_th), tag="audit calib/thresh")
            partitions["notype_train"] = nt_train
        else:
            if args.n_splits == 1:
                _fold = build_single_split_fold(fam_seqs, seed=args.seed)
                tr_idx, va_idx = _fold
            else:
                folds = sh_balanced_group_folds(fam_seqs, args.n_splits, seed=args.seed)
                tr_idx, va_idx = folds[0]
            tr = [fam_seqs[i] for i in tr_idx]
            va = [fam_seqs[i] for i in va_idx]
            f_tr, f_ca, f_th, _diag = rare_subtype_aware_split(
                tr, seed=args.seed,
                calib_frac=float(get("NEUROMAMBA_CALIB_FRAC", "0.15")),
                thresh_frac=float(get("NEUROMAMBA_THRESH_FRAC", "0.15")))
            partitions = {"train": f_tr, "calib": f_ca, "thresh": f_th,
                          "val": va, "notype_train": []}
            nt_val, _d1, _n1 = filter_notype_for_fold(notype_seqs, va, tag="audit val")
            nt_train, _d2, _n2 = filter_notype_for_fold(
                nt_val, list(f_ca) + list(f_th), tag="audit calib/thresh")
            partitions["notype_train"] = nt_train
        from collections import defaultdict as _dd
        subj = {n: set(s.subject_id for s in partitions[n]) for n in partitions}
        overlaps = []
        for a in partitions:
            for b in partitions:
                if a >= b:
                    continue
                inter = subj[a] & subj[b]
                if inter and (a in ("calib", "thresh", "val") or b in ("calib", "thresh", "val")):
                    overlaps.append(f"{a}x{b}={len(inter)}subj")
        if overlaps:
            problems.append(f"subject overlap across held-out partitions: {overlaps}")
            checks.append(("split subject disjointness", f"FAIL {overlaps}"))
        else:
            checks.append(("split subject disjointness", "OK (0 held-out overlaps)"))
        try:
            import subject_leakage_audit_dashboard as _aud
            _dup = _aud.duplicate_fingerprints(partitions)
            _crit = [d for d in _dup if d["risk"] == "CRITICAL"]
            if _crit:
                problems.append(f"{len(_crit)} duplicate fingerprints across splits")
                checks.append(("duplicate fingerprints", f"FAIL {len(_crit)}"))
            else:
                checks.append(("duplicate fingerprints", "OK"))
        except Exception as e:
            checks.append(("duplicate fingerprints", f"WARN {e}"))
    except Exception as e:
        problems.append(f"split/leak audit failed: {e}")
        checks.append(("split/leak audit", f"FAIL {e}"))

    _smoke_prefix = "smoke_" if _flag("NEUROMAMBA_SMOKE", "0") else ""
    fold_json = f"{_smoke_prefix}neuromamba_v25_fold1_{fam_name}_checkpoint.json"
    ckpt_dir = Path(get("NEUROMAMBA_CKPT_DIR", "neuromamba_v25_ckpt"))
    _fold_tid = run_tid
    if os.path.exists(fold_json):
        try:
            with open(fold_json) as f:
                prev = json.load(f)
            _pt = prev.get("training_identity")
            if _pt and _pt == _fold_tid:
                checks.append(("checkpoint identity", "MATCH (fold reusable)"))
            elif not _pt and prev.get("code_version") in TRAINING_COMPAT_VERSIONS:
                checks.append(("checkpoint identity", "MATCH (legacy compatible)"))
            else:
                msg = "MISMATCH -> fresh training required"
                checks.append(("checkpoint identity", msg))
                print(f"[audit] checkpoint identity: {msg} "
                      f"(saved {(_pt or 'legacy')[:12]} vs current {_fold_tid[:12]})", flush=True)
            _b = (prev.get("baseline_arrays") or {}).get("npz")
            if getattr(args, "run_baseline_comparator", False):
                if _b and os.path.exists(_b):
                    checks.append(("baseline artifact", "OK"))
                else:
                    msg = "MISSING (will regenerate comparator only)"
                    checks.append(("baseline artifact", msg))
                    print(f"[audit] baseline artifact: {msg}", flush=True)
        except Exception as e:
            checks.append(("checkpoint read", f"WARN {e}"))
    else:
        checks.append(("checkpoint identity", "no saved fold (fresh)"))

    _nt_subs = set(s.subject_id for s in notype_seqs)
    _sup_subs = set(s.subject_id for s in supervised)
    checks.append(("notype x supervised", f"{len(_nt_subs & _sup_subs)} shared (by-design, guard active)"))

    for name, status in checks:
        print(f"[audit] {name:32s} {status}", flush=True)
    if problems:
        print("[audit] RESULT: BLOCKED -- do NOT allocate GPU time", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        return False
    print("[audit] RESULT: OK -- safe to run (Tier B smoke then Tier C full)", flush=True)
    return True

def run_cv_main(args):
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return json.JSONEncoder.default(self, o)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[kaggle_eval] CUDA unavailable, falling back to {device}", flush=True)
    print(f"[device] requested={args.device} resolved={device} "
          f"cuda_available={torch.cuda.is_available()} "
          f"cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}", flush=True)

    sequences = load_datasets(args.datasets)
    external_seqs = load_datasets(args.external) if args.external else None
    supervised = filter_supervised_sequences(sequences)
    notype_seqs = [s for s in sequences if s.source_type == "notype"]

    _lodo_held = lodo_parse_env()
    if _lodo_held:
        if not external_seqs:
            raise RuntimeError("[lodo v25.61] NEUROMAMBA_LODO is set but --external "
                               "is empty: the held-out cohorts must be loadable")
        _ext_by_src = {}
        for _s in external_seqs:
            _ext_by_src.setdefault(_s.source_type, []).append(_s)
        _missing = [c for c in _lodo_held if c not in _ext_by_src]
        if _missing:
            raise RuntimeError(f"[lodo v25.61] held cohort(s) {_missing} produced no "
                               f"sequences (loaded externals: {sorted(_ext_by_src)})")
        lodo_namespace_guard(supervised + external_seqs)
        _held_seqs = [s for c in _lodo_held for s in _ext_by_src.get(c, [])]
        _pool_ext = [s for c, ss in sorted(_ext_by_src.items()) if c not in _lodo_held
                     for s in ss]
        _lodo_tlvmc_pool = [s for s in supervised
                            if s.source_type in _LODO_TLVMC_SOURCES]
        if _flag("NEUROMAMBA_LODO_EXCLUDE_FROZEN_VAL", "0"):
            _fz_train, _fz_calib, _fz_thresh, _ = rare_subtype_aware_split(
                list(_lodo_tlvmc_pool), seed=42,
                calib_frac=float(get("NEUROMAMBA_CALIB_FRAC", "0.15")),
                thresh_frac=float(get("NEUROMAMBA_THRESH_FRAC", "0.15")))
            _fz_val_subjects = set(s.subject_id for s in (_fz_calib + _fz_thresh))
            _before = len(_lodo_tlvmc_pool)
            _lodo_tlvmc_pool = [s for s in _lodo_tlvmc_pool
                                if s.subject_id not in _fz_val_subjects]
            print(f"[lodo v28bu2] frozen-val exclusion ACTIVE: removed "
                  f"{_before - len(_lodo_tlvmc_pool)} seqs "
                  f"({len(_fz_val_subjects)} frozen-val subjects) -- internal "
                  f"mAP check now genuinely held out for this arm", flush=True)
        supervised = supervised + _pool_ext + _held_seqs
        external_seqs = _held_seqs
        print(f"[lodo v25.61] ACTIVE: held={_lodo_held} (val+external-eval only); "
              f"train pool = TLVMC supervised + externals "
              f"{[c for c in sorted(_ext_by_src) if c not in _lodo_held]} "
              f"(+{len(_pool_ext)} seqs)", flush=True)

    if _flag("NEUROMAMBA_SMOKE", "0"):
        _sm = max(4, int(get("NEUROMAMBA_SMOKE_MAX_SUBJECTS", "8")))
        _sm_seqs = int(get("NEUROMAMBA_SMOKE_MAX_SEQS_PER_SUBJECT", "2"))
        _sm_seed = int(get("NEUROMAMBA_SMOKE_SEED", "123"))
        _sm_epochs = int(get("NEUROMAMBA_SMOKE_EPOCHS", "3"))
        _rng = np.random.RandomState(_sm_seed)
        _fam_pool = supervised
        if args.families:
            _fam_pool = [s for s in supervised if s.source_type == args.families[0]]
        if len(set(s.subject_id for s in _fam_pool)) >= 4:
            _keep = {}
            for s in _fam_pool:
                _keep.setdefault(s.subject_id, []).append(s)
        else:
            _keep = {}
            for s in supervised:
                _keep.setdefault(s.subject_id, []).append(s)
        _subs = sorted(_keep)[:_sm]
        _sel = [s for sid in _subs for s in _keep[sid][:_sm_seqs]]
        supervised = _sel
        notype_seqs = [s for s in notype_seqs if s.subject_id in set(_subs)][: _sm_seqs * _sm]
        args.epochs = min(args.epochs, _sm_epochs)
        os.environ["NEUROMAMBA_SMOKE"] = "1"
        os.environ["NEUROMAMBA_CKPT_DIR"] = "neuromamba_v25_ckpt_smoke"
        if not args.output:
            args.output = "smoke_results.json"
        print(f"[smoke v25.42.6] Tier B SMOKE: {len(supervised)} supervised seqs "
              f"({len(set(s.subject_id for s in supervised))} subjects, "
              f"family={args.families[0] if args.families else 'pooled'}), "
              f"{len(notype_seqs)} notype, epochs={args.epochs}, seed={_sm_seed} "
              f"-- FUNCTIONAL TEST ONLY, metrics not reportable", flush=True)

    if _lodo_held and _flag("NEUROMAMBA_SMOKE", "0"):
        _in_pool = {id(s) for s in supervised}
        _lost = [s for s in (external_seqs or []) if id(s) not in _in_pool]
        if _lost:
            supervised = supervised + _lost
            print(f"[lodo v25.61] smoke re-injected {len(_lost)} held sequences "
                  f"dropped by the smoke subsample", flush=True)
        _pool_tlv = [s for s in supervised
                     if s.source_type in _LODO_TLVMC_SOURCES]
        if len({s.subject_id for s in _pool_tlv}) < 4:
            _by_sub = {}
            for _s in _lodo_tlvmc_pool:
                _by_sub.setdefault(_s.subject_id, []).append(_s)
            _tlv_slice = []
            for _sid in sorted(_by_sub):
                _tlv_slice.extend(_by_sub[_sid][:2])
                if len(_tlv_slice) >= 12:
                    break
            supervised = supervised + _tlv_slice
            print(f"[lodo v25.61] smoke re-injected {len(_tlv_slice)} TLVMC "
                  f"sequences across {len({s.subject_id for s in _tlv_slice})} "
                  f"subjects (inner split needs the TLVMC slice)", flush=True)
    audit_subtype_contract(supervised + ([] if _lodo_held else (external_seqs or [])),
                           out_path="subtype_contract_audit_v25.json")

    if args.families:
        families = {f: [s for s in supervised if s.source_type == f] for f in args.families}
        dropped = [f for f, v in families.items() if not v]
        families = {k: v for k, v in families.items() if v}
        if dropped:
            print(f"[kaggle_eval v25.20] --families: no sequences for {dropped} (dropped)", flush=True)
        print(f"[kaggle_eval v25.20] --families={args.families} -> training families: {list(families.keys())}", flush=True)
    elif args.domain_split:
        families = {"defog": [s for s in supervised if s.source_type == "defog"],
                    "tdcsfog": [s for s in supervised if s.source_type == "tdcsfog"]}
    else:
        families = {"all": supervised}
        n_sh = sum(1 for s in supervised if ((s.y_subtype[:, 0] > 0.5) & s.train_mask.astype(bool) & s.subtype_observed[:, 0]).sum() > 0)
        n_subj = len(set(s.subject_id for s in supervised))
        print(f"[kaggle_eval v25] pooled family: {len(supervised)} sequences, {n_subj} subjects, "
              f"{n_sh} sequences containing observed Start Hesitation", flush=True)
        if notype_seqs:
            print(f"[kaggle_eval v25] notype: {len(notype_seqs)} sequences (Event-target training, subtypes masked)", flush=True)

    if _lodo_held:
        families = {"lodo_" + "_".join(_lodo_held): supervised}
        print(f"[lodo v25.61] families overridden -> {list(families.keys())} "
              f"({len(supervised)} pooled seqs incl. held-out val)", flush=True)

    grand = {"config": vars(args), "families": {}}
    all_maps = []

    for fam_name, fam_seqs in families.items():
        _min_fam = 2 if _flag("NEUROMAMBA_SMOKE", "0") else 20  # v25.42.6: smoke lowers the skip threshold
        if len(fam_seqs) < _min_fam:
            print(f"[kaggle_eval] family {fam_name} too few sequences ({len(fam_seqs)}), skip", flush=True)
            continue
        print("=" * 64)
        print(f"MODEL FAMILY: {fam_name}")
        print("=" * 64)
        src_labels = np.array([s.source_type for s in fam_seqs])
        groups = np.array([s.subject_id for s in fam_seqs])
        if args.n_splits == 1:
            folds = [build_single_split_fold(fam_seqs, seed=args.seed)]
            print("[kaggle_eval v25.20] contest-mimic track: 1 split "
                  f"(NEUROMAMBA_CALIB_FRAC={get('NEUROMAMBA_CALIB_FRAC', '0.15')}, "
                  f"NEUROMAMBA_THRESH_FRAC={get('NEUROMAMBA_THRESH_FRAC', '0.15')})", flush=True)
        fold_src = fam_seqs
        _split_mode = getattr(args, "split_mode", None)
        if _split_mode is None:
            if getattr(args, "family_shared_val", False) and args.n_splits == 1 \
                    and len(families) == 1 and fam_name != "all":
                _split_mode = "shared_pooled"
            elif args.n_splits > 1:
                _split_mode = "grouped_cv"
            else:
                _split_mode = "pure_family"
        if _lodo_held:
            _split_mode = "lodo"
            print(f"[split-mode v25.61] LODO protocol forced (held={_lodo_held}); "
                  f"arg.split_mode={getattr(args, 'split_mode', None)!r} ignored", flush=True)
        if _split_mode == "shared_pooled" and not (args.n_splits == 1 and len(families) == 1 and fam_name != "all"):
            print(f"[split-mode v25.42.8] shared_pooled requires exactly one family "
                  f"and n_splits==1 (got families={list(families.keys())}, "
                  f"n_splits={args.n_splits}) -- falling back to pure_family", flush=True)
            _split_mode = "pure_family"
        print(f"[split-mode v25.42.8] PRIMARY SPLIT PROTOCOL = {_split_mode.upper()} "
              f"(family={fam_name}, n_splits={args.n_splits}, n_repeats={getattr(args, 'n_repeats', 1)})",
              flush=True)
        if _split_mode == "shared_pooled":
            _svf = family_shared_val_fold(supervised, fam_seqs, seed=args.seed)
            folds = [_svf]
            fold_src = supervised
            _va_subs = len(set(s.subject_id for s in [supervised[i] for i in _svf[1]]))
            print(f"[family-shared-val v25.38] {fam_name}: train filtered to family "
                  f"({len(_svf[0])} seqs), val = SHARED pooled split "
                  f"({len(_svf[1])} seqs, {_va_subs} subjects) -- per-domain "
                  f"ensemble-compatible", flush=True)
        elif _split_mode == "pure_family":
            if args.n_splits == 1:
                folds = [build_single_split_fold(fam_seqs, seed=args.seed)]
                print(f"[split-mode v25.42.8] pure_family: train+val both drawn "
                      f"from {fam_name} subjects only (the journal primary claim)", flush=True)
            else:
                print(f"[split-mode v25.42.8] pure_family with n_splits>1: "
                      f"grouped CV on the family's own subjects", flush=True)
                folds = sh_balanced_group_folds(fam_seqs, args.n_splits, seed=args.seed)
        elif _split_mode == "lodo":
            folds = [lodo_fold_indices(fam_seqs, _lodo_held)]
        else:  # grouped_cv
            if _flag("NEUROMAMBA_SH_BALANCED_OUTER_FOLDS", "1"):  # v25.57: _flag parse
                folds = sh_balanced_group_folds(fam_seqs, args.n_splits, seed=args.seed)
            elif len(np.unique(src_labels)) > 1:
                kf = StratifiedGroupKFold(n_splits=args.n_splits)
                folds = list(kf.split(fam_seqs, y=src_labels, groups=groups))
            else:
                kf = GroupKFold(n_splits=args.n_splits)
                folds = list(kf.split(fam_seqs, groups=groups))

        if notype_seqs is not None:
            _notype_subs = set(s.subject_id for s in notype_seqs)
            _sup_subs = set(s.subject_id for s in supervised)
            _overlap = sorted(_notype_subs & _sup_subs)
            print(f"[leak-guard v25.42.3] {len(notype_seqs)} notype sequences "
                  f"({len(_notype_subs)} subjects); {len(_overlap)} subjects also in "
                  f"supervised -- per-fold val filtering below", flush=True)

        warm_seqs, warm_epochs = None, 0
        if fam_name == "defog" or fam_name.startswith("lodo_"):
            warm_family = get("NEUROMAMBA_WARM_START_FAMILY", "tdcsfog")
            warm_epochs = int(get("NEUROMAMBA_WARM_EPOCHS", "15"))
            if warm_epochs > 0:
                warm_seqs = [s for s in supervised if s.source_type == warm_family]
                if warm_seqs:
                    print(f"[warm-start v25.9] family {fam_name}: pretrain {warm_epochs} epochs on {warm_family} "
                          f"({len(warm_seqs)} sequences) then fine-tune on {fam_name} "
                          f"[v25.58 per-family fix]", flush=True)
                else:
                    warm_epochs = 0

        fold_results, ens_probs, ens_labels, ens_observed, baseline_fold_results = [], [], [], [], []
        repeat_results = []  # v25.42.7: per-repeat family summaries
        n_repeats = max(1, int(getattr(args, "n_repeats", 1) or 1))
        _folds_all = []
        for _rep in range(n_repeats):
            _rep_seed = args.seed + _rep * 10000
            if n_repeats > 1:
                print(f"[repeat v25.42.7] REPEAT {_rep + 1}/{n_repeats} "
                      f"(split seed {args.seed} -> {_rep_seed})", flush=True)
            if _split_mode == "shared_pooled":
                _svf = family_shared_val_fold(supervised, fam_seqs, seed=_rep_seed)
                _folds = [_svf]
                fold_src = supervised
            elif _split_mode == "pure_family":
                if args.n_splits == 1:
                    _folds = [build_single_split_fold(fam_seqs, seed=_rep_seed)]
                else:
                    _folds = sh_balanced_group_folds(fam_seqs, args.n_splits, seed=_rep_seed)
            elif _split_mode == "lodo":
                _folds = [lodo_fold_indices(fam_seqs, _lodo_held)]
            else:  # grouped_cv
                if _flag("NEUROMAMBA_SH_BALANCED_OUTER_FOLDS", "1"):  # v25.57: _flag parse
                    _folds = sh_balanced_group_folds(fam_seqs, args.n_splits, seed=_rep_seed)
                elif len(np.unique(src_labels)) > 1:
                    kf = StratifiedGroupKFold(n_splits=args.n_splits)
                    _folds = list(kf.split(fam_seqs, y=src_labels, groups=groups))
                else:
                    _folds = list(GroupKFold(n_splits=args.n_splits).split(fam_seqs, groups=groups))
            for _fi, _f in enumerate(_folds):
                _folds_all.append((_fi, _rep, _f))
        for fold_idx, (fold_idx_in_rep, _rep, (tr_idx, va_idx)) in enumerate(_folds_all):
            tr = [fold_src[i] for i in tr_idx]
            va = [fold_src[i] for i in va_idx]
            leak = set(s.subject_id for s in tr) & set(s.subject_id for s in va)
            assert not leak, f"Subject leakage: {leak}"
            fold_notype, _dropped_va, _n_keep = filter_notype_for_fold(
            notype_seqs, va, tag=f"fold {fold_idx + 1} val")
            fold_warm_seqs = warm_seqs
            if warm_seqs and _flag("NEUROMAMBA_WARM_EXCLUDE_VAL", "0"):
                _val_ids = set(s.subject_id for s in va)
                _n0, _s0 = len(fold_warm_seqs), len({s.subject_id for s in fold_warm_seqs})
                fold_warm_seqs = [s for s in warm_seqs if s.subject_id not in _val_ids]
                _s1 = len({s.subject_id for s in fold_warm_seqs})
                print(f"[warm-excl v43] NESTED ARM: warm pool {_n0} -> {len(fold_warm_seqs)} "
                      f"sequences ({_s0} -> {_s1} lab subjects; {_s0 - _s1} validation patients "
                      f"removed from pre-training and rehearsal)", flush=True)
            _smoke_prefix = "smoke_" if _flag("NEUROMAMBA_SMOKE", "0") else ""
            _rep_suffix = f"_r{_rep + 1}" if n_repeats > 1 else ""
            print("-" * 60)
            print(f"FOLD {fold_idx_in_rep + 1}/{args.n_splits} ({fam_name})"
                  f"{f' REPEAT {_rep + 1}/{n_repeats}' if n_repeats > 1 else ''}")
            print("-" * 60)
            fold_json = f"{_smoke_prefix}neuromamba_v25_fold{fold_idx_in_rep + 1}_{fam_name}{_rep_suffix}_checkpoint.json"
            fold_tag = f"{_smoke_prefix}{fam_name}_fold{fold_idx_in_rep + 1}{_rep_suffix}"
            _fold_tid = training_identity(
                model_type=args.model, lr=args.lr, batch_size=args.batch_size,
                epochs=args.epochs, seed=args.seed + _rep * 10000, n_seeds=args.n_seeds,
                fold_idx=fold_idx_in_rep, family=fam_name,
                use_rare_split=not args.no_rare_split,
                warm_family=get("NEUROMAMBA_WARM_START_FAMILY", "tdcsfog") if warm_seqs else None,
                warm_epochs=warm_epochs)
            if args.resume and os.path.exists(fold_json):
                try:
                    with open(fold_json) as f:
                        prev = json.load(f)
                    _prev_tid = prev.get("training_identity")
                    _prev_eid = prev.get("evaluation_identity")
                    if not _prev_tid:
                        if prev.get("code_version") in TRAINING_COMPAT_VERSIONS or \
                                prev.get("code_version") == CODE_VERSION:
                            _prev_tid = _fold_tid  # trusted legacy training config
                        else:
                            _prev_tid = "legacy-unknown"
                    _cur_eid = evaluation_identity()
                    _b_npz = (prev.get("baseline_arrays") or {}).get("npz")
                    _base_avail = None if (not getattr(args, "run_baseline_comparator", False)) \
                        else bool(_b_npz and os.path.exists(_b_npz))
                    mode, reason = resume_decision(
                        _prev_tid, _prev_eid, _fold_tid, _cur_eid,
                        has_checkpoint=prev.get("status") != "failed",
                        baseline_available=_base_avail)
                    print(f"[resume v25.42.6] fold {fold_idx + 1} ({fam_name}) "
                          f"decision={mode} device={device} "
                          f"(cuda={torch.cuda.is_available()})", flush=True)
                    if mode == "exact_resume":
                        print(f"[resume v25.42.6] fold {fold_idx + 1} ({fam_name}) "
                              f"REUSED_RESULT (training+eval identity match; "
                              f"mAP={prev.get('kaggle_map', {}).get('mAP')})", flush=True)
                        fold_results.append(prev)
                        ens_npz = (prev.get("ens_arrays") or {}).get("npz")
                        if ens_npz and os.path.exists(ens_npz):
                            try:
                                z = np.load(ens_npz)
                                ens_probs.append(np.asarray(z["sub_probs"]))
                                ens_labels.append(np.asarray(z["sub_labels"]))
                                ens_observed.append(np.asarray(z["sub_observed"]))
                            except Exception as e:
                                print(f"[resume v25.5] WARNING: cannot load {ens_npz}: {e}", flush=True)
                        if _base_avail:
                            try:
                                zb = np.load(_b_npz)
                                baseline_fold_results.append({
                                    "fold": fold_idx, "kaggle_map": None,
                                    "sub_probs": np.asarray(zb["sub_probs"]),
                                    "sub_labels": np.asarray(zb["sub_labels"]),
                                    "sub_observed": np.asarray(zb["sub_observed"]),
                                    "fog_probs": None, "model": None,
                                    "resumed": True})
                            except Exception as e:
                                print(f"[resume v25.5] WARNING: cannot load baseline {_b_npz}: {e}", flush=True)
                        continue
                    if mode == "evaluate_only":
                        print(f"[resume v25.42.6] fold {fold_idx + 1} ({fam_name}) "
                              f"REEVALUATED_FROZEN_MODEL: {reason}", flush=True)
                        _re = None
                        try:
                            _re = reevaluate_frozen_fold(prev, fold_tag, _cur_eid)
                        except Exception as e:
                            print(f"[resume v25.42.6] fold {fold_idx + 1}: frozen "
                                  f"reevaluation CRASHED: {e}", flush=True)
                            traceback.print_exc()
                        if _re is None:
                            raise RuntimeError(
                                "EVALUATE_ONLY requested but frozen predictions "
                                "unusable; refusing to fall back to training "
                                "(v25.53.7). Fix the fold cache or rerun training.")
                        fold_results.append(_re["fold_record"])
                        ens_probs.append(_re["sub_probs"])
                        ens_labels.append(_re["sub_labels"])
                        ens_observed.append(_re["sub_observed"])
                        if _base_avail:
                            try:
                                zb = np.load(_b_npz)
                                baseline_fold_results.append({
                                    "fold": fold_idx, "kaggle_map": None,
                                    "sub_probs": np.asarray(zb["sub_probs"]),
                                    "sub_labels": np.asarray(zb["sub_labels"]),
                                    "sub_observed": np.asarray(zb["sub_observed"]),
                                    "fog_probs": None, "model": None,
                                    "resumed": True})
                            except Exception as e:
                                print(f"[resume v25.5] WARNING: cannot load baseline {_b_npz}: {e}", flush=True)
                        elif getattr(args, "run_baseline_comparator", False):
                            try:
                                _b_tr, _b_ca, _b_th, _b_diag = rare_subtype_aware_split(
                                    tr, seed=args.seed + fold_idx,
                                    calib_frac=float(get("NEUROMAMBA_CALIB_FRAC", "0.15")),
                                    thresh_frac=float(get("NEUROMAMBA_THRESH_FRAC", "0.15")))
                                _rep_of_fold = 0
                                if fold_tag and "_r" in fold_tag:
                                    try:
                                        _rep_of_fold = int(fold_tag.split("_r")[1].split("_")[0]) - 1
                                    except ValueError:
                                        _rep_of_fold = 0
                                _bres = run_resnet1d_baseline_fold(
                                    _b_tr, va, device, min(args.epochs, 40),
                                    args.lr, args.batch_size, fold_idx, repeat=_rep_of_fold)
                                _b_npz2 = f"{fold_tag}_baseline_ens.npz"
                                np.savez_compressed(_b_npz2, sub_probs=_bres["sub_probs"],
                                                    sub_labels=_bres["sub_labels"],
                                                    sub_observed=_bres["sub_observed"])
                                prev["baseline_arrays"] = {"npz": _b_npz2}
                                with open(fold_json, "w") as f:
                                    json.dump(prev, f, indent=2, cls=NpEncoder)
                                baseline_fold_results.append({
                                    "fold": fold_idx, "kaggle_map": _bres["kaggle_map"],
                                    "sub_probs": _bres["sub_probs"],
                                    "sub_labels": _bres["sub_labels"],
                                    "sub_observed": _bres["sub_observed"],
                                    "fog_probs": None, "model": None,
                                    "resumed": True})
                            except Exception as e:
                                print(f"[resume v25.5] WARNING: baseline regeneration failed: {e}", flush=True)
                                traceback.print_exc()
                        continue
                    print(f"[resume v25.42.6] fold {fold_idx + 1} ({fam_name}) FRESH: {reason}", flush=True)
                except Exception as e:
                    print(f"[resume v25.5] WARNING: could not reuse {fold_json}: {e}", flush=True)
            try:
                _p1m = phase1_manifest(args, fam_name, fold_idx_in_rep, _rep,
                                       tr, va, _split_mode, _fold_tid,
                                       baseline_tid=None,
                                       calib=None, thresh=None)
                model, res, metrics, tt, ens_pack, bres, fold_record = run_fold(
                    tr, va, device, args.epochs, args.lr, args.batch_size, args.seed, fold_idx,
                    external_seqs=external_seqs, notype_seqs=fold_notype, run_baseline=args.run_baseline_comparator,
                    model_type=args.model, use_rare_split=not args.no_rare_split, n_seeds=args.n_seeds,
                    fold_tag=fold_tag, resume=args.resume,
                    warm_seqs=fold_warm_seqs, warm_epochs=warm_epochs,
                    training_identity=_fold_tid, p1_manifest=_p1m)
                if bres is not None and _p1m is not None and "protocol_hash" in _p1m:
                    bres["protocol_hash"] = _p1m["protocol_hash"]
            except Exception as e:
                print(f"FOLD {fold_idx + 1} FAILED: {e}", flush=True)
                traceback.print_exc()
                fold_results.append({"fold": fold_idx + 1, "status": "failed", "error": str(e)})
                continue
            fold_results.append(fold_record)
            if ens_pack[0] is not None:
                ens_npz = f"{fold_tag}_ens.npz"
                _sub_ids = np.array([s.subject_id for s in va for _ in range(len(s.labels))])
                _sub_src = np.array([s.source_type for s in va for _ in range(len(s.labels))])
                _ei = res.get("eval_inputs") or {}
                _seed_subs = _ei.get("seed_sub_probs") or []
                _haz_mk = int(_ei.get("hazard_stream") or 0)
                _ens_arrays = dict(sub_probs=ens_pack[0],
                                   sub_labels=ens_pack[1], sub_observed=ens_pack[2],
                                   sub_ids=_sub_ids, sub_source=_sub_src,
                                   fog_probs=_ei.get("fog_probs"),
                                   fog_labels=_ei.get("fog_labels"),
                                   fog_ids=_ei.get("fog_ids"),
                                   fog_subjects=_ei.get("fog_subjects"),
                                   fog_sources=_ei.get("fog_sources"),
                                   fog_med=_ei.get("fog_med"),
                                   seed_sub_probs=np.asarray(_seed_subs, dtype=object))
                if _haz_mk:
                    _ens_arrays["hazard_stream"] = np.int64(1)
                np.savez_compressed(ens_npz, **_ens_arrays)
                fold_record["ens_arrays"] = {"npz": ens_npz,
                                             "sub_probs_shape": list(ens_pack[0].shape)}
                if _haz_mk:
                    fold_record["hazard_stream"] = 1
                fold_record["frozen_threshold"] = float(_ei.get("frozen_threshold", res.get("frozen_threshold", 0.5)))
                fold_record["temperature"] = float(_ei.get("temperature", res.get("temperature", 1.0)))
                fold_record["postproc"] = dict(res.get("postproc") or {})
                fold_record["eval_inputs_saved"] = bool(_ei)
                if _p1m is not None:
                    with open(f"{fold_tag}_manifest.json", "w", encoding="utf-8") as _mf:
                        json.dump(_p1m, _mf, indent=2, default=str)
                    fold_record["phase1_manifest"] = _p1m
                    fold_record["protocol_hash"] = _p1m.get("protocol_hash")
            if bres is not None:
                try:
                    _b_npz = f"{fold_tag}_baseline_ens.npz"
                    np.savez_compressed(
                        _b_npz, sub_probs=bres["sub_probs"],
                        sub_labels=bres["sub_labels"],
                        sub_observed=bres["sub_observed"],
                        training_identity=bres.get("training_identity", ""),
                        baseline_seed=int(bres.get("seed", -1)),
                        model_state=np.asarray(
                            bres["model"].state_dict() if bres.get("model") is not None else {})
                        if bres.get("model") is not None else None,
                        val_subjects=np.array(
                            [s.subject_id for s in va for _ in range(len(s.labels))], dtype=object))
                    fold_record["baseline_arrays"] = {
                        "npz": _b_npz,
                        "sub_probs_shape": list(bres["sub_probs"].shape),
                        "training_identity": bres.get("training_identity", ""),
                        "baseline_seed": int(bres.get("seed", -1))}
                except Exception as e:
                    print(f"[baseline v25.42.5] WARNING: failed to save baseline sidecar: {e}", flush=True)
            try:
                fold_record["fold_json_path"] = fold_json
                with open(fold_json, "w") as f:
                    json.dump(fold_record, f, indent=2, cls=NpEncoder)
                print(f"[checkpoint] fold {fold_idx + 1} results saved to {fold_json}", flush=True)
            except Exception as e:
                print(f"[checkpoint] WARNING: failed to save fold checkpoint: {e}", flush=True)
            if ens_pack[0] is not None:
                ens_probs.append(ens_pack[0]); ens_labels.append(ens_pack[1]); ens_observed.append(ens_pack[2])
            if bres is not None:
                try:
                    _b_npz = fold_record.get("baseline_arrays", {}).get("npz")
                    if _b_npz and os.path.exists(_b_npz):
                        bres_light = {k: v for k, v in bres.items() if k != "model"}
                        bres_light["sub_probs"] = "npz:" + _b_npz
                        bres_light["sub_labels"] = None
                        bres_light["sub_observed"] = None
                        baseline_fold_results.append(bres_light)
                    else:
                        baseline_fold_results.append(bres)
                except Exception as e:
                    print(f"[baseline v25.42.5] WARNING: failed to register baseline sidecar: {e}", flush=True)
                    baseline_fold_results.append(bres)

        done = [f for f in fold_results if f.get("status") != "failed"]
        maps = [f["kaggle_map"]["mAP"] for f in done if f.get("kaggle_map")]
        event_f1s = [f["event_f1"] for f in done]
        collapse = worst_fold_collapse_check(event_f1s)
        _ef_mean, _ef_n_eval, _ef_n_tot = nan_safe_mean(event_f1s)
        fam_summary = {"n_folds_completed": len(done), "kaggle_map_mean": float(np.mean(maps)) if maps else None,
                        "kaggle_map_std": float(np.std(maps)) if maps else None,
                        "event_f1_mean": _ef_mean,
                        "event_f1_n_evaluable": _ef_n_eval,
                        "event_f1_n_not_evaluable": _ef_n_tot - _ef_n_eval,
                        "worst_fold_collapse_check": collapse, "fold_results": fold_results}
        if _ef_n_tot and _ef_n_eval < _ef_n_tot:
            print(f"[integrity v25.44] {fam_name}: {_ef_n_tot - _ef_n_eval}/{_ef_n_tot} fold(s) had a "
                  f"degenerate (empty) event-F1 -- excluded from the mean (nan-safe), NOT counted "
                  f"as perfect scores", flush=True)

        try:
            cidiff, ece_max = None, None
            if ens_probs:
                pooled_probs = np.concatenate(ens_probs, axis=0)
                pooled_labels = np.concatenate(ens_labels, axis=0)
                pooled_observed = np.concatenate(ens_observed, axis=0)
                try:
                    ens_map = compute_kaggle_map(pooled_probs, pooled_labels, pooled_observed)
                except Exception as e:
                    print(f"[FOLD-ENSEMBLE] pooled mAP scoring failed: {e}", flush=True)
                    ens_map = None
                if ens_map is not None:
                    fam_summary["fold_ensemble_map"] = ens_map
                    print(f"[kaggle-track] Fold-ensemble (pooled OOF) mAP={ens_map['mAP']:.4f} "
                          f"(internal diagnostic, NOT a journal claim)", flush=True)
                subj_groups = []
                for f_idx, (fold_idx_in_rep, _rep, (tr_idx, va_idx)) in enumerate(_folds_all):
                    va = [fold_src[i] for i in va_idx]
                    for s in va:
                        subj_groups.extend([s.subject_id] * len(s.labels))
                subj_groups = np.array(subj_groups, dtype=object)[:len(pooled_probs)]
                ci = bootstrap_kaggle_map_ci(pooled_probs, pooled_labels, pooled_observed, groups=subj_groups)
                fam_summary["bootstrap_ci"] = ci
                print(f"[kaggle-track] bootstrap 95% CI [{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}] "
                      f"(unit={ci.get('unit','subject')})", flush=True)
                if baseline_fold_results:
                    pooled_base_probs = resolve_baseline_probs(baseline_fold_results)
                    if pooled_base_probs is not None:
                        n = min(len(pooled_base_probs), len(pooled_probs))
                        print(f"[kaggle-track] baseline-diff bootstrap starting "
                              f"(NEUROMAMBA_BOOTSTRAP_N boots; minutes on 4.6M-frame "
                              f"pooled OOF -- progress prints every 50 boots)", flush=True)
                        cidiff = kaggle_map_ci_diff(pooled_probs[:n], pooled_base_probs[:n], pooled_labels[:n], pooled_observed[:n],
                                                     groups=subj_groups[:n])
                        fam_summary["baseline_ci_diff"] = cidiff
                        print(f"[kaggle-track] vs frozen ResNet1D baseline: diff_mean={cidiff['diff_mean']:.4f} "
                              f"CI=[{cidiff['ci_lo']:.4f},{cidiff['ci_hi']:.4f}] non_overlapping={cidiff['non_overlapping_zero']}", flush=True)
                    else:
                        fam_summary["baseline_ci_diff"] = {"status": "NOT_EVALUATED",
                                                           "reason": "baseline comparator artifact unavailable after resume"}
                        print(f"[kaggle-track] baseline comparator arrays unavailable "
                              f"({len(baseline_fold_results)} records) -- gate_2 = NOT_EVALUATED", flush=True)
                _ext_eces = []
                for fr in fold_results:
                    er = (fr.get("tripod_external_validation") or {}).get("cohorts") or {}
                    for c in er.values():
                        if c.get("ece") is not None and np.isfinite(c.get("ece")):
                            _ext_eces.append(c["ece"])
                ece_max = float(np.mean(_ext_eces)) if _ext_eces else float("nan")
                if not _ext_eces:
                    _val_eces = [f.get("ece") for f in done if f.get("ece") is not None]
                    ece_max = float(np.mean(_val_eces)) if _val_eces else float("nan")
                fam_summary["ece_max"] = ece_max
                print(f"[calib v25.26] gate ECE (mean per-cohort gated) = {ece_max:.4f} "
                      f"(per-cohort {[round(e, 3) for e in _ext_eces]})", flush=True)
                _ps = {}
                for _fr in done:
                    _ps = (_fr.get("per_source") or {})
                    break
                if _ps and fam_name in _ps and fam_name != "all":
                    _pf = _ps[fam_name]
                    fam_summary["pure_family_heldout"] = {
                        "family": fam_name,
                        "event_f1": _pf.get("event_f1"),
                        "farh": (_pf.get("farh") or {}).get("farh_events_per_hour"),
                        "episode_f1_frame": _pf.get("episode_f1_frame"),
                        "note": f"val frames from the {fam_name} source only "
                                f"(the pooled shared-val row mixes defog+tdcsfog)"}
                    print(f"[pure-family v25.42.7] {fam_name}-only held-out: "
                          f"eventF1={fam_summary['pure_family_heldout']['event_f1']:.3f} "
                          f"FARh={fam_summary['pure_family_heldout']['farh']:.2f}/h "
                          f"(pooled row is the ensemble-compatible number)", flush=True)
        except Exception as e:
            print(f"[FOLD-ENSEMBLE] aggregation for family {fam_name} FAILED: {e}", flush=True)
            traceback.print_exc()
            fam_summary["ensemble_error"] = str(e)

        if n_repeats > 1:
            try:
                _rep_rows = {}
                for _fr in fold_results:
                    if _fr.get("status") == "failed":
                        continue
                    _rep = _fr.get("repeat")
                    if _rep is None:
                        _rep = int((_fr.get("fold", 1) - 1) // args.n_splits) if args.n_splits else 0
                    _rep_rows.setdefault(_rep, []).append(_fr)
                _rep_sum = {}
                for _rep, _frs in sorted(_rep_rows.items()):
                    _maps = [f.get("kaggle_map", {}).get("mAP") for f in _frs
                             if f.get("kaggle_map", {}).get("mAP") is not None]
                    _ef = [f.get("event_f1") for f in _frs if f.get("event_f1") is not None]
                    _wf = [f.get("window_f1") for f in _frs if f.get("window_f1") is not None]
                    _ece = [f.get("ece") for f in _frs if f.get("ece") is not None]
                    _sh_ap = [f.get("kaggle_map", {}).get("StartHesitation") for f in _frs
                              if f.get("kaggle_map", {}).get("StartHesitation") is not None]
                    _farh = [f.get("farh", {}).get("farh_events_per_hour") for f in _frs
                             if f.get("farh", {}).get("farh_events_per_hour") is not None]
                    _brier = [f.get("brier") for f in _frs if f.get("brier") is not None]
                    _slope = [f.get("reliability_diagram_slope") if f.get("reliability_diagram_slope") is not None
                              else f.get("calib_slope") for f in _frs
                              if f.get("reliability_diagram_slope") is not None
                              or f.get("calib_slope") is not None]
                    _log_slope = [f.get("logistic_calib_slope") for f in _frs
                                  if f.get("logistic_calib_slope") is not None]
                    _sh_pow = [f.get("held_sh_positive_subjects") for f in _frs
                               if f.get("held_sh_positive_subjects") is not None]
                    _sh_ev = [f.get("held_sh_events") for f in _frs
                              if f.get("held_sh_events") is not None]
                    _sh_frac = [f.get("held_sh_max_subject_event_fraction") for f in _frs
                                if f.get("held_sh_max_subject_event_fraction") is not None]
                    _sh_rec = [f.get("sh_event_recall") for f in _frs
                               if f.get("sh_event_recall") is not None]
                    _sh_cov = [f.get("sh_subject_coverage") for f in _frs
                               if f.get("sh_subject_coverage") is not None]
                    _base_map = [f.get("baseline_map") for f in _frs
                                 if f.get("baseline_map") is not None]
                    _main_map_f = [f.get("main_map") for f in _frs
                                   if f.get("main_map") is not None]
                    _clin = [f.get("clinical_op_points") for f in _frs
                             if f.get("clinical_op_points") is not None]
                    _clin10 = [float(c["sens_at_farh10"]) for c in _clin
                               if c.get("sens_at_farh10") is not None]
                    _clin85 = [float(c["farh_at_sens85"]) for c in _clin
                               if c.get("farh_at_sens85") is not None
                               and c["farh_at_sens85"] != float("inf")]
                    _clin85_unreach = int(sum(1 for c in _clin
                                              if c.get("farh_at_sens85") in (None, float("inf"))))
                    _proto_hashes = [f.get("protocol_hash") for f in _frs
                                     if f.get("protocol_hash") is not None]
                    _n_subj = int(np.mean(_sh_pow)) if _sh_pow else 0
                    _n_ev = int(np.mean(_sh_ev)) if _sh_ev else 0
                    _max_frac = float(np.mean(_sh_frac)) if _sh_frac else 1.0
                    _sh_powered = sh_repeat_powered(
                        _n_subj, _n_ev, _max_frac,
                        min_subjects=int(os.environ.get("NEUROMAMBA_MIN_HELD_SH_SUBJECTS", "4")),
                        min_events=int(os.environ.get("NEUROMAMBA_MIN_HELD_SH_EVENTS", "8")),
                        max_subject_frac_lim=float(os.environ.get("NEUROMAMBA_MAX_SH_SUBJECT_EVENT_FRAC", "0.5")))
                    _rep_sum[_rep] = {
                        "n_folds": len(_frs),
                        "kaggle_map_mean": float(np.mean(_maps)) if _maps else None,
                        "main_map": float(np.mean(_main_map_f)) if _main_map_f else None,
                        "baseline_map": float(np.mean(_base_map)) if _base_map else None,
                        "event_f1_mean": float(np.mean(_ef)) if _ef else None,
                        "window_f1_mean": float(np.mean(_wf)) if _wf else None,
                        "ece_mean": float(np.mean(_ece)) if _ece else None,
                        "sh_ap_mean": float(np.mean(_sh_ap)) if _sh_ap else None,
                        "sh_event_recall_mean": float(np.mean(_sh_rec)) if _sh_rec else None,
                        "sh_subject_coverage_mean": float(np.mean(_sh_cov)) if _sh_cov else None,
                        "farh_mean": float(np.mean(_farh)) if _farh else None,
                        "brier_mean": float(np.mean(_brier)) if _brier else None,
                        "reliability_diagram_slope_mean": float(np.mean(_slope)) if _slope else None,
                        "logistic_calib_slope_mean": float(np.mean(_log_slope)) if _log_slope else None,
                        "sens_at_farh10": float(np.mean(_clin10)) if _clin10 else None,
                        "farh_at_sens85": float(np.mean(_clin85)) if _clin85 else None,
                        "farh_at_sens85_unreachable_folds": _clin85_unreach,
                        "clinical_feasible": bool(_clin10 and max(_clin10) >= 0.85),
                        "selection_policy_hash": (_proto_hashes[0] if _proto_hashes else None),
                        "held_sh_subjects": _n_subj,
                        "held_sh_events": _n_ev,
                        "held_sh_max_subject_event_fraction": _max_frac,
                        "sh_powered": bool(_sh_powered)}
                fam_summary["repeated_cv"] = _rep_sum
                def _dist(vals):
                    if not vals:
                        return None
                    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                            "median": float(np.median(vals)),
                            "q25": float(np.percentile(vals, 25)),
                            "q75": float(np.percentile(vals, 75)),
                            "min": float(np.min(vals)), "max": float(np.max(vals))}
                _rk = [r["kaggle_map_mean"] for r in _rep_sum.values() if r.get("kaggle_map_mean") is not None]
                _re = [r["event_f1_mean"] for r in _rep_sum.values() if r.get("event_f1_mean") is not None]
                _rw = [r["window_f1_mean"] for r in _rep_sum.values() if r.get("window_f1_mean") is not None]
                _rf = [r["farh_mean"] for r in _rep_sum.values() if r.get("farh_mean") is not None]
                _rb = [r["brier_mean"] for r in _rep_sum.values() if r.get("brier_mean") is not None]
                _rs = [r["reliability_diagram_slope_mean"] for r in _rep_sum.values()
                       if r.get("reliability_diagram_slope_mean") is not None]
                _rls = [r["logistic_calib_slope_mean"] for r in _rep_sum.values()
                        if r.get("logistic_calib_slope_mean") is not None]
                SH_REFERENCE_STABILITY_FLOOR = 0.05
                _powered = [r for r in _rep_sum.values() if r.get("sh_powered")]
                _sh_ap_pow = [r["sh_ap_mean"] for r in _powered if r.get("sh_ap_mean") is not None]
                sh_stable, _ss = sh_stability_decision(
                    _powered, floor=SH_REFERENCE_STABILITY_FLOOR)
                _ss["reference_floor"] = SH_REFERENCE_STABILITY_FLOOR
                fam_summary["repeated_cv_summary"] = {
                    "n_repeats": len(_rep_sum),
                    "n_powered_repeats": len(_powered),
                    "kaggle_map": _dist(_rk),
                    "event_f1": _dist(_re),
                    "window_f1": _dist(_rw),
                    "farh": _dist(_rf),
                    "brier": _dist(_rb),
                    "reliability_diagram_slope": _dist(_rs),
                    "logistic_calib_slope": _dist(_rls),
                    "sh_ap": _dist(_sh_ap_pow),
                    "sh_stability": _ss,
                    "note": "between-repeat distribution = split-selection variance "
                            "(repeated subject-grouped CV); underpowered SH repeats excluded"}
                _ss = fam_summary["repeated_cv_summary"]["sh_stability"]
                print(f"[repeated-cv v25.42.10] SH stability: powered={_ss['powered_repeats']}/{n_repeats} "
                      f"above_floor={_ss['above_floor']}/{_ss['powered_repeats']} "
                      f"event_recall>0={_ss['event_recall_nonzero']}/{_ss['powered_repeats']} "
                      f"coverage>0={_ss['subject_coverage_nonzero']}/{_ss['powered_repeats']} "
                      f"-> {'SH STABLE' if _ss['sh_stable'] else 'SH NOT STABLE'} "
                      f"(reference floor {SH_REFERENCE_STABILITY_FLOOR})", flush=True)
                print(f"[repeated-cv v25.42.9] {n_repeats} repeats: "
                      f"mAP {fam_summary['repeated_cv_summary']['kaggle_map']} | "
                      f"farh {fam_summary['repeated_cv_summary']['farh']} | "
                      f"brier {fam_summary['repeated_cv_summary']['brier']}", flush=True)
            except Exception as _e:
                print(f"[repeated-cv v25.42.7] summary failed: {_e}", flush=True)
                traceback.print_exc()

        if n_repeats > 1 and fam_summary.get("repeated_cv"):
            try:
                _deltas = []
                _missing_baseline = []
                for _rep, _frs in sorted(fam_summary["repeated_cv"].items()):
                    _mm = _frs.get("main_map")
                    _bm = _frs.get("baseline_map")
                    if _mm is None or _bm is None:
                        _missing_baseline.append(_rep)
                        continue
                    _deltas.append(float(_mm - _bm))
                if _missing_baseline:
                    fam_summary["repeated_cv_summary"]["main_vs_baseline"] = {
                        "error": "MISSING per-repeat baseline_map",
                        "missing_in_repeats": _missing_baseline,
                        "note": "per-repeat delta is a HARD protocol requirement; "
                                "re-run with --run-baseline-comparator (frozen sidecar)"}
                    print(f"[repeated-cv v25.42.10] ERROR: per-repeat baseline mAP "
                          f"missing in repeats {_missing_baseline} -- delta NOT computed "
                          f"(no diff_mean fallback; reviewer fix #1)", flush=True)
                elif _deltas:
                    fam_summary["repeated_cv_summary"]["main_vs_baseline"] = {
                        "delta_mean": float(np.mean(_deltas)),
                        "delta_median": float(np.median(_deltas)),
                        "delta_sd": float(np.std(_deltas)),
                        "n_repeats_positive_delta": int(sum(1 for d in _deltas if d > 0)),
                        "n_repeats_total": len(_deltas),
                        "delta_per_repeat": [float(d) for d in _deltas],
                        "baseline_map_per_repeat": [float(r.get("baseline_map"))
                                                    for r in fam_summary["repeated_cv"].values()],
                        "interpretation": "did the architecture win repeatedly? "
                                          "(n_repeats_positive_delta / n_repeats_total)"}
                    _b = fam_summary["repeated_cv_summary"]["main_vs_baseline"]
                    print(f"[repeated-cv v25.42.10] main vs baseline per repeat: "
                          f"delta_mean={_b['delta_mean']:.4f} delta_median={_b['delta_median']:.4f} "
                          f"positive_in={_b['n_repeats_positive_delta']}/{len(_deltas)} repeats", flush=True)
            except Exception as _e:
                print(f"[repeated-cv v25.42.9] baseline pairing failed: {_e}", flush=True)
                traceback.print_exc()

        if fam_summary.get("repeated_cv"):
            try:
                import csv as _csv
                _tbl = f"{fam_name}_phase1_repeats.csv"
                _rows = []
                for _rep, r in sorted(fam_summary["repeated_cv"].items()):
                    _rows.append({
                        "repeat": _rep + 1, "test_subjects": r.get("held_sh_subjects"),
                        "sh_subjects": r.get("held_sh_subjects"),
                        "sh_events": r.get("held_sh_events"),
                        "sh_powered": r.get("sh_powered"),
                        "main_map": r.get("main_map"), "baseline_map": r.get("baseline_map"),
                        "delta": (float(r["main_map"] - r["baseline_map"])
                                  if r.get("main_map") is not None and r.get("baseline_map") is not None
                                  else None),
                        "kaggle_map": r.get("kaggle_map_mean"), "sh_ap": r.get("sh_ap_mean"),
                        "event_f1": r.get("event_f1_mean"), "farh": r.get("farh_mean"),
                        "brier": r.get("brier_mean"),
                        "reliability_diagram_slope": r.get("reliability_diagram_slope_mean"),
                        "logistic_calib_slope": r.get("logistic_calib_slope_mean"),
                        "ece": r.get("ece_mean"), "window_f1": r.get("window_f1_mean"),
                        "sens_at_farh10": r.get("sens_at_farh10"),
                        "farh_at_sens85": r.get("farh_at_sens85"),
                        "clinical_feasible": r.get("clinical_feasible"),
                        "selection_policy_hash": r.get("selection_policy_hash")})
                with open(_tbl, "w", newline="", encoding="utf-8") as _f:
                    _w = _csv.DictWriter(_f, fieldnames=list(_rows[0].keys()))
                    _w.writeheader(); _w.writerows(_rows)
                fam_summary["phase1_repeats_csv"] = _tbl
                print(f"[repeated-cv v25.42.10] per-repeat table saved to {_tbl}", flush=True)
            except Exception as _e:
                print(f"[repeated-cv v25.42.10] table save failed: {_e}", flush=True)
                traceback.print_exc()

        grand["families"][fam_name] = fam_summary
        _kaggle_keys = ("kaggle_map", "kaggle_map_mean", "kaggle_map_std",
                        "fold_ensemble_map", "bootstrap_ci", "baseline_ci_diff")
        _jour = {k: v for k, v in fam_summary.items() if k != "fold_results" and k not in _kaggle_keys}
        _kag = {k: v for k, v in fam_summary.items() if k in _kaggle_keys and v is not None}
        try:
            _jf = next((f for f in fam_summary.get("fold_results", []) if f.get("status") != "failed"), None)
            if _jf is not None:
                _km = _jf.get("kaggle_map") or {}
                _farh = ((_jf.get("farh") or {}).get("farh_events_per_hour"))
                print("-" * 76, flush=True)
                print(f"[journal-metrics v25.59f] FAMILY {fam_name} (fold-level, manuscript set):", flush=True)
                print(f"  subtype mAP = {_jf.get('main_map')}"
                      + (f"  (SH={_km.get('StartHesitation'):.4f} Turn={_km.get('Turn'):.4f} "
                         f"Walking={_km.get('Walking'):.4f})" if _km else ""), flush=True)
                print(f"  event F1 = {_jf.get('event_f1')} | episode-F1(frame) = {_jf.get('episode_f1_frame')}", flush=True)
                print(f"  FARh = {_farh} /h (budget {CLINICAL_TARGET['farh_max']:g}) "
                      f"| sens@FARh<=10 = {_jf.get('clinical_op_points') and _jf['clinical_op_points'].get('sens_at_farh10')}", flush=True)
                print(f"  window F1 = {_jf.get('window_f1')} | frame F1 = {_jf.get('episode_f1_frame')}", flush=True)
                print(f"  ECE = {_jf.get('ece')} | Brier = {_jf.get('brier')} | temperature = {_jf.get('temperature')}", flush=True)
                print(f"  ICC = {json.dumps(_jf.get('icc'))}", flush=True)
                print(f"  bootstrap main-vs-baseline diff = {json.dumps(fam_summary.get('baseline_ci_diff'))}", flush=True)
                print("-" * 76, flush=True)
        except Exception as _je:
            print(f"[journal-metrics v25.59f] WARNING: consolidated block failed: {_je}", flush=True)
        print("=" * 76, flush=True)
        print(f"[journal-track] FAMILY {fam_name} SUMMARY (manuscript metrics)", flush=True)
        print(json.dumps(_jour, indent=2, default=str), flush=True)
        if _kag:
            print(f"[kaggle-track] FAMILY {fam_name} SUMMARY (internal leaderboard-mimic diagnostics ONLY -- "
                  f"not part of the journal claim)", flush=True)
            print(json.dumps(_kag, indent=2, default=str), flush=True)
        print("=" * 76, flush=True)
        if maps:
            all_maps.extend(maps)

    if all_maps:
        if _flag("NEUROMAMBA_SHOW_LEADERBOARD", "0"):
            print_leaderboard_comparison(float(np.mean(all_maps)))

    internal_map = {"mAP": float(np.mean(all_maps)), "n_comparable": 3} if all_maps else None
    baseline_cidiffs = [fs.get("baseline_ci_diff") for fs in grand["families"].values() if fs.get("baseline_ci_diff")]
    worst_f1s = [fs.get("event_f1_mean") for fs in grand["families"].values()]
    ece_maxes = [fs.get("ece_max") for fs in grand["families"].values() if fs.get("ece_max") is not None]

    ext_min_ef1 = float(os.environ.get("NEUROMAMBA_EXTERNAL_MIN_EVENT_F1", 0.3))
    ext_max_farh = float(os.environ.get("NEUROMAMBA_EXTERNAL_MAX_FARH", 30.0))
    ext_cohorts = {}
    for fs in grand["families"].values():
        for fr in fs.get("fold_results", []):
            er = fr.get("tripod_external_validation") or {}
            _er_rev = er.get("evaluation_rev")
            if _er_rev != EVALUATION_REV:
                _why = ("no evaluation_rev stamp (pre-v25.56 record)"
                        if not _er_rev else f"stamped {_er_rev} != current {EVALUATION_REV}")
                print(f"[external v25.56] REFUSED stored external rows for fold "
                      f"{fr.get('fold', '?')} ({_why}) -- stale rows cannot feed "
                      f"gate_4; a fresh training run is required after this eval "
                      f"change", flush=True)
                continue
            for src, c in (er.get("cohorts") or {}).items():
                e = ext_cohorts.setdefault(src, {"event_f1": [], "farh": [], "passed": []})
                e["event_f1"].append(c["event_f1"])
                e["farh"].append(c["farh"])
                e["passed"].append(c["passed"])
    external_results = None
    if ext_cohorts:
        external_results = []
        for src, agg in ext_cohorts.items():
            _m_ef1, _n_ef1, _t_ef1 = nan_safe_mean(agg["event_f1"])
            mean_farh = float(np.mean([v for v in agg["farh"] if np.isfinite(v)])) \
                if any(np.isfinite(v) for v in agg["farh"]) else float("nan")
            mean_ef1 = _m_ef1
            external_results.append({
                "cohort": src, "event_f1": mean_ef1, "farh": mean_farh,
                "n_folds": len(agg["event_f1"]),
                "n_evaluable_event_f1": _n_ef1,
                "passed": bool(np.isfinite(mean_ef1) and mean_ef1 >= ext_min_ef1
                               and np.isfinite(mean_farh) and mean_farh <= ext_max_farh),
            })
        grand["external_validation"] = external_results
        print("[external] cross-fold cohort summary:", flush=True)
        for r in external_results:
            print(f"[external] {r['cohort']}: eventF1={r['event_f1']:.3f} FARh={r['farh']:.2f}/h "
                  f"over {r['n_folds']} folds passed={r['passed']}", flush=True)

    gate = sota_claim_status(internal_map, baseline_cidiffs[0] if baseline_cidiffs else None, worst_f1s,
                              external_results=external_results, ece_max=(max(ece_maxes) if ece_maxes else None))
    print("=" * 76, flush=True)
    print("[journal-track] Final gate (external cohorts required for a full SOTA claim)", flush=True)
    print(json.dumps(gate, default=str), flush=True)
    if _flag("NEUROMAMBA_SHOW_LEADERBOARD", "0") and all_maps:
        print("[kaggle-track] (internal only) pooled mAP over families = "
              f"{float(np.mean(all_maps)):.4f}", flush=True)
    print("=" * 76, flush=True)

    if getattr(args, "benchmark_on", None):
        grand["benchmark"] = {}
        for name in args.benchmark_on:
            bseqs = [s for s in (external_seqs or []) if s.source_type == name]
            if not bseqs:
                print(f"[benchmark v25.27] cohort '{name}' not loaded; skipped", flush=True)
                continue
            print(f"[benchmark v25.27] training the {args.model} architecture ON {name} "
                  f"({len(bseqs)} seqs, {len(set(s.subject_id for s in bseqs))} subjects, LOSO)", flush=True)
            grand["benchmark"][name] = run_cohort_loso_benchmark(
                bseqs, device, epochs=int(get("NEUROMAMBA_BENCH_EPOCHS", "30")),
                lr=args.lr, batch_size=args.batch_size, seed=args.seed, model_type=args.model)

    if getattr(args, "journal_report", False):
        grand["dataset_characteristics"] = dataset_characteristics(
            list(sequences) + list(external_seqs or []))
        jr = build_journal_report(grand, gate,
                                  baseline_ci_diff=baseline_cidiffs[0] if baseline_cidiffs else None)
        grand["journal_report"] = jr["structured"]
        try:
            with open("journal_report.md", "w", encoding="utf-8") as f:
                f.write(jr["markdown"])
            print("[journal-track] publication report written to journal_report.md "
                  "(primary claim = clinical/generalization tables; pooled mAP is "
                  "quarantined in Appendix A, kaggle-track)", flush=True)
        except Exception as e:
            print(f"[journal-track] WARNING: report write failed: {e}", flush=True)

    if (not _flag("NEUROMAMBA_MONO_SKIP", "0")
            and not _flag("NEUROMAMBA_CALIB_USE_OOF_POOL", "0")):
        try:
            import glob as _mg
            import pooled_oof_calibration as _poc
            _oofs = [p for p in sorted(_mg.glob("*_fold*_ens.npz")) if "_baseline_" not in p]
            if len(_oofs) >= 2:
                _pool = _poc.load_concat_pool(_oofs, dedup=True)
                _mr = check_monotone_trigger_sweep(
                    _pool["probs"], _pool["ids"], labels=_pool["labels"],
                    n_points=60, groups=_pool["subjects"])
                print(f"[mono-gate] per-subject ok={_mr['ok']} "
                      f"failing={_mr['n_groups_failing']}/{len(_mr['per_group'])}", flush=True)
                if not _mr["ok"]:
                    _bad = {k: v for k, v in _mr["per_group"].items() if not v["ok"]}
                    raise AssertionError(
                        f"OOF pool is NON-monotone on {_mr['n_groups_failing']} subject "
                        f"group(s): raising the trigger turned frames ON -- the "
                        f"post-processor is broken and every operating point is "
                        f"untrustworthy. worst={list(_bad.items())[:3]}")
            else:
                print("[mono-gate] skipped: <2 OOF ensemble npz arrays present in cwd",
                      flush=True)
        except AssertionError:
            raise
        except Exception as _me:
            print(f"[mono-gate] WARNING: could not assert (non-fatal): {_me}", flush=True)

    if _flag("NEUROMAMBA_CALIB_USE_OOF_POOL", "0"):
        import glob as _glob
        _by_fam = {}
        for _p in sorted(_glob.glob("*_fold*_ens.npz")):
            if "_baseline_" not in _p:
                _by_fam.setdefault(_p.split("_fold")[0], []).append(_p)
        for _pfam, _paths in sorted(_by_fam.items()):
            if len(_paths) < 2:
                print(f"[oof-pool v25.60] {_pfam}: <2 OOF npz arrays -- skipped", flush=True)
                continue
            try:
                import pooled_oof_calibration as _poc
                _poc.run_pool_report(_paths, _pfam,
                                     out_prefix=f"pooled_oof_calibration_{_pfam}")
                print(f"[oof-pool v25.60] {_pfam}: pooled calibration written "
                      f"(pooled_oof_calibration_{_pfam}.json/.md)", flush=True)
            except Exception as _e:
                print(f"[oof-pool v25.60] {_pfam} FAILED (non-fatal): {_e}", flush=True)
                traceback.print_exc()
    out_path = args.output or "neuromamba_v25_publication_results.json"
    with open(out_path, "w") as f:
        json.dump(grand, f, indent=2, cls=NpEncoder)
    print(f"[output] Results written to {out_path}", flush=True)
    _s4 = emit_clinical_s4(grand, out_path)
    if _s4:
        print(f"[clinical-s4 v25.43] {_s4} clinical rows -> {str(out_path).replace('.json', '_clinical_s4.csv')}", flush=True)
    return grand, gate

def emit_clinical_s4(grand, results_path):
    """v25.43.0 E2: publication-required clinical table (spec:
    research/agent-clinical-standards.md). One row per repeat + pooled row.
    Column groups: provenance / ranking / frozen operating point / episode
    burden (FA/day) / latency / calibration / safety-runtime. Missing -> "".
    Tolerant walker: collects leaf dicts carrying >=2 clinical keys, deduped."""
    import csv as _csv
    _CLIN_KEYS = ("official_map", "mAP", "map", "subtype_map", "sensitivity", "sens",
                  "farh", "far", "ece", "trigger", "window_f1", "event_f1",
                  "StartHesitation", "Turn", "Walking", "val_phase", "best_epoch")
    seen, leaves = set(), []
    def _walk(obj, path):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, dict):
            hits = [k for k in _CLIN_KEYS if k in obj]
            if len(hits) >= 2:
                leaves.append((path, obj, hits))
            for k, v in obj.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]" if path else f"[{i}]")
    _walk(grand, "")
    rows = []
    for path, d, _ in leaves:
        r = {
            "provenance_code_version": d.get("code_version", ""),
            "provenance_identity": str(d.get("training_identity", ""))[:12],
            "repeat_path": path,
            "mAP": d.get("official_map", d.get("mAP", d.get("map", ""))),
            "AP_SH": d.get("StartHesitation", ""),
            "AP_Turn": d.get("Turn", ""),
            "AP_Walk": d.get("Walking", ""),
            "sens": d.get("sensitivity", d.get("sens", "")),
            "FARh": d.get("farh", d.get("far", "")),
            "far_day_at_T": d.get("far_day_at_T", ""),
            "ece_max": d.get("ece", ""),
            "trigger": d.get("trigger", ""),
            "trigger_source": d.get("trigger_source", ""),
            "window_f1": d.get("window_f1", ""),
            "event_f1": d.get("event_f1", ""),
            "val_phase": d.get("val_phase", ""),
            "sel_best_epoch": d.get("sel_best_epoch", d.get("best_epoch", "")),
        }
        rows.append(r)
    if not rows:
        print("[clinical-s4 v25.43] no clinical leaf records found in results; skipped", flush=True)
        return 0
    cols = list(rows[0].keys())
    csv_path = str(results_path).replace(".json", "_clinical_s4.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)

    out_path = args.output or "neuromamba_v25_publication_results.json"
    with open(out_path, "w") as f:
        json.dump(grand, f, indent=2, cls=NpEncoder)
    print(f"[output] Results written to {out_path}", flush=True)
    _s4 = emit_clinical_s4(grand, out_path)
    if _s4:
        print(f"[clinical-s4 v25.43] {_s4} clinical rows -> {str(out_path).replace(chr(46)+chr(106)+chr(115)+chr(111)+chr(110), chr(46)+chr(99)+chr(115)+chr(118))} ", flush=True)
    return grand, gate

def make_synthetic_sequences(n_subjects=8, n_records_per_subject=3, T=3000, fs=100.0, seed=0):
    rng = np.random.RandomState(seed)
    seqs = []
    for si in range(n_subjects):
        sid = f"SYN{si:03d}"
        for ri in range(n_records_per_subject):
            t = np.arange(T) / fs
            sig = rng.randn(T, 3).astype(np.float32) * 0.05
            sig[:, 2] += 1.0
            y_sub = np.zeros((T, 3), dtype=np.float32)
            so = np.ones((T, 3), dtype=bool)  # v25: default all-observed
            if rng.random() < 0.6:
                n_ev = rng.randint(1, 3)
                for _ in range(n_ev):
                    c = rng.randint(0, T - 200)
                    dur = rng.randint(50, 200)
                    k = rng.randint(0, 3)
                    sig[c:c + dur, 2] += 0.08 * np.sin(2 * np.pi * 6.0 * t[c:c + dur])
                    y_sub[c:c + dur, k] = 1.0
            signal = assemble_input_features(sig, fs_hz=fs)
            src = "defog" if ri % 2 == 0 else "tdcsfog"
            seq = FoGSequence(signal=signal, y_subtype=y_sub, subject_id=sid, record_id=f"{sid}_r{ri}",
                               source_type=src, subtype_observed=so, fs_hz=fs, strides=1.0 / fs)
            seqs.append(seq)
    return seqs

def run_selftest(device="cpu", epochs=3):
    print("=" * 70)
    print("NeuroMamba v25 SELF-TEST (synthetic data, CPU)")
    print("=" * 70)
    ok = True

    print("[selftest 1/10] feature assembly / physics freeze-band response...")
    try:
        rng = np.random.RandomState(7)
        T, fs = 4000, 100.0
        t = np.arange(T) / fs
        sig = rng.randn(T, 3).astype(np.float32) * 0.05
        sig[:, 2] += 1.0
        sig[1500:2200, 2] += 0.15 * np.sin(2 * np.pi * 6.0 * t[1500:2200]).astype(np.float32)
        feats = assemble_input_features(sig, fs_hz=fs)
        assert feats.shape == (T, N_CHANNELS) and np.isfinite(feats).all()
        if N_CHANNELS == 11:
            fi_burst = feats[1800, 7]; fi_calm = feats[300, 7]
            dom_f_burst = feats[1800, 10]
            assert fi_burst > fi_calm, f"freeze index must rise in burst {fi_burst:.3f} vs {fi_calm:.3f}"
            assert abs(dom_f_burst - 6.0) < 0.6, f"dominant frequency should be 6 Hz, got {dom_f_burst}"
            print(f"  OK: logFI burst={fi_burst:.3f} vs calm={fi_calm:.3f}, domF={dom_f_burst:.2f} Hz")
        else:
            print(f"  OK: {N_CHANNELS}-channel input, shape {feats.shape}")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 2/10] ConvBiGRU + ResNet1D + winner forward pass...")
    try:
        _sanity_test_resnet1d()
        _sanity_test_convbigru()
        _sanity_test_winner()
        print("  OK: ConvBiGRU + ResNet1D + winner forward pass")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 3/10] subtype_observed contract + ASL per-class mask...")
    try:
        seqs = make_synthetic_sequences(n_subjects=6, n_records_per_subject=2, T=1200, seed=1)
        assert seqs[0].subtype_observed.shape == (1200, 3)
        assert seqs[0].subtype_observed.all()
        logits = torch.randn(50, 3)
        targets = torch.zeros(50, 3); targets[0, 0] = 1.0; targets[5, 1] = 1.0
        mask_full = torch.ones(50, 3)
        loss_full = subtype_asl_loss(logits, targets, mask_full)
        mask_no_sh = mask_full.clone(); mask_no_sh[:, 0] = 0.0
        loss_no_sh = subtype_asl_loss(logits, targets, mask_no_sh)
        assert torch.isfinite(loss_full) and loss_full.item() > 0
        print(f"  OK: subtype_observed shape OK, ASL full={loss_full.item():.4f} no_sh={loss_no_sh.item():.4f}")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 4/10] loss_weight_schedule (sub_w floor from epoch 0)...")
    try:
        fog_w_0, sub_w_0 = loss_weight_schedule(0, 80)
        exp_floor = float(os.environ.get("NEUROMAMBA_SUB_W_FLOOR", 0.35))
        assert abs(sub_w_0 - exp_floor) < 1e-9, f"sub_w at epoch 0 must be the floor {exp_floor}, got {sub_w_0}"
        fog_w_5, sub_w_5 = loss_weight_schedule(5, 80)
        assert sub_w_5 > exp_floor, f"sub_w at epoch 5 must be > floor, got {sub_w_5}"
        print(f"  OK: epoch 0 sub_w={sub_w_0} (floor {exp_floor}), epoch 5 sub_w={sub_w_5:.3f}")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 5/10] rare_subtype_aware_split (1st-place trick)...")
    try:
        seqs = make_synthetic_sequences(n_subjects=12, n_records_per_subject=2, T=1500, seed=2)
        train, calib, thresh, diag = rare_subtype_aware_split(seqs, seed=0)
        assert len(train) > 0 and len(calib) > 0 and len(thresh) > 0
        assert diag["split_method"] == "rare_subtype_aware"
        assert "held_sh_frac" in diag and "held_walk_frac" in diag
        print(f"  OK: rare_split diag={json.dumps({k:v for k,v in diag.items() if k != 'split_method'})}")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 6/10] audit_subtype_contract (containment invariant)...")
    try:
        seqs = make_synthetic_sequences(n_subjects=4, n_records_per_subject=1, T=1000, seed=3)
        audit = audit_subtype_contract(seqs)
        assert "defog" in audit or "tdcsfog" in audit
        print(f"  OK: audit passed for {len(audit)} sources, no containment violations")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 7/10] train_one_config end-to-end, ALL model types (CPU, few epochs)...")
    try:
        for k in ["NEUROMAMBA_USE_SPECTRAL", "NEUROMAMBA_USE_ATTENTION"]:
            os.environ[k] = "0"
        os.environ["NEUROMAMBA_HARD_FAIL_FALLBACK"] = "0"
        os.environ["NEUROMAMBA_EARLY_STOP_MIN_EPOCH"] = "1"
        os.environ["NEUROMAMBA_WINDOW_ROLL"] = "0"  # speed up selftest
        seqs = make_synthetic_sequences(n_subjects=10, n_records_per_subject=2, T=1500)
        train, calib, thresh, _ = rare_subtype_aware_split(seqs, seed=0)
        os.environ["NEUROMAMBA_WINNER_D"] = "32"
        os.environ["NEUROMAMBA_WINNER_HEADS"] = "4"
        os.environ["NEUROMAMBA_WINNER_DEPTH"] = "2"
        for mt in ["sequence", "resnet1d", "winner"]:
            model, results = train_one_config(train, calib, thresh, device=device, epochs=epochs, lr=1e-3,
                                               batch_size=4, seed=0, fog_warmup_epochs=1, ramp_epochs=1,
                                               model_type=mt)
            assert "frozen_threshold" in results and "final_event_f1" in results
            assert "use_walking_gate" in results and results["use_walking_gate"]
            print(f"  OK [{mt}]: final_event_f1={results['final_event_f1']:.3f} "
                  f"frozen_threshold={results['frozen_threshold']:.3f} walking_gate=ON")
        os.environ["NEUROMAMBA_WINDOW_ROLL"] = "1"  # restore default
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 8/10] per-class observed mAP (unknown frames masked)...")
    try:
        sub_labels = np.zeros((100, 3), dtype=np.float32)
        sub_labels[:20, 0] = 1.0      # SH positive
        sub_labels[30:40, 1] = 1.0    # Turn positive
        sub_labels[60:70, 2] = 1.0    # Walking positive
        sub_probs = np.random.rand(100, 3).astype(np.float32)
        sub_probs[:20, 0] = 0.9       # high prob for SH true positives
        sub_probs[30:40, 1] = 0.9     # high prob for Turn true positives
        sub_probs[60:70, 2] = 0.9     # high prob for Walking true positives
        observed = np.ones((100, 3), dtype=bool)
        observed[50:, 0] = False      # SH unobserved on last 50 (but SH positives are all <20, still scored)
        m = compute_official_like_map(sub_labels, sub_probs, observed)
        assert m["comparable"], "all 3 subtypes have positives in observed -> comparable"
        assert m["StartHesitation"] > 0.5, f"SH AP should be high (probs=0.9 on positives), got {m['StartHesitation']:.3f}"
        assert all(np.isfinite(m.get(n, 0.0)) for n in SUBTYPE_NAMES)
        print(f"  OK: per-class observed mAP={m['mAP']:.4f}, SH={m['StartHesitation']:.3f} "
              f"Turn={m['Turn']:.3f} Walk={m['Walking']:.3f} (unknown SH frames on 50:100 masked out)")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 9/10] subject-level bootstrap CI...")
    try:
        rng = np.random.RandomState(0)
        n = 1000
        sub_probs = rng.rand(n, 3).astype(np.float32)
        sub_labels = np.zeros((n, 3), dtype=np.float32)
        sub_labels[::20, 0] = 1.0
        sub_observed = np.ones((n, 3), dtype=bool)
        groups = np.array([f"s{i // 100}" for i in range(n)], dtype=object)  # 10 subjects
        ci = bootstrap_kaggle_map_ci(sub_probs, sub_labels, sub_observed, n_boot=50, groups=groups)
        assert ci["unit"] == "subject"
        assert ci["ci_lo"] <= ci["mean"] <= ci["ci_hi"]
        print(f"  OK: subject-level bootstrap CI=[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}] mean={ci['mean']:.4f} unit={ci['unit']}")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("[selftest 10/10] walking-gate + window-roll inference...")
    try:
        seqs = make_synthetic_sequences(n_subjects=4, n_records_per_subject=1, T=2000, seed=5)
        model = ResNet1DPerFrameAdapter(in_channels=N_CHANNELS, use_walking_gate=True).to(device)
        probs, labels, ids = run_inference_cached(model, seqs, device, window_roll=True)
        assert len(probs) == len(labels)
        gated = _apply_walking_gate_inference(model, seqs, probs, device, window_roll=False)
        assert gated.shape == probs.shape
        assert (gated <= probs + 1e-6).all(), "walking-gate must attenuate (not amplify) probs"
        print(f"  OK: window-roll inference shape={probs.shape}, walking-gate attenuated (max gating={1 - (gated.mean()/probs.mean()):.3f})")
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc(); ok = False

    print("=" * 70)
    print(f"SELF-TEST {'PASSED' if ok else 'FAILED'}")
    print("=" * 70)
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NeuroMamba v25.2 -- SOTA & Clinical (winner-style sequence model)")
    ap.add_argument("--mode", type=str, default="cv", choices=["cv", "selftest", "audit-only"])
    ap.add_argument("--audit-only", action="store_true",
                    help="v25.42.6 Tier A preflight: verify dataset/cache/split/leak/"
                         "checkpoint/baseline identity WITHOUT allocating GPU time. "
                         "Exit 0 = safe to run; exit 1 = blocking issue.")
    ap.add_argument("--model", type=str, default="sequence",
                    choices=["sequence", "resnet1d", "winner"],
                    help="sequence = Conv+BiGRU primary member (DEFAULT). "
                         "resnet1d = frozen ResNet1D baseline. winner = Transformer+BiLSTM comparator.")
    ap.add_argument("--datasets", nargs="+", default=["tlvmc"])
    ap.add_argument("--external", nargs="+", default=[])
    ap.add_argument("--domain-split", action="store_true",
                    help="train separate defog/tdcsfog models (1st+2nd place approach)")
    ap.add_argument("--families", nargs="+", default=None,
                    help="v25.20: run ONLY these families, e.g. --families tdcsfog "
                         "(overrides --domain-split; saves GPU hours on measurement runs)")
    ap.add_argument("--family-shared-val", action="store_true",
                    help="v25.38: with exactly one --families family and --n-splits 1, "
                         "filter only the TRAIN side to the family and score the SHARED "
                         "pooled val split -- per-domain (defog/tdcsfog) models "
                         "ensemble-compatible, the 1st-place structure. Combine the "
                         "runs with ensemble_families.py. DEPRECATED alias for "
                         "--split-mode shared_pooled.")
    ap.add_argument("--split-mode", type=str, default=None,
                    choices=["pure_family", "shared_pooled", "grouped_cv"],
                    help="v25.42.8 (Phase-1 protocol hardening, reviewer Blocker A): "
                         "explicit primary split protocol. "
                         "pure_family  = train AND evaluate on the requested family only "
                         "               (its own subject split; the journal primary claim). "
                         "shared_pooled= train filtered to the family, val = the pooled "
                         "               defog+tdcsfog split (ensemble-compatible, contest "
                         "               mimic; requires one family + n_splits==1). "
                         "grouped_cv   = n_splits-fold subject-grouped CV on the family. "
                         "Default: pure_family for n_splits==1; shared_pooled when "
                         "--family-shared-val is passed; grouped_cv for n_splits>1.")
    ap.add_argument("--no-rare-split", action="store_true",
                    help="disable rare-subtype-aware validation split (1st-place trick); use random subject split")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="number of seeds for ensemble (1st place used 4+4 per domain)")
    ap.add_argument("--n-repeats", type=int, default=1,
                    help="v25.42.7 (Phase 1): repeated subject-grouped outer CV. "
                         "Each repeat rebuilds the folds with a fresh split seed "
                         "(args.seed + repeat*10000), so the between-repeat mean/SD "
                         "quantifies split-selection variance -- replaces the "
                         "single-split estimate. Requires fresh training per repeat "
                         "(different splits -> different identity).")
    ap.add_argument("--run-baseline-comparator", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip completed folds and resume interrupted in-fold training from checkpoints")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--journal-report", action="store_true",
                    help="v25.22: write journal_report.md -- publication-grade "
                         "generalization report (internal CV, frozen external "
                         "daphnet/xuanwu/weargait, deployment gap, adaptation, "
                         "clinical operating points, ICC, gate). Clinical metrics "
                         "headline; no leaderboard/mAP framing. Defaults "
                         "--external to xuanwu weargait daphnet.")
    ap.add_argument("--benchmark-on", nargs="+", default=None,
                    help="v25.27: train the SAME architecture ON one external "
                         "cohort with LOSO CV (e.g. --benchmark-on daphnet) for "
                         "fair comparison with published trained-on-cohort "
                         "numbers (DeepFoG 0.83/0.88; Lin & Grewal F1 0.9999). "
                         "Labeled 'trained on cohort', NOT frozen transfer.")
    args = ap.parse_args()

    try:
        with open(__file__, "rb") as _f:
            _sha_full = hashlib.sha256(_f.read()).hexdigest()
            _sha = _sha_full[:16]
    except Exception:
        _sha_full, _sha = "?", "?"
    _man_env = ["NEUROMAMBA_CALIB_FRAC", "NEUROMAMBA_THRESH_FRAC", "NEUROMAMBA_TRAIN_WINDOW",
                "NEUROMAMBA_WINNER_TOKEN_LOSS", "NEUROMAMBA_MLLS_CORRECT",
                "NEUROMAMBA_ONSET_WINDOW_S", "NEUROMAMBA_ONSET_TARGET_MODE",
                "NEUROMAMBA_ONSET_TAU_S", "NEUROMAMBA_TOKEN_POS_WEIGHT_MODE",
                "NEUROMAMBA_ROLL_WINDOW", "NEUROMAMBA_ROLL_STRIDE",
                "NEUROMAMBA_SCORE_ANCHOR", "NEUROMAMBA_SEED_DIVERSITY",
                "NEUROMAMBA_SELECT_MAP_BLEND", "NEUROMAMBA_NOTYPE_PSEUDO",
                "NEUROMAMBA_TOKEN_ASL", "NEUROMAMBA_RARE_BOOST",   # v25.59 provenance
                "NEUROMAMBA_HARD_NEG_CROP_PROB", "NEUROMAMBA_N_CHANNELS",
                "NEUROMAMBA_LONG_CONTEXT_RECAL", "NEUROMAMBA_LONG_CONTEXT_MODE",
                "NEUROMAMBA_ADAM_BETA2", "NEUROMAMBA_SEL_F1_GRID",
                "NEUROMAMBA_GRAD_CLIP", "NEUROMAMBA_GRAD_CLIP_MODE",
                "NEUROMAMBA_MIN_EPOCH_MAP_SEL", "NEUROMAMBA_FOG_TOLERANCE",
                "NEUROMAMBA_MASK_DEFOG_SH", "NEUROMAMBA_DEFOG_SH_WEAK",
                "NEUROMAMBA_DOMAIN_DANN",
                "NEUROMAMBA_CONFORMAL_SSBC", "NEUROMAMBA_MIN_CALIB_SUBJ_POSTPROC",
                "NEUROMAMBA_EVENT_BUDGET_CONFORMAL", "NEUROMAMBA_BOOTSTRAP_N",
                "NEUROMAMBA_SUBTYPE_FINETUNE", "NEUROMAMBA_SUBTYPE_FT_EPOCHS",
                "NEUROMAMBA_SUBTYPE_FT_LR", "NEUROMAMBA_SUBTYPE_FT_MONITOR",
                "NEUROMAMBA_SUBTYPE_FT_RESTORE_BEST",
                "NEUROMAMBA_PARETO_SELECT", "NEUROMAMBA_PARETO_ALPHA",
                "NEUROMAMBA_PARETO_MIN_SH_AP",
                "NEUROMAMBA_LOGIT_ADJUST_TAU", "NEUROMAMBA_TTA_FLIP",
                "NEUROMAMBA_JOINT_CERTIFY"]
    _eff_cfg = {k: os.environ.get(k, d) for k, d in TRAINING_ENV_DEFAULTS.items()}
    for k, d in TRAINING_ARCH_ENV.items():
        v = os.environ.get(k, d)
        if v:
            _eff_cfg[k] = v
    _run_tid = training_identity(
        model_type=args.model, lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed, n_seeds=args.n_seeds, fold_idx=0,
        family=(args.families[0] if args.families else "all"),
        use_rare_split=not args.no_rare_split)
    print("[selection-policy v25.42.9] FROZEN DEVELOPMENT PROTOCOL for this run:", flush=True)
    print(f"  model-selection metric : {'windowF1' if _flag('NEUROMAMBA_SELECT_BY_WINDOW_F1', '1') else 'calibSubtypeMAP'}", flush=True)
    print(f"  early-stop rule        : patience={EARLY_STOP_PATIENCE} min_epoch={EARLY_STOP_MIN_EPOCH}", flush=True)
    print(f"  calibration method     : temperature fit on calib only (gated first); T-grid [0.10,12]x240, boundary fits accepted (v25.43.0)", flush=True)
    print(f"  threshold objective    : {get('NEUROMAMBA_THRESH_OBJECTIVE', 'windowF1 (variant) -> max-epF1 fallback on calib')}", flush=True)
    print(f"  postprocessing space   : confirm={get('NEUROMAMBA_POSTPROC_CONFIRM', 'auto>=3')} cooldown={get('NEUROMAMBA_POSTPROC_COOLDOWN', 'auto>=30')} sustain/merge from calib", flush=True)
    print(f"  swa/ema policy         : aswa from epoch {SWA_START}; variants raw/ema/swa selected on calib", flush=True)
    print(f"  ensemble rule          : {args.n_seeds}-seed mean; non-final seeds EXCLUDED (v25.42.7)", flush=True)
    print(f"  split policy           : split_mode={getattr(args, 'split_mode', None)} n_repeats={getattr(args, 'n_repeats', 1)} (locked before first repeat)", flush=True)
    _run_device = args.device if torch.cuda.is_available() else "cpu"
    print("[manifest] " + json.dumps({
        "code": CODE_VERSION, "file_sha256": _sha, "file_sha256_full": _sha_full,
        "n_channels": N_CHANNELS,
        "seed": args.seed, "n_splits": args.n_splits, "n_seeds": args.n_seeds,
        "model": args.model, "device": _run_device, "device_requested": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "epochs": args.epochs,
        "families": args.families, "family_shared_val": args.family_shared_val,
        "split_mode": getattr(args, "split_mode", None),
        "training_identity": _run_tid,
        "env": {k: os.environ.get(k) for k in _man_env if os.environ.get(k) is not None},
        "effective_config": _eff_cfg}),
        flush=True)

    if args.mode == "selftest":
        ok = run_selftest(device="cpu", epochs=int(os.environ.get("NEUROMAMBA_SELFTEST_EPOCHS", 3)))
        sys.exit(0 if ok else 1)
    elif args.mode == "audit-only" or args.audit_only:
        ok = preflight_audit(args, _run_tid)
        sys.exit(0 if ok else 1)
    else:
        if args.journal_report and not args.external:
            args.external = ["xuanwu", "weargait", "daphnet"]
            print("[journal v25.22] --journal-report: external cohorts default to "
                  "xuanwu, weargait, daphnet (4-dataset protocol: TLVMc train + "
                  "3 frozen external sites)", flush=True)
        _cv = run_cv_main(args)
        if _cv is None or (isinstance(_cv, tuple) and len(_cv) > 0 and _cv[0] is None):
            print("[journal v25.46] run_cv_main returned None (all folds failed) -- "
                  "exiting 2 (see per-fold errors above)", flush=True)
            sys.exit(2)
        grand, gate = _cv