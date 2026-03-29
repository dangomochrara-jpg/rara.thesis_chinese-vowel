#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import csv
import os
import sys
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# Enable Japanese font rendering for matplotlib (labels, legends, etc.)
try:
    import japanize_matplotlib  # type: ignore
except Exception:
    print("[warn] japanize_matplotlib not available; Japanese glyphs may not render.", file=sys.stderr)
# Force white backgrounds globally for figures and axes
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# Prefer a system Japanese font to guarantee glyph coverage
def _configure_japanese_font() -> None:
    try:
        # Common JP fonts by OS
        if sys.platform.startswith("win"):
            candidates = [
                "Yu Gothic UI", "Yu Gothic", "Meiryo", "MS Gothic", "MS Mincho",
            ]
        elif sys.platform == "darwin":
            candidates = [
                "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Osaka",
            ]
        else:
            candidates = [
                "Noto Sans CJK JP", "Noto Serif CJK JP", "IPAexGothic", "VL Gothic",
            ]

        available_families = {f.name for f in fm.fontManager.ttflist}
        chosen = None
        for name in candidates:
            if name in available_families:
                chosen = name
                break

        # Always fix minus sign rendering for CJK environments
        plt.rcParams["axes.unicode_minus"] = False

        if chosen:
            # Set as preferred sans-serif font
            current = plt.rcParams.get("font.sans-serif", [])
            if isinstance(current, str):
                current = [current]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [chosen] + list(current)
            print(f"[info] Using Japanese font: {chosen}")
        else:
            print("[warn] No JP-specific font found; relying on default/japanize_matplotlib.", file=sys.stderr)
    except Exception as e:
        print(f"[warn] Failed to configure JP font: {e}", file=sys.stderr)

_configure_japanese_font()
import librosa
import librosa.display
import sounddevice as sd
from scipy.signal import lfilter, get_window

# Optional: shape alignment for trajectory similarity
_HAVE_PROCRUSTES = True
try:
    from scipy.spatial import procrustes as _procrustes  # type: ignore
except Exception:
    _HAVE_PROCRUSTES = False
    _procrustes = None  # type: ignore

# ------- Optional: Praat / Parselmouth for robust formants -------
_HAVE_PARSEL = True
try:
    import parselmouth
except Exception:
    _HAVE_PARSEL = False  # fallback to LPC
    parselmouth = None

# ---------------------- Parameters ----------------------
SR = 16000
WIN_LEN = int(0.04 * SR)   # 40 ms window
HOP_LEN = int(0.01 * SR)   # 10 ms hop
N_FFT  = 1024
ROLL_SEC = 3.0             # seconds of rolling display
ROLL_SAMPLES = int(ROLL_SEC * SR)
MAX_Y_HZ = 4000            # display limit for spectrogram
SMOOTH_N = 5               # moving average window for F1/F2 (frames)

DEFAULT_VOWELS = ["/a/", "/i/", "/u/", "/e/", "/o/", "/æ/", "/ɪ/", "/ʊ/"]

# Optional CSV means import: search these filenames in current directory
MEANS_CSV_CANDIDATES = [
    "vowel_means.csv",
    "means.csv",
    "formant_means.csv",
]
FLAT_TEMPLATE_POINTS = 7  # number of points for flat template made from means

# Optional: Paste your table text directly here. Leave empty if not used.
#   a.wav,1022.32,1542.13
#   Ü (yu).wav\t827.16\t2142.63
PASTED_TABLE: str = ""

# Hardcoded means dictionary: label or filename -> {"F1": value, "F2": value}
HARDCODED_MEANS: Dict[str, Dict[str, float]] = {}

# ---------------------- Visualization Settings ----------------------
# Set to False to hide spectrogram and keep a clean white background
SHOW_SPECTROGRAM = False

# ---------------------- Templates ----------------------
def _example_templates() -> Dict[str, np.ndarray]:
    
    return {
        "/a/": np.array([[800,1200],[750,1300],[700,1400],[650,1500],[600,1600]], dtype=float),
        "/i/": np.array([[300,2300],[320,2400],[340,2500],[330,2450],[320,2400]], dtype=float),
        "/u/": np.array([[350,700],[340,720],[330,740],[340,720],[350,700]], dtype=float),
        "/e/": np.array([[400,2000],[420,2050],[450,2100],[430,2050],[410,2000]], dtype=float),
        "/o/": np.array([[500,900],[520,950],[540,1000],[520,950],[500,900]], dtype=float),
        "/æ/": np.array([[700,1700],[680,1750],[660,1800],[680,1750],[700,1700]], dtype=float),
        "/ɪ/": np.array([[350,2000],[360,2050],[370,2100],[360,2050],[350,2000]], dtype=float),
        "/ʊ/": np.array([[400,900],[390,920],[380,940],[390,920],[400,900]], dtype=float),
    }

def _sanitize_vowel_label(name: str) -> str:
    """Normalize label from file-like names.

    - Remove trailing .wav/.WAV
    - Strip whitespace at ends
    - Wrap with slashes like "/a/"
    """
    n = name.strip()
    if n.lower().endswith(".wav"):
        n = n[: -4]
    if not n.startswith("/"):
        n = "/" + n
    if not n.endswith("/"):
        n = n + "/"
    return n

def _load_means_csv() -> Dict[str, np.ndarray]:
    """Load per-vowel average F1/F2 from a CSV if present.

    Expected columns (case-insensitive, flexible):
      - first text column as the label (e.g., file name)
      - a column containing "F1" and a column containing "F2" (Hz)

    Returns a mapping of vowel label -> (K,2) ndarray where values are flat at
    the provided means. If CSV not found or unreadable, returns empty dict.
    """
    base_dir = os.path.dirname(__file__)
    path = None
    for cand in MEANS_CSV_CANDIDATES:
        p = os.path.join(base_dir, cand)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return {}
    result: Dict[str, np.ndarray] = {}
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # Identify likely columns
            headers = [h or "" for h in reader.fieldnames or []]
            # Prefer explicit Japanese header for file name if present
            name_keys = [h for h in headers if h.strip() and h.strip().lower() in ("file", "filename", "name") or "ファイル" in h]
            name_key = name_keys[0] if name_keys else (headers[0] if headers else None)
            f1_key = next((h for h in headers if "f1" in h.lower()), None)
            f2_key = next((h for h in headers if "f2" in h.lower()), None)
            if not (name_key and f1_key and f2_key):
                print(f"[warn] CSV means missing required columns: {path}", file=sys.stderr)
                return {}
            for row in reader:
                raw_name = str(row.get(name_key, "")).strip()
                if not raw_name:
                    continue
                try:
                    f1 = float(str(row.get(f1_key, "")).strip())
                    f2 = float(str(row.get(f2_key, "")).strip())
                except Exception:
                    continue
                if not (np.isfinite(f1) and np.isfinite(f2) and f1 > 0 and f2 > 0):
                    continue
                label = _sanitize_vowel_label(raw_name)
                flat = np.tile(np.array([[f1, f2]], dtype=float), (FLAT_TEMPLATE_POINTS, 1))
                result[label] = flat
        if result:
            print(f"[info] loaded CSV means from {os.path.basename(path)}: {list(result.keys())}")
    except Exception as e:
        print(f"[warn] failed to read CSV means: {e}", file=sys.stderr)
        return {}
    return result

def _parse_pasted_table(pasted: str) -> Dict[str, Dict[str, float]]:
    """Parse pasted table text with 3 columns: name, F1, F2.

    Supports comma, tab, or whitespace separation. Ignores header lines.
    Returns a dict compatible with HARDCODED_MEANS.
    """
    if not pasted.strip():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    lines = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
    for ln in lines:
        # Skip header-ish lines
        low = ln.lower()
        if ("f1" in low and "f2" in low) or ("hz" in low):
            continue
        # Split by comma, tab, or multiple spaces
        if "," in ln:
            parts = [p.strip() for p in ln.split(",")]
        elif "\t" in ln:
            parts = [p.strip() for p in ln.split("\t")]
        else:
            parts = [p for p in ln.split()]
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            f1 = float(parts[1])
            f2 = float(parts[2])
        except Exception:
            # Try to salvage numbers appearing later in the row
            nums = []
            for p in parts[1:]:
                try:
                    nums.append(float(p))
                except Exception:
                    pass
            if len(nums) >= 2:
                f1, f2 = nums[0], nums[1]
            else:
                continue
        out[name] = {"F1": f1, "F2": f2}
    return out

def _means_to_flat_templates(means: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for raw_name, vals in means.items():
        try:
            f1 = float(vals.get("F1", np.nan))
            f2 = float(vals.get("F2", np.nan))
        except Exception:
            continue
        if not (np.isfinite(f1) and np.isfinite(f2) and f1 > 0 and f2 > 0):
            continue
        # Round to 2 decimals as requested
        f1 = round(f1, 2)
        f2 = round(f2, 2)
        label = _sanitize_vowel_label(raw_name)
        flat = np.tile(np.array([[f1, f2]], dtype=float), (FLAT_TEMPLATE_POINTS, 1))
        out[label] = flat
    return out

def load_templates() -> Dict[str, np.ndarray]:
    base = {}
    path = os.path.join(os.path.dirname(__file__), "templates.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, seq in raw.items():
                arr = np.array(seq, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    base[k] = arr
            if base:
                print(f"[info] loaded templates.json with vowels: {list(base.keys())}")
        except Exception as e:
            print(f"[warn] failed to load templates.json: {e}", file=sys.stderr)
    if not base:
        print("[info] using built-in demo templates")
        base = _example_templates()

    # Merge CSV-derived means as flat templates (overrides duplicates)
    csv_templates = _load_means_csv()
    if csv_templates:
        base.update(csv_templates)
        print(f"[info] merged {len(csv_templates)} vowels from CSV means")
    # Merge pasted table (if provided)
    pasted_means = _parse_pasted_table(PASTED_TABLE)
    if pasted_means:
        pt = _means_to_flat_templates(pasted_means)
        base.update(pt)
        print(f"[info] merged {len(pt)} vowels from pasted table")
    # Merge hardcoded means (highest priority)
    hc_templates = _means_to_flat_templates(HARDCODED_MEANS)
    if hc_templates:
        base.update(hc_templates)
        print(f"[info] merged {len(hc_templates)} vowels from HARDCODED_MEANS")
    return base

TEMPLATES = load_templates()
VOWEL_LIST = sorted(TEMPLATES.keys(), key=lambda s: DEFAULT_VOWELS.index(s) if s in DEFAULT_VOWELS else 999)

# ---------------------- Formant estimation ----------------------
def lpc_formants(y: np.ndarray, sr: int, order: int = 14, n_formants: int = 2) -> np.ndarray:
    """LPC estimate of formants (simple, real-time friendly)."""
    if len(y) < order + 1:
        return np.array([np.nan] * n_formants)
    y = lfilter([1, -0.97], 1, y)
    try:
        A = librosa.lpc(y, order=order)
        roots = np.roots(A)
        roots = roots[np.imag(roots) >= 0]
        angs = np.arctan2(np.imag(roots), np.real(roots))
        freqs = np.sort(angs * (sr / (2*np.pi)))
        freqs = freqs[freqs > 0]
        if len(freqs) < n_formants:
            pad = np.array([np.nan] * (n_formants - len(freqs)))
            return np.concatenate([freqs, pad])[:n_formants]
        return freqs[:n_formants]
    except Exception:
        return np.array([np.nan] * n_formants)

def praat_formants_center(y: np.ndarray, sr: int, max_formant: float = 5500.0) -> Tuple[float, float]:
    """Use Praat (parselmouth) to get F1/F2 at center of the given frame."""
    if not _HAVE_PARSEL:
        return (np.nan, np.nan)
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        formant = parselmouth.praat.call(snd, "To Formant (burg)", 0.025, 5.0, max_formant, 0.025, 50)
        t = snd.xmin + (snd.xmax - snd.xmin) / 2
        f1 = parselmouth.praat.call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = parselmouth.praat.call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        return float(f1), float(f2)
    except Exception:
        return (np.nan, np.nan)

def estimate_f1f2_frame(frame: np.ndarray, sr: int) -> Tuple[float, float]:
    """Try Praat; fallback to LPC."""
    f1, f2 = praat_formants_center(frame, sr)
    if np.isnan(f1) or np.isnan(f2) or f1 <= 0 or f2 <= 0:
        ff = lpc_formants(frame, sr, order=14, n_formants=2)
        if len(ff) == 2 and not np.any(np.isnan(ff)):
            return float(ff[0]), float(ff[1])
        else:
            return (np.nan, np.nan)
    return (f1, f2)

def moving_average(x: List[float], n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    kernel = np.ones(n)/n
    y = np.convolve(x, kernel, mode='same')
    return y

# ---------------------- Similarity (shape) ----------------------
def _resample_traj(traj: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a (K,2) trajectory to num_points along its own time axis [0,1]."""
    if traj.ndim != 2 or traj.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    k = len(traj)
    if k < 2:
        return np.empty((0, 2), dtype=float)
    t = np.linspace(0.0, 1.0, k)
    ti = np.linspace(0.0, 1.0, num_points)
    f1 = np.interp(ti, t, traj[:, 0])
    f2 = np.interp(ti, t, traj[:, 1])
    return np.stack([f1, f2], axis=1)

def extract_f1f2_trajectory(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame F1/F2 trajectory for the given mono audio.

    Returns (t, f1s, f2s) where t is seconds for each frame center based on
    global WIN_LEN/HOP_LEN. Applies simple mean-removal, peak normalization,
    and moving-average smoothing.
    """
    if y.ndim > 1:
        y = y[:, 0]
    if len(y) < WIN_LEN * 2:
        return np.array([]), np.array([]), np.array([])
    frames = librosa.util.frame(y, frame_length=WIN_LEN, hop_length=HOP_LEN).T
    f1s: List[float] = []
    f2s: List[float] = []
    for fr in frames:
        fr = fr - fr.mean()
        denom = float(np.max(np.abs(fr)) + 1e-8)
        fr = fr / denom
        f1, f2 = estimate_f1f2_frame(fr.astype(np.float32), sr)
        if np.isnan(f1) or np.isnan(f2) or f1 <= 0 or f2 <= 0:
            f1s.append(np.nan)
            f2s.append(np.nan)
        else:
            f1s.append(float(f1))
            f2s.append(float(f2))
    f1s_arr = moving_average(f1s, SMOOTH_N)
    f2s_arr = moving_average(f2s, SMOOTH_N)
    t = np.arange(len(f1s_arr)) * (HOP_LEN / SR)
    return t, f1s_arr, f2s_arr

def build_templates_from_wavdir(
    wav_dir: str,
    points: int = 60,
    recursive: bool = True,
    sr: int = SR,
) -> Dict[str, np.ndarray]:
    """Scan a directory for .wav files and build per-file F1/F2 templates.

    - File name (without extension) becomes the label, normalized to "/label/".
    - Each wav is converted to a cleaned F1/F2 trajectory and resampled to
      `points` along a normalized time axis.
    - Returns a dictionary suitable for templates.json.
    """
    base_path = Path(wav_dir)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    wav_paths: List[Path] = []
    if recursive:
        wav_paths = list(base_path.rglob("*.wav")) + list(base_path.rglob("*.WAV"))
    else:
        wav_paths = list(base_path.glob("*.wav")) + list(base_path.glob("*.WAV"))

    result: Dict[str, np.ndarray] = {}
    for p in sorted(wav_paths):
        try:
            y, s = librosa.load(str(p), sr=sr, mono=True)
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}", file=sys.stderr)
            continue

        t, f1s, f2s = extract_f1f2_trajectory(y, s)
        if t.size == 0:
            print(f"[warn] insufficient audio for {p}", file=sys.stderr)
            continue
        live = np.stack([f1s, f2s], axis=1)
        mask = np.isfinite(live).all(axis=1)
        live = live[mask]
        if live.shape[0] < 3:
            print(f"[warn] too few valid frames for {p}", file=sys.stderr)
            continue
        traj = _resample_traj(live.astype(float), max(3, int(points)))
        label = _sanitize_vowel_label(p.stem)
        result[label] = traj
        print(f"[info] built template: {label} from {p.name} ({traj.shape[0]} pts)")
    return result

def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    s = np.where(s < 1e-8, 1.0, s)
    return (x - m) / s

def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    try:
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size < 2:
            return 0.0
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return 0.0
        if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0

def _dtw_distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Simple DTW distance for two (N,2) sequences in z-score space.
    Returns average path cost (normalized by path length).
    Complexity: O(N^2) for N=num_points. With N≈50 it's fine real-time.
    """
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    # cost matrix
    diff = a[:, None, :] - b[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    # cumulative matrix
    C = np.full((n + 1, m + 1), np.inf, dtype=float)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = D[i - 1, j - 1]
            C[i, j] = c + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    # approximate path length by n + m (upper bound) -> use average
    path_len = n + m if (n + m) > 0 else 1
    return float(C[n, m] / path_len)

# Tunables for sensitivity
SIMILARITY_RESAMPLED_POINTS = 60
SIMILARITY_ALPHA = 1.6  
SIMILARITY_BETA = 0.6   
WEIGHT_PROCRUSTES = 0.3
WEIGHT_CORRELATION = 0.5
WEIGHT_DTW = 0.2

def compute_shape_similarity_percent(template: np.ndarray, f1s: np.ndarray, f2s: np.ndarray, num_points: int = SIMILARITY_RESAMPLED_POINTS) -> Optional[int]:
    """Compute integer percent similarity of live F1/F2 trajectory shape vs template.

    - Time-normalize both to the same number of points.
    - Prefer Procrustes (translation/scale/rotation-invariant). Fallback to z-score RMSE.
    Returns None if insufficient data.
    """
    if template is None or template.ndim != 2 or template.shape[1] != 2:
        return None
    if f1s.size == 0 or f2s.size == 0:
        return None
    live = np.stack([f1s, f2s], axis=1)
    # Drop rows with NaNs
    mask = np.isfinite(live).all(axis=1)
    live = live[mask]
    if live.shape[0] < 3:
        return None
    # Resample both to the same length
    tpl_rs = _resample_traj(template.astype(float), num_points)
    live_rs = _resample_traj(live.astype(float), num_points)
    if tpl_rs.shape[0] == 0 or live_rs.shape[0] == 0:
        return None

    # Procrustes similarity
    sim_proc = None
    if _HAVE_PROCRUSTES:
        try:
            _, _, disparity = _procrustes(tpl_rs, live_rs)
            sim_proc = float(np.exp(-SIMILARITY_ALPHA * float(disparity)))
        except Exception:
            sim_proc = None

    # Correlation similarity (shape-only per dimension)
    tpl_z = _zscore(tpl_rs)
    live_z = _zscore(live_rs)
    r1 = _pearsonr_safe(tpl_z[:, 0], live_z[:, 0])
    r2 = _pearsonr_safe(tpl_z[:, 1], live_z[:, 1])
    sim_corr = float(np.clip(((r1 + r2) * 0.5 + 1.0) * 0.5, 0.0, 1.0))

    # DTW similarity on z-score space
    dtw_avg_cost = _dtw_distance_2d(tpl_z, live_z)
    if not np.isfinite(dtw_avg_cost):
        sim_dtw = 0.0
    else:
        sim_dtw = float(np.exp(-SIMILARITY_BETA * dtw_avg_cost))

    # Combine
    weights_sum = WEIGHT_CORRELATION + WEIGHT_DTW + (WEIGHT_PROCRUSTES if sim_proc is not None else 0.0)
    if weights_sum <= 0:
        combined = sim_corr  # fallback
    else:
        w_proc = WEIGHT_PROCRUSTES if sim_proc is not None else 0.0
        combined = (
            w_proc * (sim_proc if sim_proc is not None else 0.0)
            + WEIGHT_CORRELATION * sim_corr
            + WEIGHT_DTW * sim_dtw
        ) / weights_sum

    pct = int(np.clip(np.rint(100.0 * combined), 0, 100))
    return pct

# ---------------------- Audio + UI ----------------------
import tkinter as tk
from tkinter import ttk

@dataclass
class State:
    current_vowel: str = "/a/"
    running: bool = False

class VowelLiveApp:
    def __init__(self, master):
        self.master = master
        master.title("母音ライブ比較 — ytj lab")
        # Set white background for Tk root and ttk widgets
        master.configure(bg="white")
        style = ttk.Style(master)
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("TButton", background="white")
        style.configure("TCombobox", fieldbackground="white", background="white")
        self.state = State(current_vowel=VOWEL_LIST[0] if VOWEL_LIST else "/a/")

        # UI Controls
        frm = ttk.Frame(master)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(frm, text="母音:").pack(side=tk.LEFT)
        self.vowel_var = tk.StringVar(value=self.state.current_vowel)
        # Dynamic width based on longest label length (min 6)
        dyn_width = max(6, max((len(v) for v in VOWEL_LIST), default=6))
        self.vowel_menu = ttk.Combobox(frm, textvariable=self.vowel_var, values=VOWEL_LIST, state="readonly", width=dyn_width)
        self.vowel_menu.pack(side=tk.LEFT, padx=6)
        self.vowel_menu.bind("<<ComboboxSelected>>", self.on_select_vowel)

        self.btn_start = ttk.Button(frm, text="開始", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=6)
        self.btn_stop  = ttk.Button(frm, text="停止", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=6)

        self.info_var = tk.StringVar(value=" 準備完了")
        ttk.Label(master, textvariable=self.info_var).pack(side=tk.TOP, anchor="w", padx=8)

        # Matplotlib Figure: split F1 / F2 into separate subplots
        self.fig, (self.ax_f1, self.ax_f2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self.fig.patch.set_facecolor("white")
        self.ax_f1.set_facecolor("white")
        self.ax_f2.set_facecolor("white")
        self.fig.suptitle("フォルマント（F1／F2）")
        self.ax_f1.set_ylabel("口の開き（F1）")
        self.ax_f2.set_ylabel("舌の前後（F2）")
        self.ax_f2.set_xlabel("時間（秒）")
        self.ax_f1.set_ylim(0, MAX_Y_HZ)
        self.ax_f2.set_ylim(0, MAX_Y_HZ)
        # Show semantic scale labels for mouth opening on F1 axis
        self.ax_f1.text(-0.08, 0.0, "小さく", transform=self.ax_f1.transAxes, va="bottom", ha="right")
        self.ax_f1.text(-0.08, 1.0, "大きく", transform=self.ax_f1.transAxes, va="top", ha="right")
        # Show semantic scale labels for tongue position on F2 axis
        self.ax_f2.text(-0.08, 0.0, "「あ」みたいに", transform=self.ax_f2.transAxes, va="bottom", ha="right")
        self.ax_f2.text(-0.08, 1.0, "「い」みたいに", transform=self.ax_f2.transAxes, va="top", ha="right")
        self.canvas = None  # will be embedded after creating tk agg canvas

        # Using matplotlib's interactive window instead of embedding for simplicity:
        plt.ion()
        self.fig.show()

        # Audio buffers
        self.ring = deque(maxlen=ROLL_SAMPLES)
        self.stream = None
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)

        # Key bindings 1..9 for quick vowel switch
        for i in range(1, min(10, len(VOWEL_LIST)+1)):
            master.bind(str(i), self._make_hotkey_handler(VOWEL_LIST[i-1]))

        self.update_loop()  # UI update loop

    def _make_hotkey_handler(self, vowel):
        def handler(event):
            self.vowel_var.set(vowel)
            self.on_select_vowel(None)
        return handler

    def on_select_vowel(self, event):
        self.state.current_vowel = self.vowel_var.get()
        self.info_var.set(f"母音を選択: {self.state.current_vowel}")
        # refresh overlay immediately on next update
        self._redraw(force_overlay_only=True)

    # -------------------------- Audio --------------------------
    def _audio_callback(self, indata, frames, time, status):
        if status:
            # Could print status; avoid spamming
            pass
        # Push a small copy to queue to avoid blocking
        try:
            self.audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def start(self):
        if self.state.running:
            return
        self.state.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.info_var.set(f"実行中… 現在の母音: {self.state.current_vowel}")

        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1, samplerate=SR,
            blocksize=HOP_LEN
        )
        self.stream.start()

    def stop(self):
        if not self.state.running:
            return
        self.state.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.info_var.set("停止しました")
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
        self.ring.clear()

    # ---------------------- Visualization ----------------------
    def _compute_formant_traj(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return t, F1s, F2s for the audio window."""
        if len(audio) < WIN_LEN*2:
            return np.array([]), np.array([]), np.array([])
        frames = librosa.util.frame(audio, frame_length=WIN_LEN, hop_length=HOP_LEN).T
        f1s, f2s = [], []
        for fr in frames:
            fr = fr - fr.mean()
            denom = np.max(np.abs(fr)) + 1e-8
            fr = fr / denom
            f1, f2 = estimate_f1f2_frame(fr.astype(np.float32), SR)
            if np.isnan(f1) or np.isnan(f2) or f1 <= 0 or f2 <= 0:
                f1s.append(np.nan); f2s.append(np.nan)
            else:
                f1s.append(f1); f2s.append(f2)
        f1s = moving_average(f1s, SMOOTH_N)
        f2s = moving_average(f2s, SMOOTH_N)
        t = np.arange(len(f1s)) * (HOP_LEN / SR)
        return t, f1s, f2s

    def _overlay_native_template(self, t_max: float):
        """Draw the native template for the selected vowel stretched to [0, t_max]."""
        v = self.state.current_vowel
        if v not in TEMPLATES or len(TEMPLATES[v]) < 2:
            return
        traj = TEMPLATES[v]  # shape: (K,2) -> F1,F2
        K = len(traj)
        if t_max <= 0:
            t_max = 1.0
        tx = np.linspace(0, t_max, K)
        # Plot dashed lines for native template to respective axes
        self.ax_f1.plot(tx, traj[:, 0], linestyle="--", linewidth=1, label="母語 F1（テンプレ）")
        self.ax_f2.plot(tx, traj[:, 1], linestyle="--", linewidth=1, label="母語 F2（テンプレ）")

    def _redraw(self, force_overlay_only: bool = False):
        # Draw trajectories on separate axes
        self.ax_f1.clear()
        self.ax_f2.clear()
        self.ax_f1.set_facecolor("white")
        self.ax_f2.set_facecolor("white")
        self.fig.suptitle(f"フォルマント（F1／F2）｜母音: {self.state.current_vowel}")
        self.ax_f1.set_ylabel("口の開き（F1）")
        self.ax_f2.set_ylabel("舌の前後（F2）")
        self.ax_f2.set_xlabel("時間（秒）")
        self.ax_f1.set_ylim(0, MAX_Y_HZ)
        self.ax_f2.set_ylim(0, MAX_Y_HZ)
        # Show semantic scale labels for mouth opening on F1 axis
        self.ax_f1.text(-0.08, 0.0, "小さく", transform=self.ax_f1.transAxes, va="bottom", ha="right")
        self.ax_f1.text(-0.08, 1.0, "大きく", transform=self.ax_f1.transAxes, va="top", ha="right")
        # Show semantic scale labels for tongue position on F2 axis
        self.ax_f2.text(-0.08, 0.0, "「あ」みたいに", transform=self.ax_f2.transAxes, va="bottom", ha="right")
        self.ax_f2.text(-0.08, 1.0, "「い」みたいに", transform=self.ax_f2.transAxes, va="top", ha="right")

        audio = np.array(self.ring, dtype=np.float32)
        t, f1s, f2s = np.array([]), np.array([]), np.array([])

        if len(audio) >= WIN_LEN*2 and not force_overlay_only:
            # Compute live F1/F2 trajectories
            t, f1s, f2s = self._compute_formant_traj(audio)

        # Overlay native template
        t_max = float(t[-1]) if t.size > 0 else 1.0
        self._overlay_native_template(t_max)

        # Overlay live trajectories
        if t.size > 0:
            self.ax_f1.plot(t, f1s, linewidth=2, label="話者 F1（ライブ）")
            self.ax_f2.plot(t, f2s, linewidth=2, label="話者 F2（ライブ）")

        # Compute and display shape similarity as integer percent
        similarity_pct: Optional[int] = None
        try:
            v = self.state.current_vowel
            if t.size > 0 and v in TEMPLATES:
                similarity_pct = compute_shape_similarity_percent(
                    TEMPLATES[v], f1s, f2s, num_points=50
                )
        except Exception:
            similarity_pct = None

        if similarity_pct is not None:
            base_msg = (
                f"実行中… 現在の母音: {self.state.current_vowel}"
                if self.state.running else f"母音: {self.state.current_vowel}"
            )
            self.info_var.set(f"{base_msg} ｜ 発音の類似度: {similarity_pct}%")

        # Legends (avoid duplicates) for each axis
        for axis in (self.ax_f1, self.ax_f2):
            handles, labels = axis.get_legend_handles_labels()
            if labels:
                seen = {}
                uniq_handles = []
                uniq_labels = []
                for h, l in zip(handles, labels):
                    if l not in seen:
                        seen[l] = True
                        uniq_handles.append(h); uniq_labels.append(l)
                axis.legend(uniq_handles, uniq_labels, loc="upper right")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_loop(self):
        """Main UI loop to read audio chunks and refresh plot."""
        # Pull audio chunks from queue and append to ring buffer
        chunks = 0
        while not self.audio_q.empty():
            data = self.audio_q.get_nowait()
            self.ring.extend(data.tolist())
            chunks += 1

        # Redraw figure periodically
        if chunks > 0 or not self.state.running:
            self._redraw()

        # Schedule next iteration
        self.master.after(50, self.update_loop)


def main():
    parser = argparse.ArgumentParser(description="母音ライブ比較")
    parser.add_argument(
        "--build-templates",
        dest="build_templates_dir",
        type=str,
        default=None,
        help="templates.json を作る元 WAV が入っているディレクトリ",
    )
    parser.add_argument(
        "--points",
        dest="tpl_points",
        type=int,
        default=60,
        help="WAV からテンプレを作るときの点数（既定: 60）",
    )
    parser.add_argument(
        "--output",
        dest="output_json",
        type=str,
        default=None,
        help="出力 templates.json のパス（既定: スクリプトと同じフォルダ）",
    )
    parser.add_argument(
        "--no-recursive",
        dest="no_recursive",
        action="store_true",
        help="WAV を再帰的に検索しない",
    )

    args = parser.parse_args()

    if args.build_templates_dir:
        try:
            tpl = build_templates_from_wavdir(
                args.build_templates_dir,
                points=args.tpl_points,
                recursive=not args.no_recursive,
                sr=SR,
            )
            if not tpl:
                print("[warn] no templates were built. Nothing to write.", file=sys.stderr)
                return
            out_path = (
                Path(args.output_json)
                if args.output_json
                else Path(__file__).parent / "templates.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({k: v.tolist() for k, v in tpl.items()}, f, ensure_ascii=False, indent=2)
            print(f"[info] wrote templates to {out_path}")
        except Exception as e:
            print(f"[error] failed to build templates: {e}", file=sys.stderr)
        return

    root = tk.Tk()
    app = VowelLiveApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

if __name__ == "__main__":
    main()
