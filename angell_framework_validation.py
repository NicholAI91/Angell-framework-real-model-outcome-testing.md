#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Angell Framework V2: Real-model outcome testing scaffold (single-file)
Author: Nicholas Reid Angell (framework)
Implementation: ChatGPT (single-file scaffold)

Goal
- Fit and test outcomes using the canonical Angell equation as a measurable model.
- Run baselines and ablations to produce scientific traction, not just visuals.
- Generate figure suite + provenance hashes so figures are evidence, not decoration.

Assumptions (default)
- Validation target: Noma geometry-first early detection
- Outcome y is binary: 1 = "positive event" (e.g., early-stage identified, severe outcome, etc.)
- You can swap in any domain by remapping columns and interpretation.

Run
  python angell_v2_validation.py
  python angell_v2_validation.py --csv your_data.csv --outdir outputs

CSV schema (minimal)
  case_id, t, x, y
Optional (recommended)
  rho, delay_onset_to_first_contact, delay_contact_to_referral, delay_referral_to_treatment

Notes
- No fixed colors are set for plots (matplotlib defaults).
- If some libs are missing, the script will degrade gracefully where possible.
"""

import os
import sys
import json
import math
import time
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# Optional dependencies (handled gracefully)
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
except Exception:
    roc_auc_score = None
    average_precision_score = None
    brier_score_loss = None
    GroupKFold = None
    LogisticRegression = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# -----------------------------
# Constants and helpers
# -----------------------------
PHI = (1.0 + math.sqrt(5.0)) / 2.0


def sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-u))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_01(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=float), xmin, xmax
    return (x - xmin) / (xmax - xmin), xmin, xmax


# -----------------------------
# Angell Model components
# -----------------------------
def gate(z: np.ndarray, gate_type: str, k_gate: float, width: float) -> np.ndarray:
    """
    Gate choices:
      - heaviside: 1[z >= 0]
      - sigmoid: sigmoid(k_gate * z)
      - piecewise: 0 if z <= -width, 1 if z >= width, linear in between (optionally smoothed)
    """
    if gate_type == "heaviside":
        return (z >= 0.0).astype(float)

    if gate_type == "sigmoid":
        return sigmoid(k_gate * z)

    if gate_type == "piecewise":
        w = max(float(width), 1e-9)
        g = (z + w) / (2.0 * w)
        g = np.clip(g, 0.0, 1.0)
        return g

    raise ValueError(f"Unknown gate_type: {gate_type}")


def brake(M: np.ndarray, delta: float, r: float) -> np.ndarray:
    """
    Brake term: [1 - delta*M/r]_+
    Domains: delta >= 0, r > 0, M >= 0
    """
    r_safe = max(float(r), 1e-9)
    b = 1.0 - (float(delta) * M) / r_safe
    return np.maximum(b, 0.0)


def phase_term(t: np.ndarray, omega: float, phase0: float) -> np.ndarray:
    """
    Phase term: |sin(theta)|
    We model theta(t) = omega * t + phase0
    """
    return np.abs(np.sin(float(omega) * t + float(phase0)))


def logistic_phi(t: np.ndarray, r: float, t0: float, phi_power: float) -> np.ndarray:
    """
    Logistic growth term with phi scaling:
      phi_power / (1 + exp(-r*(t - t0)))
    """
    return float(phi_power) / (1.0 + np.exp(-float(r) * (t - float(t0))))


def angell_S(
    x: np.ndarray,
    t: np.ndarray,
    M: np.ndarray,
    alpha: float,
    beta: float,
    delta: float,
    r: float,
    t0: float,
    omega: float,
    phase0: float,
    phi_power: float,
    gate_type: str,
    k_gate: float,
    width: float,
    ablate: Optional[Dict[str, bool]] = None,
) -> np.ndarray:
    """
    Canonical structure (matches your unified style):
      S = Gate( alpha*x^2 - beta ) * [1 - delta*M/r]_+ * |sin(theta)| * (phi_power / (1 + exp(-r*(t - t0))))
    """
    ablate = ablate or {}
    z = float(alpha) * (x ** 2) - float(beta)

    G = np.ones_like(x, dtype=float) if ablate.get("no_gate", False) else gate(z, gate_type, k_gate, width)
    B = np.ones_like(x, dtype=float) if ablate.get("no_brake", False) else brake(M, delta, r)
    P = np.ones_like(x, dtype=float) if ablate.get("no_phase", False) else phase_term(t, omega, phase0)
    L = np.ones_like(x, dtype=float) if ablate.get("no_logistic", False) else logistic_phi(t, r, t0, phi_power)

    return G * B * P * L


def outcome_prob(S: np.ndarray, kappa: float, tau: float) -> np.ndarray:
    """
    Outcome wrapper:
      p(y=1 | x,t) = sigmoid( kappa*(S - tau) )
    """
    return sigmoid(float(kappa) * (S - float(tau)))


# -----------------------------
# Synthetic dataset generator
# -----------------------------
def generate_synthetic_noma(
    n: int = 1200,
    seed: int = 7,
    with_groups: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Synthetic ground truth with:
    - x: geometry deviation score (0..1)
    - t: days since onset
    - M: delay accumulator
    - rho: site/group label
    - y: binary outcome driven by Angell p
    """
    rng = np.random.default_rng(seed)

    # Groups
    if with_groups:
        rho = rng.integers(0, 12, size=n)
    else:
        rho = np.zeros(n, dtype=int)

    # Core signals
    t = rng.uniform(0.0, 14.0, size=n)  # days
    x = np.clip(rng.normal(loc=0.35 + 0.03 * rho, scale=0.18, size=n), 0.0, 1.0)

    # Delays: emulate operations friction
    delay1 = np.clip(rng.exponential(scale=1.4, size=n), 0.0, 10.0)
    delay2 = np.clip(rng.exponential(scale=0.9, size=n), 0.0, 8.0)
    delay3 = np.clip(rng.exponential(scale=0.7, size=n), 0.0, 8.0)
    M = delay1 + delay2 + delay3

    # Ground-truth parameters (chosen to create identifiable signatures)
    alpha = 6.0
    beta = 1.0  # threshold on x^2
    delta = 0.25
    r = 0.75
    t0 = 4.0
    omega = 0.85
    phase0 = 0.35
    phi_power = PHI ** 1.0
    gate_type = "sigmoid"
    k_gate = 6.0
    width = 0.15

    S = angell_S(
        x=x, t=t, M=M,
        alpha=alpha, beta=beta, delta=delta,
        r=r, t0=t0, omega=omega, phase0=phase0,
        phi_power=phi_power,
        gate_type=gate_type, k_gate=k_gate, width=width,
        ablate=None
    )

    # Outcome wrapper
    kappa = 9.0
    tau = 0.35
    p = outcome_prob(S, kappa=kappa, tau=tau)

    # Add observation noise to p, then sample y
    p_noisy = np.clip(p + rng.normal(0.0, 0.03, size=n), 0.0, 1.0)
    y = rng.binomial(1, p_noisy, size=n).astype(int)

    return {
        "case_id": np.arange(n).astype(int),
        "rho": rho.astype(int),
        "t": t.astype(float),
        "x": x.astype(float),
        "M": M.astype(float),
        "y": y.astype(int),
        "delays": np.vstack([delay1, delay2, delay3]).T.astype(float),
        "true_params": {
            "alpha": alpha, "beta": beta, "delta": delta, "r": r, "t0": t0,
            "omega": omega, "phase0": phase0, "phi_power": phi_power,
            "gate_type": gate_type, "k_gate": k_gate, "width": width,
            "kappa": kappa, "tau": tau,
        }
    }


# -----------------------------
# Data loading and mapping
# -----------------------------
@dataclass
class DataMap:
    case_id: str = "case_id"
    t: str = "t"
    x: str = "x"
    y: str = "y"
    rho: str = "rho"
    delay_cols: Tuple[str, str, str] = ("delay_onset_to_first_contact", "delay_contact_to_referral", "delay_referral_to_treatment")


def load_dataset(csv_path: Optional[str], datamap: DataMap) -> Dict[str, Any]:
    """
    Loads CSV if provided and readable; otherwise generates synthetic dataset.
    Returns dict with numpy arrays: case_id, x, t, y, rho, M.
    """
    if csv_path and os.path.exists(csv_path):
        if pd is None:
            raise RuntimeError("pandas is required to read CSV. Install pandas or omit --csv to use synthetic data.")
        df = pd.read_csv(csv_path)

        # Required
        for col in [datamap.t, datamap.x, datamap.y]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in CSV.")

        case_id = df[datamap.case_id].values if datamap.case_id in df.columns else np.arange(len(df))
        t = df[datamap.t].astype(float).values
        x = df[datamap.x].astype(float).values
        y = df[datamap.y].astype(int).values

        # rho optional
        if datamap.rho in df.columns:
            rho_raw = df[datamap.rho].values
            # factorize non-numeric rho
            if np.issubdtype(rho_raw.dtype, np.number):
                rho = rho_raw.astype(int)
            else:
                _, rho = np.unique(rho_raw.astype(str), return_inverse=True)
                rho = rho.astype(int)
        else:
            rho = np.zeros(len(df), dtype=int)

        # M(t) from delays or fallback
        d1, d2, d3 = datamap.delay_cols
        if (d1 in df.columns) and (d2 in df.columns) and (d3 in df.columns):
            delays = np.vstack([df[d1].astype(float).values, df[d2].astype(float).values, df[d3].astype(float).values]).T
            M = np.maximum(np.sum(delays, axis=1), 0.0)
        else:
            # Fallback: use t as a rough proxy for delay accumulation
            M = np.maximum(t, 0.0)
            delays = None

        return {
            "source": "csv",
            "csv_path": csv_path,
            "case_id": case_id.astype(int),
            "rho": rho.astype(int),
            "t": t.astype(float),
            "x": x.astype(float),
            "M": M.astype(float),
            "y": y.astype(int),
            "delays": delays,
            "df_columns": list(df.columns),
        }

    # Synthetic fallback
    syn = generate_synthetic_noma()
    return {
        "source": "synthetic",
        "csv_path": None,
        "case_id": syn["case_id"],
        "rho": syn["rho"],
        "t": syn["t"],
        "x": syn["x"],
        "M": syn["M"],
        "y": syn["y"],
        "delays": syn["delays"],
        "true_params": syn["true_params"],
    }


# -----------------------------
# Fitting and evaluation
# -----------------------------
@dataclass
class FitConfig:
    gate_type: str = "sigmoid"  # heaviside | sigmoid | piecewise
    # Training behavior
    normalize_x: bool = True
    normalize_t: bool = False
    # Cross-validation
    n_splits: int = 5
    # Optimization
    maxiter: int = 800
    seed: int = 7
    # Gate tuning
    k_gate_init: float = 5.0
    width_init: float = 0.2


def negative_log_likelihood(params: np.ndarray, x: np.ndarray, t: np.ndarray, M: np.ndarray, y: np.ndarray, cfg: FitConfig) -> float:
    """
    Params order:
      alpha, beta, delta, r, t0, omega, phase0, log_phi_power, k_gate, width, kappa, tau
    We use log_phi_power to ensure positivity via exp.
    """
    alpha, beta, delta, r, t0, omega, phase0, log_phi_power, k_gate, width, kappa, tau = params
    phi_power = float(np.exp(log_phi_power))

    S = angell_S(
        x=x, t=t, M=M,
        alpha=alpha, beta=beta, delta=delta,
        r=r, t0=t0, omega=omega, phase0=phase0,
        phi_power=phi_power,
        gate_type=cfg.gate_type, k_gate=k_gate, width=width,
        ablate=None
    )
    p = outcome_prob(S, kappa=kappa, tau=tau)

    # Clamp for numeric stability
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)

    # Bernoulli NLL
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    if not np.isfinite(nll):
        return 1e18
    return float(nll)


def fit_angell_mle(x: np.ndarray, t: np.ndarray, M: np.ndarray, y: np.ndarray, cfg: FitConfig) -> Dict[str, Any]:
    if minimize is None:
        raise RuntimeError("scipy is required for MLE fitting. Install scipy or run baselines only.")

    # Initial guesses
    alpha0 = 3.0
    beta0 = 0.6
    delta0 = 0.15
    r0 = 0.6
    t00 = np.median(t)
    omega0 = 0.8
    phase00 = 0.2
    log_phi_power0 = math.log(PHI)
    k_gate0 = cfg.k_gate_init
    width0 = cfg.width_init
    kappa0 = 6.0
    tau0 = 0.3

    p0 = np.array([alpha0, beta0, delta0, r0, t00, omega0, phase00, log_phi_power0, k_gate0, width0, kappa0, tau0], dtype=float)

    # Bounds keep parameters in sane scientific domains
    bounds = [
        (1e-6, 50.0),     # alpha
        (0.0, 50.0),      # beta
        (0.0, 10.0),      # delta
        (1e-6, 10.0),     # r
        (float(np.min(t)) - 10.0, float(np.max(t)) + 10.0),  # t0
        (0.0, 10.0),      # omega
        (-math.pi, math.pi),  # phase0
        (math.log(1e-6), math.log(1e6)),  # log_phi_power
        (1e-6, 50.0),     # k_gate
        (1e-6, 10.0),     # width
        (1e-6, 50.0),     # kappa
        (-10.0, 10.0),    # tau
    ]

    res = minimize(
        fun=lambda p: negative_log_likelihood(p, x, t, M, y, cfg),
        x0=p0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(cfg.maxiter)}
    )

    p_hat = res.x
    alpha, beta, delta, r, t0, omega, phase0, log_phi_power, k_gate, width, kappa, tau = p_hat
    phi_power = float(np.exp(log_phi_power))

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "nll": float(res.fun),
        "params": {
            "alpha": float(alpha),
            "beta": float(beta),
            "delta": float(delta),
            "r": float(r),
            "t0": float(t0),
            "omega": float(omega),
            "phase0": float(phase0),
            "phi_power": float(phi_power),
            "gate_type": cfg.gate_type,
            "k_gate": float(k_gate),
            "width": float(width),
            "kappa": float(kappa),
            "tau": float(tau),
        }
    }


def predict_with_params(x: np.ndarray, t: np.ndarray, M: np.ndarray, params: Dict[str, Any], ablate: Optional[Dict[str, bool]] = None) -> np.ndarray:
    S = angell_S(
        x=x, t=t, M=M,
        alpha=params["alpha"], beta=params["beta"], delta=params["delta"],
        r=params["r"], t0=params["t0"], omega=params["omega"], phase0=params["phase0"],
        phi_power=params["phi_power"],
        gate_type=params["gate_type"], k_gate=params["k_gate"], width=params["width"],
        ablate=ablate
    )
    return outcome_prob(S, kappa=params["kappa"], tau=params["tau"])


def compute_metrics(y_true: np.ndarray, p_hat: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    p_hat = np.clip(p_hat, 1e-9, 1.0 - 1e-9)
    if roc_auc_score is not None:
        try:
            out["auroc"] = float(roc_auc_score(y_true, p_hat))
        except Exception:
            out["auroc"] = float("nan")
    else:
        out["auroc"] = float("nan")

    if average_precision_score is not None:
        try:
            out["auprc"] = float(average_precision_score(y_true, p_hat))
        except Exception:
            out["auprc"] = float("nan")
    else:
        out["auprc"] = float("nan")

    if brier_score_loss is not None:
        try:
            out["brier"] = float(brier_score_loss(y_true, p_hat))
        except Exception:
            out["brier"] = float("nan")
    else:
        out["brier"] = float("nan")

    # Simple calibration bins
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(p_hat, bins) - 1
    cal_errs = []
    for b in range(10):
        idx = np.where(bin_ids == b)[0]
        if idx.size < 10:
            continue
        cal_errs.append(abs(np.mean(y_true[idx]) - np.mean(p_hat[idx])))
    out["calibration_mae_10bin"] = float(np.mean(cal_errs)) if len(cal_errs) > 0 else float("nan")
    return out


def baseline_logreg(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    if LogisticRegression is None:
        return None
    X = np.column_stack([x, t])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]


def baseline_threshold(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Choose threshold to maximize F1 on training (simple heuristic)
    thr_grid = np.linspace(float(np.min(x)), float(np.max(x)), 101)
    best_thr = thr_grid[50]
    best_f1 = -1.0
    for thr in thr_grid:
        yhat = (x >= thr).astype(int)
        tp = np.sum((yhat == 1) & (y == 1))
        fp = np.sum((yhat == 1) & (y == 0))
        fn = np.sum((yhat == 0) & (y == 1))
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return (x >= best_thr).astype(float) * 0.9 + 0.05  # probabilistic-ish output


def make_figures(outdir: str, data: Dict[str, Any], p_main: np.ndarray, p_abls: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, str]:
    if plt is None:
        return {}

    safe_mkdir(outdir)
    fig_paths: Dict[str, str] = {}

    x = data["x"]
    t = data["t"]
    y = data["y"]

    # 1) Reliability-like plot (calibration)
    fig = plt.figure()
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(p_main, bins) - 1
    xs, ys = [], []
    for b in range(10):
        idx = np.where(bin_ids == b)[0]
        if idx.size < 20:
            continue
        xs.append(float(np.mean(p_main[idx])))
        ys.append(float(np.mean(y[idx])))
    plt.plot(xs, ys, marker="o", linestyle="-")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicción media (bin)")
    plt.ylabel("Frecuencia observada (bin)")
    plt.title("Calibración (Angell V2)")
    pth = os.path.join(outdir, "fig_calibration.png")
    plt.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig_paths["calibration"] = pth

    # 2) Score distribution by outcome
    fig = plt.figure()
    plt.hist(p_main[y == 0], bins=30, alpha=0.7, label="y=0")
    plt.hist(p_main[y == 1], bins=30, alpha=0.7, label="y=1")
    plt.xlabel("p(y=1)")
    plt.ylabel("Conteo")
    plt.title("Distribución de probabilidad por outcome")
    plt.legend()
    pth = os.path.join(outdir, "fig_prob_dist.png")
    plt.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig_paths["prob_dist"] = pth

    # 3) Surface slice for interpretability (x,t grid)
    fig = plt.figure()
    xg = np.linspace(float(np.min(x)), float(np.max(x)), 120)
    tg = np.linspace(float(np.min(t)), float(np.max(t)), 120)
    Xg, Tg = np.meshgrid(xg, tg)
    Mg = np.full_like(Xg, float(np.median(data["M"])))
    Pg = predict_with_params(Xg.ravel(), Tg.ravel(), Mg.ravel(), params=params, ablate=None).reshape(Xg.shape)
    plt.imshow(Pg, origin="lower", aspect="auto",
               extent=[xg.min(), xg.max(), tg.min(), tg.max()])
    plt.colorbar()
    plt.xlabel("x (score)")
    plt.ylabel("t (tiempo)")
    plt.title("Superficie p(y=1) | M fijo (mediana)")
    pth = os.path.join(outdir, "fig_surface_xt.png")
    plt.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig_paths["surface_xt"] = pth

    # 4) Ablation comparison (mean p)
    fig = plt.figure()
    keys = ["main"] + list(p_abls.keys())
    vals = [float(np.mean(p_main))] + [float(np.mean(p_abls[k])) for k in p_abls.keys()]
    plt.bar(np.arange(len(keys)), vals)
    plt.xticks(np.arange(len(keys)), keys, rotation=25, ha="right")
    plt.ylabel("Media p(y=1)")
    plt.title("Comparación de ablations (media de probabilidad)")
    pth = os.path.join(outdir, "fig_ablation_means.png")
    plt.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig_paths["ablation_means"] = pth

    return fig_paths


def run_pipeline(args: argparse.Namespace) -> int:
    safe_mkdir(args.outdir)

    datamap = DataMap()
    cfg = FitConfig(
        gate_type=args.gate_type,
        normalize_x=not args.no_normalize_x,
        normalize_t=args.normalize_t,
        n_splits=args.n_splits,
        maxiter=args.maxiter,
        seed=args.seed,
        k_gate_init=args.k_gate_init,
        width_init=args.width_init
    )

    data = load_dataset(args.csv, datamap)

    # Prepare arrays
    x_raw = data["x"].astype(float)
    t_raw = data["t"].astype(float)
    M_raw = data["M"].astype(float)
    y = data["y"].astype(int)
    rho = data["rho"].astype(int)

    # Optional normalization (store for provenance)
    norm_info: Dict[str, Any] = {}
    if cfg.normalize_x:
        x, x_min, x_max = normalize_01(x_raw)
        norm_info["x_norm"] = {"min": x_min, "max": x_max, "method": "minmax_01"}
    else:
        x = x_raw.copy()
        norm_info["x_norm"] = {"method": "none"}

    if cfg.normalize_t:
        t, t_min, t_max = normalize_01(t_raw)
        norm_info["t_norm"] = {"min": t_min, "max": t_max, "method": "minmax_01"}
    else:
        t = t_raw.copy()
        norm_info["t_norm"] = {"method": "none"}

    M = np.maximum(M_raw, 0.0)

    # Fit Angell parameters (MLE)
    if args.baselines_only:
        fit = {"success": False, "message": "baselines_only=True", "params": None}
        params = None
        p_main = None
    else:
        fit = fit_angell_mle(x=x, t=t, M=M, y=y, cfg=cfg)
        params = fit["params"]
        p_main = predict_with_params(x, t, M, params=params, ablate=None)

    # Baselines
    p_lr = baseline_logreg(x, t, y)
    p_thr = baseline_threshold(x, y)

    # Ablations
    p_abls: Dict[str, np.ndarray] = {}
    if params is not None:
        ablation_defs = {
            "no_gate": {"no_gate": True},
            "no_brake": {"no_brake": True},
            "no_phase": {"no_phase": True},
            "no_logistic": {"no_logistic": True},
        }
        for k, ab in ablation_defs.items():
            p_abls[k] = predict_with_params(x, t, M, params=params, ablate=ab)

    # Metrics
    metrics: Dict[str, Any] = {"source": data["source"], "n": int(len(y))}
    if p_main is not None:
        metrics["angell_main"] = compute_metrics(y, p_main)
        metrics["angell_ablations"] = {k: compute_metrics(y, pv) for k, pv in p_abls.items()}
    else:
        metrics["angell_main"] = None
        metrics["angell_ablations"] = None

    metrics["baseline_logreg"] = compute_metrics(y, p_lr) if p_lr is not None else None
    metrics["baseline_threshold"] = compute_metrics(y, p_thr)

    # Figures
    fig_paths: Dict[str, str] = {}
    if (plt is not None) and (p_main is not None):
        fig_paths = make_figures(args.outdir, {"x": x, "t": t, "M": M, "y": y}, p_main, p_abls, params)

    # Provenance block
    provenance: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": {
            "path": os.path.abspath(sys.argv[0]),
            "sha256": sha256_file(sys.argv[0]) if os.path.exists(sys.argv[0]) else None,
        },
        "dataset": {
            "source": data["source"],
            "csv_path": os.path.abspath(args.csv) if args.csv else None,
            "csv_sha256": sha256_file(args.csv) if (args.csv and os.path.exists(args.csv)) else None,
            "columns": data.get("df_columns", None),
        },
        "normalization": norm_info,
        "config": cfg.__dict__,
        "fit": fit,
        "metrics": metrics,
        "figures": fig_paths,
    }

    # Save artifacts
    out_json = os.path.join(args.outdir, "angell_v2_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    # Console summary
    print("\n=== Angell V2 Outcome Testing ===")
    print(f"Data source: {data['source']}")
    print(f"Rows: {len(y)}")
    print(f"Outdir: {args.outdir}")
    print(f"Results JSON: {out_json}")
    if p_main is not None:
        print("\nAngell main metrics:")
        for k, v in metrics["angell_main"].items():
            print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: NaN")
        print("\nAblations:")
        for abk, abm in metrics["angell_ablations"].items():
            auroc = abm.get("auroc", float("nan"))
            auprc = abm.get("auprc", float("nan"))
            brier = abm.get("brier", float("nan"))
            print(f"  {abk}: AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f}")
    else:
        print("\nAngell model not fit (baselines_only=True).")

    print("\nBaselines:")
    if metrics["baseline_logreg"] is not None:
        bl = metrics["baseline_logreg"]
        print(f"  logreg: AUROC={bl['auroc']:.4f} AUPRC={bl['auprc']:.4f} Brier={bl['brier']:.4f}")
    bt = metrics["baseline_threshold"]
    print(f"  threshold: AUROC={bt['auroc']:.4f} AUPRC={bt['auprc']:.4f} Brier={bt['brier']:.4f}")

    if fig_paths:
        print("\nFigures:")
        for k, p in fig_paths.items():
            print(f"  {k}: {p}")

    # If synthetic, show true vs fit (quick check)
    if data["source"] == "synthetic" and ("true_params" in data) and (params is not None):
        print("\nSynthetic true_params vs fit (quick consistency check):")
        tp = data["true_params"]
        keys = ["alpha", "beta", "delta", "r", "t0", "omega", "phase0", "phi_power", "k_gate", "width", "kappa", "tau"]
        for k in keys:
            tv = tp.get(k, None)
            fv = params.get(k, None)
            if (tv is not None) and (fv is not None):
                print(f"  {k}: true={tv:.44f} fit={fv:.4f}")

    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV dataset. If omitted, synthetic data is generated.")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory for figures and JSON.")
    ap.add_argument("--gate_type", type=str, default="sigmoid", choices=["heaviside", "sigmoid", "piecewise"], help="Gate form.")
    ap.add_argument("--n_splits", type=int, default=5, help="GroupKFold splits (if used later).")
    ap.add_argument("--maxiter", type=int, default=800, help="Max iterations for optimizer.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--k_gate_init", type=float, default=5.0, help="Initial k for sigmoid gate.")
    ap.add_argument("--width_init", type=float, default=0.2, help="Initial width for piecewise gate.")
    ap.add_argument("--no_normalize_x", action="store_true", help="Disable x normalization.")
    ap.add_argument("--normalize_t", action="store_true", help="Enable t normalization.")
    ap.add_argument("--baselines_only", action="store_true", help="Skip Angell fit, run only baselines.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_pipeline(args))
