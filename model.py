from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class MethodConfig:
    """
    User-edited method + design configuration for the 2x2 scouting design.
    Units must be respected consistently.
    """

    # Design levels (for now: exactly 2x2)
    T1: float                 # [K]
    T2: float                 # [K]
    tG1: float                # [min]
    tG2: float                # [min]

    # System timing / flow
    flow_mL_min: float        # [mL/min]
    delay_volume_mL: float    # [mL] (instrument dwell/delay volume)
    t0_col_min: float         # [min] (column dead time)

    # Prediction grid resolution
    n_T: int = 100
    n_tG: int = 100

    @property
    def t_dwell_min(self) -> float:
        """Dwell time from delay volume and flow."""
        return self.delay_volume_mL / self.flow_mL_min

    @property
    def t0_total_min(self) -> float:
        """Total dead time used in the retention model (column + dwell)."""
        return self.t0_col_min + self.t_dwell_min

    @property
    def T_levels(self) -> Tuple[float, float]:
        return (self.T1, self.T2)

    @property
    def tG_levels(self) -> Tuple[float, float]:
        return (self.tG1, self.tG2)


def fit_models(config: MethodConfig, measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the retention-time, peak-width, and asymmetry models for each analyte
    from the 2x2 scouting measurements.

    Parameters
    ----------
    config : MethodConfig
        Method/design settings (T1, T2, tG1, tG2, dead times, etc.).
    measurements : pd.DataFrame
        DataFrame indexed by analyte, with at least these columns:
        - tR_T1_G1, tR_T1_G2, tR_T2_G1, tR_T2_G2
        - w_T1_G1,  w_T1_G2,  w_T2_G1,  w_T2_G2
        - A_T1_G1,  A_T1_G2,  A_T2_G1,  A_T2_G2

    Returns
    -------
    pd.DataFrame
        One row per analyte, with columns:
        a1, a2, a0, b1, b2, b0, c1, c2, c0
    """

    required_cols = [
        "tR_T1_G1", "tR_T1_G2", "tR_T2_G1", "tR_T2_G2",
        "w_T1_G1",  "w_T1_G2",  "w_T2_G1",  "w_T2_G2",
        "A_T1_G1",  "A_T1_G2",  "A_T2_G1",  "A_T2_G2",
    ]

    missing = [col for col in required_cols if col not in measurements.columns]
    if missing:
        raise ValueError(f"measurements is missing required columns: {missing}")

    if measurements.index.has_duplicates:
        raise ValueError("measurements index must contain unique analyte names.")

    T1, T2 = config.T1, config.T2
    tG1, tG2 = config.tG1, config.tG2
    t0_total = config.t0_total_min

    if T1 <= 0 or T2 <= 0 or tG1 <= 0 or tG2 <= 0:
        raise ValueError("Temperatures and gradient times must be strictly positive.")

    d_inv_T = (1.0 / T2) - (1.0 / T1)
    d_inv_tG = (1.0 / tG2) - (1.0 / tG1)

    rows = []

    for analyte, row in measurements.iterrows():
        # Retention times
        tr11, tr12, tr21, tr22 = row[
            ["tR_T1_G1", "tR_T1_G2", "tR_T2_G1", "tR_T2_G2"]
        ].to_numpy(dtype=float)

        # Peak widths
        w11, w12, w21, w22 = row[
            ["w_T1_G1", "w_T1_G2", "w_T2_G1", "w_T2_G2"]
        ].to_numpy(dtype=float)

        # Peak asymmetry / tailing
        A11, A12, A21, A22 = row[
            ["A_T1_G1", "A_T1_G2", "A_T2_G1", "A_T2_G2"]
        ].to_numpy(dtype=float)

        # Sanity checks
        tr_adj = np.array([tr11, tr12, tr21, tr22]) - t0_total
        if np.any(tr_adj <= 0):
            raise ValueError(
                f"{analyte}: all retention times must be > total dead time "
                f"({t0_total:.4f} min)."
            )

        if np.any(np.array([w11, w12, w21, w22]) <= 0):
            raise ValueError(f"{analyte}: all peak widths must be strictly positive.")

        if np.any(np.array([A11, A12, A21, A22]) <= 0):
            raise ValueError(f"{analyte}: all asymmetry values must be strictly positive.")

        # Retention-time model
        a1 = (np.log(tr22 - t0_total) - np.log(tr12 - t0_total)) / d_inv_T
        a2 = (np.log(tr12 - t0_total) - np.log(tr11 - t0_total)) / d_inv_tG
        a0 = np.log(tr11 - t0_total) - (a1 / T1) - (a2 / tG1)

        # Peak-width model
        b1 = np.log(w22 / w12) / d_inv_T
        b2 = np.log(w12 / w11) / d_inv_tG
        b0 = np.log(w11) - (b1 / T1) - (b2 / tG1)

        # Peak-asymmetry model
        c1 = np.log(A22 / A12) / d_inv_T
        c2 = np.log(A12 / A11) / d_inv_tG
        c0 = np.log(A11) - (c1 / T1) - (c2 / tG1)

        rows.append(
            {
                "analyte": analyte,
                "a1": a1, "a2": a2, "a0": a0,
                "b1": b1, "b2": b2, "b0": b0,
                "c1": c1, "c2": c2, "c0": c0,
            }
        )

    return pd.DataFrame(rows).set_index("analyte")

# --- Score refactor scaffold (xarray-based) ---
# Goal: one implementation per score that works for both:
#   - grid predictions (N, M, L)  => dims ("analyte","T","tG")
#   - single-point predictions    => dims ("analyte","T","tG") with len(T)=len(tG)=1


# ---------- helpers ----------
def predictions_to_xr(
    retention: dict,
    width: dict,
    asymmetry: dict,
    *,
    T_vals: np.ndarray,
    tG_vals: np.ndarray,
    analytes: list | None = None,
    tR_name: str = "tR",
    w_name: str = "w",
    A_name: str = "A",
) -> xr.Dataset:
    """
    Convert dict-of-(M,L) predictions into an xarray Dataset with dims (analyte, T, tG).
    """
    if analytes is None:
        analytes = list(retention.keys())

    # stack into (N, M, L)
    tR = np.stack([retention[a] for a in analytes], axis=0)
    w  = np.stack([width[a]     for a in analytes], axis=0)
    A  = np.stack([asymmetry[a]   for a in analytes], axis=0)

    if tR.shape[1:] != (len(T_vals), len(tG_vals)):
        raise ValueError(
            f"Prediction grid shape mismatch: got (M,L)={tR.shape[1:]}, "
            f"expected ({len(T_vals)},{len(tG_vals)})."
        )

    return xr.Dataset(
        data_vars={
            tR_name: (("analyte", "T", "tG"), tR),
            w_name:  (("analyte", "T", "tG"), w),
            A_name:  (("analyte", "T", "tG"), A),
        },
        coords={
            "analyte": analytes,
            "T": np.asarray(T_vals),
            "tG": np.asarray(tG_vals),
        },
    )


def _stack_points(da: xr.DataArray, analyte_dim: str = "analyte"):
    """Stack all non-analyte dims into a single 'point' dim, for vectorized per-point sorting."""
    method_dims = [d for d in da.dims if d != analyte_dim]
    if not method_dims:
        # degenerate: only analyte dim exists
        return da.expand_dims(point=[0]), ["point"]
    return da.stack(point=method_dims), method_dims


# ---------- per-score functions (vectorized over any number of method dims) ----------
def score_eta(
    pred: xr.Dataset,
    *,
    column_dead_time: float,
    width_penalty_coeff: float = 10.0,
    analyte_dim: str = "analyte",
    tG_dim: str = "tG",
    w_name: str = "w",
) -> xr.DataArray:
    """
    η = ((tG - t0 - w̄)/(tG - t0))^k
    Returns dims: all method dims (e.g. T, tG)
    """
    w_bar = pred[w_name].mean(analyte_dim)
    denom = (pred.coords[tG_dim] - column_dead_time)
    # broadcast denom to method dims automatically
    return ((denom - w_bar) / denom) ** width_penalty_coeff


def score_Sbar(
    pred: xr.Dataset,
    *,
    analyte_dim: str = "analyte",
    A_name: str = "A",
) -> xr.DataArray:
    """
    S_i = {A_i if A_i <= 1 else 1/A_i}
    S̄ = mean(S_i) over analytes.
    """
    S = xr.where(pred[A_name] <= 1, pred[A_name], 1 / pred[A_name])
    return S.mean(analyte_dim)


def score_We(
    pred: xr.Dataset,
    *,
    column_dead_time: float,
    analyte_dim: str = "analyte",
    tG_dim: str = "tG",
    tR_name: str = "tR",
) -> xr.DataArray:
    """
    W_e = (max(tR) - min(tR)) / (tG - t0)
    """
    tR_max = pred[tR_name].max(analyte_dim)
    tR_min = pred[tR_name].min(analyte_dim)
    denom = (pred.coords[tG_dim] - column_dead_time)
    return (tR_max - tR_min) / denom


def score_critical_resolution(
    pred: xr.Dataset,
    *,
    analyte_dim: str = "analyte",
    tR_name: str = "tR",
    w_name: str = "w",
) -> xr.DataArray:
    """
    Critical resolution: min over adjacent pairs after sorting by tR.
    Notebook formula:
        Rs_i = (tR_{i+1}-tR_i) / (w_{i+1}+w_i)
    """
    tR = pred[tR_name]
    w  = pred[w_name]

    # stack method dims => (analyte, point)
    tR_s, method_dims = _stack_points(tR, analyte_dim=analyte_dim)
    w_s,  _           = _stack_points(w,  analyte_dim=analyte_dim)

    # NumPy sort per point along analyte axis (axis=0)
    tR_vals = np.asarray(tR_s.data)  # (N, P)
    w_vals  = np.asarray(w_s.data)   # (N, P)

    order = np.argsort(tR_vals, axis=0)  # (N, P) integer indices
    tR_sorted = np.take_along_axis(tR_vals, order, axis=0)
    w_sorted  = np.take_along_axis(w_vals,  order, axis=0)

    # adjacent resolution (N-1, P)
    dt = tR_sorted[1:, :] - tR_sorted[:-1, :]
    ws = w_sorted[1:, :]  + w_sorted[:-1, :]
    Rs = dt / ws

    crit_vals = np.min(Rs, axis=0)  # (P,)

    crit = xr.DataArray(
        crit_vals,
        dims=("point",),
        coords={"point": tR_s.coords["point"]},
        name="Rs_crit",
    )

    # unstack back to original method dims
    crit = crit.unstack("point") if method_dims != ["point"] else crit.squeeze("point")
    return crit


def score_DU(
    pred: xr.Dataset,
    *,
    analyte_dim: str = "analyte",
    tR_name: str = "tR",
) -> xr.DataArray:
    """
    Gap-uniformity DU score using the "Alternate proposal" definition:

      Δj  = tR_{j+1} - tR_j  (after sorting by tR)
      W   = sum_j Δj = tR_last - tR_first
      Δ   = W / (n-1)
      σΔ  = sqrt( (1/(n-2)) * sum_j (Δj - Δ)^2 )   (corrected sample std over gaps)
      CVΔ = σΔ / Δ

    Alternate proposal:
      DU = 1 - (CVΔ / CVΔ_max)

    with CVΔ_max given in the document (it simplifies to sqrt(n-1) for the corrected-sample σΔ). :contentReference[oaicite:0]{index=0}

    Returns
    -------
    xr.DataArray
        DU over all method dims (e.g. (T, tG)); compatible with single-point (len dims = 1).
    """
    tR = pred[tR_name]

    # Stack all method dims => (analyte, point)
    tR_s, method_dims = _stack_points(tR, analyte_dim=analyte_dim)

    tR_vals = np.asarray(tR_s.data)  # (N, P)
    if np.any(~np.isfinite(tR_vals)):
        raise ValueError("Non-finite values found in tR.")

    N, P = tR_vals.shape

    # Edge cases:
    # - N <= 1: not meaningful, return 1
    # - N == 2: only one gap => perfectly uniform by construction, return 1
    if N <= 2:
        du_vals = np.ones((P,), dtype=float)
    else:
        # Sort by elution order per point
        order = np.argsort(tR_vals, axis=0)                 # (N, P)
        tR_sorted = np.take_along_axis(tR_vals, order, 0)   # (N, P)

        gaps = np.diff(tR_sorted, axis=0)  # (N-1, P)

        # Total window W = tR_last - tR_first (== sum gaps)
        W = tR_sorted[-1, :] - tR_sorted[0, :]  # (P,)

        # Ideal gap Δ = W/(N-1)
        Delta = W / (N - 1)

        # Corrected sample std over gaps (denominator N-2)
        # σΔ^2 = (1/(N-2)) * sum (gap - Delta)^2
        # Broadcasting Delta over the (N-1) gap axis
        centered = gaps - Delta[None, :]
        sigma = np.sqrt(np.sum(centered * centered, axis=0) / (N - 2))  # (P,)

        # CVΔ = σΔ / Δ ; handle degeneracy W=0 => Δ=0 (all coelute) => DU=0
        CV = np.full((P,), np.nan, dtype=float)
        ok = Delta > 0
        CV[ok] = sigma[ok] / Delta[ok]

        # CVΔ_max from the document; simplifies to sqrt(N-1) with corrected σΔ. :contentReference[oaicite:1]{index=1}
        CV_max = ((N - 1) / np.sqrt(N - 2)) * np.sqrt(
            (1 - 1 / (N - 1)) ** 2 + (N - 2) * (1 / (N - 1)) ** 2
        )

        du_vals = np.zeros((P,), dtype=float)
        # For non-degenerate windows: DU = 1 - CV/CVmax, clipped to [0,1]
        du_vals[ok] = 1.0 - (CV[ok] / CV_max)
        du_vals = np.clip(du_vals, 0.0, 1.0)

        # If Delta==0 (all coelute), du_vals stays 0 by design.

    du = xr.DataArray(
        du_vals,
        dims=("point",),
        coords={"point": tR_s.coords["point"]},
        name="DU",
    )

    # Unstack back to original method dims
    du = du.unstack("point") if method_dims != ["point"] else du.squeeze("point")
    return du


def score_CPO(
    pred: xr.Dataset,
    *,
    analyte_dim: str = "analyte",
    tR_name: str = "tR",
    w_name: str = "w",
    A_name: str = "A",
    Rs_clip_divisor: float = 2.0,
) -> xr.DataArray:
    """
    Critical Pair Order (CPO), vectorized over any method grid.

    Matches the notebook logic in cell 17, except for the final aggregation:
      - resolution for adjacent pairs (after sorting by tR):
            Rs_i = (tR_{i+1} - tR_i) / (w_{i+1} + w_i)
      - tailing ranking:
            r_i = clip(S_{i+1} / S_i, 0, 1)
      - critical pair order per adjacent pair:
            cpo_i = clip(Rs_i / Rs_clip_divisor, 0, 1) * r_i
      - final CPO score:
            CPO = min_i cpo_i     (user-requested change; was mean in notebook)

    Parameters
    ----------
    pred : xr.Dataset
        Must contain tR, w, S with dims (analyte, ...method dims...)
        e.g. (analyte, T, tG).
    Rs_clip_divisor : float
        The "divide by 2" factor from your current notebook (cell 17). Keep at 2.0 unless changed.

    Returns
    -------
    xr.DataArray
        CPO over method dims (e.g. (T, tG)); compatible with single-point (len dims = 1).
    """
    tR = pred[tR_name]
    w  = pred[w_name]
    A  = pred[A_name]

    # stack method dims => (analyte, point)
    tR_s, method_dims = _stack_points(tR, analyte_dim=analyte_dim)
    w_s,  _           = _stack_points(w,  analyte_dim=analyte_dim)
    A_s,  _           = _stack_points(A,  analyte_dim=analyte_dim)

    tR_vals = np.asarray(tR_s.data)  # (N, P)
    w_vals  = np.asarray(w_s.data)   # (N, P)
    A_vals  = np.asarray(A_s.data)   # (N, P)

    if np.any(~np.isfinite(tR_vals)):
        raise ValueError("Non-finite values found in tR.")
    if np.any(~np.isfinite(w_vals)) or np.any(w_vals <= 0):
        raise ValueError("Peak widths must be finite and strictly positive.")
    if np.any(~np.isfinite(A_vals)) or np.any(A_vals <= 0):
        raise ValueError("Tailing/asymmetry values must be finite and strictly positive.")

    N, P = tR_vals.shape

    # Edge case: if <2 analytes, no adjacent pair exists -> define CPO as NaN
    if N < 2:
        cpo_vals = np.full((P,), np.nan, dtype=float)
    else:
        # Sort by elution order per point
        order = np.argsort(tR_vals, axis=0)  # (N, P)
        tR_sorted = np.take_along_axis(tR_vals, order, axis=0)
        w_sorted  = np.take_along_axis(w_vals,  order, axis=0)
        A_sorted  = np.take_along_axis(A_vals,  order, axis=0)

        # Adjacent resolution (N-1, P)
        dt = tR_sorted[1:, :] - tR_sorted[:-1, :]
        ws = w_sorted[1:, :]  + w_sorted[:-1, :]
        Rs = dt / ws

        # Tailing ranking (N-1, P)
        tail_ratio = A_sorted[1:, :] / A_sorted[:-1, :]
        tail_rank = np.clip(tail_ratio, 0.0, 1.0)

        # Pair-wise CPO (N-1, P)
        pair_cpo = np.clip(Rs / float(Rs_clip_divisor), 0.0, 1.0) * tail_rank

        # Requested aggregation: minimum over adjacent pairs
        cpo_vals = np.min(pair_cpo, axis=0)

    cpo = xr.DataArray(
        cpo_vals,
        dims=("point",),
        coords={"point": tR_s.coords["point"]},
        name="CPO",
    )

    # Unstack back to original method dims
    cpo = cpo.unstack("point") if method_dims != ["point"] else cpo.squeeze("point")
    return cpo


# ---------- convenience aggregator ----------
def compute_scores(
    pred: xr.Dataset,
    *,
    column_dead_time: float,
    width_penalty_coeff: float = 10.0,
) -> xr.Dataset:
    """
    Bundle the scores you currently use into a single xarray Dataset.
    DU and CPO are left uncomputed on purpose (definition pending).
    """
    return xr.Dataset(
        data_vars=dict(
            eta=score_eta(pred, column_dead_time=column_dead_time, width_penalty_coeff=width_penalty_coeff),
            Sbar=score_Sbar(pred),
            We=score_We(pred, column_dead_time=column_dead_time),
            Rs_crit=score_critical_resolution(pred),
            DUr=score_DU(pred),
            CPO=score_CPO(pred),
        ),
        coords=pred.coords,
    )


# ---------- usage (grid) ----------
# You must provide 1D coordinate vectors, not meshgrids:
#   T_vals = temperature_range
#   tG_vals = gradient_time_range
#
# pred_xr = predictions_to_xr(
#     predicted_retention_time,
#     predicted_peak_width,
#     predicted_peak_tailing,
#     T_vals=temperature_range,
#     tG_vals=gradient_time_range,
#     analytes=list(predicted_retention_time.keys()),
# )
# scores = compute_scores(pred_xr, column_dead_time=column_dead_time, width_penalty_coeff=10)
# scores

# ---------- usage (single point) ----------
# Build a degenerate grid by passing length-1 coordinate arrays and (N,1,1) values.
# If your single-point predictions are scalars per analyte (dict -> float), wrap them as [[value]]:
#
# sp_retention_time_111 = {k: np.array([[v]]) for k, v in sp_retention_time.items()}
# sp_peak_width_111     = {k: np.array([[v]]) for k, v in sp_peak_width.items()}
# sp_peak_tailing_111   = {k: np.array([[v]]) for k, v in sp_peak_tailing.items()}
#
# pred_sp = predictions_to_xr(
#     sp_retention_time_111,
#     sp_peak_width_111,
#     sp_peak_tailing_111,
#     T_vals=np.array([temperature_selected]),
#     tG_vals=np.array([gradient_time_selected]),
#     analytes=list(sp_retention_time.keys()),
# )
# scores_sp = compute_scores(pred_sp, column_dead_time=column_dead_time, width_penalty_coeff=10)
# scores_sp

def predict_grid_from_params(
    model_params,
    *,
    T_vals: np.ndarray,
    tG_vals: np.ndarray,
    t0_total: float,
):
    """
    Predict tR, w, S on a (T, tG) grid from fitted model parameters.

    Model forms:
      ln(tR - t0_total) = a0 + a1/T + a2/tG
      ln(w)             = b0 + b1/T + b2/tG
      ln(S)             = c0 + c1/T + c2/tG

    Returns dicts {analyte: ndarray(M,L)} for tR, w, S.
    """
    analytes = list(model_params.index)
    T_vals = np.asarray(T_vals, dtype=float)
    tG_vals = np.asarray(tG_vals, dtype=float)

    if T_vals.ndim != 1 or tG_vals.ndim != 1:
        raise ValueError("T_vals and tG_vals must be 1D arrays (not meshgrids).")

    invT = (1.0 / T_vals)[:, None]     # (M,1)
    invtG = (1.0 / tG_vals)[None, :]   # (1,L)

    a0 = model_params["a0"].to_numpy(float)[:, None, None]
    a1 = model_params["a1"].to_numpy(float)[:, None, None]
    a2 = model_params["a2"].to_numpy(float)[:, None, None]

    b0 = model_params["b0"].to_numpy(float)[:, None, None]
    b1 = model_params["b1"].to_numpy(float)[:, None, None]
    b2 = model_params["b2"].to_numpy(float)[:, None, None]

    c0 = model_params["c0"].to_numpy(float)[:, None, None]
    c1 = model_params["c1"].to_numpy(float)[:, None, None]
    c2 = model_params["c2"].to_numpy(float)[:, None, None]

    # broadcast (N, M, L)
    tR_adj = np.exp(a0 + a1 * invT[None, :, :] + a2 * invtG[None, :, :])
    tR = float(t0_total) + tR_adj
    w  = np.exp(b0 + b1 * invT[None, :, :] + b2 * invtG[None, :, :])
    S  = np.exp(c0 + c1 * invT[None, :, :] + c2 * invtG[None, :, :])

    pred_tR = {analytes[i]: tR[i, :, :] for i in range(len(analytes))}
    pred_w  = {analytes[i]: w[i, :, :]  for i in range(len(analytes))}
    pred_S  = {analytes[i]: S[i, :, :]  for i in range(len(analytes))}
    return pred_tR, pred_w, pred_S

def predict_single_point_from_params(
    model_params,
    *,
    T: float,
    tG: float,
    t0_total: float,
):
    """
    Return single-point predictions as dicts {analyte: float} for:
      - retention time tR [min]
      - peak width w [min]
      - tailing/asymmetry A [-]
    Model forms assumed consistent with your fit_models() cell:
      ln(tR - t0_total) = a0 + a1/T + a2/tG
      ln(w)             = b0 + b1/T + b2/tG
      ln(A)             = c0 + c1/T + c2/tG
    """
    analytes = list(model_params.index)

    a0 = model_params["a0"].to_numpy(float)
    a1 = model_params["a1"].to_numpy(float)
    a2 = model_params["a2"].to_numpy(float)

    b0 = model_params["b0"].to_numpy(float)
    b1 = model_params["b1"].to_numpy(float)
    b2 = model_params["b2"].to_numpy(float)

    c0 = model_params["c0"].to_numpy(float)
    c1 = model_params["c1"].to_numpy(float)
    c2 = model_params["c2"].to_numpy(float)

    invT = 1.0 / float(T)
    invtG = 1.0 / float(tG)

    tR_adj = np.exp(a0 + a1 * invT + a2 * invtG)
    tR = t0_total + tR_adj
    w = np.exp(b0 + b1 * invT + b2 * invtG)
    S = np.exp(c0 + c1 * invT + c2 * invtG)

    pred_tR = {analytes[i]: float(tR[i]) for i in range(len(analytes))}
    pred_w  = {analytes[i]: float(w[i])  for i in range(len(analytes))}
    pred_S  = {analytes[i]: float(S[i])  for i in range(len(analytes))}

    return pred_tR, pred_w, pred_S

# --- VISUALISE: score surfaces from scores_grid (robust to dim order) ---

def plot_score_surface(
    scores,
    name: str,
    *,
    T_dim: str = "T",
    tG_dim: str = "tG",
    use_pcolormesh: bool = True,
):
    if name not in scores:
        raise KeyError(f"{name!r} not in scores dataset. Available: {list(scores.data_vars)}")

    da = scores[name]

    # Allow either (T, tG) or (tG, T); transpose if needed.
    if set(da.dims) != {T_dim, tG_dim}:
        raise ValueError(f"{name} must have dims containing {{{T_dim},{tG_dim}}}, got {da.dims}")

    da = da.transpose(T_dim, tG_dim)

    T = scores.coords[T_dim].values
    tG = scores.coords[tG_dim].values
    Z = da.values  # (len(T), len(tG))

    plt.figure(figsize=(3.25, 2.5))
    if use_pcolormesh:
        plt.pcolormesh(tG, T, Z, shading="nearest")
    else:
        plt.imshow(
            Z,
            aspect="equal",
            origin="lower",
            extent=(float(tG.min()), float(tG.max()), float(T.min()), float(T.max())),
        )

    plt.xlabel(tG_dim)
    plt.ylabel(T_dim)
    plt.title(name)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
