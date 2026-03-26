import numpy as np
import xarray as xr



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

    method_dims = [d for d in pred[w_name].dims if d != analyte_dim]
    denom = pred.coords[tG_dim] - column_dead_time

    eta = ((denom - w_bar) / denom) ** width_penalty_coeff
    return eta.transpose(*method_dims)


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
        Must contain tR, w, A with dims (analyte, ...method dims...)
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


def score_SQF(
    pred: xr.Dataset,
    *,
    column_dead_time: float,
    width_penalty_coeff: float = 10.0,
) -> xr.DataArray:
    """
    Separation Quality Factor (SQF), defined as the geometric mean of:
        {eta, Sbar, We, DUr, CPO}
    """
    eta  = score_eta(
        pred,
        column_dead_time=column_dead_time,
        width_penalty_coeff=width_penalty_coeff,
    )
    Sbar = score_Sbar(pred)
    We   = score_We(pred, column_dead_time=column_dead_time)
    DUr  = score_DU(pred)
    CPO  = score_CPO(pred)

    factors = xr.concat([eta, Sbar, We, DUr, CPO], dim="metric")

    if bool((factors < 0).any()):
        raise ValueError("SQF is undefined for negative score components.")

    # geometric mean; zeros are allowed and give SQF = 0
    sqf = np.exp(np.log(factors).mean("metric"))
    sqf = sqf.where(factors.min("metric") > 0, 0.0)
    sqf.name = "SQF"
    return sqf


# ---------- convenience aggregator ----------
def compute_scores(
    pred: xr.Dataset,
    *,
    column_dead_time: float,
    width_penalty_coeff: float = 10.0,
) -> xr.Dataset:
    """
    Bundle the scores you currently use into a single xarray Dataset.
    """
    eta = score_eta(
        pred,
        column_dead_time=column_dead_time,
        width_penalty_coeff=width_penalty_coeff,
    )
    Sbar = score_Sbar(pred)
    We = score_We(pred, column_dead_time=column_dead_time)
    DUr = score_DU(pred)
    CPO = score_CPO(pred)
    Rs_crit = score_critical_resolution(pred)

    factors = xr.concat([eta, Sbar, We, DUr, CPO], dim="metric")
    if bool((factors < 0).any()):
        raise ValueError("SQF is undefined for negative score components.")
    SQF = np.exp(np.log(factors).mean("metric"))
    SQF = SQF.where(factors.min("metric") > 0, 0.0)
    SQF.name = "SQF"

    return xr.Dataset(
        data_vars=dict(
            eta=eta,
            Sbar=Sbar,
            We=We,
            DUr=DUr,
            CPO=CPO,
            SQF=SQF,
            Rs_crit=Rs_crit,
        ),
        coords=pred.drop_vars("analyte").coords,  # keep method dims, drop analyte dim
    )
