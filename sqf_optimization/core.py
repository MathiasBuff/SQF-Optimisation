from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr

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

        # TODO: move validation to i/o, assume core gets clean data
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

def predict_grid_from_params(
    model_params,
    *,
    T_vals: np.ndarray,
    tG_vals: np.ndarray,
    t0_total: float,
):
    """
    Predict tR, w, A on a (T, tG) grid from fitted model parameters.

    Model forms:
      ln(tR - t0_total) = a0 + a1/T + a2/tG
      ln(w)             = b0 + b1/T + b2/tG
      ln(A)             = c0 + c1/T + c2/tG

    Returns dicts {analyte: ndarray(M,L)} for tR, w, A.
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
    A  = np.exp(c0 + c1 * invT[None, :, :] + c2 * invtG[None, :, :])

    # pred_tR = {analytes[i]: tR[i, :, :] for i in range(len(analytes))}
    # pred_w  = {analytes[i]: w[i, :, :]  for i in range(len(analytes))}
    # pred_A  = {analytes[i]: A[i, :, :]  for i in range(len(analytes))}
    
    # """
    # Convert dict-of-(M,L) predictions into an xarray Dataset with dims (analyte, T, tG).
    # """
    if analytes is None:
        analytes = list(tR.keys())

    # # stack into (N, M, L)
    # tR = np.stack([pred_tR[a] for a in analytes], axis=0)
    # w  = np.stack([pred_w[a]     for a in analytes], axis=0)
    # A  = np.stack([pred_A[a]   for a in analytes], axis=0)

    if tR.shape[1:] != (len(T_vals), len(tG_vals)):
        raise ValueError(
            f"Prediction grid shape mismatch: got (M,L)={tR.shape[1:]}, "
            f"expected ({len(T_vals)},{len(tG_vals)})."
        )

    return xr.Dataset(
        data_vars={
            "tR": (("analyte", "T", "tG"), tR),
            "w":  (("analyte", "T", "tG"), w),
            "A":  (("analyte", "T", "tG"), A),
        },
        coords={
            "analyte": analytes,
            "T": np.asarray(T_vals),
            "tG": np.asarray(tG_vals),
        },
    )
    
    
    # return pred_tR, pred_w, pred_A

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

