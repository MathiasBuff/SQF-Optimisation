import numpy as np
import matplotlib.pyplot as plt

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

    fig = plt.figure(figsize=(3.25, 2.5))
    if use_pcolormesh:
        plt.pcolormesh(tG, T, Z, shading="nearest", cmap="nipy_spectral")
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
    return fig

# --- Optimum-point synthetic chromatogram (transcribed to the xarray single-point test) ---

def plot_synthetic_chromatogram(grid, T_index, tG_index, measurements):

    T_val = grid.coords["T"].values[T_index]
    tG_val = grid.coords["tG"].values[tG_index]

    # Extract per-analyte scalars at the single point (T=0, tG=0)
    tR_opt = grid.isel(T=T_index, tG=tG_index)["tR"].values  # (N,)
    w_opt  = grid.isel(T=T_index, tG=tG_index)["w"].values   # (N,)
    A_opt  = grid.isel(T=T_index, tG=tG_index)["A"].values   # (N,)
    analytes = grid.isel(T=T_index, tG=tG_index).coords["analyte"].values.tolist()

    areas = np.array(measurements["area"], dtype=float)  # (N,)

    # Convert (w, S) into asymmetric left/right widths (same as old notebook)
    w_left  = 2.0 * w_opt / (A_opt + 1.0)
    w_right = 2.0 * w_opt * A_opt / (A_opt + 1.0)

    # Time axis for simulated chromatogram
    x = np.linspace(0.0, float(tG_val), 2000)  # increase points for smoothness

    # Broadcasted peak model:
    # y_i(x) = a_i * exp( -4 * (tR_i - x)^2 / w_i(x)^2 )
    # where w_i(x) = w_left_i if x < tR_i else w_right_i
    rt = tR_opt[:, None]            # (N,1)
    xx = x[None, :]                # (1,X)
    wl = w_left[:, None]           # (N,1)
    wr = w_right[:, None]          # (N,1)
    aa = areas[:, None]             # (N,1)

    w_piece = np.where(xx < rt, wl, wr)  # (N,X)
    y = aa * np.exp(-4.0 * ((rt - xx) ** 2) / (w_piece ** 2))  # (N,X)
    y_total = np.sum(y, axis=0)  # (X,)


    # Overlay individual peaks (comment out if too cluttered)
    fig = plt.figure(figsize=(15, 6))
    plt.plot(x, y_total, linewidth=2.0, label="Total", color="black")
    for i, k in enumerate(analytes):
        plt.plot(x, y[i], label=k, alpha=0.6)
    plt.title(f"Predicted chromatogram at tG={tG_val:.2f} min, T={T_val:.2f} K")
    plt.xlabel("Time [min]")
    plt.ylabel("Signal (a.u.)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()

    # Simple total chromatogram plot (replace the above if too cluttered)
    # fig = plt.figure(figsize=(15, 6))
    # plt.plot(x, y_total)
    # plt.title(f"Predicted chromatogram at tG={tG_test:.2f} min, T={T_test:.2f} K")
    # plt.xlabel("Time [min]")
    # plt.ylabel("Signal (a.u.)")
    # plt.tight_layout()

    return fig

