import numpy as np
import logging
import matplotlib.pyplot as plt
LOGGER = logging.getLogger(__name__)
from .config import RMConfig, FitInitConfig, get_mcf, fixed_parameters
import at

# ----------------------- PLOTING OPTICS AND ORMs -----------------------

def plot_beta(s_pos, bx, by, save_path=None):
    def rms(x):
        return np.sqrt(np.mean(x ** 2))
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 4),
        sharey=False
    )

    # Normal quadrupoles

    # axes[0].scatter(s_pos, bx*100, s=15, color='darkblue')
    axes[0].plot(s_pos, bx * 100, linewidth=0.8, color='darkorange')
    axes[0].set_xlabel("S [m]")
    axes[0].set_ylabel(r"$\Delta \beta_x  \beta_x \%$")
    axes[0].set_title("Horizontal beta beating")

    text_q = (
        f"Max = {(bx.max()) * 100:.2e}\n"
        f"RMS = {rms(bx) * 100:.2e} %"
    )
    axes[0].text(
        0.98, 0.95, text_q,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Skew quadrupoles

    # axes[1].scatter(s_pos, by* 100, s=15, color='darkblue')
    axes[1].plot(s_pos, by * 100, linewidth=0.8, color='darkblue')
    axes[1].set_xlabel("S [m]")
    axes[1].set_ylabel(r"$\Delta \beta_y  \beta_y \%$")
    axes[1].set_title("Vertical beta beating")
    text_q = (
        f"Max = {(by.max()) * 100:.2e}\n"
        f"RMS = {rms(by) * 100:.2e} %"
    )

  
    axes[1].text(
        0.98, 0.95, text_q,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)



def plot_eta(s_pos, dx, dy, save_path=None):
    def rms(x):
        return np.sqrt(np.mean(x ** 2))
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 4),
        sharey=False
    )

    # Normal quadrupoles

    # axes[0].scatter(s_pos, bx*100, s=15, color='darkblue')
    axes[0].plot(s_pos, dx * 1000, linewidth=0.8, color='darkorange')
    axes[0].set_xlabel("S [m]")
    axes[0].set_ylabel(r"$\Delta \eta_x  [mm]$")
    axes[0].set_title("Horizontal dispersion")


    text_q = (
        f"Max = {(dx.max()) * 1000:.2e}\n"
        f"RMS = {rms(dx) * 1000:.2e} mm"
    )
    axes[0].text(
        0.98, 0.95, text_q,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Skew quadrupoles

    # axes[1].scatter(s_pos, by* 100, s=15, color='darkblue')
    axes[1].plot(s_pos, dy * 1000, linewidth=0.8, color='darkblue')
    axes[1].set_xlabel("S [m]")
    axes[1].set_ylabel(r"$\Delta \eta_y  [mm]$")
    axes[1].set_title("Vertical dispersion")

    text_q = (
        f"Max = {(dy.max()) * 1000:.2e}\n"
        f"RMS = {rms(dy) * 1000:.2e} mm"
    )
    axes[1].text(
        0.98, 0.95, text_q,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)

def calculate_dispersion(ring, fixed_parameters, bpm_ords, calculator = 'Linear', rfStep = 200, RFAttr = "Frequency"):

    C = 2.99792458e8

    if calculator == 'Linear':

        f_rf = fixed_parameters.Frequency
        h_rf = fixed_parameters.HarmNumber

        _, ORBITPLUS = at.find_sync_orbit(
            ring,
            (-C * rfStep * h_rf / f_rf ** 2) / 2,
            refpts=bpm_ords
        )

        dx = ORBITPLUS[:, 0]
        dy = ORBITPLUS[:, 2]

        _, ORBIT0 = at.find_sync_orbit(
            ring,
            (C * rfStep * h_rf / f_rf ** 2) / 2,
            refpts=bpm_ords
        )

        dx0 = ORBIT0[:, 0]
        dy0 = ORBIT0[:, 2]

        dispersion_meas = np.concatenate((dx - dx0, dy - dy0))

    else:

        if bidirectional == True:

            shift_rf(ring, cav_ords,+rfStep / 2,attr=RFAttr)

            _,orbit = at.find_orbit4(ring, 0, bpm_ords)
            orbit_plus_x = orbit[:,0]
            orbit_plus_y = orbit[:,2]

            shift_rf(ring, cav_ords,-rfStep/2, attr=RFAttr)
            shift_rf(ring, cav_ords,-rfStep / 2, attr=RFAttr)

            _,orbit = at.find_orbit4(ring, 0, bpm_ords)
            orbit_minus_x = orbit[:,0]
            orbit_minus_y = orbit[:,2]

            shift_rf(ring, cav_ords,+rfStep / 2, attr=RFAttr)  # Restore

            dx = orbit_plus_x - orbit_minus_x - orbit0_x
            dy = orbit_plus_y - orbit_minus_y - orbit0_y

        else:
            shift_rf(ring, cav_ords,+rfStep, attr=RFAttr)

            _,orbit = at.find_orbit4(ring, 0, bpm_ords)
            orbit_new_x = orbit[:, 0]
            orbit_new_y = orbit[:,2]

            shift_rf(ring, cav_ords, -rfStep, attr=RFAttr)  # Restore

            dx = orbit_new_x - orbit0_x
            dy = orbit_new_y - orbit0_y

        dispersion_meas = np.concatenate((dx, dy))

    return dispersion_meas


import numpy as np
import matplotlib.pyplot as plt


def plot_matrices(
    *matrices,
    titles=None,
    cmap="viridis",
    plot_type="2d",
    save_path=None,
    same_scale=True,
):
    """
    Plot one or more matrices in either 2D or 3D.

    Parameters
    ----------
    matrices : ndarray
        Matrices to plot.
    titles : list of str, optional
        Titles of each subplot.
    cmap : str
        Matplotlib colormap.
    plot_type : {"2d", "3d"}
        Type of visualization.
    save_path : str or Path, optional
        Save figure if provided.
    same_scale : bool
        Use the same color scale for all matrices.
    """

    n = len(matrices)

    if n == 0:
        raise ValueError("At least one matrix must be provided.")

    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(n)]
    elif len(titles) < n:
        titles += [f"Matrix {i+1}" for i in range(len(titles), n)]

    # -------------------------------------------------------
    # Same color scale for all matrices
    # -------------------------------------------------------
    if same_scale:
        vmin = min(np.nanmin(m) for m in matrices)
        vmax = max(np.nanmax(m) for m in matrices)
    else:
        vmin = vmax = None

    fig = plt.figure(figsize=(7 * n, 6))

    for i, matrix in enumerate(matrices):

        if plot_type.lower() == "3d":

            ax = fig.add_subplot(1, n, i + 1, projection="3d")

            X, Y = np.meshgrid(
                np.arange(matrix.shape[1]),
                np.arange(matrix.shape[0])
            )

            surf = ax.plot_surface(
                X,
                Y,
                matrix,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0,
                antialiased=True,
                shade=True,
                rcount=min(250, matrix.shape[0]),
                ccount=min(250, matrix.shape[1]),
            )

            ax.view_init(elev=28, azim=-60)

            ax.set_title(titles[i], fontsize=15, weight="bold")

            ax.set_xlabel("Correctors", fontsize=12, labelpad=10)
            ax.set_ylabel("BPMs", fontsize=12, labelpad=10)
            ax.set_zlabel("Response [m]", fontsize=12, labelpad=10)

            ax.tick_params(labelsize=10)

            # cleaner background
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.grid(True, alpha=0.3)

            cbar = fig.colorbar(
                surf,
                ax=ax,
                shrink=0.75,
                aspect=25,
                pad=0.08,
            )
            cbar.set_label("Response [m]", fontsize=11)

        else:

            ax = fig.add_subplot(1, n, i + 1)

            im = ax.imshow(
                matrix,
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )

            ax.set_title(titles[i], fontsize=15, weight="bold")
            ax.set_xlabel("Correctors", fontsize=12)
            ax.set_ylabel("BPMs", fontsize=12)

            ax.tick_params(labelsize=10)

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
            cbar.set_label("Response [m]", fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)

def plot_orm_bars_simone_style(
    R_dir, R_coup,
    R_model=None,
    labels=None,
    title="Steerers response",
    ylabel="std [m/rad]",
    save_path=None
):

    std_dir = np.std(R_dir , axis=0)
    std_coup = np.std(R_coup, axis=0)
    std_model = np.std(R_model, axis=0) if R_model is not None else None

    x = np.arange(len(std_dir))
    width = 0.75

    fig, ax = plt.subplots(figsize=(8, 2))



    ax.bar(
        x,
        std_dir,
        width=width * 0.85,
        color="green",
        alpha=0.85,
        label="direct",
        zorder=2
    )

    ax.bar(
        x,
        std_coup,
        width=width * 0.55,
        color="red",
        alpha=0.85,
        label="coupling",
        zorder=3
    )

    if std_model is not None:
        ax.bar(
            x,
            std_model,
            width=width,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            label="expected",
            zorder=1
        )

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    #ax.set_ylim(4, 6)
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)

    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)
