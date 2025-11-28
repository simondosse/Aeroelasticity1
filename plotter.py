#%%
from cProfile import label
import os
from re import L
from tkinter import font
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import matplotlib.patches as mpatches
from pyparsing import col

from data_manager import _load_npz
from typing import Tuple, Optional 
import ROM

mpl.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["Computer Modern Roman"],
})

# Colors used both in python plots and in LaTeX documents (on Inkscape figures for examples)
with open("colors_set.json", "r") as f:
    colors_set = json.load(f)

# figsize = (5.5,4)  # Taille des figures (format article)

''' FREQ & DAMPING PLOTTING FUNCTIONS '''

def plot_modal_data_single(f,
                           damping,
                           par=None,
                           U: Optional[np.ndarray] = None,
                           suptitle: Optional[str] = None,
                           Uc: Optional[float] = None,
                           colors: Optional[list] = None,
                           figsize: bool = False,
                           save: bool = False,
                           filename: str = 'modal_data'):
    """
    Trace un seul jeu de données (fréquences et amortissements) à partir des tableaux passés.

    Entrées
    - f: (nU,) ou (nU, n_modes) fréquences en Hz
    - damping: (nU,) ou (nU, n_modes) amortissements zeta
    - par: ModelParameters optionnel (utilisé pour U et le style du modèle)
    - U: optionnel si par n'est pas fourni
    - suptitle: titre global de la figure
    """
    f = np.asarray(f)
    z = np.asarray(damping)

    if figsize :
        figsize = figsize # on laisse la possibilité de passer une taille personnalisée, on change le paramètre global figsize
    else:
        figsize = (5.5,2.25)
    if U is None:
        if par is None or not hasattr(par, 'U'):
            raise ValueError("Provide U or a ModelParameters 'par' with U.")
        U = par.U
    U = np.asarray(U).ravel()

    # Style selon le modèle (si dispo)
    if par is not None and hasattr(par, 'model_aero'):
        model_name = 'Theodorsen' if str(par.model_aero).lower().startswith('theod') else 'QuasiSteady'
    else:
        model_name = 'Theodorsen'
    linestyle = '-' if model_name == 'Theodorsen' else '--'

    # Harmoniser dimensions à 2D
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if f.shape[0] != U.size or z.shape[0] != U.size:
        raise ValueError("First dimension of f/damping must match len(U).")

    n_modes = f.shape[1]

    # Couleurs et labels par mode
    try:
        base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    except Exception:
        base_colors = []

    if colors:
        colors = colors
    else:
        colors = base_colors[:n_modes] if len(base_colors) >= n_modes else [f"C{i}" for i in range(n_modes)]
    mode_labels = [fr"$v^{{{j+1}}}$" for j in range(n_modes)]

    # Figure
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=figsize)
    if suptitle is not None:
        fig.suptitle(suptitle)

    # Fréquences
    for j in range(n_modes):
        if not np.all(np.isnan(f[:, j])):
            ax[0].plot(U, f[:, j], color=colors[j], linestyle=linestyle, lw=1.2)
    ax[0].set_ylabel('$f$ [Hz]')
    ax[0].grid(True, linewidth=0.3, alpha=0.5)

    # Amortissement + légende
    for j in range(n_modes):
        ax[1].plot(U, z[:, j], color=colors[j], linestyle=linestyle, lw=1.2, label=mode_labels[j])

    ax[1].set_xlabel(r'$U[\mathrm{m}.\mathrm{s}^{-1}$]')
    ax[1].set_ylabel(r'$\zeta$')
    ax[1].grid(True, linewidth=0.3, alpha=0.5)
    ax[1].legend(frameon=False, ncols=min(4, n_modes))

    if Uc is not None:
        for a in ax:
            a.axvline(Uc, color='k', lw=1.1, ls=':', alpha=0.7)
            ymax = ax[0].get_ylim()[1]

        x_offset = 0.02  # décale de 2% de la largeur du graphe
        xmin, xmax = ax[0].get_xlim()
        dx = x_offset * (xmax - xmin)
        ax[0].text(
            Uc+dx, ymax*0.90,              # position du texte
            r"$U_c$",                   # texte LaTeX
            ha='center', va='top',
            fontsize=9
        )



    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    plt.show()

def plot_modal_data_two(fa,
                        za,
                        fb,
                        zb,
                        par_a=None,
                        par_b=None,
                        U: Optional[np.ndarray] = None,
                        labels: Optional[Tuple[str, str]] = None,
                        save: bool = False,
                        filename: str = 'modal_comparison'):
    """
    Superpose deux simulations (fa, za) et (fb, zb).

    - fa, fb: (nU,) ou (nU, n_modes)
    - za, zb: (nU,) ou (nU, n_modes)
    - par_a, par_b: ModelParameters optionnels pour style et/ou U
    - U: optionnel si non fourni via par_a/par_b
    - labels: labels pour la légende, sinon model_aero utilisé si dispo
    """
    fa = np.asarray(fa)
    fb = np.asarray(fb)
    za = np.asarray(za)
    zb = np.asarray(zb)

    # Récupérer U
    if U is None:
        U_candidates = []
        if par_a is not None and hasattr(par_a, 'U'):
            U_candidates.append(np.asarray(par_a.U).ravel())
        if par_b is not None and hasattr(par_b, 'U'):
            U_candidates.append(np.asarray(par_b.U).ravel())
        if len(U_candidates) == 0:
            raise ValueError("Provide U or ModelParameters par_a/par_b with U.")
        U = U_candidates[0]
        for Uc in U_candidates[1:]:
            if Uc.shape != U.shape or not np.allclose(Uc, U):
                raise ValueError("U arrays differ between par_a and par_b; cannot plot on same x-axis.")
    U = np.asarray(U).ravel()

    # Harmoniser dimensions à 2D
    def to_2d(a):
        return a.reshape(-1, 1) if a.ndim == 1 else a
    fa, fb, za, zb = map(to_2d, (fa, fb, za, zb))

    # Tailles et cohérences
    if fa.shape[0] != U.size or fb.shape[0] != U.size or za.shape[0] != U.size or zb.shape[0] != U.size:
        raise ValueError("First dimension of fa/fb/za/zb must match len(U).")

    n_modes = min(fa.shape[1], fb.shape[1], za.shape[1], zb.shape[1])

    # Styles par modèle
    def model_and_style(par):
        if par is not None and hasattr(par, 'model_aero'):
            name = 'Theodorsen' if str(par.model_aero).lower().startswith('theod') else 'QuasiSteady'
        else:
            name = None
        style = '-' if (name or 'Theodorsen') == 'Theodorsen' else '--'
        return name, style

    name_a, style_a = model_and_style(par_a)
    name_b, style_b = model_and_style(par_b)

    la = labels[0] if labels else (name_a or 'A')
    lb = labels[1] if labels else (name_b or 'B')

    # Couleurs et labels par mode
    colors = ['blue', 'red', 'C2', 'C3', 'C4', 'C5']
    mode_labels = [f"Mode {j+1}" for j in range(n_modes)]

    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    # Fréquences
    for j in range(n_modes):
        ax[0].plot(U, fa[:, j], color=colors[j % len(colors)], linestyle=style_a, lw=1.2)
        ax[0].plot(U, fb[:, j], color=colors[j % len(colors)], linestyle=style_b, lw=1.2)
    ax[0].set_ylabel('f [Hz]')
    ax[0].grid(True, linewidth=0.3, alpha=0.5)

    # Amortissements + légende
    for j in range(n_modes):
        ax[1].plot(U, za[:, j], color=colors[j % len(colors)], linestyle=style_a, lw=1.2, label=f"{mode_labels[j]} - {la}")
        ax[1].plot(U, zb[:, j], color=colors[j % len(colors)], linestyle=style_b, lw=1.2, label=f"{mode_labels[j]} - {lb}")

    ax[1].set_xlabel(r'$U [\mathrm{m}.\mathrm{s}^{-1}]$')
    ax[1].set_ylabel(r'$\zeta [-]$')
    ax[1].grid(True, linewidth=0.3, alpha=0.5)
    ax[1].legend(frameon=False, ncols=2)

    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    plt.show()

def plot_params_table(par, save: bool = False, filename: str = 'param_table'):
    # Construire un dict lisible des principaux paramètres
    sig = 4
    def round_sig(x, sig=4):
        if x == 0:
            return 0
        return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

    items = [
        ("model_aero", par.model_aero),
        (r"$s$", round_sig(par.s,sig=sig)),
        (r"$c$", round_sig(par.airfoil.c,sig=sig)),
        (r"$x_{ea}$", round_sig(par.airfoil.x_ea,sig=sig)),
        (r"$x_{cg}$", round_sig(par.airfoil.x_cg,sig=sig)),
        (r"$m$", round_sig(par.airfoil.m,sig=sig)),
        (r"$EI_x$", round_sig(par.EIx,sig=sig)),
        (r"GJ", round_sig(par.GJ,sig=sig)),
        (r"$I_{\alpha,EA}$", round_sig(par.airfoil.Ialpha_EA,sig=sig)),
        (r"$\zeta_w$", round_sig(par.eta_w,sig=sig)),
        (r"$\zeta_{\alpha}$", round_sig(par.eta_alpha,sig=sig)),
        (r"$dC_L$", round_sig(par.dCL,sig=sig)),
        (r"$dC_M$", round_sig(par.dCM,sig=sig)),
        (r"$M_t$", round_sig(par.Mt,sig=sig)),
        (r"$I_{\alpha,t}$", round_sig(par.I_alpha_t,sig=sig)),
        (r"$x_t$", round_sig(par.x_t,sig=sig)),
        (r"$N_w$", par.Nw),
        (r"$N_{\alpha}$", par.Nalpha),
        (r"$U_{max}$", round_sig(par.Umax)),
        (r"$nU$", par.Ustep),
    ]
    # Filtrer None
    table_data = [[k, str(v)] for k, v in items if v is not None]

    if not table_data:
        print("No parameters found to display.")
        return

    fig, ax = plt.subplots(figsize=(6, len(table_data)*0.4 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", f"Value"],
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    ax.set_title("Model Parameters", fontsize=11, pad=10)

    plt.tight_layout()
    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    plt.show()

''' EIGVECTOR COEFFICIENTS PLOTTING FUNCTIONS '''

def plot_vi(vi: np.ndarray,
            Nw: int,
            Nalpha: int,
            kind: str = 'abs',              # 'abs' | 'real_imag' | 'mag_phase'
            normalize: Optional[str] = None,# None | 'max' | 'l2'
            align_phase: bool = False,
            ref_idx: Optional[int] = None,
            figsize: Optional[Tuple[float, float]] = None,
            bending_label: str = 'vw (bending)',
            torsion_label: str = 'va (torsion)',
            ax: Optional[plt.Axes] = None,  # NEW: draw into an existing axis
            save: bool = False,
            filename: str = 'vi'
):
    
    if figsize is None:
        figsize = (7, 2.25)
    vi = np.asarray(vi).reshape(-1)
    assert vi.size == Nw + Nalpha, "vi must have length Nw+Nalpha"
    v_plot = vi.copy()

    # Optional phase alignment
    if align_phase and v_plot.size > 0:
        if ref_idx is None:
            ref_idx = int(np.argmax(np.abs(v_plot)))
        ang = np.angle(v_plot[ref_idx])
        v_plot = v_plot * np.exp(-1j * ang)

    # normalization across the full vector (vw+va)
    if normalize == 'max':
        a = np.max(np.abs(v_plot)) or 1.0
        v_plot = v_plot / a
    elif normalize == 'l2':
        a = np.sqrt(np.vdot(v_plot, v_plot).real) or 1.0
        v_plot = v_plot / a

    vw = v_plot[:Nw]
    va = v_plot[Nw:]

    # mag_phase needs two axes -> only allowed when ax is None
    if kind == 'mag_phase' and ax is not None:
        raise ValueError("kind='mag_phase' requires ax=None (creates a 2-row figure)")

    if kind == 'mag_phase':
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
        ax_mag, ax_phi = axes
        ax_mag.bar(np.arange(Nw), np.abs(vw), color='#4C78A8', label=bending_label)
        ax_mag.bar(Nw + np.arange(Nalpha), np.abs(va), color='#F58518', label=torsion_label)
        ax_mag.set_xticks(np.arange(Nw + Nalpha))
        ax_mag.set_xticklabels([rf"$\phi_{{w,{k+1}}}$" for k in range(Nw)] + [rf"$\phi_{{\alpha,{k+1}}}$" for k in range(Nalpha)])
        ax_mag.set_ylabel("|coeff|"); ax_mag.legend(); ax_mag.set_title("Modal coefficients magnitude")

        phase = np.angle(np.concatenate([vw, va]))
        ax_phi.plot(np.arange(Nw + Nalpha), phase, 'o-', color='#6F4E7C')
        ax_phi.axhline(0.0, color='k', lw=0.8, alpha=0.5)
        ax_phi.set_xticks(np.arange(Nw + Nalpha))
        ax_phi.set_xticklabels([fr"$\phi_{{w,{k+1}}}$" for k in range(Nw)] + [rf"$\phi_{{\alpha,{k+1}}}$" for k in range(Nalpha)])
        ax_phi.set_ylabel("phase [rad]"); ax_phi.set_title("Modal coefficients phase")
        if save:
            os.makedirs('images', exist_ok=True)
            fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
        return fig, axes

    # Single-axis variants
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 2), constrained_layout=True)
        created_fig = True

    idx_w = np.arange(Nw)
    idx_a = Nw + np.arange(Nalpha)

    if kind == 'abs':
        # ax.bar(idx_w, np.abs(vw), color="#0026FF", label=bending_label)
        # ax.bar(idx_a, np.abs(va), color="#A200FF", label=torsion_label)
        ax.bar(idx_w, np.abs(vw), color=colors_set['strong_blue'], label=bending_label)
        ax.bar(idx_a, np.abs(va), color = colors_set['strong_purple'], label=torsion_label)
        ax.set_ylabel(r"$|v^i_{\_,j}|$")
        ax.set_title("Modal coefficients (magnitude)")
    elif kind == 'real_imag':
        width = 0.38
        ax.bar(idx_w - width/2, vw.real, width, color='#4C78A8', label=f"Re {bending_label}")
        ax.bar(idx_w + width/2, vw.imag, width, color='#72B7B2', label=f"Im {bending_label}")
        ax.bar(idx_a - width/2, va.real, width, color='#F58518', label=f"Re {torsion_label}")
        ax.bar(idx_a + width/2, va.imag, width, color='#E45756', label=f"Im {torsion_label}")
        ax.set_ylabel("value")
        ax.set_title("Modal coefficients (real/imag)")
        ax.axhline(0.0, color='k', lw=0.8, alpha=0.5)
    else:
        raise ValueError("kind must be 'abs', 'real_imag', or 'mag_phase'")

    ax.set_xticks(np.arange(Nw + Nalpha))
    ax.set_xticklabels([fr"$v_{{w,{k+1}}}$" for k in range(Nw)] + [rf"$v_{{\alpha,{k+1}}}$" for k in range(Nalpha)])
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if created_fig:
        ax.legend()
        # if save:
        #     os.makedirs('images', exist_ok=True)
        #     fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
        return fig, ax
    else:
        # if save:
        #     # save the parent figure of the provided axis
        #     try:
        #         fig_to_save = ax.figure
        #         os.makedirs('images', exist_ok=True)
        #         fig_to_save.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
        #     except Exception:
        #         pass
        return ax

def plot_vi_grid(Vq: np.ndarray, Nw: int, Nalpha: int,
                 mode_indices=None,
                 freqs_hz: Optional[np.ndarray] = None,
                 kind: str = 'abs',              # 'abs' | 'real_imag'
                 normalize: Optional[str] = 'l2',# None | 'max' | 'l2' (per mode)
                 align_phase: bool = False,
                 sharey: bool = True,
                 figsize: Optional[Tuple[float, float]] = None,
                 suptitle: Optional[str] = None,
                 show: bool = True,
                 save: bool = False,
                 filename: str = 'vi_grid',
                 bending_label: str = r'$v_{w,j}$',
                 torsion_label: str = r'$v_{\alpha,j}$',
):
    Vq = np.asarray(Vq)
    assert Vq.shape[0] == Nw + Nalpha, "Vq must have Nw+Nalpha rows"

    # Determine modes to display (accept 0-based or 1-based)
    n_modes_total = Vq.shape[1]
    if mode_indices is None:
        sel_modes = list(range(n_modes_total))
    else:
        sel_modes = list(mode_indices)
    n_modes = len(sel_modes)

    if figsize is None:
        figsize = (max(5.0, 3.0 * n_modes), 2)

    fig, axes = plt.subplots(1, n_modes, sharey=sharey, figsize=figsize, constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    for i, jm in enumerate(sel_modes):
        ax = axes[i]
        plot_vi(
            vi=Vq[:, jm],
            Nw=Nw, Nalpha=Nalpha,
            kind=kind,
            normalize=normalize,
            align_phase=align_phase,
            bending_label=bending_label,
            torsion_label=torsion_label,
            ax=ax,  # reuse single-axis variant
        )
        title = f"Mode {jm+1} : $v^{{{jm+1}}}$"
        if freqs_hz is not None:
            title += f" (f={float(freqs_hz[jm]):.2f} Hz)"
        ax.set_title(title)

        # Only keep the y-label on the leftmost subplot to avoid repetition
        if i > 0:
            ax.set_ylabel("")

        if i == n_modes - 1:
            ax.legend(frameon=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes

def plot_vi_grid_over_U(U: np.ndarray,Vq_U: np.ndarray,
                        Nw: int,Nalpha: int,
                        mode_indices=None,n_samples: int = 10,
                        kind: str = 'abs', # 'abs' | 'real_imag' (mag_phase not supported here)
                        normalize = None, # None | 'max' | 'l2'
                        align_phase = False,
                        f_modes_U= None,
                        sharey=True,
                        figsize = None,suptitle= None,show=True,
                        save: bool = False, filename: str = 'vi_grid_over_U',
                        bending_label = 'vw (bending)',
                        torsion_label = 'va (torsion)',
):
    U = np.asarray(U).ravel()
    assert U.ndim == 1 and U.size > 0, "U must be 1D with at least one entry"

    Vq_U = np.asarray(Vq_U)
    Nq = Nw + Nalpha
    assert Vq_U.ndim == 3, "Vq_U must be 3D (nU, ?, n_modes)"

    # Accept either Vq_U shape (nU, Nq, Nq) or eigvecs_U shape (nU, 2*Nq, Nq)
    if Vq_U.shape[1] == Nq:
        Vq_only = Vq_U
    elif Vq_U.shape[1] == 2 * Nq:
        Vq_only = Vq_U[:, :Nq, :]
    else:
        raise ValueError("Vq_U second dim must be Nq or 2*Nq")

    nU, _, n_modes_total = Vq_only.shape
    assert nU == U.size, "len(U) must match Vq_U.shape[0]"

    # Determine modes to display (accept 1-based)
    if mode_indices is None:
        sel_modes = list(range(n_modes_total))
    else:
        sel_modes = list(mode_indices)
        if len(sel_modes) > 0 and min(sel_modes) >= 1 and max(sel_modes) <= n_modes_total:
            sel_modes = [m - 1 for m in sel_modes]  # convert to 0-based

    # Choose U samples
    n_rows = min(int(n_samples), nU)
    if n_rows <= 1:
        row_idx = [0]
    else:
        row_idx = np.unique(np.linspace(0, nU - 1, n_rows).astype(int)).tolist()
        n_rows = len(row_idx)

    n_cols = len(sel_modes)
    if n_cols == 0:
        raise ValueError("No modes selected to plot")

    if kind == 'mag_phase':
        raise ValueError("kind='mag_phase' not supported in grid; use plot_vi for a single U")

    if figsize is None:
        figsize = (max(5.0, 3.0 * n_cols), max(2.6, 2.6 * n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, sharey=sharey, figsize=figsize, constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, iu in enumerate(row_idx):
        for c, jm in enumerate(sel_modes):
            ax = axes[r, c]
            vi = Vq_only[iu, :, jm]
            plot_vi(
                vi=vi,
                Nw=Nw, Nalpha=Nalpha,
                kind=kind,
                normalize=normalize,
                align_phase=align_phase,
                bending_label=bending_label,
                torsion_label=torsion_label,
                ax=ax
            )
            # Titles and labels
            if r == 0:
                ax.set_title(f"Mode {jm+1}")
            else:
                ax.set_title("")
            # Annotate frequency inside subplot (consistent with plot_mode_shapes_grid_over_U)
            if f_modes_U is not None:
                try:
                    fval = float(f_modes_U[iu, jm])
                    if np.isfinite(fval):
                        ax.text(
                            0.98, 0.06, f"f = {fval:.2f} Hz",
                            transform=ax.transAxes,
                            ha='right', va='bottom', fontsize=9, color='0.35'
                        )
                except Exception:
                    pass
            if c == 0:
                ax.set_ylabel(rf"$U={U[iu]:.2f} \mathrm{{m}}.\mathrm{{s}}^{-1}$")
            # X-ticks only on bottom row
            if r < n_rows - 1:
                ax.set_xticklabels([])

    if suptitle:
        fig.suptitle(suptitle)

    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes

def plot_vi_contribution_over_U(U: np.ndarray,
                                Vq_U: np.ndarray,
                                Nw: int,
                                Nalpha: int,
                                mode_index: int = 0,
                                norm: str = None,   # 'l2' or 'max'
                                figsize: Tuple[float, float] = (6, 4),
                                suptitle: Optional[str] = None,
                                show: bool = True,
                                save: bool = False,
                                filename: str = 'nonfichier'):
    """
    Trace l'évolution, pour un mode donné, du rapport |vw|/|va| en fonction de U.

    Entrées
    - U: vecteur des vitesses (nU,)
    - Vq_U: tenseur (nU, Nq, n_modes) ou (nU, 2*Nq, n_modes) contenant les vecteurs propres
            (format compatible avec `plot_vi_grid_over_U`).
    - Nw, Nalpha: tailles des sous-vecteurs vw et va (Nq = Nw+Nalpha)
    - mode_index: index du mode (0-based)
    - norm: méthode pour réduire un vecteur à un scalaire ('l2' pour norme L2, 'max' pour max(|.|)).
    - show/save/filename: comportement d'affichage et sauvegarde (PDF dans `images/`).

    Retour
    - fig, ax
    """
    U = np.asarray(U).ravel()
    if U.ndim != 1 or U.size == 0:
        raise ValueError("U must be a 1D array with at least one element")

    Vq_U = np.asarray(Vq_U)
    Nq = Nw + Nalpha
    if Vq_U.ndim != 3:
        raise ValueError("Vq_U must be 3D with shape (nU, Nq or 2*Nq, n_modes)")

    # Accept either Vq_U shape (nU, Nq, n_modes) or eigvecs_U shape (nU, 2*Nq, n_modes)
    if Vq_U.shape[1] == Nq:
        Vq_only = Vq_U
    elif Vq_U.shape[1] == 2 * Nq:
        Vq_only = Vq_U[:, :Nq, :]
    else:
        raise ValueError("Vq_U second dim must be Nq or 2*Nq")

    nU, _, n_modes_total = Vq_only.shape
    if nU != U.size:
        raise ValueError("len(U) must match Vq_U.shape[0]")

    # accept 1-based indices
    mi = int(mode_index)


    # compute ratios
    ratios = np.zeros(nU, dtype=float)
    tiny = 1e-12
    for iu in range(nU):
        vi = Vq_only[iu, :, mi]
        vw = vi[:Nw]
        va = vi[Nw:]
        if norm == 'max':
            num = float(np.max(np.abs(vw)))
            den = float(np.max(np.abs(va)))
        else:
            num = float(np.linalg.norm(vw))
            den = float(np.linalg.norm(va))
        if den == 0.0:
            ratios[iu] = np.nan
        else:
            ratios[iu] = num / max(den, tiny)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.plot(U, ratios, color='tab:blue')
    ax.set_xlabel(r'$U [\mathrm{m}.\mathrm{s}^{-1}]$')
    ax.set_ylabel(r'$|\mathbf{v}_w| / |\mathbf{v}_a|$')
    
    title = r"Relative bending-torsion contribution ($|\mathbf{v}_w|/|\mathbf{v}_a|$)"+f" - Mode {mi+1}"
    if suptitle:
        title = suptitle
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax

def plot_vi_wa_phase_over_U(par, U, phase_w_a_list, idx_modes=None, save: bool = False, filename: str = 'phase_w_a_over_U'):
    '''
    Trace la phase entre la partie de flexion et de torsion pour un ou plusieurs modes.
    Paramètres
    ----------
    par : objet
        Contient par.U (vecteur des vitesses).
    U : array_like
        Vitesses utilisées pour le tracé (écrase par.U si fourni).
    phase_w_a_list : list of array_like
        Liste contenant les vecteurs de phase pour chaque mode.
    idx_modes : list of int, optional
        Indices des modes (0-based). Si None, les indices seront [0, 1, 2, ...].
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    U = np.asarray(par.U if U is None else U).ravel()
    n_modes = len(phase_w_a_list)
    if idx_modes is None:
        idx_modes = list(range(n_modes))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
    fig.suptitle(r'Phase angle $\theta_w$ by which the bending part leads the torsion part')

    # Couleurs par défaut (cyclage)
    colors = plt.cm.tab10(np.linspace(0, 1, n_modes))
    if len(phase_w_a_list) == 2:
        colors = ['tab:blue', 'tab:orange']

    for phase_w_a, mode_idx, color in zip(phase_w_a_list, idx_modes, colors):
        phase_w_a = np.asarray(phase_w_a).ravel()
        ax.plot(U, phase_w_a, '-', lw=1.3, color=color, label=rf'Mode {mode_idx+1}')

    ax.set_xlabel(r'$U [\mathrm{m}.\mathrm{s}^{-1}]$')
    ax.set_ylabel(r'Phase angle $\theta_w$ [rad]')
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)

    ax.axhline(y=np.pi/2, linestyle='--', color='k', lw=0.8, alpha=0.5)
    ax.axhline(y=-np.pi/2, linestyle='--', color='k', lw=0.8, alpha=0.5)

    ax.legend()
    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    plt.show()

''' w(y), alpha(y) plotting functions '''

def plot_w_alpha_fields(par, t, w_map, a_map, U = None, times_to_plot=None, cmap='viridis', return_maps=False):
    """
    Reconstruit et trace w(y,t) et alpha(y,t) à partir de l'état X(t).
    Suppose q = [qv (optionnel), qw (Nw), qa (Nalpha)] puis [vitesse...].

    Paramètres
    ----------
    par : ModelParameters
        Doit fournir y, Nw, Nalpha, (optionnellement Nv).
    t : ndarray (nt,)
        Temps utilisés lors de l'intégration.
    w_map : ndarray (nt, Ny)
        Champ w(y,t) à tracer.
    a_map : ndarray (nt, Ny)
        Champ alpha(y,t) à tracer.
    times_to_plot : list[float] ou None
        Instants sélectionnés pour coupes w(y, t_i) et alpha(y, t_i). Si None, choisi 4 instants.
    cmap : str
        Colormap pour les cartes temps-envergure.
    return_maps : bool
        Si True, retourne aussi (w_map, alpha_map) de taille (nt, Ny).

    Retours
    -------
    (optionnel) w_map : ndarray (nt, Ny), alpha_map : ndarray (nt, Ny)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float)
    nt = t.size

    Nw = par.Nw
    Nalpha = par.Nalpha
    Nv = par.Nv

    # Choix des instants pour coupes
    if times_to_plot is None:
        # 4 instants répartis (début/extrémités incluses)
        idx = np.linspace(0, nt - 1, 4, dtype=int)
    else:
        times_to_plot = np.asarray(times_to_plot, dtype=float)
        # projection aux indices les plus proches
        idx = np.array([np.argmin(np.abs(t - ti)) for ti in times_to_plot], dtype=int)

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    if U>=0: # si U = None alors ça renvoie False par défaut
        fig.suptitle(f'U = {U} m/s')
    # Cartes temps–envergure
    im0 = axes[0, 0].imshow(w_map, aspect='auto', origin='lower',
                            extent=[par.y[0], par.y[-1], t[0], t[-1]], cmap=cmap)
    axes[0, 0].set_title("w(y,t)")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0, 0], label='w')

    im1 = axes[1, 0].imshow(a_map, aspect='auto', origin='lower',
                            extent=[par.y[0], par.y[-1], t[0], t[-1]], cmap=cmap)
    axes[1, 0].set_title("alpha(y,t)")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1, 0], label='alpha')

    # Coupes w(y, t_i)
    for j in idx:
        axes[0, 1].plot(par.y, w_map[j, :], label=f"t={t[j]:.3g}")
    axes[0, 1].set_title("Coupes w(y, t_i)")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("w")
    axes[0, 1].legend(loc='best')

    # Coupes alpha(y, t_i)
    for j in idx:
        axes[1, 1].plot(par.y, a_map[j, :], label=f"t={t[j]:.3g}")
    axes[1, 1].set_title("Coupes alpha(y, t_i)")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("alpha")
    axes[1, 1].legend(loc='best')

    plt.show()

# this one is not used anymore
def plot_mode_shapes_grid(y, freqs_hz, W=None, ALPHA=None,extras=None,normalize=False,colors=None,styles=None,sharey=True,figsize=None,suptitle=None,show=True, save: bool = False, filename: str = 'nonfichier'):
    '''
    Trace les formes modales par mode, en colonnes :
    | Mode 1 (f=...) | Mode 2 (f=...) | Mode 3 (f=...) | ...

    Chaque subplot (colonne) superpose w_i(y), alpha_i(y) et, si fourni, des champs supplémentaires (extras).

    Paramètres
    ----------
    y : (Ny,) array
        Abscisses spanwise.
    freqs_hz : (n_modes,) array
        Fréquences par mode (Hz).
    W : (n_modes, Ny) or None
        Formes en flexion w_i(y).
    ALPHA : (n_modes, Ny) or None
        Formes en torsion alpha_i(y).
    extras : dict[str, np.ndarray] or None
        Champs additionnels par mode, ex. {'v': V} avec V shape (n_modes, Ny).
    normalize : {'per_mode','per_field', None}
        - 'per_mode'  : normalise toutes les courbes d'un même mode par le max absolu parmi les champs présents
        - 'per_field' : normalise chaque champ indépendamment (par son propre max absolu)
        - None        : pas de normalisation
    colors : dict[str, str] or None
        Couleurs par champ, ex. {'w': 'C0','alpha': 'C1','v':'C2'}.
    styles : dict[str, str] or None
        Styles de ligne par champ, ex. {'w':'-','alpha':'--','v':':' }.
    sharey : bool
        Partage de l'axe Y entre subplots.
    figsize : tuple or None
        Taille figure (L, H). Défaut calculé sur le nb de modes.
    suptitle : str or None
        Titre global de la figure.
    show : bool
        Appelle plt.show() si True.

    Retour
    ------
    fig, axes : matplotlib Figure et Axes


    En vrai l'arg "normalize" ne sert à rien comme on traite l'amplitude des ces vecteurs en amont
    '''



    # Construire la collection de champs à tracer
    fields = []
    if W is not None:
        fields.append(('w', np.asarray(W)))
    if ALPHA is not None:
        fields.append(('alpha', np.asarray(ALPHA)))
    if extras:
        for name, mat in extras.items():
            fields.append((str(name), np.asarray(mat)))

    if len(fields) == 0:
        raise ValueError("Aucun champ fourni (W, ALPHA ou extras).")

    # Vérifications de dimensions et harmonisation
    y = np.asarray(y).ravel()
    Ny = y.size

    # Nombre de modes à tracer = min(nb colonnes disponibles, len(freqs_hz))
    n_modes_available = [f[1].shape[0] for f in fields]
    n_modes = int(np.min([np.min(n_modes_available), np.asarray(freqs_hz).size]))

    # Vérifie la dimension Ny
    for name, mat in fields:
        if mat.shape[1] != Ny:
            raise ValueError(f"Le champ '{name}' a Ny={mat.shape[1]} différent de len(y)={Ny}.")

    # Couleurs / styles par défaut
    if colors is None:
        colors = {}
    if styles is None:
        styles = {}

    default_palette = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    default_styles = ['-', '--', ':', '-.']

    # Assigner des couleurs/styles par champ s’ils manquent
    for idx, (name, _) in enumerate(fields):
        colors.setdefault(name, default_palette[idx % len(default_palette)])
        styles.setdefault(name, default_styles[idx % len(default_styles)])

    # Figure
    if figsize is None:
        figsize = (max(5.0, 3.0 * n_modes), 3.4)  # largeur ~3 par mode
    fig, axes = plt.subplots(1, n_modes, sharey=sharey, figsize=figsize, constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    # Boucle sur les modes (colonnes)
    for i in range(n_modes):
        ax = axes[i]
        # Traces des champs
        for name, mat in fields:
            curve = np.array(mat[i, :], dtype=float)
            ax.plot(y, curve, color=colors[name], linestyle=styles[name], lw=1.3, label=name)

        # Titres / axes
        fi = float(freqs_hz[i])
        ax.set_title(f"Mode {i+1} (f={fi:.2f} Hz)")
        if i == 0:
            ax.set_ylabel("Amplitude [a.u.]")
        ax.set_xlabel("y [m]")
        ax.grid(True, linewidth=0.3, alpha=0.5)

        # Légende sur le dernier subplot uniquement (évite la répétition)
        if i == n_modes - 1:
            ax.legend(frameon=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


# this one is not used anymore
def plot_mode_shapes_grid_over_U(y, U, WU=None, ALPHAU=None, f_modes_U=None,
                                 mode_indices=None, n_samples=10,
                                 colors=None, styles=None,
                                 sharey=True, figsize=None,
                                 suptitle=None, show=True, save: bool = False, filename: str = 'nonfichier'):
    """
    Plot spatial mode shapes (w and alpha) for multiple wind speeds in a grid.

    Inputs
    ------
    y : array-like, shape (Ny,)
        Spanwise coordinate array.
    U : array-like, shape (nU,)
        Wind speed samples corresponding to WU/ALPHAU.
    WU : array-like or None, shape (nU, n_modes, Ny)
        Bending shapes w_i(y) reconstructed at each U. If None, only alpha is plotted.
    ALPHAU : array-like or None, shape (nU, n_modes, Ny)
        Torsion shapes alpha_i(y) reconstructed at each U. If None, only w is plotted.
    f_modes_U : array-like or None, shape (nU, n_modes)
        Modal frequencies (Hz) per mode and U. If provided, each subplot
        is annotated with its corresponding frequency.
    mode_indices : list[int] or None
        Modes to plot. Defaults to all available modes.
        - Accepts 0-based indices (e.g., [0, 1, 2]).
        - Also accepts 1-based indices (e.g., [1, 2, 3]); detection is automatic:
        if all(idx >= 1) and max(idx) <= n_modes, they are treated as 1-based.
    n_samples : int
        Number of U samples to plot, evenly spaced from U[0] to U[-1]. Clipped to len(U).
    colors : dict[str, str] or None
        Colors per field name. Defaults: {'w':'C0', 'alpha':'C1'}.
    styles : dict[str, str] or None
        Line styles per field name. Defaults: {'w':'-', 'alpha':'--'}.
    sharey : bool
        Share Y axis across subplots. Keep False if you already normalized upstream.
    figsize : (float, float) or None
        Figure size. Defaults to (3.0 * n_modes, 2.6 * n_rows), clamped to reasonable minimums.
    suptitle : str or None
        Global figure title.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array of shape (n_rows, n_cols)

    Notes
    -----
    - WU/ALPHAU are expected as (nU, n_modes, Ny) from your ModalParamDyn.
    - Each subplot overlays w and alpha for a single (mode, U) pair.
    Columns = modes; rows = selected U's.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if WU is None and ALPHAU is None:
        raise ValueError("Provide at least one of WU or ALPHAU.")

    if WU is not None:
        WU = np.asarray(WU)
    if ALPHAU is not None:
        ALPHAU = np.asarray(ALPHAU)

    U = np.asarray(U).ravel()
    y = np.asarray(y).ravel()
    nU = U.size
    Ny = y.size

    # Infer n_modes and validate shapes
    n_modes_candidates = []
    if WU is not None:
        if WU.ndim != 3:
            raise ValueError("WU must have shape (nU, n_modes, Ny).")
        if WU.shape[0] != nU or WU.shape[2] != Ny:
            raise ValueError(f"WU shape mismatch: got {WU.shape}, expected (nU={nU}, n_modes, Ny={Ny}).")
        n_modes_candidates.append(WU.shape[1])
    if ALPHAU is not None:
        if ALPHAU.ndim != 3:
            raise ValueError("ALPHAU must have shape (nU, n_modes, Ny).")
        if ALPHAU.shape[0] != nU or ALPHAU.shape[2] != Ny:
            raise ValueError(f"ALPHAU shape mismatch: got {ALPHAU.shape}, expected (nU={nU}, n_modes, Ny={Ny}).")
        n_modes_candidates.append(ALPHAU.shape[1])
    if f_modes_U is not None:
        f_modes_U = np.asarray(f_modes_U)
        if f_modes_U.ndim != 2 or f_modes_U.shape[0] != nU:
            raise ValueError(
                f"f_modes_U must have shape (nU, n_modes). Got {getattr(f_modes_U, 'shape', None)} with nU={nU}."
            )
        n_modes_candidates.append(f_modes_U.shape[1])

    if not n_modes_candidates:
        raise ValueError("Cannot infer number of modes. Provide WU and/or ALPHAU with valid shapes.")
    n_modes_total = int(min(n_modes_candidates))  # safe choice if shapes differ slightly

    # Select modes
    if mode_indices is None:
        mode_indices_0 = list(range(n_modes_total))
    else:
        idx = np.asarray(mode_indices, dtype=int).ravel().tolist()
        if len(idx) > 0 and min(idx) >= 1 and max(idx) <= n_modes_total:
            mode_indices_0 = [k - 1 for k in idx]  # convert 1-based to 0-based
        else:
            mode_indices_0 = idx
        for k in mode_indices_0:
            if k < 0 or k >= n_modes_total:
                raise ValueError(f"Mode index {k} out of range [0, {n_modes_total-1}].")

    n_cols = len(mode_indices_0)
    if n_cols == 0:
        raise ValueError("No modes selected to plot.")

    # Pick evenly spaced U indices
    n_rows = int(min(max(1, n_samples), nU))
    idx_rows = np.linspace(0, nU - 1, n_rows, dtype=int)
    idx_rows = np.unique(idx_rows)
    n_rows = idx_rows.size

    # Colors / styles defaults
    if colors is None:
        colors = {}
    if styles is None:
        styles = {}
    colors.setdefault('w', 'C0')
    colors.setdefault('alpha', 'C1')
    styles.setdefault('w', '-')
    styles.setdefault('alpha', '--')

    # Figure sizing
    if figsize is None:
        figsize = (max(5.0, 3.0 * n_cols), max(3.0, 1.3 * n_rows))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        sharey=sharey,
        figsize=figsize,
        constrained_layout=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Plot
    for r, iu in enumerate(idx_rows):
        Uval = float(U[iu])
        for c, kmode in enumerate(mode_indices_0):
            ax = axes[r, c]

            # w field
            if WU is not None:
                ax.plot(
                    y, np.asarray(WU[iu, kmode, :], dtype=float),
                    color=colors.get('w', 'C0'),
                    linestyle=styles.get('w', '-'),
                    lw=1.3,
                    label='w'
                )
            # alpha field
            if ALPHAU is not None:
                ax.plot(
                    y, np.asarray(ALPHAU[iu, kmode, :], dtype=float),
                    color=colors.get('alpha', 'C1'),
                    linestyle=styles.get('alpha', '--'),
                    lw=1.3,
                    label='alpha'
                )

            # Titles / labels
            if r == 0:
                ax.set_title(f"Mode {kmode+1}")
            if c == 0:
                ax.set_ylabel(rf"$U = {Uval:.2f} \mathrm{{m}}.\mathrm{{s}}^{-1}")
            if r == n_rows - 1:
                ax.set_xlabel("y [m]")

            ax.grid(True, linewidth=0.3, alpha=0.5)

            # Annotate frequency if provided
            if f_modes_U is not None:
                try:
                    fval = float(f_modes_U[iu, kmode])
                    if np.isfinite(fval):
                        ax.text(
                            0.98, 0.06, f"f = {fval:.2f} Hz",
                            transform=ax.transAxes,
                            ha='right', va='bottom', fontsize=9, color='0.35'
                        )
                except Exception:
                    pass

            # Legend only on the last subplot
            if (r == n_rows - 1) and (c == n_cols - 1):
                ax.legend(frameon=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return fig, axes

'''POWER, ENERGY, ETC. PLOTS'''

def plot_aero_work_distribution(par, U, E_w, E_a, E,
                                suptitle: Optional[str] = None,
                                save: bool = False,
                                filename: str = 'aero_work_distribution'):
    '''
    Plot the distribution of work done by aerodynamic forces along the span.
    
    Parameters
    ----------
    par : 
        Contains model.y (spanwise locations) and model.s (span length).
    U : float
        Wind speed [m/s].
    E_w : (Ny,) array
        Work contribution from bending [J].
    E_a : (Ny,) array
        Work contribution from torsion [J].
    E : (Ny,) array
        Total work [J].
    Returns
    -------
    None
    '''    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    ax.plot(E_w, par.y, lw=1, color = colors_set['strong_blue'],  label=r'bending work')
    ax.plot(E_a, par.y, lw=1, color = colors_set['strong_purple'], label=r'torsion work')
    ax.plot(E, par.y, lw=2, color='k', label='total work')
    

    # Positive side (right)
    mask_pos = E > 0
    ax.fill_betweenx(par.y, 0, E,
                     where=mask_pos,
                     color=colors_set['light_green'],
                     alpha=0.7,
                     interpolate=True)
    # Negative side (left)
    mask_neg = E < 0
    ax.fill_betweenx(par.y, 0, E,
                     where=mask_neg,
                     color=colors_set['light_red'],
                     alpha=0.7,
                     interpolate=True)
    #area legend patches
    '''
    fill_betweenx ne crée pas automatiquement de labels utilisables par la légende
    donc on doit créer un "faux objet graphique" juste pour la légende
    '''
    positive_patch = mpatches.Patch(color=colors_set['light_green'], 
                                    label=r'work $> 0$', 
                                    alpha=0.7)

    negative_patch = mpatches.Patch(color=colors_set['light_red'], 
                                    label=r'work $< 0$', 
                                    alpha=0.7)


    ax.vlines(0, ymin=0, ymax=par.s, colors='k', linestyles='--', lw=0.8)
    ax.set_xlim((-max(np.abs(E)*1.3),max(np.abs(E)*1.3)))
    ax.set_xlabel('Work of aerodynamic forces on a period [J]')
    ax.set_ylabel('Spanwise location y [m]')
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Fusion des légendes
    handles, labels = ax.get_legend_handles_labels()
    handles += [positive_patch, negative_patch]

    ax.legend(handles=handles,fontsize=8)

    if suptitle:
        fig.suptitle(suptitle)

    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')
    plt.show()

def _plot_aero_work_on_ax(ax, par, U, E_w, E_a, E, show_labels=True, show_legend=True):
    ax.plot(E_w, par.y, lw=1, color=colors_set['strong_blue'],  label=r'bending work')
    ax.plot(E_a, par.y, lw=1, color=colors_set['strong_purple'], label=r'torsion work')
    ax.plot(E, par.y, lw=2, color='k', label='total work')

    mask_pos = E > 0
    mask_neg = E < 0

    ax.fill_betweenx(par.y, 0, E,
                     where=mask_pos,
                     color=colors_set['light_green'],
                     alpha=0.7,
                     interpolate=True)

    ax.fill_betweenx(par.y, 0, E,
                     where=mask_neg,
                     color=colors_set['light_red'],
                     alpha=0.7,
                     interpolate=True)

    # Patches pour la légende
    positive_patch = mpatches.Patch(color=colors_set['light_green'],
                                    label=r'work $> 0$',
                                    alpha=0.7)

    negative_patch = mpatches.Patch(color=colors_set['light_red'],
                                    label=r'work $< 0$',
                                    alpha=0.7)

    ax.vlines(0, ymin=0, ymax=par.s, colors='k', linestyles='--', lw=0.8)
    ax.set_xlim((-max(np.abs(E)*1.3), max(np.abs(E)*1.3)))
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if show_labels:
        ax.set_xlabel('Work of aerodynamic forces on a period [J]')
        ax.set_ylabel('Spanwise location y [m]')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles += [positive_patch, negative_patch]
        ax.legend(handles=handles, fontsize=9, loc ='best')    

def plot_two_aero_work_distributions(par, 
                                      U1, E_w1, E_a1, E1,
                                      U2, E_w2, E_a2, E2,
                                      suptitle: Optional[str] = None,
                                      save=False,
                                      filename="aero_work_distribution_double"):

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    # Graphe de gauche : avec labels + légende
    _plot_aero_work_on_ax(axs[0], par, U1, E_w1, E_a1, E1,
                          show_labels=True,
                          show_legend=True)

    # Graphe de droite : sans labels, sans légende
    _plot_aero_work_on_ax(axs[1], par, U2, E_w2, E_a2, E2,
                          show_labels=False,
                          show_legend=False)
    axs[1].set_xlabel('Work of aerodynamic forces on a period [J]')

    if suptitle is not None:
        fig.suptitle(suptitle)

    if save:
        os.makedirs('images', exist_ok=True)
        fig.savefig(os.path.join('images', f"{filename}.pdf"), bbox_inches='tight')

    plt.show()


''' ANIMATION PLOTS'''
def animate_beam(par,t,X,U=None,n_stations=15,interval=30,
                 scale_w=1.0,scale_alpha=1.0,scale_chord=1.0,
                 show_airfoil=False,airfoil_points=60,
                 x_ref='EA',            # 'EA' ou 'CG' comme point de rotation visuelle
                 ylim=None,
                 xlim=None,
                 save_path=None,        # '.mp4' ou '.gif' si tu veux enregistrer
                 repeat=True,
):
    """
    Anime la poutre à partir de X(t): ligne d'axe w(y,t) + cordes locales orientées par alpha(y,t).

    - par.airfoil fournit la géométrie NACA (c, x_ea, x_cg, h(x)).
    - On projette en 2D (plan y–z). La torsion crée un décalage vertical ~ (x - x_ea)*sin(alpha).

    Paramètres
    ----------
    par : ModelParameters (cf. input.py)
    t : (nt,) array
    X : (nt, 2*(Nv+Nw+Nalpha)) array (sortie integrate_state_rk)
    U : float ou None (annot)
    n_stations : nb de stations spanwise où dessiner les cordes locales
    interval : ms entre frames (matplotlib.animation)
    scale_w : facteur d’échelle sur w (visuel)
    scale_alpha : facteur d’échelle sur alpha (visuel)
    scale_chord : facteur d’échelle visuel de la corde (longueur affichée)
    show_airfoil : si True, projette un petit contour 2D (haut/bas) à chaque station
    airfoil_points : nb de points le long de la corde pour le contour 2D
    x_ref : 'EA' ou 'CG' (point de pivot pour la visualisation)
    ylim, xlim : limites axes (z et y)
    save_path : chemin pour enregistrer (None pour ne pas enregistrer)
    repeat : boucle l’animation

    Retour
    ------
    fig, ani
    """
    from ROM import _modal_to_physical_fields, X_to_q  # import local pour éviter cycles

    t = np.asarray(t).ravel()
    nt = t.size
    if X.shape[0] != nt:
        raise ValueError("X and t must have the same dimensions.")

    # we get the generalized coordinates from the state vector X(t)
    qw_t, qa_t = X_to_q(par = par, X=X, t=t)

    # Reconstruction champs (nt, Ny)
    w_map, a_map = _modal_to_physical_fields(par, qw_t, qa_t)
    y = np.asarray(par.y, dtype=float).ravel()
    Ny = y.size
    if w_map.shape[0] != nt or w_map.shape[1] != Ny:
        raise ValueError("Dimensions de w_map inattendues.")
    if a_map.shape[0] != nt or a_map.shape[1] != Ny:
        raise ValueError("Dimensions de a_map inattendues.")

    # Géométrie NACA
    af = par.airfoil
    c = float(af.c)
    x_ea = float(af.x_ea)
    x_cg = float(af.x_cg)
    pivot_x = x_ea if (str(x_ref).upper() == 'EA') else x_cg

    # Stations pour dessiner les cordes
    n_st = int(max(2, min(n_stations, Ny)))
    idx_st = np.linspace(0, Ny - 1, n_st, dtype=int)
    y_st = y[idx_st]

    # Prépare figure
    fig, ax = plt.subplots(figsize=(10, 4))
    title = ax.set_title("Structural animation")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    if xlim is None:
        ax.set_xlim(y[0], y[-1])
    else:
        ax.set_xlim(*xlim)

    # Estimation basique des bornes z si non données
    z_est = scale_w * np.max(np.abs(w_map))
    chord_span = scale_chord * (c) * 0.5  # excursion verticale max approx due à torsion
    z_pad = 0.1 * max(1.0, z_est + chord_span)
    if ylim is None:
        ax.set_ylim(-z_est - chord_span - z_pad, z_est + chord_span + z_pad)
    else:
        ax.set_ylim(*ylim)

    # Ligne d'axe (w)
    center_line, = ax.plot([], [], 'k-', lw=1.8, label='centerline w(y,t)')

    # Cordes locales (LE–TE) + marqueurs EA/CG
    chord_lines = []
    ea_points = []
    cg_points = []
    for _ in idx_st:
        ln, = ax.plot([], [], color='tab:blue', lw=1.2, alpha=0.9)
        chord_lines.append(ln)
        ea, = ax.plot([], [], 'ro', ms=3, alpha=0.8)  # EA
        ea_points.append(ea)
        cg, = ax.plot([], [], 'go', ms=3, alpha=0.8)  # CG
        cg_points.append(cg)

    # Option: petit contour airfoil projeté (yz)
    af_lines = []
    if show_airfoil:
        x_samp = np.linspace(0.0, c, int(max(airfoil_points, 20)))
        z_upper = +np.interp(x_samp, af.x, af.h)  # h(x) déjà en mètres
        z_lower = -z_upper
        for _ in idx_st:
            up, = ax.plot([], [], color='0.5', lw=0.8, alpha=0.6)
            lo, = ax.plot([], [], color='0.5', lw=0.8, alpha=0.6)
            af_lines.append((up, lo))
    else:
        x_samp = None
        z_upper = None
        z_lower = None

    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc='upper right', frameon=False)

    def init():
        center_line.set_data([], [])
        for ln in chord_lines:
            ln.set_data([], [])
        for p in ea_points + cg_points:
            p.set_data([], [])
        if show_airfoil:
            for up, lo in af_lines:
                up.set_data([], [])
                lo.set_data([], [])
        return [center_line, *chord_lines, *ea_points, *cg_points] + ([l for pair in af_lines for l in pair] if show_airfoil else [])

    def update(frame):
        w = scale_w * w_map[frame, :]               # (Ny,)
        a = scale_alpha * a_map[frame, :]           # (Ny,)

        # Ligne d’axe
        center_line.set_data(y, w)

        # Cordes locales
        for k, j in enumerate(idx_st):
            yj = y_st[k]
            wj = w[j]
            aj = a[j]

            # Segment LE–TE (projection 2D yz) pivoté autour de pivot_x
            # z(x) = wj + (x - pivot_x) * sin(alpha_j) ; x projeté via scale_chord
            x_le, x_te = 0.0, c
            z_le = wj + scale_chord * (x_le - pivot_x) * np.sin(aj)
            z_te = wj + scale_chord * (x_te - pivot_x) * np.sin(aj)
            chord_lines[k].set_data([yj, yj], [z_le, z_te])

            # EA / CG (z au pivot_x et au x_cg)
            z_ea = wj + scale_chord * (x_ea - pivot_x) * np.sin(aj)
            z_cg = wj + scale_chord * (x_cg - pivot_x) * np.sin(aj)
            ea_points[k].set_data([yj], [z_ea])
            cg_points[k].set_data([yj], [z_cg])

            # Contour airfoil projeté
            if show_airfoil:
                # rotation petite mais on garde sin/cos pour robustesse
                z_up = wj + scale_chord * (x_samp - pivot_x) * np.sin(aj) + z_upper * np.cos(aj)
                z_lo = wj + scale_chord * (x_samp - pivot_x) * np.sin(aj) + z_lower * np.cos(aj)
                # on trace en yz: y = const, z = f(x)
                # pour alléger, on ne montre que l’épaisseur (projection z)
                af_lines[k][0].set_data(np.full_like(x_samp, yj), z_up)
                af_lines[k][1].set_data(np.full_like(x_samp, yj), z_lo)

        # Titre
        if U is not None:
            title.set_text(f"Structural mode animation  |  U = {U} m/s  |  t = {t[frame]:.3f} s")
        else:
            title.set_text(f"Structural mode animation  |  t = {t[frame]:.3f} s")

        return [center_line, *chord_lines, *ea_points, *cg_points] + ([l for pair in af_lines for l in pair] if show_airfoil else [])

    ani = animation.FuncAnimation(
        fig, update, frames=nt, init_func=init, interval=interval, blit=True, repeat=repeat
    )

    if save_path:
        ext = str(save_path).lower().rsplit('.', 1)[-1]
        if ext == 'gif':
            ani.save(save_path, writer='pillow', fps=max(1, int(1000 / interval)))
        else:
            ani.save(save_path, writer='ffmpeg', fps=max(1, int(1000 / interval)))

    return fig, ani


# x = np.linspace(0,2*np.pi,100)
# y1 = np.cos(x)
# y2 = np.cos(x+np.pi/2)

# fig, ax = plt.subplots()
# ax.plot(x, y1, label='cos(x)')
# ax.plot(x, y2, label='cos(x + π/2)')
# fig.savefig('images/trig_functions.pdf')
