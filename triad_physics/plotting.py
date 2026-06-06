"""Plotting helpers shared by the example simulations.

Every figure here is rendered from a real solver run. No synthetic curves,
no analytic placeholders. Inputs come straight out of `integrate(...)`.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_density_spacetime(result: dict, title: str, out_path: str,
                           cmap: str = 'viridis', logscale: bool = False,
                           vmax_pct: float = 99.5) -> str:
    """Heatmap of |Psi(x,t)|^2 — the canonical Triad spacetime view."""
    rho = result['density']           # (N, T_rec)
    t = result['t']
    x = result['x']
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = rho.copy()
    if logscale:
        data = np.log10(data + 1e-12)
        cbar_label = r'$\log_{10}|\Psi(x,t)|^2$'
    else:
        cbar_label = r'$|\Psi(x,t)|^2$'
    vmax = np.percentile(data, vmax_pct)
    vmin = data.min()
    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], x[0], x[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('time $t$')
    ax.set_ylabel('position $x$')
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_density_snapshots(result: dict, times, title: str, out_path: str) -> str:
    """Several |Psi(x,t)|^2 snapshots overlaid for direct visual comparison."""
    rho = result['density']
    t = result['t']
    x = result['x']
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for tq in times:
        idx = int(np.argmin(np.abs(t - tq)))
        ax.plot(x, rho[:, idx], label=f't = {t[idx]:.2f}')
    ax.set_xlabel('position $x$')
    ax.set_ylabel(r'$|\Psi(x,t)|^2$')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_observables_vs_time(result: dict, observables_dict: dict,
                             title: str, out_path: str) -> str:
    """Plot one or more scalar observables computed per recorded frame.

    observables_dict: {label: function(psi_slice, dx) -> float}
    psi_slice is reconstructed from sqrt(rho) — this is fine for
    quantities that depend only on |Psi|, but for phase-aware ones
    use the dedicated per-step instrumentation in the example script
    rather than this helper.
    """
    rho = result['density']
    t = result['t']
    dx = float(result['dx'])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, fn in observables_dict.items():
        series = np.array([fn(np.sqrt(rho[:, j]).astype(np.complex128), dx)
                           for j in range(rho.shape[1])])
        ax.plot(t, series, label=label)
    ax.set_xlabel('time $t$')
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_density_2d(result_2d: dict, title: str, out_path: str,
                    cmap: str = 'magma') -> str:
    """Render the final 2D density from `integrate_2d`."""
    psi_final = result_2d['psi_final']
    rho = np.abs(psi_final) ** 2
    x = result_2d['x']
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(rho.T, origin='lower', extent=[x[0], x[-1], x[0], x[-1]],
                   cmap=cmap, interpolation='bilinear')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$|\Psi(x,y)|^2$')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
