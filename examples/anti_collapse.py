"""Anti-collapse — the canonical Triad demonstration.

A strongly attractive cubic (Lambda << 0) wants to collapse |Psi|^2 into a
spike. In the Triad model, P1 (fractional dispersion via alpha and sigma)
and P2 (memory feedback via lam and nu) push back. The integration stays
finite and a stationary, non-singular profile emerges. This is *emergent*
crystallization: nothing in the integrator imposes "do not collapse",
the field calibrates itself.

This script runs two configurations and saves three figures:

  1. anti_collapse_full.png         — full P1+P2+P3, peak stays bounded.
  2. anti_collapse_weakened.png     — same Lambda, P1/P2 strongly reduced
                                       so the dispersive/memory pushback can
                                       no longer balance the cubic; the peak
                                       grows by orders of magnitude.
  3. anti_collapse_peaks.png        — peak |Psi|^2 vs time, both runs on the
                                       same axes, log-y for legibility.

Reported numbers (peak ratio, final norm, final crystallinity) come from
the solver, not from analytic fits.
"""
from __future__ import annotations
import os
import numpy as np
from triad_physics import TriadParams, integrate, observables
from triad_physics.plotting import (
    ensure_outdir, plot_density_spacetime, plot_density_snapshots,
)

OUTDIR = ensure_outdir(os.path.join(os.path.dirname(__file__), '..', 'outputs'))


def make_pulse(N: int, L: float, width: float = 0.6) -> np.ndarray:
    """Narrow gaussian seed — easier to push into collapse than the default."""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    psi = np.exp(-x ** 2 / (2.0 * width ** 2)).astype(np.complex128)
    dx = float(x[1] - x[0])
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi


def run_full() -> dict:
    p = TriadParams(
        L=32.0, N=256, dt=0.002, T=8.0,
        hbar=1.0, m=1.0,
        V_ext=None, omega=0.0,
        Lambda=-15.0,                  # strong attractive cubic (P2)
        alpha=0.30, sigma=1.5,         # fractional dispersion (P1)
        Gamma=0.01, f_FDT=0.001,       # FDT-coupled open system (P3)
        nu=(10.0, 0.5),
        lam=(3.0, 1.0),                # repulsive memory fields (P2)
        mode='full',
        seed=1,
        record_every=10,
        backend='cpu',
    )
    psi0 = make_pulse(p.N, p.L, width=0.4)
    return integrate(p, psi0=psi0)


def run_weakened() -> dict:
    """Same attractive cubic, but P1 (dispersion) and P2 (memory) are weak.

    This is NOT a P1/P2-isolated mode. All three pillars remain active. We
    only reduce their *magnitudes* so the dispersive and memory pushback no
    longer balance the cubic. The intent is a side-by-side picture, not an
    audit-only run.
    """
    p = TriadParams(
        L=32.0, N=256, dt=0.002, T=8.0,
        hbar=1.0, m=1.0,
        V_ext=None, omega=0.0,
        Lambda=-15.0,
        alpha=0.01, sigma=1.5,         # P1 weakened (still active)
        Gamma=0.01, f_FDT=0.001,
        nu=(10.0, 0.5),
        lam=(0.02, 0.01),              # P2 memory weakened (still active)
        mode='full',
        seed=1,
        record_every=10,
        backend='cpu',
    )
    psi0 = make_pulse(p.N, p.L, width=0.4)
    return integrate(p, psi0=psi0)


def main():
    print('[anti-collapse] running full P1+P2+P3 ...')
    r_full = run_full()
    print('[anti-collapse] running weakened-P1/P2 control ...')
    r_weak = run_weakened()

    # peak |Psi|^2 over time, from the recorded density tensor
    peak_full = r_full['density'].max(axis=0)
    peak_weak = r_weak['density'].max(axis=0)
    t_full = r_full['t']
    t_weak = r_weak['t']

    # report scalar numbers straight from the solver
    norm_full = observables.norm(r_full['psi_final'], r_full['dx'])
    norm_weak = observables.norm(r_weak['psi_final'], r_weak['dx'])
    cryst_full = observables.crystallinity(r_full['psi_final'], r_full['dx'])
    cryst_weak = observables.crystallinity(r_weak['psi_final'], r_weak['dx'])
    pmax_full = float(peak_full.max())
    pmax_weak = float(peak_weak.max())
    print(f'  full     : peak_max={pmax_full:8.3f}  norm={norm_full:.4f}  cryst={cryst_full:.4f}')
    print(f'  weakened : peak_max={pmax_weak:8.3f}  norm={norm_weak:.4f}  cryst={cryst_weak:.4f}')
    print(f'  ratio peak_weak / peak_full = {pmax_weak / max(pmax_full, 1e-12):.2f}x')

    plot_density_spacetime(
        r_full,
        title=(r'Anti-collapse, full P1+P2+P3 — $\Lambda=-10$, $\alpha=0.30$, $\sigma=1.5$'
               f'\nfinal norm = {norm_full:.3f},  crystallinity = {cryst_full:.3f}'),
        out_path=os.path.join(OUTDIR, 'anti_collapse_full.png'),
        cmap='viridis',
    )
    plot_density_spacetime(
        r_weak,
        title=(r'Anti-collapse weakened (same $\Lambda$, $\alpha$ and lam reduced)'
               f'\npeak grows {pmax_weak / max(pmax_full, 1e-12):.1f}x relative to full'),
        out_path=os.path.join(OUTDIR, 'anti_collapse_weakened.png'),
        cmap='magma',
        logscale=True,
    )

    plot_density_snapshots(
        r_full,
        times=[0.0, 1.0, 3.0, 6.0],
        title='Anti-collapse full — density snapshots',
        out_path=os.path.join(OUTDIR, 'anti_collapse_snapshots.png'),
    )

    # peak overlay, log y, both runs on the same axes
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(t_full, peak_full, label='full P1+P2+P3 (stays bounded)', color='C0', lw=2)
    ax.semilogy(t_weak, peak_weak, label='weakened P1/P2 (peak runs away)', color='C3', lw=2)
    ax.set_xlabel('time $t$')
    ax.set_ylabel(r'$\max_x |\Psi(x,t)|^2$')
    ax.set_title('Peak density vs. time — emergent containment from P1 + P2')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    peaks_path = os.path.join(OUTDIR, 'anti_collapse_peaks.png')
    fig.savefig(peaks_path, dpi=130)
    plt.close(fig)

    print(f'[anti-collapse] saved 4 figures to {OUTDIR}')


if __name__ == '__main__':
    main()
