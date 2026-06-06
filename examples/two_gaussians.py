"""Two-gaussian collision.

Two counter-propagating gaussian wave packets are launched toward the center
of the box. The Triad system shows three things at once:

  - P1 (kinetic + fractional dispersion) carries the packets across the box
    and produces the interference fringes when they overlap.
  - P2 (Lambda|Psi|^2 plus memory fields y_j) deforms the fringe pattern,
    holds part of the overlap region as a longer-lived structure, and
    feeds the field back into itself.
  - P3 (Gamma + FDT noise) keeps the integration as an open system. The
    norm decays slowly and noise prevents a perfectly clean recurrence.

Output: a spacetime heatmap of |Psi(x,t)|^2 plus three density snapshots
(before / during / after the collision).
"""
from __future__ import annotations
import os
import numpy as np
from triad_physics import TriadParams, integrate, observables
from triad_physics.plotting import (
    ensure_outdir, plot_density_spacetime, plot_density_snapshots,
)

OUTDIR = ensure_outdir(os.path.join(os.path.dirname(__file__), '..', 'outputs'))


def two_gaussians(N: int, L: float, x0: float = 8.0, k0: float = 4.0,
                  width: float = 1.2) -> np.ndarray:
    """Two equal-amplitude gaussians at +/- x0, carrying opposite momenta +/- k0."""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    g_right = np.exp(-(x + x0) ** 2 / (2.0 * width ** 2)) * np.exp(1j * k0 * x)
    g_left  = np.exp(-(x - x0) ** 2 / (2.0 * width ** 2)) * np.exp(-1j * k0 * x)
    psi = (g_right + g_left).astype(np.complex128)
    dx = float(x[1] - x[0])
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi


def main():
    p = TriadParams(
        L=32.0, N=256, dt=0.002, T=8.0,
        hbar=1.0, m=1.0,
        V_ext=None, omega=0.0,
        Lambda=-0.8,                   # mild attractive nonlinearity (P2)
        alpha=0.10, sigma=1.6,         # fractional dispersion (P1)
        Gamma=0.02, f_FDT=0.0008,      # open system (P3)
        nu=(2.0, 0.3),
        lam=(-0.2, -0.1),
        mode='full',
        seed=2,
        record_every=8,
        backend='cpu',
    )
    psi0 = two_gaussians(p.N, p.L, x0=8.0, k0=4.0, width=1.2)
    print('[two-gaussians] integrating ...')
    r = integrate(p, psi0=psi0)

    nrm = observables.norm(r['psi_final'], r['dx'])
    cry = observables.crystallinity(r['psi_final'], r['dx'])
    pk  = observables.peak_density(r['psi_final'])
    print(f'  final: norm={nrm:.4f}  crystallinity={cry:.4f}  peak={pk:.4f}')
    print(f'  density tensor: {r["density"].shape}  (N, T_recorded)')

    plot_density_spacetime(
        r,
        title=(r'Two counter-propagating gaussians, full P1+P2+P3'
               '\ninterference fringes plus memory-driven rebinding'),
        out_path=os.path.join(OUTDIR, 'two_gaussians_spacetime.png'),
        cmap='inferno',
    )
    plot_density_snapshots(
        r,
        times=[0.0, 2.0, 4.0, 6.0, 8.0],
        title='Two gaussians — density snapshots before, during and after collision',
        out_path=os.path.join(OUTDIR, 'two_gaussians_snapshots.png'),
    )

    # also save the real and imaginary parts of psi_final to show phase structure.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = r['x']; psi = r['psi_final']
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(x, np.real(psi), label=r'$\mathrm{Re}\,\Psi$', color='C0')
    axes[0].plot(x, np.imag(psi), label=r'$\mathrm{Im}\,\Psi$', color='C3')
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)
    axes[0].set_ylabel(r'$\Psi(x, T)$')
    axes[0].set_title('Final wavefunction: real + imaginary components')
    axes[1].plot(x, np.abs(psi) ** 2, color='black')
    axes[1].set_xlabel('position $x$'); axes[1].set_ylabel(r'$|\Psi|^2$')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'two_gaussians_phase.png'), dpi=130)
    plt.close(fig)
    print(f'[two-gaussians] saved 3 figures to {OUTDIR}')


if __name__ == '__main__':
    main()
