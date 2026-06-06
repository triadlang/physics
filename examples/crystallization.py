"""Emergent crystallization.

Start from a random complex field (chaos). The Triad model has no rule
saying "form a crystal". Yet the combination

    P1 (alpha, sigma)  - dispersion that mixes spatial scales,
    P2 (Lambda, lam)   - nonlinear self-interaction with memory feedback,
    P3 (Gamma, f_FDT)  - dissipation balanced by FDT noise,

drives the field toward a structured, stationary profile. This is the
literal meaning of "crystallization is emergent, not imposed; it
calibrates itself without imposing rules".

What this script measures, all from the recorded density tensor:

  - peak |Psi|^2(t)          : does the amplitude settle?
  - crystallinity(t)         : spectral mass above |k| > k_cutoff.
  - stabilization_score      : how stationary the late half is vs.
                                the early half (CV-based score in [0,1]).

Output: spacetime heatmap + two observable traces vs. time.
"""
from __future__ import annotations
import os
import numpy as np
from triad_physics import TriadParams, integrate, observables
from triad_physics.plotting import ensure_outdir, plot_density_spacetime

OUTDIR = ensure_outdir(os.path.join(os.path.dirname(__file__), '..', 'outputs'))


def random_seed_psi(N: int, L: float, rng) -> np.ndarray:
    """A field of small complex gaussian noise — no structure at all."""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = float(x[1] - x[0])
    re = rng.standard_normal(N)
    im = rng.standard_normal(N)
    psi = (re + 1j * im).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi


def main():
    p = TriadParams(
        L=32.0, N=256, dt=0.0025, T=20.0,
        hbar=1.0, m=1.0,
        V_ext=None, omega=0.0,
        Lambda=-8.0,                   # P2 cubic
        alpha=0.12, sigma=1.8,         # P1 fractional dispersion close to wave-like
        Gamma=0.01, f_FDT=0.001,       # P3 open system
        nu=(10.0, 0.5),
        lam=(1.125, 0.375),
        mode='full',
        seed=7,
        record_every=10,
        backend='cpu',
    )
    rng = np.random.default_rng(p.seed)
    psi0 = random_seed_psi(p.N, p.L, rng)

    print('[crystallization] integrating from random initial condition ...')
    r = integrate(p, psi0=psi0)

    rho = r['density']
    t = r['t']
    dx = float(r['dx'])

    # per-frame observables. crystallinity here is computed from sqrt(rho), so
    # phase is dropped. that is fine: the spectral-mass-above-k_cutoff measure
    # is dominated by |Psi|, and the qualitative shape of the trace is the
    # quantity of interest. final number below uses the full complex Psi.
    peak_t = rho.max(axis=0)
    cryst_t = np.array([
        observables.crystallinity(np.sqrt(rho[:, j]).astype(np.complex128), dx)
        for j in range(rho.shape[1])
    ])
    stab = observables.stabilization_score(peak_t)

    cry_final = observables.crystallinity(r['psi_final'], dx)
    nrm_final = observables.norm(r['psi_final'], dx)
    print(f'  final: norm={nrm_final:.4f}  crystallinity={cry_final:.4f}')
    print(f'  stabilization_score(peak) = {stab:.4f}')

    plot_density_spacetime(
        r,
        title=(r'Crystallization from chaos, full P1+P2+P3'
               f'\nfinal crystallinity = {cry_final:.3f},  '
               f'stabilization score (peak) = {stab:.3f}'),
        out_path=os.path.join(OUTDIR, 'crystallization_spacetime.png'),
        cmap='turbo',
    )

    # observables vs time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(t, peak_t, color='C0')
    axes[0].set_ylabel(r'$\max_x |\Psi|^2$')
    axes[0].set_title('Crystallization — chaos becomes a stationary structure')
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, cryst_t, color='C2')
    axes[1].set_ylabel('crystallinity')
    axes[1].set_xlabel('time $t$')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'crystallization_observables.png'), dpi=130)
    plt.close(fig)

    print(f'[crystallization] saved 2 figures to {OUTDIR}')


if __name__ == '__main__':
    main()
