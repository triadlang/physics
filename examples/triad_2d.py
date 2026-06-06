"""2D Triad — anti-collapse and crystallization on a plane.

Two configurations in 2D, both with full P1+P2+P3:

  1. anti_collapse_2d : narrow gaussian seed, strong attractive cubic,
                        fractional dispersion plus memory feedback hold
                        the peak finite and shape it into a structured
                        2D profile.
  2. crystal_2d       : random-field seed, the solver settles into a
                        spatially structured pattern.

For each run we save the final |Psi(x,y)|^2 as an image and, where
record_density is set, also dump a short sequence of frames so the
"chaos -> stabilization" arc is visible as a side-by-side panel.
"""
from __future__ import annotations
import os
import numpy as np
from triad_physics import TriadParams, integrate_2d, observables
from triad_physics.plotting import ensure_outdir, plot_density_2d

OUTDIR = ensure_outdir(os.path.join(os.path.dirname(__file__), '..', 'outputs'))


def gauss_2d(N: int, L: float, width: float = 1.0) -> np.ndarray:
    xs = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    psi = np.exp(-(X * X + Y * Y) / (2.0 * width ** 2)).astype(np.complex128)
    dx = float(xs[1] - xs[0])
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx * dx)
    return psi


def random_field_2d(N: int, L: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = float(xs[1] - xs[0])
    psi = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx * dx)
    return psi


def run_anti_collapse_2d():
    p = TriadParams(
        L=16.0, N=96, dt=0.002, T=3.0,
        hbar=1.0, m=1.0,
        V_ext=None, omega=0.0,
        Lambda=-8.0,
        alpha=0.25, sigma=1.5,
        Gamma=0.02, f_FDT=0.001,
        nu=(8.0, 0.5),
        lam=(2.0, 0.8),
        mode='full',
        seed=3,
        record_every=20,
        backend='cpu',
        D=2,
    )
    psi0 = gauss_2d(p.N, p.L, width=0.6)
    print('[2d-anti-collapse] integrating ...')
    r = integrate_2d(p, psi0=psi0, record_density=True)
    rho = np.abs(r['psi_final']) ** 2
    print(f'  final: peak={rho.max():.4f}  norm={(rho.sum() * r["dx"] ** 2):.4f}')
    plot_density_2d(
        r,
        title=r'2D anti-collapse, final $|\Psi(x,y)|^2$',
        out_path=os.path.join(OUTDIR, 'anti_collapse_2d_final.png'),
        cmap='inferno',
    )
    _save_density_strip(r, os.path.join(OUTDIR, 'anti_collapse_2d_strip.png'),
                        title='2D anti-collapse — density frames over time',
                        cmap='inferno')
    return r


def run_crystal_2d():
    p = TriadParams(
        L=16.0, N=96, dt=0.0025, T=25.0,
        hbar=1.0, m=1.0,
        V_ext=None, omega=0.0,
        Lambda=-6.0,
        alpha=0.10, sigma=1.8,
        Gamma=0.02, f_FDT=0.0001,
        nu=(10.0, 0.5),
        lam=(1.5, 0.5),
        mode='full',
        seed=11,
        record_every=40,
        backend='cpu',
        D=2,
    )
    psi0 = random_field_2d(p.N, p.L, seed=11)
    print('[2d-crystal] integrating from random initial condition ...')
    r = integrate_2d(p, psi0=psi0, record_density=True)
    rho = np.abs(r['psi_final']) ** 2
    print(f'  final: peak={rho.max():.4f}  norm={(rho.sum() * r["dx"] ** 2):.4f}')
    plot_density_2d(
        r,
        title=r'2D crystallization from chaos, final $|\Psi(x,y)|^2$',
        out_path=os.path.join(OUTDIR, 'crystal_2d_final.png'),
        cmap='magma',
    )
    _save_density_strip(r, os.path.join(OUTDIR, 'crystal_2d_strip.png'),
                        title='2D crystallization — chaos becomes structure',
                        cmap='magma')
    return r


def _save_density_strip(r, out_path, title, cmap, n_frames: int = 5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if 'density' not in r:
        return
    rho_t = r['density']               # (N, N, T_rec)
    T_rec = rho_t.shape[2]
    idxs = np.linspace(0, T_rec - 1, n_frames).round().astype(int)
    fig, axes = plt.subplots(1, n_frames, figsize=(2.6 * n_frames, 2.8))
    vmax = float(rho_t[..., idxs].max())
    x = r['x']
    for ax, j in zip(axes, idxs):
        ax.imshow(rho_t[..., j].T, origin='lower',
                  extent=[x[0], x[-1], x[0], x[-1]],
                  cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f't = {r["t"][j]:.2f}', fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    run_anti_collapse_2d()
    run_crystal_2d()
    print(f'[2d] saved figures to {OUTDIR}')


if __name__ == '__main__':
    main()
