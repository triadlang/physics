"""Regime gallery.

Every named regime in `triad_physics.regimes` is the same Triad equation
with different parameters. We run them all on the 1D solver and tile
their spacetime |Psi(x,t)|^2 heatmaps side by side, so the visual
fingerprint of each regime is directly comparable.

This intentionally skips the 3D regime (`B0_3d`) since it doesn't share
the same spacetime view.
"""
from __future__ import annotations
import os
import numpy as np
from triad_physics import integrate, list_regimes, resolve_regime, observables

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    regimes = [r for r in list_regimes() if r != 'B0_3d']
    print(f'[gallery] running {len(regimes)} regimes ...')

    # use compact shared grid for runtime, all defaults otherwise.
    results = []
    for name in regimes:
        p = resolve_regime(name, N=128, dt=0.005)
        p.backend = 'cpu'
        p.record_every = max(1, int(round(p.T / p.dt / 80)))   # ~80 frames
        r = integrate(p)
        cry = observables.crystallinity(r['psi_final'], r['dx'])
        nrm = observables.norm(r['psi_final'], r['dx'])
        print(f'  {name:24s}  norm={nrm:7.3f}  cryst={cry:.4f}  '
              f'frames={r["density"].shape[1]}')
        results.append((name, r, cry, nrm))

    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.6 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis('off')
    for k, (name, r, cry, nrm) in enumerate(results):
        ax = axes[k // cols, k % cols]
        ax.axis('on')
        rho = r['density']
        t = r['t']; x = r['x']
        vmax = np.percentile(rho, 99.5)
        ax.imshow(rho, aspect='auto', origin='lower',
                  extent=[t[0], t[-1], x[0], x[-1]],
                  cmap='viridis', vmin=0.0, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(f'{name}\ncryst={cry:.2f}, norm={nrm:.1f}', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(r'Triad regime gallery — $|\Psi(x,t)|^2$ for every named regime',
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'regime_gallery.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[gallery] saved {out}')


if __name__ == '__main__':
    main()
