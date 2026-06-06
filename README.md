# triad-physics

The Triad-equation PDE solver and its visualisations, extracted from
[TriadLang](https://triadlang.com) into a self-contained Python package.

If you want to run the physics, study the regimes, or generate figures
for a paper or a talk, this is the smallest repo that does it. It carries
no compiler, no parser, no .tri frontend, no ML stack. Just the equation,
the integrator, the observables, and a handful of simulations.

## The equation

```
i*hbar * d_t Psi = [ -hbar^2/(2m) * laplacian
                     + V_ext
                     + Lambda * |Psi|^2
                     + V_mem(y_j)
                     + alpha * (-laplacian)^(sigma/2)
                     - i*Gamma ] Psi  +  eta
```

The three pillars are always active together:

| pillar | role                          | parameters                |
|--------|-------------------------------|---------------------------|
| **P1** | phase, oscillation, dispersion | `alpha`, `sigma`, `m`, `hbar` |
| **P2** | self-reference, memory         | `Lambda`, `lam`, `nu`     |
| **P3** | coupling, dissipation, noise   | `Gamma`, `f_FDT`, `kT`    |

`mode='full'` is the only physically used mode. `linear` and `thermal`
exist solely as an audit harness in the parent project; they are not a
regime selector and are never reached from user code.

## Install

```bash
git clone <this-repo> triad-physics
cd triad-physics
pip install -e .
```

Dependencies: `numpy`, `matplotlib`. Optional: `cupy-cuda12x` for GPU.

## Smallest possible example

```python
from triad_physics import resolve_regime, integrate, observables

p = resolve_regime('B0', seed=0)        # the calibrated baseline regime
r = integrate(p)                         # runs the real PDE solver
print(observables.norm(r['psi_final'], r['dx']))           # ~ 25.65
print(observables.crystallinity(r['psi_final'], r['dx']))  # ~ 0.84
```

These two numbers (norm ~25.65, crystallinity ~0.84) come straight from
the solver. They are not analytic, not invented, not curve-fits. The
smoke test in `tests/test_smoke.py` re-runs the regime and verifies them.

## Example simulations

All scripts under `examples/` drive the real solver and write PNGs to
`outputs/`. They use no synthetic data anywhere.

```bash
PYTHONPATH=. python examples/anti_collapse.py
PYTHONPATH=. python examples/two_gaussians.py
PYTHONPATH=. python examples/crystallization.py
PYTHONPATH=. python examples/triad_2d.py
PYTHONPATH=. python examples/regime_gallery.py
```

| script | what it shows | figures |
|--------|---------------|---------|
| `anti_collapse.py`   | strong attractive cubic balanced by P1+P2; a side-by-side run with weakened P1/P2 shows the peak running away | spacetime, log-y peak overlay, snapshots |
| `two_gaussians.py`   | counter-propagating gaussians, interference fringes, memory-driven rebinding | spacetime, snapshots, real/imag of final Psi |
| `crystallization.py` | random initial field stabilises into structure; chaos becomes a stationary pattern | spacetime + peak and crystallinity vs. time |
| `triad_2d.py`        | the same two phenomena in 2D, with frame strips over time | final 2D density + 5-frame strip |
| `regime_gallery.py`  | one tile per named regime, every one is a real solver run | a 4x4 panel |

## The principle, in one sentence

Start with chaos. Stabilise. The crystallisation is emergent, not
imposed. The system calibrates itself without imposing rules — the only
thing imposed is the equation, and the equation is the same in every
regime.

## What lives where

```
triad_physics/
  backend.py        numpy / cupy dispatch
  solver.py         the Strang split-step Fourier integrator (1D, 2D, 3D)
  observables.py    norm, peak, crystallinity, stabilization, energy, ...
  regimes.py        16 named regimes, all P1+P2+P3-compliant
  plotting.py       shared matplotlib helpers for the examples
examples/           runnable simulations
outputs/            figures (.gitignore-friendly)
tests/test_smoke.py B0 verifies against the published numbers
native_c/           drop libtriad_rt.so here (optional, env: TRIAD_NATIVE_SO)
```

## Repository origin

This repo carries only the physics layer of TriadLang. The full project
(parser, compiler, ML, CLI, examples in `.tri`) lives at the upstream
TriadLang repository. The solver and regimes here are an unmodified
copy of `runtime/core/solver.py`, `runtime/physics/observables.py`,
`runtime/backend.py` and `stdlib/regimes.py` with imports rewritten to
the flat package layout and the GPU-fused / native-C resolution made
self-contained.
