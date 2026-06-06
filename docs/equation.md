# Triad equation — reference

```
i*hbar * d_t Psi = [ -hbar^2/(2m) * laplacian       (P1)
                     + V_ext                          (P1)
                     + Lambda * |Psi|^2               (P2)
                     + V_mem(y_j)                     (P2)
                     + alpha * (-laplacian)^(sigma/2) (P1, fractional)
                     - i*Gamma ] Psi                  (P3)
                   +  eta                             (P3, FDT noise)
```

Memory fields evolve as an Ornstein-Uhlenbeck process driven by the
current density:

```
d_t y_j = -nu_j * y_j + nu_j * |Psi|^2
V_mem  = sum_j lam_j * y_j
```

The FDT coupling sets the noise amplitude in 1D as

```
f_FDT = 2 * Gamma * dx * kT / hbar
```

so the dissipation and the noise are linked, as required for a thermal
open system.

## Integrator

Strang split-step Fourier method. One step `dt` of the full
P1+P2+P3 propagator is:

1. half kinetic + half fractional + half dissipation, in k-space
2. update memory fields `y_j` by half (exponential decay toward rho)
3. full nonlinear + memory + external potential, in real space
4. update memory fields `y_j` by the other half
5. add FDT noise (complex gaussian, amplitude from `f_FDT`)
6. half kinetic + half fractional + half dissipation, in k-space
7. apply absorbing-boundary mask if enabled

This is unitary in the linear, dissipation-free limit (audit-only mode
`linear`); it conserves the norm to floating-point precision in that
case. With Gamma and FDT noise on, the norm drifts in the way the
fluctuation-dissipation balance requires.

## Modes

- `full`     — the physical model. All three pillars active. Used by
                every regime and every user-facing call.
- `linear`   — Lambda, alpha, Gamma, f_FDT, lam all forced to 0. Exists
                only for the unitarity / dt-convergence audit test in
                the parent TriadLang repository.
- `thermal`  — Lambda, alpha, lam forced to 0; keeps Gamma and f_FDT.
                Same audit role, for the dissipation half-step alone.

`linear` and `thermal` are NOT regimes. They are not selected by any
regime, template, or program. They are visible here only because the
audit test in the upstream TriadLang project needs them.

## Observables shipped here

From `triad_physics.observables`:

- `norm(psi, dx)`            — integral of |Psi|^2 dx
- `peak_density(psi)`        — max of |Psi|^2
- `fwhm(psi, dx)`            — full-width-at-half-maximum of |Psi|^2
- `ipr(psi, dx)`             — inverse participation ratio
- `power_spectrum(psi, dx)`  — (k, |FFT(psi)|^2) sorted by k
- `dominant_wavenumber(psi, dx)`
- `crystallinity(psi, dx)`   — fraction of spectral mass above k_cutoff
- `stabilization_score(obs_t)` — late vs. early coefficient-of-variation
                                  ratio, in [0, 1]
- `energy(psi, dx, hbar, m, Lambda, V_ext, V_mem)` — Hamiltonian energy
- ... and the time-series analysis helpers
  (`time_delay_embedding`, `correlation_dimension`,
   `lyapunov_proxy`, `reservoir_memory_capacity`, etc.)

## Verified numbers

From the B0 regime at N=128, T=20, dt=0.005, seed=0:

```
norm(psi_final, dx)         ~ 25.65
crystallinity(psi_final, dx) ~ 0.84
stabilization_score(peak_t)  ~ 0.82
```

These are reproduced by `tests/test_smoke.py`.
