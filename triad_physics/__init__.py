"""triad_physics — the PDE core of TriadLang as a standalone physics package.

Exports the Triad solver and observables so that simulations and figures
can be built without dragging in the full TriadLang frontend, compiler,
or ML stack.

The model is the Triad equation:

    i*hbar*d_t Psi = [ -hbar^2/(2m) * laplacian
                       + V_ext
                       + Lambda * |Psi|^2
                       + V_mem(y_j)
                       + alpha * (-laplacian)^(sigma/2)
                       - i*Gamma ] Psi + eta

with FDT-coupled noise eta. The three pillars are always active together:

    P1: kinetic + fractional dispersion (alpha, sigma)
    P2: cubic self-interaction Lambda|Psi|^2 + memory fields y_j (lam, nu)
    P3: dissipation Gamma + FDT noise (f_FDT, kT)
"""
from .solver import (
    TriadParams,
    integrate,
    integrate_2d,
    integrate_3d,
    integrate_adaptive,
)
from .regimes import resolve_regime, list_regimes
from . import observables

__all__ = [
    'TriadParams',
    'integrate',
    'integrate_2d',
    'integrate_3d',
    'integrate_adaptive',
    'resolve_regime',
    'list_regimes',
    'observables',
]
