"""Smoke test — runs the verified B0 regime and checks the published numbers.

Ground truth (from extensive prior testing on TriadLang):
  N=128, T=20, regime_B0 -> crystallinity ~ 0.84, norm ~ 25.6.
"""
from __future__ import annotations
import math
from triad_physics import integrate, resolve_regime, observables


def test_b0_matches_published_numbers():
    p = resolve_regime('B0', seed=0)
    assert p.N == 128 and p.T == 20.0 and p.mode == 'full'
    r = integrate(p)
    nrm = observables.norm(r['psi_final'], r['dx'])
    cry = observables.crystallinity(r['psi_final'], r['dx'])
    # tolerances around the verified values
    assert math.isclose(nrm, 25.65, rel_tol=0.01), f'norm out of band: {nrm}'
    assert math.isclose(cry, 0.84, abs_tol=0.02), f'crystallinity out of band: {cry}'


def test_full_mode_keeps_all_three_pillars():
    """mode='full' must not zero any of the P1/P2/P3 coefficients."""
    from triad_physics.solver import _effective_params
    p = resolve_regime('B0')
    eff = _effective_params(p)
    assert eff['Lambda'] != 0.0,  'P2 (Lambda) was zeroed in full mode'
    assert eff['alpha']  != 0.0,  'P1 (alpha) was zeroed in full mode'
    assert eff['Gamma']  != 0.0,  'P3 (Gamma) was zeroed in full mode'


if __name__ == '__main__':
    test_b0_matches_published_numbers()
    test_full_mode_keeps_all_three_pillars()
    print('smoke tests passed.')
