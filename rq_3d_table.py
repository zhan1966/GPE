#!/usr/bin/env python3
"""Tables from the three-dimensional spectrum runs of rq_3d.py.

Reads the JSON files written by rq_3d.py (one per run, in
results/spectrum3d/) and prints five things:

  1. the scalar table: lambda_1, the two gaps, sigma_T and sigma_T/lambda_1^{1/3},
     which the paragraph after Corollary 3.11 predicts to approach
     mu_1 = 2.4105;
  2. the per-metric table of exact min rho, max rho and kappa against the
     bounds of Theorem 3.4 with the exact zeta_- and zeta_+, for every run;
  3. the log-log slopes in beta of the harmonic-trap sweep, against the
     Thomas-Fermi scalings in d = 3;
  4. zeta_- of the modified H^1 and a_0 metrics against beta (Section 5.3);
  5. the comparison of the measured kappa on the stirrer at 499^3 with the
     iteration counts of Figure 1, through the asymptotic rate of gradient
     descent and of conjugate gradients (Section 5.2).

Usage:  python3 rq_3d_table.py [dir]      (default results/spectrum3d)
"""
import json
import math
import os
import sys

MU1 = 2.4105          # Corollary 3.11
NAMES = ['h1', 'a0', 'a0mod', 'au', 'a2u', 'a3u', 'tf']
LABEL = {'h1': 'modified $H^1$', 'a0': '$a_0$', 'a0mod': 'modified $a_0$',
         'au': '$a_\\bu$', 'a2u': 'modified $a_\\bu$, $c=2$',
         'a3u': 'modified $a_\\bu$, $c=3$', 'tf': 'Thomas--Fermi'}
# Iterations to 1e-12 (gradient descent, conjugate gradients) of the runs
# stirrer_tf_n499 behind Figure 1 of the paper: 499^3, stirrer, beta = 1600.
TAB2 = {'h1': (169, 31), 'a0': (111, 29), 'a0mod': (122, 31), 'au': (35, 16),
        'a3u': (25, 14), 'a2u': (24, 13), 'tf': (8, 7)}
SWEEP = ['harm1', 'harm10', 'harm1e2', 'harm3e2', 'harm1e3', 'harm3e3',
         'harm1e4', 'harm3e4', 'harm1e5', 'harm3e5']
MESH = ['stir99', 'stir199', 'stir499']


def load(d, tag):
    p = os.path.join(d, f'{tag}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def scalars(runs):
    print('\n=== scalars ===')
    print(f"{'run':14s} {'n':>4s} {'beta':>8s} {'L':>6s} {'lam1':>9s} "
          f"{'lam2-lam1':>10s} {'lam1(A3b)-lam1':>14s} {'sigma_T':>9s} "
          f"{'sT/lam1^1/3':>11s} {'|F|_h':>8s}")
    for t, d in runs:
        l1 = d['lam1']
        print(f"{t:14s} {d['n']:4d} {d['beta']:8.0f} {d['L']:6.2f} {l1:9.4f} "
              f"{d['lam2']-l1:10.4f} {d['lam1_A3']-l1:14.4f} "
              f"{d['sigma_T']:9.4f} {d['sigma_T']/l1**(1/3):11.4f} "
              f"{d['ground_state']['newton_res']:8.1e}")


def metrics(runs):
    for t, d in runs:
        l1, sT = d['lam1'], d['sigma_T']
        print(f"\n=== {t}: n = {d['n']}, beta = {d['beta']:g}, "
              f"lam1 = {l1:.4f}, sigma_T = {sT:.4f}, "
              f"1 - lam1/lam2 = {1-l1/d['lam2']:.3e} ===")
        print(f"  {'metric':7s} {'min rho':>10s} {'max rho':>10s} {'kappa':>9s} | "
              f"{'lower bd':>9s} {'upper bd':>8s} {'kappa bd':>8s} | "
              f"{'zeta-':>8s} {'zeta+':>9s}")
        for m in NAMES:
            v = d['metrics'].get(m)
            if v is None:
                continue
            ub, kb = v['ub'], v['kappa_bound']
            print(f"  {m:7s} {v['rmin']:10.6f} {v['rmax']:10.5f} "
                  f"{v['kappa']:9.4g} | {v['lb']:9.6f} "
                  f"{'--' if ub is None else format(ub, '8.4f')} "
                  f"{'--' if kb is None else format(kb, '8.4g')} | "
                  f"{max(0.0, -v['zeta_min']):8.3f} "
                  f"{max(0.0, v['zeta_max']):9.3f}")


def slopes(d):
    runs = [x for x in [(t, load(d, t)) for t in SWEEP] if x[1]]
    if len(runs) < 3:
        return
    D = [r[1] for r in runs]
    lb = [math.log(x['beta']) for x in D]
    print('\n=== log-log slopes in beta, harmonic trap ===')
    print(f"  lambda_1(A_beta)      {slope(lb, [math.log(x['lam1']) for x in D]):+.4f}"
          f"    (Thomas-Fermi in d = 3: 2/5 = +0.4000)")
    print(f"  lambda_2 - lambda_1   "
          f"{slope(lb, [math.log(x['lam2']-x['lam1']) for x in D]):+.4f}"
          f"    (lambda_1^-1, so -2/5)")
    print(f"  sigma_T               {slope(lb, [math.log(x['sigma_T']) for x in D]):+.4f}"
          f"    (lambda_1^1/3, so 2/15 = +0.1333)")
    for m in NAMES:
        if m not in D[0]['metrics']:
            continue
        k = slope(lb, [math.log(x['metrics'][m]['kappa']) for x in D])
        r = slope(lb, [math.log(x['metrics'][m]['rmin']) for x in D])
        note = '    (a_u: lambda_1^2/3, so 4/15 = +0.2667)' if m == 'au' else ''
        print(f"  kappa_{m:6s}          {k:+.4f}   min rho {r:+.4f}{note}")
    print('\n  sigma_T/lambda_1^(1/3) against mu_1 = 2.4105, and kappa_a_u '
          'against (2/mu_1) lambda_1^(2/3):')
    for x in D:
        p = (2 / MU1) * x['lam1'] ** (2 / 3)
        print(f"    beta {x['beta']:8.0f}: {x['sigma_T']/x['lam1']**(1/3):6.4f}"
              f"     kappa {x['metrics']['au']['kappa']:7.3f} vs {p:7.3f}"
              f"  (ratio {x['metrics']['au']['kappa']/p:.3f})")
    print('\n  the gap bound against the exact min rho of the a_u metric:')
    for x in D:
        g = 1 - x['lam1'] / x['lam2']
        e = x['metrics']['au']['rmin']
        print(f"    beta {x['beta']:8.0f}: 1 - lam1/lam2 = {g:.4e}  exact "
              f"{e:.4f}  factor {e/g:8.1f}   lambda_1^(4/3) = "
              f"{x['lam1']**(4/3):8.1f}")


def against_table2(d):
    x = load(d, 'stir499')
    if x is None:
        return
    print('\n=== measured kappa against the iteration counts of Figure 1 '
          '(499^3, stirrer, beta = 1600) ===')
    print('  If the asymptotic rate governs, 2 k ln(1/r) is the same constant')
    print('  for every metric, with r = (kappa-1)/(kappa+1) for the flow and')
    print('  r = (sqrt(kappa)-1)/(sqrt(kappa)+1) for conjugate gradients.')
    print(f"\n  {'metric':24s} {'kappa':>7s} | {'GD':>4s} {'2k ln(1/r)':>10s} | "
          f"{'CG':>4s} {'2k ln(1/r)':>10s}")
    acc = []
    for m in ['tf', 'a2u', 'a3u', 'au', 'a0', 'a0mod', 'h1']:
        v = x['metrics'].get(m)
        if v is None or m not in TAB2:
            continue
        k = v['kappa']
        r = (k - 1) / (k + 1)
        rc = (math.sqrt(k) - 1) / (math.sqrt(k) + 1)
        g, c = TAB2[m]
        A, B = 2 * g * math.log(1 / r), 2 * c * math.log(1 / rc)
        acc.append((m, A, B))
        print(f"  {LABEL[m]:24s} {k:7.3f} | {g:4d} {A:10.2f} | {c:4d} {B:10.2f}")
    mid = [a for a in acc if a[0] not in ('tf', 'h1')]
    if mid:
        print(f"\n  over the five metrics with 4.5 <= kappa <= 25: "
              f"GD {min(a[1] for a in mid):.1f}-{max(a[1] for a in mid):.1f}, "
              f"CG {min(a[2] for a in mid):.1f}-{max(a[2] for a in mid):.1f}")


def h1_vs_a0(d):
    runs = [x for x in [(t, load(d, t)) for t in SWEEP] if x[1]]
    if not runs:
        return
    print('\n=== why the H^1 flow does not degrade with beta and a_0 does ===')
    print('  Theorem 3.4 gives min rho >= sigma_T/(sigma_T + zeta_-). For the')
    print('  H^1 metric zeta_- <= alpha uniformly in beta, because V + 3 beta u^2')
    print('  >= lambda_1 on the Thomas-Fermi region and V >= lambda_1 outside it;')
    print('  for the a_0 metric zeta_- = lambda_1(A_beta) exactly, attained where')
    print('  the density vanishes. sigma_T grows like lambda_1^{1/3}, so min rho')
    print('  tends to 1 for the first and to 0 for the second.')
    print(f"\n  {'beta':>8s} {'lam1':>8s} {'sigma_T':>8s} | "
          f"{'zeta-(h1)':>9s} {'/alpha':>7s} {'min rho':>8s} {'kappa':>8s} | "
          f"{'zeta-(a0)':>9s} {'/lam1':>7s} {'min rho':>8s} {'kappa':>8s}")
    for _, x in runs:
        h, a = x['metrics']['h1'], x['metrics']['a0']
        print(f"  {x['beta']:8.0f} {x['lam1']:8.3f} {x['sigma_T']:8.4f} | "
              f"{-h['zeta_min']:9.3f} {-h['zeta_min']/x['alpha']:7.4f} "
              f"{h['rmin']:8.4f} {h['kappa']:8.2f} | "
              f"{-a['zeta_min']:9.3f} {-a['zeta_min']/x['lam1']:7.4f} "
              f"{a['rmin']:8.4f} {a['kappa']:8.2f}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results',
        'spectrum3d')
    tags = SWEEP + ['harm1e5_n399'] + MESH
    runs = [(t, load(d, t)) for t in tags]
    runs = [(t, x) for t, x in runs if x is not None]
    if not runs:
        sys.exit(f'no run JSON found in {d}')
    scalars(runs)
    metrics(runs)
    slopes(d)
    h1_vs_a0(d)
    against_table2(d)


if __name__ == '__main__':
    main()
