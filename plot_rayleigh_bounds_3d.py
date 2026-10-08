"""Figure A.1 (fig:spectrum3d) of the paper: the three-dimensional counterpart of plot_rayleigh_bounds_1d.py.

Reads the harmonic-trap sweep written by rq_3d.py (results/spectrum3d/harm*.json)
and draws, against lambda_1(A_beta):

  left   the exact smallest Rayleigh quotient of the Riemannian Hessian for
         every metric, with the two lower bounds for the a_u metric, so that
         the collapse of the gap-only bound is visible;
  right  the exact condition number for every metric, with the reference slopes
         lambda_1^{2/3} and lambda_1^{5/3} that Remark 3.12 predicts for the
         a_u metric and for the Euclidean ones, which carry one extra factor
         lambda_1 in max rho on top of the same lambda_1^{-2/3} in min rho.

The point of the right panel is the Thomas-Fermi metric: its condition number
does not grow. Colours and markers are those of make_figures.py, so a
metric is the same colour here as in Figures 1 to 6.

Writes $FIG_DIR/rayleigh_bounds_3d.png (default figures/) or the path given as
the first argument, and prints the table of what it drew.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'results', 'spectrum3d')
FIG_DIR = os.environ.get('FIG_DIR', os.path.join(HERE, 'figures'))
OUT = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(FIG_DIR, 'rayleigh_bounds_3d.png'))

# (label, colour, marker) -- the palette of make_figures.py
STYLE = {
    'h1':    (r'modified $H^1$',            'tab:blue',   'o'),
    'a0':    (r'$a_0$',                     'tab:red',    's'),
    'a0mod': (r'modified $a_0$',            'tab:green',  '^'),
    'au':    (r'$a_u$',                     'tab:purple', 'v'),
    'a3u':   (r'modified $a_u$',     'tab:orange', 'D'),
    'tf':    ('Thomas-Fermi',               'tab:cyan',   'P'),
}
ORDER = ['h1', 'a0', 'a0mod', 'au', 'a3u', 'tf']   # c=2 not drawn

# Every point on the same grid. Several values of beta have been run at more
# than one resolution as mesh checks; keeping a finer one for a single beta
# would put a resolution change inside a curve whose slope is the whole point,
# so the modal grid wins and the others are reported as checks instead.
cand = {}
for f in sorted(glob.glob(os.path.join(DATA, 'harm*.json'))):
    d = json.load(open(f))
    if d.get('potential') != 'harmonic':
        continue
    cand.setdefault(d['beta'], []).append(d)
if not cand:
    sys.exit(f'no harmonic-trap runs in {DATA}')
ns = [d['n'] for v in cand.values() for d in v]
grid = max(set(ns), key=ns.count)
runs, checks = [], []
for b in sorted(cand):
    same = [d for d in cand[b] if d['n'] == grid]
    if same:
        runs.append(same[0])
        checks += [d for d in cand[b] if d['n'] != grid]
    else:                              # no run at the modal grid for this beta
        runs.append(sorted(cand[b], key=lambda d: d['n'])[-1])
runs.sort(key=lambda d: d['beta'])

beta = np.array([d['beta'] for d in runs])
lam1 = np.array([d['lam1'] for d in runs])
lam2 = np.array([d['lam2'] for d in runs])
lam3 = np.array([d['lam1_A3'] for d in runs])
Lam = np.maximum(lam2, lam3)
lb_gap = 1 - lam1 / lam2
lb_new = 1 - lam1 / Lam
have = [m for m in ORDER if all(m in d['metrics'] for d in runs)]

print(f'all points on the {grid}^3 grid'
      + (f'; {len(checks)} mesh check(s) held out: '
         + ', '.join(f"beta={c['beta']:g} at n={c['n']}" for c in checks)
         if checks else ''))
print(f"{'beta':>9s} {'n':>4s} {'lam1':>8s} {'lb_gap':>10s} {'lb_new':>8s} | "
      + ' '.join(f'{m:>8s}' for m in have))
for i, d in enumerate(runs):
    print(f"{d['beta']:9.0f} {d['n']:4d} {lam1[i]:8.3f} {lb_gap[i]:10.3e} "
          f"{lb_new[i]:8.4f} | "
          + ' '.join(f"{d['metrics'][m]['kappa']:8.3f}" for m in have))

XLAB = r'$\lambda_1(A_\beta(\mathbf{u}^*))$'


def fmt_beta(b):
    e = int(np.floor(np.log10(b) + 1e-9))
    m = b / 10.0 ** e
    if e <= 1:
        return f'${b:g}$'
    if abs(m - 1) < 1e-9:
        return r'$10^{%d}$' % e
    return r'$%g{\times}10^{%d}$' % (m, e)


def beta_axis(ax, lam1, betas):
    """A second abscissa naming the beta each marker was computed at.

    Both panels are plotted against lambda_1(A_beta(u*)), which is what the
    bounds are stated in and what the layer scale lambda_1^{1/3} refers to,
    but that leaves a reader unable to say which interaction strength a marker
    is. The two increase together, so the ticks sit at the computed lambda_1
    and carry beta as their label. Minor ticks are off: a decade of this axis
    is not a decade of beta. The labels are rotated because in three
    dimensions lambda_1 grows like beta^{2/5}, so ten values of beta spanning
    five decades land inside 1.6 decades of the abscissa.
    """
    top = ax.secondary_xaxis('top')
    top.set_xticks(list(lam1))
    top.set_xticklabels([fmt_beta(b) for b in betas], fontsize=7,
                        rotation=55, ha='left', rotation_mode='anchor')
    top.minorticks_off()
    top.tick_params(axis='x', pad=1.5)
    top.set_xlabel(r'interaction strength $\beta$', fontsize=9, labelpad=4)


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]

print('\nlog-log slopes of kappa against lambda_1:')
for m in have:
    k = np.array([d['metrics'][m]['kappa'] for d in runs])
    print(f"  {m:6s} {slope(lam1, k):+.3f}"
          + ('   (a_u: +2/3 predicted)' if m == 'au' else '')
          + ('   (bounded, so 0)' if m == 'tf' else ''))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

for m in have:
    lab, col, mk = STYLE[m]
    r = np.array([d['metrics'][m]['rmin'] for d in runs])
    ax[0].loglog(lam1, r, color=col, marker=mk, ls='-', mfc='none',
                 ms=5, lw=1.2, label=lab)
ax[0].loglog(lam1, lb_new, 'k--', lw=1.1,
             label=r'bound $1-\lambda_1/\Lambda$, $a_u$')
ax[0].loglog(lam1, lb_gap, 'k:', lw=1.4,
             label=r'gap bound $1-\lambda_1/\lambda_2$')
ax[0].set_xlabel(XLAB)
ax[0].set_ylabel(r'exact $\rho_{\min}$ at the ground state')
ax[0].legend(fontsize=7.5, loc='lower left', ncol=2)
ax[0].grid(alpha=.3, which='both')

for m in have:
    lab, col, mk = STYLE[m]
    k = np.array([d['metrics'][m]['kappa'] for d in runs])
    ax[1].loglog(lam1, k, color=col, marker=mk, ls='-', mfc='none',
                 ms=5, lw=1.2, label=lab)
# the held-out mesh checks, as open black rings: if the grid mattered they
# would sit off the curves
for c in checks:
    for m in have:
        if m in c['metrics']:
            ax[1].loglog([c['lam1']], [c['metrics'][m]['kappa']], 'ko',
                         mfc='none', ms=10, mew=0.8, zorder=5,
                         label=(r'mesh checks, $%d^3$ and $%d^3$'
                                % (min(x['n'] for x in checks),
                                   max(x['n'] for x in checks))
                                if (c is checks[0] and m == have[0]) else None))
kau = np.array([d['metrics']['au']['kappa'] for d in runs])
ka0 = np.array([d['metrics']['a0']['kappa'] for d in runs])
ax[1].loglog(lam1, kau[-1] * (lam1 / lam1[-1]) ** (2 / 3), 'k--', lw=1.1,
             label=r'$\propto\lambda_1^{2/3}$')
ax[1].loglog(lam1, ka0[-1] * (lam1 / lam1[-1]) ** (5 / 3), 'k:', lw=1.4,
             label=r'$\propto\lambda_1^{5/3}$')
ax[1].set_xlabel(XLAB)
ax[1].set_ylabel(r'condition number $\kappa$ of the Riemannian Hessian')
ax[1].legend(fontsize=7.5, loc='upper left', ncol=2)
ax[1].grid(alpha=.3, which='both')

for a in ax:
    beta_axis(a, lam1, beta)
plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
plt.savefig(OUT, dpi=160)
print('\nsaved', OUT)
