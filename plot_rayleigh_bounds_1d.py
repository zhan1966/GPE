"""The d = 1 rows of Table A.1 (tab:spectrum) of the paper, from the two 1D sweep logs written by rq_1d.py:
   results/spectrum1d/run_small.log  (beta = 1, 10, 100; N = 3000)
   results/spectrum1d/run_large.log  (beta = 1e3 ... 3e5; N = 4000)
Prints the table and the local log-log slopes between consecutive beta (Appendix A of the paper quotes the slopes over
[1e4, 3e5], taken from the same table), writes results/spectrum1d/table1d_rows.tex, and draws the one-dimensional
counterpart of Figure A.1, which is not in the paper, to $FIG_DIR/rayleigh_bounds_1d.png (default figures/) or to the
path given as the first argument."""
import re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'results', 'spectrum1d')
FIG_DIR = os.environ.get('FIG_DIR', os.path.join(HERE, 'figures'))
OUT_PNG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIG_DIR, 'rayleigh_bounds_1d.png')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = {}
for fn in ['run_small.log', 'run_large.log']:
    try:
        txt = open(os.path.join(DATA, fn)).read()
    except FileNotFoundError:
        continue
    blocks = txt.split('=== beta = ')[1:]
    for b in blocks:
        beta = float(b.split()[0])
        N = int(re.search(r'N = (\d+)', b).group(1))
        lam1 = float(re.search(r'lam1\(A_b\) = ([\d.eE+-]+)', b).group(1))
        lam2 = float(re.search(r'lam2\(A_b\) = ([\d.eE+-]+)', b).group(1))
        lam1_A3 = float(re.search(r'lam1\(A_3b\) = ([\d.eE+-]+)', b).group(1))
        mu1 = float(re.search(r'mu1 \(constrained L2-Hessian gap\) = ([\d.eE+-]+)', b).group(1))
        lbp = float(re.search(r'paper 1-lam1/lam2 = ([\d.eE+-]+)', b).group(1))
        lam2 = lam1 / (1.0 - lbp)          # recover lam2 with full precision from the printed bound
        m = {}
        for name in ['a_u', 'a_0', 'H1_0']:
            mm = re.search(r'\[' + re.escape(name) + r'\s*\] rho_min = ([\d.eE+-]+)\s+rho_max = ([\d.eE+-]+)\s+kappa = ([\d.eE+-]+)', b)
            m[name] = tuple(float(g) for g in mm.groups())
        rows[beta] = dict(N=N, lam1=lam1, lam2=lam2, lam1_A3=lam1_A3, mu1=mu1, **{k: v for k, v in m.items()})

betas = sorted(rows)
lam1 = np.array([rows[b]['lam1'] for b in betas])
lam2 = np.array([rows[b]['lam2'] for b in betas])
lam1_A3 = np.array([rows[b]['lam1_A3'] for b in betas])
rmin_au = np.array([rows[b]['a_u'][0] for b in betas]); rmax_au = np.array([rows[b]['a_u'][1] for b in betas])
rmin_a0 = np.array([rows[b]['a_0'][0] for b in betas]); rmax_a0 = np.array([rows[b]['a_0'][1] for b in betas])
lb_old = 1 - lam1 / lam2
lb_new = 1 - lam1 / lam1_A3
Lam = np.maximum(lam2, lam1_A3)
lb_new_max = 1 - lam1 / Lam

print(" beta      N    lam1      lam2-lam1   lam1(A3)-lam1   rho_min(a_u)  LB_paper   LB_new     ratio_new/exact   rho_max(a_u)  kappa(a_u)  kappa_bound(a_u)  rho_min(a_0)  kappa(a_0)")
for i, b in enumerate(betas):
    kb = (1 + lam1[i] / lam2[i]) * Lam[i] / (Lam[i] - lam1[i])
    print(f" {b:8.0f} {rows[b]['N']:5d}  {lam1[i]:9.3f}  {lam2[i]-lam1[i]:10.4g}  {lam1_A3[i]-lam1[i]:12.4g}   {rmin_au[i]:.4e}   {lb_old[i]:.2e}  {lb_new_max[i]:.4e}   {lb_new_max[i]/rmin_au[i]:.3f}            {rmax_au[i]:.4f}      {rmax_au[i]/rmin_au[i]:8.2f}    {kb:8.2f}        {rmin_a0[i]:.4e}   {rmax_a0[i]/rmin_a0[i]:.3e}")

# local slopes
if len(betas) >= 3:
    sl = np.diff(np.log(rmin_au)) / np.diff(np.log(lam1))
    sl3 = np.diff(np.log(lam1_A3 - lam1)) / np.diff(np.log(lam1))
    slg = np.diff(np.log(lam2 - lam1)) / np.diff(np.log(lam1))
    print("\nlocal log-log slopes vs lam1 between consecutive beta:")
    print("  d log rho_min(a_u) / d log lam1 :", np.round(sl, 3), " (boundary-layer prediction -2/3)")
    print("  d log (lam1(A_3b)-lam1) / d log lam1 :", np.round(sl3, 3), " (prediction +1/3)")
    print("  d log (lam2-lam1) / d log lam1 :", np.round(slg, 3), " (prediction -1)")

# The two metrics drawn here keep the colour and marker they have in
# plot_rayleigh_bounds_3d.py and in make_figures.py, and the two bounds are
# black, dashed for the one of Corollary 3.5 and dotted for the gap-only one,
# as in the three-dimensional figure; a reader moving between the two figures
# meets the same conventions.
# The a_0 marker is the larger of the two and is drawn first, so that both are
# visible where the two curves agree, which in the left panel is everywhere:
# the two metrics have the same smallest Rayleigh quotient to within three per
# cent, and a reader should see that rather than one curve.
AU = dict(color='tab:purple', marker='v', ls='-', mfc='none', ms=5, lw=1.2)
A0 = dict(color='tab:red', marker='s', ls='-', mfc='none', ms=8, lw=1.2)
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
    is not a decade of beta.
    """
    top = ax.secondary_xaxis('top')
    top.set_xticks(list(lam1))
    top.set_xticklabels([fmt_beta(b) for b in betas], fontsize=7,
                        rotation=55, ha='left', rotation_mode='anchor')
    top.minorticks_off()
    top.tick_params(axis='x', pad=1.5)
    top.set_xlabel(r'interaction strength $\beta$', fontsize=9, labelpad=4)


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(lam1, rmin_a0, label=r'$a_0$', **A0)
ax[0].loglog(lam1, rmin_au, label=r'$a_u$', **AU)
ax[0].loglog(lam1, lb_new_max, 'k--', lw=1.1,
             label=r'bound $1-\lambda_1/\Lambda$, $a_u$')
ax[0].loglog(lam1, lb_old, 'k:', lw=1.4,
             label=r'gap bound $1-\lambda_1/\lambda_2$')
ref = lam1 ** (-2 / 3) * rmin_au[-1] / lam1[-1] ** (-2 / 3)
ax[0].loglog(lam1, ref, 'k-.', lw=1.1, label=r'$\propto\lambda_1^{-2/3}$')
ax[0].set_xlabel(XLAB)
ax[0].set_ylabel(r'exact $\rho_{\min}$ at the ground state')
ax[0].legend(fontsize=7.5, loc='lower left'); ax[0].grid(alpha=.3, which='both')
kb = (1 + lam1 / lam2) * Lam / (Lam - lam1)
ax[1].loglog(lam1, rmax_a0 / rmin_a0, label=r'$a_0$', **A0)
ax[1].loglog(lam1, rmax_au / rmin_au, label=r'$a_u$', **AU)
ax[1].loglog(lam1, kb, 'k--', lw=1.1,
             label=r'bound $(1+\lambda_1/\lambda_2)\,\Lambda/(\Lambda-\lambda_1)$, $a_u$')
ax[1].loglog(lam1, (1 + lam1 / lam2) / lb_old, 'k:', lw=1.4,
             label=r'gap bound $(\lambda_2+\lambda_1)/(\lambda_2-\lambda_1)$')
ax[1].set_xlabel(XLAB)
ax[1].set_ylabel(r'condition number $\kappa$ of the Riemannian Hessian')
ax[1].legend(fontsize=7.5, loc='upper left'); ax[1].grid(alpha=.3, which='both')
for a in ax:
    beta_axis(a, lam1, betas)
plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT_PNG)), exist_ok=True)
plt.savefig(OUT_PNG, dpi=160)
print('saved', OUT_PNG)
with open(os.path.join(DATA, 'table1d_rows.tex'), 'w') as f:
    f.write(r"\begin{tabular}{rrrrrrrrrr}" + "\n" + r"\toprule" + "\n")
    f.write(r"$\beta$ & $N$ & $\lambda_1(A_\beta)$ & $\lambda_2-\lambda_1$ & $\lambda_1(A_{3\beta})-\lambda_1$ & $\min\rho$ ($a_u$) & $1-\lambda_1/\lambda_2$ & $1-\lambda_1/\Lambda$ & $\max\rho$ ($a_u$) & $\kappa_{a_u}$ (bound) \\" + "\n" + r"\midrule" + "\n")
    for i, b in enumerate(betas):
        kb = (1 + lam1[i] / lam2[i]) * Lam[i] / (Lam[i] - lam1[i])
        f.write(f"{b:g} & {rows[b]['N']} & {lam1[i]:.3f} & {lam2[i]-lam1[i]:.3g} & {lam1_A3[i]-lam1[i]:.3g} & {rmin_au[i]:.4f} & {lb_old[i]:.2e} & {lb_new_max[i]:.4f} & {rmax_au[i]:.4f} & {rmax_au[i]/rmin_au[i]:.1f} ({kb:.1f}) \\\\\n")
    f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
print('wrote', os.path.join(DATA, 'table1d_rows.tex'))
