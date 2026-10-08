#!/usr/bin/env python3
"""Figures 1 to 6 of the paper, the comparison of the metrics.

Reads the error histories in results/data and writes to $FIG_DIR (default
figures/):

    metrics_<tag>.png    the energy error against the iteration count and
                         against GPU wall clock, gradient descent in the top
                         row and conjugate gradients in the bottom one;
                         Figures 1 and 2 of the paper for the two tags in
                         PAPER_TAGS
    metrics_beta_gd.png          the error curves against beta, one row per
    metrics_beta_cg.png          value, split by algorithm (Figures 3 and 4)
    metrics_domain_gd.png        the error curves on nested boxes, one row
    metrics_domain_cg.png        per L, split by algorithm (Figures 5 and 6)

    python3 make_figures.py              the six figures of the paper
    python3 make_figures.py tag ...      the 2x2 figure of the given runs

Colour encodes the metric and the drawing style the algorithm: a solid line
with open markers for gradient descent, filled markers alone for conjugate
gradients, so a reader sees at a glance which pairs share a metric. Needs only NumPy and
Matplotlib -- no GPU, no JAX.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'results', 'data')
OUT = os.environ.get('FIG_DIR', os.path.join(HERE, 'figures'))

# method key -> (label, colour, marker, linestyle)
# The c=2 member of the modified a_u family is run but not drawn: it sits on
# top of c=3 and only crowds the panels. The H^1 entries say "modified",
# which is what alpha = 20 makes them: the H^1 seminorm is alpha = 0, and it is
# reported in the text alone.
STYLE = {
    'h1_gd':    (r'modified $H^1$ flow',              'tab:blue',   'o', '-'),
    'h1_cg':    (r'Riemannian CG, modified $H^1$',    'tab:blue',   'o', 'none'),
    'a0_gd':    (r'$a_0$ flow',                       'tab:red',    's', '-'),
    'a0_cg':    (r'Riemannian CG, $a_0$',             'tab:red',    's', 'none'),
    'a0mod_gd': (r'modified $a_0$ flow',              'tab:green',  '^', '-'),
    'a0mod_cg': (r'Riemannian CG, modified $a_0$',    'tab:green',  '^', 'none'),
    'au_gd':    (r'$a_u$ flow',                        'tab:purple', 'v', '-'),
    'au_cg':    (r'Riemannian CG, $a_u$',              'tab:purple', 'v', 'none'),
    'a3u_gd':   (r'modified $a_u$ flow',         'tab:orange', 'D', '-'),
    'a3u_cg':   (r'Riemannian CG, modified $a_u$', 'tab:orange','D', 'none'),
    'tf_gd':    ('Thomas–Fermi flow',                 'tab:cyan',   'P', '-'),
    'tf_cg':    ('Riemannian CG, Thomas–Fermi',       'tab:cyan',   'P', 'none'),
}
ORDER = list(STYLE)
GPU = 'GPU time on one GH200 (s)'


def load(tag):
    """{method key: {column: array}} for one run."""
    out = {}
    for k in ORDER:
        path = os.path.join(DATA, f'{tag}_{k}.csv')
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        out[k] = {c: np.array([float(r[c]) for r in rows]) for c in rows[0]}
    return out


ALGO = [('gd', 'Riemannian gradient descent'),
        ('cg', 'Riemannian conjugate gradient')]

# Upper bounds for the beta error-curve figures, chosen the way the nested-box
# ones were: wide enough to show where the fast methods finish, not so wide
# that the one method which diverges owns the panel.
BETA_XLIM = {'gd': {'iter': 150, 'time': 400},
             'cg': {'iter': 60, 'time': 150}}


def build(tag):
    """The comparison figure, 2x2: one row per algorithm and one column per
    cost measure.

    Ten curves in one panel hide exactly the differences the figure is for, so
    the row splits them by algorithm -- five gradient flows above, the five
    conjugate gradient methods below -- and the columns are the energy error
    against the iteration count and against wall clock. The two rows keep
    independent axes: the flows run five times as long as their conjugate
    gradient counterparts, and forcing a common range would squeeze the lower
    row into the left fifth of its panels, which is the crowding this layout
    removes. Colour and marker encode the metric and are the same in both rows,
    so a column can be read down; the drawing style still encodes the
    algorithm, a solid line with open markers for descent and filled markers
    alone for conjugate gradients, so the figure matches the rest of the paper.
    """
    res = load(tag)
    if not res:
        raise SystemExit(f'no CSVs for {tag} in {DATA}')
    cols = [('iter', 'iteration'), ('time', GPU)]
    nrows, ncols = len(ALGO), len(cols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.6 * ncols, 5.2 * nrows), squeeze=False)
    # one error range for the whole figure, so a curve that stops early in one
    # row is seen to stop early and not merely to leave its own panel
    ylo = min(max(min(h['error']), 1e-17) for h in res.values())
    yhi = max(max(h['error']) for h in res.values())
    drawn = {}
    for r, (alg, title) in enumerate(ALGO):
        keys = [k for k in ORDER if k in res and k.endswith('_' + alg)]
        for c, (xk, xlabel) in enumerate(cols):
            ax = axes[r][c]
            for k in keys:
                label, colour, marker, ls = STYLE[k]
                h = res[k]
                y = np.maximum(h['error'], 1e-17)
                ln, = ax.semilogy(h[xk], y, color=colour, marker=marker, ls=ls,
                                  markevery=max(1, len(y) // 18), label=label,
                                  fillstyle='none' if ls == '-' else 'full',
                                  lw=2.0, markersize=7)
                if c == 0:
                    drawn[k] = ln
            ax.set_ylim(0.4 * ylo, 2.5 * yhi)
            ax.set_xlabel(xlabel, fontsize=17)
            ax.set_ylabel(r'$|E^k - E^*|$', fontsize=17)
            ax.set_title(title, fontsize=18)
            ax.tick_params(labelsize=14)
            ax.grid(True, which='both', alpha=0.3)
    # One legend row per metric: the flow on the left, its conjugate gradient
    # variant on the right, mirroring the panels. Matplotlib fills a
    # two-column legend column by column, so the descent entries are emitted
    # first and a blank is emitted for any method that was not run -- without
    # the blanks the two groups are of unequal size, the split lands mid-group,
    # and a flow ends up at the top of the conjugate gradient column where a
    # reader who knows this convention will not look for it. That is exactly
    # what happened to the modified a_u flow in the 999^3 figure. A blank also
    # says something true: this metric has no curve here.
    metrics = []
    for k in ORDER:
        m = k.rsplit('_', 1)[0]
        if k in res and m not in metrics:
            metrics.append(m)
    blank = plt.Line2D([], [], linestyle='none', marker='')
    handles, labels = [], []
    for alg, _ in ALGO:
        for m in metrics:
            k = f'{m}_{alg}'
            handles.append(drawn.get(k, blank))
            labels.append(STYLE[k][0] if k in drawn else '')
    ncol = 2
    lrows = -(-len(labels) // ncol)
    reserve = (0.34 * lrows + 0.18) / (5.2 * nrows)
    plt.tight_layout(rect=(0, reserve, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=17,
               handlelength=2.6, columnspacing=3.0, labelspacing=0.4)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'metrics_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}  ({nrows}x{ncols} panels, {len(labels)} curves)')


# --------------------------------------------------------------------------
def stack(name, rows, alg=None, xlim=None):
    """One figure with a row per run: energy error against the iteration count
    and against wall clock, in the panels and the convention of build().

    rows is a list of (tag, title). Used for the domain sweep, where the three
    boxes have to be read against each other curve by curve and not only
    through their endpoints.

    alg restricts to one algorithm, 'gd' or 'cg'. Six metrics times two
    algorithms in one panel is unreadable, and the two algorithms have their
    own vertical scales anyway, so each sweep is drawn as two figures.
    Passing alg=None draws both algorithms in one figure.

    xlim = {'iter': hi, 'time': hi} sets the upper bound of each column by
    hand. Letting the data set it hands the whole width to the one method that
    diverges -- the modified H^1 flow needs 600 iterations at L=32 where every
    other method needs under 40 -- and squeezes the comparison the figure is
    for into the left tenth. Curves that leave the panel are the ones that do
    not converge inside it, which the captions say.

    Only the methods run in every box are drawn. A curve present in one row
    alone would still be listed in the legend, which is shared, and would read
    as a method that failed in the other boxes rather than as one that was
    never run there.
    """
    data = [(load(tag), title) for tag, title in rows]
    missing = [t for (res, _), (t, _) in zip(data, rows) if not res]
    if missing:
        raise SystemExit(f'no CSVs for {missing} in {DATA}')
    common = [k for k in ORDER if all(k in res for res, _ in data)
              and (alg is None or k.endswith('_' + alg))]
    left_out = sorted({k for res, _ in data for k in res
                       if alg is None or k.endswith('_' + alg)} - set(common))
    if left_out:
        print(f'  {name}: {left_out} not run in every box, left out')
    data = [({k: res[k] for k in common}, title) for res, title in data]
    nrows, ncols = len(rows), 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.6 * ncols, 5.2 * nrows), squeeze=False)
    # One range per column and one for the whole figure in y, so that the rows
    # are read against each other: a curve that needs twice the iterations in
    # a larger box has to look twice as long.
    xhi = {xk: max(max(h[xk]) for res, _ in data for h in res.values())
           for xk in ('iter', 'time')}
    ylo = min(max(min(h['error']), 1e-17) for res, _ in data
              for h in res.values())
    yhi = max(max(h['error']) for res, _ in data for h in res.values())
    for r, (res, title) in enumerate(data):
        for c, (xk, xlabel) in enumerate((('iter', 'iteration'), ('time', GPU))):
            ax = axes[r][c]
            for k in ORDER:
                if k not in res:
                    continue
                label, colour, marker, ls = STYLE[k]
                h = res[k]
                y = np.maximum(h['error'], 1e-17)
                ax.semilogy(h[xk], y, color=colour, marker=marker, ls=ls,
                            markevery=max(1, len(y) // 18), label=label,
                            fillstyle='none' if ls == '-' else 'full',
                            lw=2.0, markersize=7)
            hi = (xlim or {}).get(xk, xhi[xk])
            ax.set_xlim(-0.02 * hi, 1.02 * hi)
            ax.set_ylim(0.4 * ylo, 2.5 * yhi)
            ax.set_xlabel(xlabel, fontsize=17)
            ax.set_ylabel(r'$|E^k - E^*|$', fontsize=17)
            ax.set_title(title, fontsize=18)
            ax.tick_params(labelsize=14)
            ax.grid(True, which='both', alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = 2
    lrows = -(-len(labels) // ncol)
    plt.tight_layout(rect=(0, (0.34 * lrows + 0.18) / (5.2 * nrows), 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=17, handlelength=2.6,
               columnspacing=3.0, labelspacing=0.4)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + '.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}  ({nrows}x{ncols} panels, {len(labels)} curves)')


# The two runs drawn as standalone 2x2 figures, Figures 1 and 2 of the paper.
# The other runs are rows of the stacked figures below. Pass a tag on the
# command line to build the standalone figure of any run in results/data.
PAPER_TAGS = ['stirrer_tf_n499', 'stirrer_tf_n999_q20']


if __name__ == '__main__':
    for t in sys.argv[1:] or PAPER_TAGS:
        build(t)
    if sys.argv[1:]:
        raise SystemExit
    # Figures 5 and 6: the nested boxes, all three in double precision, so
    # that the seconds may be read across a row. The L=16 box is the
    # beta=1600 run of Figures 3 and 4.
    boxes = [('stirrer_tf_n499', r'$L=8$'),
             ('stirrer_tf_n499_L16_beta1600', r'$L=16$'),
             ('stirrer_tf_n499_L32_beta1600', r'$L=32$')]
    stack('metrics_domain_gd', boxes, alg='gd',
          xlim={'iter': 150, 'time': 100})
    stack('metrics_domain_cg', boxes, alg='cg',
          xlim={'iter': 40, 'time': 45})
    # Figures 3 and 4: the same treatment for beta, on (-16,16)^3 with the
    # same grid, so that only beta varies down a column.
    bet = [('stirrer_tf_n499_L16_beta1600', r'$\beta=1600$'),
           ('stirrer_tf_n499_L16_beta6400', r'$\beta=6400$'),
           ('stirrer_tf_n499_L16_beta102400', r'$\beta=102400$')]
    stack('metrics_beta_gd', bet, alg='gd', xlim=BETA_XLIM['gd'])
    stack('metrics_beta_cg', bet, alg='cg', xlim=BETA_XLIM['cg'])
