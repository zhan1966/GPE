#!/usr/bin/env python3
"""Compare Riemannian gradient descent and CG under seven Sobolev metrics.

One problem, fourteen methods:
{modified H^1, a_0, modified a_0, a_u, modified a_u c=2, c=3, Thomas-Fermi} x {GD, CG}.
The test problem is the stirrer potential on (-8,8)^3 with beta=1600, Example 2
of Li, Zheng and Zhang (2026), started from the Thomas-Fermi profile.

    python3 run_metrics.py --n 499 --methods all --maxit 400 --verbose 10

Writes one CSV per method to --outdir, named <tag>_<method>.csv with the columns
of solvers.COLS, plus <tag>_meta.json. Nothing is plotted here; the figures come
from make_figures.py in this directory.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'results', 'data')

# key -> (metric, algorithm, label used in the legends)
# The labels are the ones the paper uses: $a_u$ rather than $a_\mathbf{u}$,
# and "modified $H^1$" for the metric with the shift alpha, which is what every
# run here uses. They are stored in the meta index and drawn by
# make_figures.py, whose STYLE map must agree with this one.
METHODS = {
    'h1_gd':    ('h1',    'gd', r'modified $H^1$ flow'),
    'h1_cg':    ('h1',    'cg', r'Riemannian CG, modified $H^1$'),
    'a0_gd':    ('a0',    'gd', r'$a_0$ flow'),
    'a0_cg':    ('a0',    'cg', r'Riemannian CG, $a_0$'),
    'a0mod_gd': ('a0mod', 'gd', r'modified $a_0$ flow'),
    'a0mod_cg': ('a0mod', 'cg', r'Riemannian CG, modified $a_0$'),
    'au_gd':    ('au',    'gd', r'$a_u$ flow'),
    'au_cg':    ('au',    'cg', r'Riemannian CG, $a_u$'),
    'a3u_gd':   ('a3u',   'gd', r'modified $a_u$ flow, $c=3$'),
    'a3u_cg':   ('a3u',   'cg', r'Riemannian CG, modified $a_u$, $c=3$'),
    'a2u_gd':   ('a2u',   'gd', r'modified $a_u$ flow, $c=2$'),
    'a2u_cg':   ('a2u',   'cg', r'Riemannian CG, modified $a_u$, $c=2$'),
    'tf_gd':    ('tf',    'gd', r'Thomas--Fermi flow'),
    'tf_cg':    ('tf',    'cg', r'Riemannian CG, Thomas--Fermi'),
}
ALL = list(METHODS)
# The six methods of the first comparison; the others complete the table and
# cost one job each.
REQUESTED = ['h1_gd', 'au_gd', 'h1_cg', 'a0_gd', 'au_cg', 'a0mod_cg']


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=99)
    p.add_argument('--potential', default='stirrer',
                   choices=('stirrer', 'lattice'))
    p.add_argument('--init', default='tf', choices=('tf', 'const', 'linear'))
    p.add_argument('--beta', type=float, default=1600.0)
    p.add_argument('--alpha', type=float, default=20.0,
                   help='shift in the modified H^1 metric (-Lap + alpha I)')
    p.add_argument('--L', type=float, default=8.0)
    p.add_argument('--order', type=int, default=2)
    p.add_argument('--disc', default='fd', choices=('fd', 'sem'),
                   help="'fd' is the second-order finite difference scheme; "
                        "'sem' is Q^q spectral elements with Gauss-Lobatto "
                        "quadrature, n = q*Ncell - 1")
    p.add_argument('--q-order', type=int, default=20)
    p.add_argument('--maxit', type=int, default=400)
    p.add_argument('--methods', default='requested',
                   help="'all', 'requested', or a comma-separated list of "
                        + ','.join(ALL))
    p.add_argument('--pcg-tol', type=float, default=1e-8,
                   help='relative residual for the inner PCG of the a_0, a_u '
                        'and the two modified a_u metrics')
    p.add_argument('--pcg-max', type=int, default=200)
    p.add_argument('--cg-dir-fp32', action='store_true',
                   help='store the conjugate gradient search direction in '
                        'single precision, so that the inner solve of the next '
                        'gradient has 4 GB more at n=999. The direction is '
                        'stored as itself rather than as a difference, so this '
                        'is not the cancellation that --lean gets wrong.')
    p.add_argument('--lean', action='store_true',
                   help='keep the applied metric in single precision: no fp64 '
                        'refinement and no fp64 copy of G u or G grad E. '
                        'Kept as an experiment, not as a mode to use: the '
                        'Riemannian gradient is a cancellation that loses '
                        'about seven digits near convergence, and the a_0 '
                        'flow stalls at |E-E*| = 6e-9 at n=499. Implies '
                        '--fp32-inner.')
    p.add_argument('--fp32-inner', action='store_true',
                   help='hold the Krylov vectors of the inner solves of '
                        'the a_0, a_u and modified a_u metrics in single '
                        'precision, with one fp64 refinement '
                        'pass. Needed at n=999, where an fp64 inner solve does '
                        'not fit on one card.')
    p.add_argument('--refine', type=int, default=1,
                   help='fp64 refinement passes after the fp32 inner solve. '
                        'One is enough for the a_0 and a_u metrics; a metric '
                        'the separable preconditioner matches poorly may need '
                        'more, since a stagnating fp32 solve cannot be fixed '
                        'by a single correction.')
    p.add_argument('--tf-f', type=float, default=0.5,
                   help='the floor parameter f of the Thomas-Fermi metric, '
                        'eq. (11) of the paper, where it is called ell: B_TF '
                        'gets + f ell^(1/3) I. Default 1/2.')
    p.add_argument('--tf-freeze-ell', action='store_true',
                   help='freeze the scalar ell of the Thomas-Fermi metric at '
                        'its value from the initial guess, instead of taking '
                        'the Rayleigh quotient of the current iterate each '
                        'outer iteration. Frozen, the metric is a smooth '
                        'function of u; updated, it converges to lambda_1.')
    p.add_argument('--tmax', type=float, default=5.0,
                   help='upper bound of the line search over the step size')
    p.add_argument('--backend', default='jax', choices=('jax', 'numpy'))
    p.add_argument('--solver', default='dense', choices=('dense', 'fft'))
    p.add_argument('--precision', default='fp64', choices=('fp64', 'fp32'))
    p.add_argument('--error-tol', type=float, default=1e-12)
    p.add_argument('--res-tol', type=float, default=1e-12)
    p.add_argument('--ref-method', default='dys',
                   choices=('dys', 'rgf', 'rcg_h1', 'rcg_a0mod'),
                   help="how the reference is computed: 'dys' is Davis-Yin "
                        "splitting, robust but slow when max V is large, since "
                        "its step is 1/max(V+3 beta u^2); 'rcg_h1' and "
                        "'rcg_a0mod' are the Riemannian CG of this paper, "
                        "which reach machine precision in a few dozen "
                        "iterations and are the right choice on a large domain")
    p.add_argument('--ref-maxit', type=int, default=6000)
    p.add_argument('--outdir', default=DEFAULT_OUT)
    p.add_argument('--cache-dir', default=None,
                   help='where u* is READ from, if it is there; without it the '
                        'ground state error is not recorded, which is what '
                        'every timed run wants. Nothing is written here unless '
                        '--save-uref is also given.')
    p.add_argument('--save-uref', action='store_true',
                   help='also WRITE u* into --cache-dir when a reference has '
                        'to be computed. Off by default: a ground state is '
                        '1 GB at n=499 and 8 GB at n=999, and no experiment in '
                        'this project needs one, since they all measure the '
                        'energy error against E* from the reference JSON.')
    p.add_argument('--tag', default=None)
    p.add_argument('--verbose', type=int, default=10)
    p.add_argument('--no-warmup', action='store_true')
    return p.parse_args(argv)


def get_ref(P, a, outdir):
    """E*, lambda* and optionally u*, cached beside the curves."""
    import numpy as np
    import solvers
    # alpha is a parameter of the H^1 metric, not of the problem, so the
    # reference does not depend on it and is shared across an alpha sweep
    dtag = '' if a.disc == 'fd' else f'_q{a.q_order}'
    ltag = '' if a.L == 8.0 else f'_L{a.L:g}'
    key = (f"ref_{a.potential}_n{a.n}_beta{a.beta:g}_o{a.order}{ltag}{dtag}")
    js = os.path.join(outdir, key + '.json')
    # u* is READ if it happens to be there, and WRITTEN only with --save-uref.
    # No experiment here needs it: every run measures the energy error,
    # u_error is NaN in all of them, and a ground state is 1 GB at n=499 and
    # 8 GB at n=999.
    npy = None if a.cache_dir is None else os.path.join(a.cache_dir, key + '.npy')
    npy_w = npy if a.save_uref else None
    if os.path.exists(js):
        with open(js) as fh:
            d = json.load(fh)
        if npy and os.path.exists(npy):
            u = P['to_dev'](P['xp'].asarray(np.load(npy), dtype=P['dtype']))
            return dict(E=float(d['E']), lam=float(d['lam']), u=u), 'cache'
        print('  reference from cache: E* and lambda* only, no u*', flush=True)
        return dict(E=float(d['E']), lam=float(d['lam']), u=None), 'cache'

    print(f'Computing the reference ({a.ref_method} from Thomas-Fermi) ...',
          flush=True)
    t0 = time.time()
    if a.ref_method.startswith('rcg'):
        import metrics
        u_tf, desc = solvers.make_init(P, 'tf')
        lam_tf = float(P['h3'] * P['xp'].sum(u_tf * P['grad_E'](u_tf)))
        # a reference has to be tighter than anything measured against it, so
        # it is driven to the residual floor rather than to the 1e-12 the
        # comparison stops at
        h, u_ref = metrics.rcg(P, u_tf, min(a.ref_maxit, 400),
                               dict(E=0.0, lam=0.0, u=None),
                               metric=a.ref_method[4:],
                               tols=solvers._tols(0.0, 1e-13), verbose=25,
                               pcg_tol=a.pcg_tol, pcg_max=a.pcg_max,
                               tmax=a.tmax, return_u=True)
        iters = int(h['iter'][-1])
        ref = dict(E=float(h['energy'][-1]), lam=float(h['eigenvalue'][-1]),
                   u=u_ref)
        print(f"  reference by {a.ref_method}: {iters} iterations, "
              f"residual {float(h['residue'][-1]):.2e}", flush=True)
    else:
        ref, iters, lam_tf = solvers.reference_solution(
            P, method=a.ref_method, maxit=a.ref_maxit, verbose=200)
    dt = time.time() - t0
    print(f"E* = {ref['E']:.15f}  lambda* = {ref['lam']:.12f}  "
          f"({iters} iterations, {dt:.1f}s)", flush=True)
    with open(js, 'w') as fh:
        json.dump({'E': ref['E'], 'lam': ref['lam'], 'lam_TF': lam_tf,
                   'potential': a.potential, 'n': a.n, 'L': a.L, 'beta': a.beta,
                   'alpha': a.alpha, 'order': a.order, 'method': a.ref_method,
                   'iters': iters, 'seconds': dt}, fh, indent=2)
    if npy_w:
        os.makedirs(a.cache_dir, exist_ok=True)
        np.save(npy_w, np.asarray(ref['u'], dtype=np.float64))
        print(f'  cached u* -> {npy_w}  ({os.path.getsize(npy_w)/2**30:.1f} GB)',
              flush=True)
    return ref, 'computed'


def save_one(h, outdir, tag, key):
    import solvers
    cols = list(solvers.COLS)
    path = os.path.join(outdir, f'{tag}_{key}.csv')
    with open(path, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for i in range(len(h['iter'])):
            fh.write(','.join(f'{h[c][i]:.15g}' for c in cols) + '\n')
    return path


def main(argv=None):
    a = parse(argv)
    sys.path.insert(0, HERE)
    import gpe3d
    import metrics
    import solvers

    if a.methods == 'all':
        keys = list(ALL)
    elif a.methods == 'requested':
        keys = list(REQUESTED)
    else:
        keys = [k.strip() for k in a.methods.split(',') if k.strip()]
    bad = [k for k in keys if k not in METHODS]
    if bad:
        raise SystemExit(f'unknown methods {bad}; choose from {ALL}')

    outdir = os.path.abspath(a.outdir)
    os.makedirs(outdir, exist_ok=True)
    dtag = '' if a.disc == 'fd' else f'_q{a.q_order}'
    ltag = '' if a.L == 8.0 else f'_L{a.L:g}'
    tag = a.tag or f'{a.potential}_{a.init}_n{a.n}{ltag}{dtag}'

    disc_desc = ('second-order FD' if a.disc == 'fd'
                 else f'Q^{a.q_order} spectral elements')
    print(f'Building {a.potential} at n={a.n} ({disc_desc}), beta={a.beta:g}, '
          f'alpha={a.alpha:g}, {a.precision} on {a.backend} ...', flush=True)
    t0 = time.time()
    P = gpe3d.build(n=a.n, L=a.L, beta=a.beta, alpha=a.alpha,
                    potential=a.potential, order=a.order, backend=a.backend,
                    solver=a.solver, precision=a.precision,
                    disc=a.disc, q_order=a.q_order)
    print(f'  built in {time.time()-t0:.1f}s', flush=True)

    P32 = None
    if (a.fp32_inner or a.lean) and a.precision == 'fp64' \
            and any(METHODS[k][0] in ('a0', 'au', 'a2u', 'a3u', 'tf')
                    for k in keys):
        t32 = time.time()
        P32 = gpe3d.build(n=a.n, L=a.L, beta=a.beta, alpha=a.alpha,
                          potential=a.potential, order=a.order,
                          backend=a.backend, solver=a.solver,
                          precision='fp32', disc=a.disc, q_order=a.q_order)
        P['sync'](P32['solve_V1'](P32['u_const']()))
        print(f'  fp32 twin for the inner solve, built in '
              f'{time.time()-t32:.1f}s', flush=True)

    ref, src = get_ref(P, a, outdir)
    u0, init_desc = solvers.make_init(P, a.init)
    print(f'  initial guess: {init_desc}', flush=True)

    if not a.no_warmup:
        t0 = time.time()
        u = P['normalize'](u0)
        P['sync'](P['solve_G'](u))
        P['sync'](P['solve_V1'](u))
        P['sync'](P['grad_E'](u))
        P['sync'](P['apply_Au'](u, u))
        P['sync'](P['ls_coeffs_fused'](u, u))
        print(f'  warm-up {time.time()-t0:.1f}s', flush=True)

    tols = solvers._tols(a.error_tol, a.res_tol)
    meta_path = os.path.join(outdir, f'{tag}_meta.json')
    old = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh).get('methods', {})
    meta = {'tag': tag, 'potential': a.potential, 'init': a.init, 'n': a.n,
            'L': a.L, 'beta': a.beta, 'alpha': a.alpha, 'order': a.order,
            'disc': a.disc, 'q_order': a.q_order,
            'backend': a.backend, 'solver': a.solver, 'precision': a.precision,
            'maxit': a.maxit, 'pcg_tol': a.pcg_tol, 'pcg_max': a.pcg_max,
            'tmax': a.tmax, 'tf_f': a.tf_f, 'refine': a.refine,
            'tf_freeze_ell': bool(a.tf_freeze_ell),
            'fp32_inner': bool(a.fp32_inner or a.lean),
            'lean': bool(a.lean),
            'E_ref': ref['E'], 'lam_ref': ref['lam'],
            'ref_source': src, 'methods': dict(old)}

    mine = {}                      # only what THIS process measured
    for key in keys:
        metric, alg, label = METHODS[key]
        print(f'\n-- {key}: {label} ({alg.upper()}, {metric} metric) --',
              flush=True)
        t0 = time.time()
        run = metrics.rgd if alg == 'gd' else metrics.rcg
        h = run(P, u0, a.maxit, ref, metric=metric, tols=tols,
                verbose=a.verbose, pcg_tol=a.pcg_tol, pcg_max=a.pcg_max,
                tmax=a.tmax, P32=P32, lean=a.lean,
                tf_f=a.tf_f, tf_freeze=a.tf_freeze_ell, n_refine=a.refine,
                **({'dir32': a.cg_dir_fp32} if alg == 'cg' else {}))
        dt = time.time() - t0
        peak = float('nan')
        try:
            import jax
            peak = (jax.local_devices()[0].memory_stats()['peak_bytes_in_use']
                    / 2 ** 30)
        except Exception:
            pass
        path = save_one(h, outdir, tag, key)
        its = int(h['iter'][-1])
        mine[key] = meta['methods'][key] = {
            'label': label, 'metric': metric, 'algorithm': alg,
            'iterations': its, 'seconds': float(h['time'][-1]),
            'wall_seconds': dt, 'final_error': float(h['error'][-1]),
            'final_residue': float(h['residue'][-1]),
            'linear_solves': int(h['linsys'][-1]),
            'inner_pcg_iterations': int(h.get('_inner', 0)),
            'peak_gpu_gb': peak,
            # The inner-solver settings THIS process used. The top-level
            # fields above record only the last process to write the index,
            # so a table whose rows come from jobs with different flags --
            # the Thomas-Fermi bursts of --pcg-max 25 --refine 6, say --
            # cannot be read off them.
            'pcg_max': a.pcg_max, 'refine': a.refine,
            'fp32_inner': bool(a.fp32_inner or a.lean),
            'lean': bool(a.lean),
            'cg_dir_fp32': bool(a.cg_dir_fp32) and alg == 'cg',
            'maxit': a.maxit, 'error_tol': a.error_tol}
        print(f'   {its} iterations, {h["time"][-1]:.1f}s, '
              f'|E-E*|={h["error"][-1]:.3e}, res={h["residue"][-1]:.3e}, '
              f'{int(h["linsys"][-1])} solves, peak {peak:.1f} GB '
              f'-> {os.path.basename(path)}', flush=True)
        # Re-read before writing, and let ONLY this job's own results win.
        # Two jobs on the same tag -- one method each, which is the recommended
        # way to run -- otherwise both load the index at startup, each adds its
        # own method, and the second to finish drops the first one's entry.
        # Merging meta['methods'] over the fresh read is NOT enough, and that
        # was the bug behind the second round of this: meta['methods'] is
        # seeded from a startup snapshot, so it carries a stale copy of every
        # other method and re-imposes it over whatever a concurrent job has
        # written since. Only `mine` may overwrite. The CSVs are per method and
        # were never at risk; only this index was.
        on_disk = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    on_disk = json.load(fh).get('methods', {})
            except (ValueError, OSError):
                on_disk = {}
        on_disk.update(mine)
        meta['methods'] = on_disk
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)

    print('\n=== summary ===', flush=True)
    print(f'{"method":10s} {"iters":>6s} {"seconds":>9s} {"solves":>8s} '
          f'{"inner":>7s} {"|E-E*|":>10s}')
    for k, m in meta['methods'].items():
        print(f'{k:10s} {m["iterations"]:6d} {m["seconds"]:9.1f} '
              f'{m["linear_solves"]:8d} {m.get("inner_pcg_iterations",0):7d} '
              f'{m["final_error"]:10.2e}')
    print('=== DONE ===', flush=True)


if __name__ == '__main__':
    main()
