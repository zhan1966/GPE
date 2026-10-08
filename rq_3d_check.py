"""Cross-check rq_3d.py's matrix-free LOBPCG against a dense solver at tiny n.

Builds N and B column by column at n^3 <= 4096 unknowns, forms an orthonormal
basis of the tangent space with a Householder reflector exactly as rq_1d.py
does, and calls scipy's dense symmetric-definite solver. The discretization is
too coarse to mean anything physically; the point is that the algebra the
3D code performs is the same algebra the 1D code performs.
"""
import os, sys, time
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpe3d, rq_3d

n = int(os.environ.get('CHECK_N', '15'))
L, beta = 8.0, float(os.environ.get('CHECK_BETA', '1600'))
P = gpe3d.build(n, L=L, beta=beta, alpha=20.0, potential='stirrer',
                backend='numpy', solver='dense', precision='fp64', disc='fd')
u, lam1, gs = rq_3d.ground_state(P, verbose=False)
print(f'n={n}, lam1={lam1:.10f}, |F|={gs["newton_res"]:.2e}')
# Two configurations: the settings the production jobs actually used, and a
# much tighter one, so the comparison against the dense solver separates the
# error of the elimination (there is none) from the error of the iteration.
CFG = os.environ.get('CHECK_CFG', 'production')
if CFG == 'production':
    INNER, ITOL, BLK = 40, 1e-4, 2
    TOL, MAXIT, TOLX, MAXITX = 1e-8, 400, 1e-6, 300
else:
    INNER, ITOL, BLK = 120, 1e-9, 4
    TOL, MAXIT, TOLX, MAXITX = 1e-11, 3000, 1e-11, 3000
print(f'config {CFG}: inner={INNER}@{ITOL:g}, block={BLK}, '
      f'tol={TOL:g}/{TOLX:g}, maxit={MAXIT}/{MAXITX}')
ops = rq_3d.make_ops(P, u, lam1, ell=lam1, tf_f=0.5, inner=INNER, inner_tol=ITOL)
N3 = n ** 3


def dense(op):
    A = np.zeros((N3, N3))
    e = np.zeros((n, n, n))
    for i in range(N3):
        e.flat[i] = 1.0
        A[:, i] = np.asarray(op(e)).ravel()
        e.flat[i] = 0.0
    return 0.5 * (A + A.T)


def tangent_basis(uv):
    v = uv / np.linalg.norm(uv)
    w = v.copy()
    w[0] -= 1.0 if v[0] >= 0 else -1.0
    w /= np.linalg.norm(w)
    return (np.eye(v.size) - 2.0 * np.outer(w, w))[:, 1:]


t = time.time()
Nd = dense(ops['N'])
Q = tangent_basis(np.asarray(u).ravel())
NT = Q.T @ Nd @ Q
NT = 0.5 * (NT + NT.T)
print(f'dense N built and projected ({time.time()-t:.0f}s), '
      f'sigma_T(dense) = {sla.eigh(NT, eigvals_only=True, subset_by_index=[0,0])[0]:.10f}')
ths, _, _ = rq_3d.lobpcg_min(P, ops['N'], lambda w: w, lambda w: rq_3d.pcg(
    P, ops['N'], ops['prec_V1'], w, tol=ITOL, maxit=INNER)[0], k=BLK,
    constraint=u, tol=TOL, maxit=MAXIT, label='sigma_T')
print(f'                              sigma_T(lobpcg) = {ths[0]:.10f}')

for name in ['h1', 'a0', 'a0mod', 'au', 'a2u', 'a3u', 'tf']:
    B, Binv, exact = ops['metric'](name)
    Bd = dense(B)
    DT = Q.T @ Bd @ Q
    DT = 0.5 * (DT + DT.T)
    ev = sla.eigh(NT, DT, eigvals_only=True)
    rmin_d, rmax_d = ev[0], ev[-1]

    def prec_N(w):
        return rq_3d.pcg(P, ops['N'], ops['prec_V1'], w, tol=ITOL, maxit=INNER)[0]

    th1, _, i1 = rq_3d.lobpcg_min(P, ops['N'], B, prec_N, k=BLK, constraint=u,
                                  tol=TOL, maxit=MAXIT, label=f'{name} min')
    th2, _, i2 = rq_3d.lobpcg_min(P, B, ops['N'], Binv, k=BLK, constraint=u,
                                  tol=TOLX, maxit=MAXITX, label=f'{name} max',
                                  osc=True, seed_bump=u * u)
    rmin, rmax = float(th1[0]), 1.0 / float(th2[0])
    print(f'[{name:6s}] dense  rho_min {rmin_d:.10f}  rho_max {rmax_d:.10f}  '
          f'kappa {rmax_d/rmin_d:.6f}')
    print(f'          lobpcg rho_min {rmin:.10f}  rho_max {rmax:.10f}  '
          f'kappa {rmax/rmin:.6f}   rel.err '
          f'{abs(rmin-rmin_d)/rmin_d:.2e} / {abs(rmax-rmax_d)/rmax_d:.2e}  '
          f'({i1["iters"]}/{i2["iters"]} its {i1["why"]}/{i2["why"]})')
