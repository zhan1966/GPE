"""How exact is the rank-one penalty that rq_2d.py uses for the constraint?

rq_2d.py imposes <eta,u>_h = 0 by adding c m m^T to the numerator rather than
by building a basis of the tangent space. That is exact only as c -> infinity
in general, at rate O(1/c). This script measures the error against the truth --
the same pencil reduced onto an explicit Householder basis, as rq_1d.py does --
on a grid small enough for a dense solver, and explains why the measured error
is not O(1/c) at all for the harmonic trap:

  * the ground state is radial, so N and B commute with rotations and the
    spectrum splits by angular momentum;
  * m is proportional to u, hence in the l = 0 sector, so every l =/= 0
    eigenvector is orthogonal to m automatically;
  * the constrained minimizer is the l = +-1 doublet, so it satisfies
    (N + c m m^T) v = N v = rho B v exactly, for every c.

The only effect of a large c is then conditioning, and the measured error
*grows* with c. On a trap without that symmetry the O(1/c) branch is the
operative one and c has to be chosen with care.

Run:  python3 rq_2d_check.py [n]     (default 40, i.e. 1600 unknowns)
"""
import os
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq_2d

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
beta = 100.0
R = (2 * beta / np.pi) ** 0.25
L = max(6.0, R + 3.5)
h, X, Y, V, Lap = rq_2d.build(beta, L, n)
u, lam1, its, res = rq_2d.ground_state(beta, h, X, Y, V, Lap)
print(f'n = {n}^2 = {n*n}, beta = {beta:g}, lam1 = {lam1:.10f}, |F| = {res:.1e}')

N = (Lap + sp.diags(V + 3 * beta * u ** 2 - lam1)).toarray()
N = 0.5 * (N + N.T)
B = (Lap + sp.diags(V + beta * u ** 2)).toarray()      # the a_u metric
B = 0.5 * (B + B.T)
m = u / np.linalg.norm(u)

# the truth: explicit Householder basis of m-perp, dense generalized eigh
w = m.copy()
w[0] -= 1.0 if m[0] >= 0 else -1.0
w /= np.linalg.norm(w)
Q = (np.eye(m.size) - 2.0 * np.outer(w, w))[:, 1:]
evT = sla.eigh(Q.T @ N @ Q, Q.T @ B @ Q, eigvals_only=True)
print(f'exact (basis elimination) : min rho = {evT[0]:.12f}   '
      f'max rho = {evT[-1]:.12f}')

# the unconstrained pencil, for contrast, and the angular structure
ev, vec = sla.eigh(N, B)
print(f'unconstrained pencil      : min rho = {ev[0]:.12f}   '
      f'max rho = {ev[-1]:.12f}')
r = np.hypot(X, Y)
print('\nfirst four unconstrained eigenpairs: parity under x -> -x tells the'
      '\nangular sector, and <m,v> whether the constraint binds')
for k in range(4):
    v = vec[:, k] / np.linalg.norm(vec[:, k])
    par = float(v @ v.reshape(n, n)[::-1, ::-1].ravel())
    print(f'  k={k}  rho={ev[k]:.12f}  parity={par:+.4f}  '
          f'|<m,v>|={abs(m @ v):.1e}  peak at r={r[np.argmax(abs(v))]:.2f}')

print(f'\n{"c/(lam1(A_3b)+8/h^2)":>22s} {"min rho (penalty)":>20s} '
      f'{"rel err":>10s} {"|<m,v>|/|v|":>12s}')
base = lam1 + 4 * 2 / h ** 2
for mult in [1e0, 1e2, 1e4, 1e6, 1e8]:
    c = mult * base
    Np = N + c * np.outer(m, m)
    val, vecp = sla.eigh(0.5 * (Np + Np.T), B, subset_by_index=[0, 0])
    v = vecp[:, 0]
    tag = '  (the value rq_2d.py uses)' if mult == 1e4 else ''
    print(f'{mult:22.0e} {val[0]:20.12f} '
          f'{abs(val[0]-evT[0])/evT[0]:10.2e} '
          f'{abs(m @ v)/np.linalg.norm(v):12.2e}{tag}')
