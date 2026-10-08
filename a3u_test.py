"""Test of the 'full-Hessian' metric B = A_{3beta}(u) = -Lap + V + 3 beta u^2 against a_u (B = A_beta):
(1) exact extreme eigenvalues of the Riemannian Hessian at the ground state (pencil (H, B) on the tangent space),
(2) iteration counts of Riemannian gradient descent with exact line search to |E - E*| < 1e-12,
    from the Thomas-Fermi start and from a perturbed start."""
import numpy as np, scipy.linalg as sla, sys, time
from scipy.optimize import minimize_scalar
import rq_1d as R

def apply_op(diag, o, w):
    r = diag * w
    r[:-1] += o * w[1:]
    r[1:] += o * w[:-1]
    return r

def gd_iterations(beta, h, x, V, lap_d, lap_o, ustar, Estar, u0, metric, tol=1e-12, maxit=5000):
    """Riemannian GD, metric 'au' (B = A_beta(u)) or 'a3u' (B = A_3beta(u)), exact line search on tau in (0, 2]."""
    E = lambda v: R.energy(v, h, V, lap_d, lap_o, beta)
    u = u0 / np.sqrt(h * u0 @ u0)
    solves = 0
    for it in range(maxit):
        if E(u) - Estar < tol:
            return it, solves
        Abu = apply_op(lap_d + V + beta * u**2, lap_o, u)          # A_beta(u) u  (= nabla E_h up to M)
        if metric == 'au':
            Bd = lap_d + V + beta * u**2
            gradX = u                                              # B^{-1} A_beta u = u
            Gu = R.banded_solve(Bd, lap_o, u); solves += 1
        else:
            Bd = lap_d + V + 3 * beta * u**2
            gradX = R.banded_solve(Bd, lap_o, Abu); solves += 1
            Gu = R.banded_solve(Bd, lap_o, u); solves += 1
        c = (h * gradX @ u) / (h * Gu @ u)                          # <w,u>_h / <Gu,u>_h
        g = gradX - c * Gu                                          # Riemannian gradient
        def phi(t):
            v = u - t * g; v = v / np.sqrt(h * v @ v); return E(v)
        t = minimize_scalar(phi, bounds=(0.0, 2.0), method='bounded', options={'xatol': 1e-8}).x
        u = u - t * g; u /= np.sqrt(h * u @ u)
    return maxit, solves

for beta in [float(b) for b in sys.argv[1:]] or [100.0, 1000.0, 10000.0]:
    Rtf = (3 * beta / 4) ** (1 / 3); L = max(6.0, 1.25 * Rtf + 4); N = 2000
    h, x, V, lap_d, lap_o = R.setup(beta, L, N)
    u, lam1 = R.ground_state(beta, h, x, V, lap_d, lap_o, verbose=False)
    bu2 = beta * u**2
    Q = R.tangent_basis(u)
    NT = Q.T @ R.dense_tridiag(lap_d + V + 3 * bu2 - lam1, lap_o) @ Q; NT = 0.5 * (NT + NT.T)
    sigT = sla.eigh(NT, subset_by_index=[0, 0], eigvals_only=True)[0]
    res = {}
    for name, dg in (('a_u', lap_d + V + bu2), ('a_3u', lap_d + V + 3 * bu2)):
        DT = Q.T @ R.dense_tridiag(dg, lap_o) @ Q; DT = 0.5 * (DT + DT.T)
        rmin, _, rmax = R.extreme_pencil(NT, DT)
        res[name] = (rmin, rmax)
    print(f"\n=== beta = {beta:g}: lam1 = {lam1:.4f}, sigma_T = {sigT:.4f}, lam1/(lam1+sigma_T) = {lam1/(lam1+sigT):.5f} ===")
    for name, (rmin, rmax) in res.items():
        k = rmax / rmin
        print(f"  {name:5s}: rho_min = {rmin:.5f}  rho_max = {rmax:.5f}  kappa = {k:7.2f}  "
              f"rate tau=1: {max(1-rmin, rmax-1):.5f}   optimal-tau rate (kappa-1)/(kappa+1): {(k-1)/(k+1):.5f}")
    print(f"  prediction for a_3u: rho_min = sigma_T/(lam1+sigma_T) = {sigT/(lam1+sigT):.5f}, kappa ~ 1 + lam1/sigma_T = {1+lam1/sigT:.2f}")
    Estar = R.energy(u, h, V, lap_d, lap_o, beta)
    uTF = np.sqrt(np.maximum(Rtf**2 - V, 0.0) / beta)
    rng = np.random.default_rng(0)
    upert = u * (1 + 0.05 * rng.standard_normal(u.size)) + 0.02 * np.exp(-(x - Rtf)**2)   # excites the boundary layer
    for label, u0 in (('TF start', uTF), ('perturbed start', upert)):
        for metric in ('au', 'a3u'):
            t0 = time.time()
            its, solves = gd_iterations(beta, h, x, V, lap_d, lap_o, u, Estar, u0, metric)
            print(f"  GD exact line search, {label:15s} {metric:4s}: {its:4d} iterations, {solves:4d} banded solves  ({time.time()-t0:.1f}s)")
