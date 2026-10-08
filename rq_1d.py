"""
Exact Riemannian Hessian spectrum at the 1D GPE ground state (harmonic trap,
2nd-order finite differences, M = h I), for the a_u, a_0 and H^1_alpha metrics,
compared with the lower bounds of the paper (Theorem 3.4 and Corollary 3.5):

   L_gamma  = 1 - lam1/lam2(A_beta)                 (the spectral-gap bound, all metrics)
   L_mu     = mu1 / (mu1 + ||Z_-||_inf)             (Theorem 3.4; mu1 = sigma_T = min_T <(A_3b - lam1) eta,eta>/<eta,eta>)
   L_3      = 1 - lam1/lam1(A_3beta)                (Corollary 3.5 when lam1(A_3beta) >= lam2; a_u and a_0)

with Z := A_{3beta} - lam1 - G^{-1} the diagonal defect of the metric.
"""
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize_scalar
import sys, time

def setup(beta, L, N):
    h = 2.0 * L / (N + 1)
    x = -L + h * np.arange(1, N + 1)
    V = x**2
    lap_d = np.full(N, 2.0 / h**2)
    lap_o = np.full(N - 1, -1.0 / h**2)
    return h, x, V, lap_d, lap_o

def banded_solve(d, o, rhs):
    ab = np.zeros((3, d.size))
    ab[0, 1:] = o
    ab[1, :] = d
    ab[2, :-1] = o
    return sla.solve_banded((1, 1), ab, rhs)

def energy(u, h, V, lap_d, lap_o, beta):
    Su = lap_d * u
    Su[:-1] += lap_o * u[1:]
    Su[1:] += lap_o * u[:-1]
    return h * (0.5 * u @ Su + 0.5 * u @ (V * u) + 0.25 * beta * np.sum(u**4))

def apply_A(u, w, h, V, lap_d, lap_o, beta, factor=1.0):
    # (-Lap_h + V + factor*beta*u^2) w
    r = (lap_d + V + factor * beta * u**2) * w
    r[:-1] += lap_o * w[1:]
    r[1:] += lap_o * w[:-1]
    return r

def ground_state(beta, h, x, V, lap_d, lap_o, tol=1e-11, maxit=3000, verbose=False):
    N = x.size
    # Thomas-Fermi start
    R = (3 * beta / 4) ** (1 / 3)
    mu = R**2
    u = np.sqrt(np.maximum(mu - V, 0.0) / beta)
    if beta < 1:  # for tiny beta TF is poor; use Gaussian
        u = np.exp(-x**2 / 2)
    u /= np.sqrt(h * u @ u)

    def E(v):
        return energy(v, h, V, lap_d, lap_o, beta)

    # a_u Riemannian GD with exact line search
    for it in range(maxit):
        d = lap_d + V + beta * u**2
        Gu = banded_solve(d, lap_o, u)
        lam = 1.0 / (h * u @ Gu)
        g = u - lam * Gu                      # grad E in a_u metric (nabla_X E = u)
        res = np.sqrt(h * g @ (d * g + np.r_[lap_o * g[1:], 0] + np.r_[0, lap_o * g[:-1]]))  # a_u norm of grad
        if res < tol:
            break
        def phi(t):
            v = u - t * g
            v = v / np.sqrt(h * v @ v)
            return E(v)
        r = minimize_scalar(phi, bounds=(0.0, 2.0), method='bounded', options={'xatol': 1e-6})
        t = r.x
        u = u - t * g
        u /= np.sqrt(h * u @ u)
    if verbose:
        print(f"  GD iters {it}, a_u-grad norm {res:.2e}")
    # Newton polish on F(u,lam) = A_beta(u)u - lam u, (u^T M u - 1)/2 = 0
    for k in range(6):
        F = apply_A(u, u, h, V, lap_d, lap_o, beta) - lam * u
        c = 0.5 * (h * u @ u - 1.0)
        J_d = lap_d + V + 3 * beta * u**2 - lam
        y1 = banded_solve(J_d, lap_o, F)
        y2 = banded_solve(J_d, lap_o, u)
        Mu = h * u
        dlam = (Mu @ y1 - c) / (Mu @ y2)
        du = -y1 + dlam * y2
        u = u + du
        lam = lam + dlam
        nrm = np.sqrt(h * F @ F)
        if nrm < 1e-13:
            break
    u /= np.sqrt(h * u @ u)
    lam = h * u @ apply_A(u, u, h, V, lap_d, lap_o, beta)
    F = apply_A(u, u, h, V, lap_d, lap_o, beta) - lam * u
    if verbose:
        print(f"  Newton residual {np.sqrt(h*F@F):.2e}, lam1 = {lam:.10f}, min u = {u.min():.2e}")
    return u, lam

def tangent_basis(u):
    # Householder reflector mapping u/|u| to e_1; columns 2..N span u-perp (Euclidean = h-orthogonal)
    N = u.size
    v = u / np.linalg.norm(u)
    w = v.copy()
    w[0] -= 1.0 if v[0] >= 0 else -1.0
    w /= np.linalg.norm(w)
    H = np.eye(N) - 2.0 * np.outer(w, w)
    Q = H[:, 1:]
    return Q

def dense_tridiag(d, o):
    A = np.diag(d)
    idx = np.arange(d.size - 1)
    A[idx, idx + 1] = o
    A[idx + 1, idx] = o
    return A

def extreme_pencil(NT, DT):
    n = NT.shape[0]
    lo = sla.eigh(NT, DT, subset_by_index=[0, 0], eigvals_only=False)
    hi = sla.eigh(NT, DT, subset_by_index=[n - 1, n - 1], eigvals_only=True)
    return lo[0][0], lo[1][:, 0], hi[0]

def run(beta, L, N, alphas=(0.0,), verbose=True):
    h, x, V, lap_d, lap_o = setup(beta, L, N)
    t0 = time.time()
    u, lam1 = ground_state(beta, h, x, V, lap_d, lap_o, verbose=verbose)
    bu2 = beta * u**2
    # linear spectral quantities (tridiagonal eigen-solvers)
    ev_Ab = sla.eigh_tridiagonal(lap_d + V + bu2, lap_o, eigvals_only=True, select='i', select_range=(0, 2))
    ev_A3 = sla.eigh_tridiagonal(lap_d + V + 3 * bu2, lap_o, eigvals_only=True, select='i', select_range=(0, 1))
    ev_A0 = sla.eigh_tridiagonal(lap_d + V, lap_o, eigvals_only=True, select='i', select_range=(0, 1))
    lam2 = ev_Ab[1]
    lam1_A3 = ev_A3[0]
    lam1_A0 = ev_A0[0]
    assert abs(ev_Ab[0] - lam1) < 1e-6 * max(1, lam1), (ev_Ab[0], lam1)

    Q = tangent_basis(u)
    A3m = dense_tridiag(lap_d + V + 3 * bu2 - lam1, lap_o)   # constrained Hessian operator (M = hI)
    NT = Q.T @ A3m @ Q
    NT = 0.5 * (NT + NT.T)
    mu1 = sla.eigh(NT, subset_by_index=[0, 0], eigvals_only=True)[0]

    out = dict(beta=beta, L=L, N=N, h=h, lam1=lam1, lam2=lam2, lam1_A3=lam1_A3, lam1_A0=lam1_A0,
               mu1=mu1, bu2max=bu2.max(), R_TF=np.sqrt(lam1), umin=u.min())

    metrics = {}
    # a_u
    D = dense_tridiag(lap_d + V + bu2, lap_o)
    DT = Q.T @ D @ Q; DT = 0.5 * (DT + DT.T)
    rmin, vmin, rmax = extreme_pencil(NT, DT)
    eta = Q @ vmin
    metrics['a_u'] = dict(rmin=rmin, rmax=rmax, x_eta=x[np.argmax(np.abs(eta))], eta=eta,
                          Zneg=np.max(np.maximum(lam1 - 2 * bu2, 0)), Zpos=np.max(np.maximum(2 * bu2 - lam1, 0)),
                          lamT=lam2)  # tangent min of G^{-1} = lam2 for a_u
    # a_0
    D = dense_tridiag(lap_d + V, lap_o)
    DT = Q.T @ D @ Q; DT = 0.5 * (DT + DT.T)
    rmin, vmin, rmax = extreme_pencil(NT, DT)
    lamT_A0 = sla.eigh(DT, subset_by_index=[0, 0], eigvals_only=True)[0]
    metrics['a_0'] = dict(rmin=rmin, rmax=rmax, x_eta=x[np.argmax(np.abs(Q @ vmin))], eta=Q @ vmin,
                          Zneg=np.max(np.maximum(lam1 - 3 * bu2, 0)), Zpos=np.max(np.maximum(3 * bu2 - lam1, 0)),
                          lamT=lamT_A0, lam1G=lam1_A0)
    # H^1_alpha
    for al in alphas:
        D = dense_tridiag(lap_d + al, lap_o)
        DT = Q.T @ D @ Q; DT = 0.5 * (DT + DT.T)
        rmin, vmin, rmax = extreme_pencil(NT, DT)
        lam1_lap = sla.eigh_tridiagonal(lap_d, lap_o, eigvals_only=True, select='i', select_range=(0, 0))[0]
        metrics[f'H1_{al:g}'] = dict(rmin=rmin, rmax=rmax, x_eta=x[np.argmax(np.abs(Q @ vmin))], eta=Q @ vmin,
                                     Zneg=np.max(np.maximum(lam1 + al - V - 3 * bu2, 0)),
                                     Zpos=np.max(np.maximum(V + 3 * bu2 - lam1 - al, 0)),
                                     lamT=None, lam1G=lam1_lap + al, alpha=al,
                                     cond_old=(al <= np.min(V + bu2)))
    out['metrics'] = metrics
    out['time'] = time.time() - t0
    return out, x, u

def report(out):
    lam1, lam2, lam1_A3, mu1 = out['lam1'], out['lam2'], out['lam1_A3'], out['mu1']
    print(f"\n=== beta = {out['beta']:g}   (L = {out['L']}, N = {out['N']}, h = {out['h']:.4f}, time {out['time']:.1f}s) ===")
    print(f"  lam1(A_b) = {lam1:.6g}   lam2(A_b) = {lam2:.6g}   gap = {lam2-lam1:.4g}   R_TF = sqrt(lam1) = {out['R_TF']:.3f}")
    print(f"  lam1(A_3b) = {lam1_A3:.6g}   lam1(A_3b)-lam1 = {lam1_A3-lam1:.4g}   mu1 (constrained L2-Hessian gap) = {mu1:.4g}")
    print(f"  lam1(A_0) = {out['lam1_A0']:.6g}   beta*max u^2 = {out['bu2max']:.4g}  (max principle: <= lam1)  min u = {out['umin']:.1e}")
    Lg = 1 - lam1 / lam2
    L3 = 1 - lam1 / lam1_A3
    for name, m in out['metrics'].items():
        Lmu = mu1 / (mu1 + m['Zneg'])
        line = f"  [{name:6s}] rho_min = {m['rmin']:.5g}  rho_max = {m['rmax']:.5g}  kappa = {m['rmax']/m['rmin']:.4g}"
        line += f"  | argmax|eta_min| at x = {m['x_eta']:+.2f}"
        print(line)
        lb = f"           lower bounds: paper 1-lam1/lam2 = {Lg:.3g}"
        if name.startswith('H1') and not m['cond_old']:
            lb += " (paper's condition alpha<=min(V+beta u^2) FAILS)"
        lb += f"   new mu1/(mu1+|Z-|) = {Lmu:.4g}"
        if name in ('a_u', 'a_0'):
            lb += f"   new explicit 1-lam1/lam1(A_3b) = {L3:.4g}"
        print(lb)
        if name == 'a_u':
            print(f"           upper bounds: paper 3 ;  monotone 1+lam1/lam2 = {1+lam1/lam2:.4g} ;  1+|Z+|/lam2 = {1+m['Zpos']/lam2:.4g}")
        elif name == 'a_0':
            print(f"           upper bounds: paper 1+3 beta max u^2/lam1(A_0) = {1+3*out['bu2max']/m['lam1G']:.4g} ; monotone 1+2 lam1/lam1(A_0) = {1+2*lam1/m['lam1G']:.4g} ; 1+|Z+|/lam_T(A_0) = {1+m['Zpos']/m['lamT']:.4g}")
        else:
            print(f"           upper bounds: paper 1+max(V+3 beta u^2)/(lam1(-Lap)+alpha) = {1+np.max(m['Zpos']+lam1+m['alpha'])/m['lam1G']:.4g} ; 1+|Z+|/lam1(G^-1) = {1+m['Zpos']/m['lam1G']:.4g}")

if __name__ == '__main__':
    betas = [float(b) for b in sys.argv[1:]] if len(sys.argv) > 1 else [1, 10, 100, 1e3, 1e4, 1e5]
    results = []
    for beta in betas:
        R = (3 * beta / 4) ** (1 / 3)
        L = max(6.0, 1.25 * R + 4)
        N = int(__import__("os").environ.get("RQ_N", "3000"))
        out, x, u = run(beta, L, N, alphas=(0.0,), verbose=False)
        report(out)
        results.append(out)
    # scaling summary
    print("\n=== scaling summary (a_u metric) ===")
    print(" beta      lam1      rho_min    rho_max   kappa    paper_LB   new_LB(mu)  new_LB(A3)  rho_min*lam1^(2/3)")
    for o in results:
        m = o['metrics']['a_u']
        print(f" {o['beta']:8.0f}  {o['lam1']:8.3f}  {m['rmin']:.4e}  {m['rmax']:.4f}  {m['rmax']/m['rmin']:7.3f}  {1-o['lam1']/o['lam2']:.3e}  {o['mu1']/(o['mu1']+o['lam1']):.3e}  {1-o['lam1']/o['lam1_A3']:.3e}  {m['rmin']*o['lam1']**(2/3):.4f}")
    print("\n=== scaling summary (a_0 metric) ===")
    print(" beta      lam1      rho_min    rho_max   kappa    paper_LB   new_LB(mu)  new_LB(A3)")
    for o in results:
        m = o['metrics']['a_0']
        print(f" {o['beta']:8.0f}  {o['lam1']:8.3f}  {m['rmin']:.4e}  {m['rmax']:.4f}  {m['rmax']/m['rmin']:7.3f}  {1-o['lam1']/o['lam2']:.3e}  {o['mu1']/(o['mu1']+m['Zneg']):.3e}  {1-o['lam1']/o['lam1_A3']:.3e}")
    print("\n=== scaling summary (H^1_0 metric) ===")
    print(" beta      lam1      rho_min    rho_max   kappa    paper_LB   new_LB(mu)")
    for o in results:
        m = o['metrics']['H1_0']
        print(f" {o['beta']:8.0f}  {o['lam1']:8.3f}  {m['rmin']:.4e}  {m['rmax']:.4f}  {m['rmax']/m['rmin']:7.3f}  {1-o['lam1']/o['lam2']:.3e}  {o['mu1']/(o['mu1']+m['Zneg']):.3e}")
