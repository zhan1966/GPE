"""
2D check (harmonic trap V = x^2 + y^2, 5-point FD, M = h^2 I): extreme Riemannian Hessian
eigenvalues at the ground state for the a_u and a_0 metrics, via sparse shift-invert ARPACK on the
tangent space (constraint <eta,u>_h = 0 imposed by a large rank-one penalty on the numerator).
Compared with the spectral-gap bound 1 - lam1/lam2 and the bound 1 - lam1/lam1(A_3beta), which is the
1 - lam1/Lambda of Corollary 3.5 when lam1(A_3beta) >= lam2.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, time, sys, os
from scipy.optimize import minimize_scalar

def build(beta, L, n):
    h = 2.0 * L / (n + 1)
    x = -L + h * np.arange(1, n + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    V = (X**2 + Y**2).ravel()
    T = sp.diags([np.full(n - 1, -1.0), np.full(n, 2.0), np.full(n - 1, -1.0)], [-1, 0, 1]) / h**2
    I = sp.identity(n)
    Lap = (sp.kron(T, I) + sp.kron(I, T)).tocsr()   # -Delta_h
    return h, X.ravel(), Y.ravel(), V, Lap

def energy(u, h, V, Lap, beta):
    return h**2 * (0.5 * u @ (Lap @ u) + 0.5 * u @ (V * u) + 0.25 * beta * np.sum(u**4))

def ground_state(beta, h, X, Y, V, Lap, tol=1e-10, maxit=2000):
    N = V.size
    R = (2 * beta / np.pi) ** 0.25
    u = np.sqrt(np.maximum(R**2 - V, 0.0) / beta)
    if beta < 5:
        u = np.exp(-(X**2 + Y**2) / 2)
    u /= np.sqrt(h**2 * u @ u)
    for it in range(maxit):
        A = Lap + sp.diags(V + beta * u**2)
        lu = spla.splu(A.tocsc())
        Gu = lu.solve(u)
        lam = 1.0 / (h**2 * u @ Gu)
        g = u - lam * Gu
        res = np.sqrt(h**2 * g @ (A @ g))
        if res < tol:
            break
        def phi(t):
            v = u - t * g
            v = v / np.sqrt(h**2 * v @ v)
            return energy(v, h, V, Lap, beta)
        t = minimize_scalar(phi, bounds=(0.0, 2.0), method='bounded', options={'xatol': 1e-6}).x
        u = u - t * g
        u /= np.sqrt(h**2 * u @ u)
    # Newton polish (bordered system via Schur complement)
    for k in range(6):
        A = Lap + sp.diags(V + beta * u**2)
        F = A @ u - lam * u
        c = 0.5 * (h**2 * u @ u - 1.0)
        J = (Lap + sp.diags(V + 3 * beta * u**2 - lam)).tocsc()
        lu = spla.splu(J)
        y1 = lu.solve(F); y2 = lu.solve(u)
        Mu = h**2 * u
        dlam = (Mu @ y1 - c) / (Mu @ y2)
        u = u - y1 + dlam * y2
        lam = lam + dlam
        if np.sqrt(h**2 * F @ F) < 1e-12:
            break
    u /= np.sqrt(h**2 * u @ u)
    A = Lap + sp.diags(V + beta * u**2)
    lam = h**2 * u @ (A @ u)
    return u, lam, it, np.sqrt(h**2 * np.sum((A @ u - lam * u)**2))

def smallest_eigs(A, k=2):
    # smallest eigenvalues of symmetric sparse A via shift-invert at 0 (A must be PD)
    lu = spla.splu(A.tocsc())
    op = spla.LinearOperator(A.shape, matvec=lu.solve, dtype=float)
    vals = spla.eigsh(A, k=k, sigma=0.0, which='LM', OPinv=op, return_eigenvectors=False, tol=1e-10)
    return np.sort(vals)

def pencil_extremes(Nmat, Dmat, m, c):
    """extreme eigenvalues of  Nmat eta = rho Dmat eta  on {eta : m.eta = 0}, via penalty Nmat + c m m^T.
    Smallest: shift-invert at sigma=0 with Sherman-Morrison. Largest: eigsh(which='LA') with M=Dmat, Minv via splu."""
    n = Nmat.shape[0]
    luN = spla.splu(Nmat.tocsc())
    w = luN.solve(m)
    denom = 1.0 + c * (m @ w)
    def solve_pen(b):   # (N + c m m^T)^{-1} b
        y = luN.solve(b)
        return y - w * (c * (m @ y) / denom)
    Npen = spla.LinearOperator((n, n), matvec=lambda v: Nmat @ v + c * m * (m @ v), dtype=float)
    OPinv = spla.LinearOperator((n, n), matvec=solve_pen, dtype=float)
    lo, vlo = spla.eigsh(Npen, k=1, M=Dmat, sigma=0.0, which='LM', OPinv=OPinv, tol=1e-9, maxiter=5000)
    # largest: standard mode needs M^{-1}
    luD = spla.splu(Dmat.tocsc())
    Minv = spla.LinearOperator((n, n), matvec=luD.solve, dtype=float)
    hi, vhi = spla.eigsh(Npen, k=2, M=Dmat, Minv=Minv, which='LA', tol=1e-8, maxiter=20000)
    order = np.argsort(hi)          # hi[order[-1]] is the penalty mode (huge); hi[order[0]] is rho_max
    assert hi[order[-1]] > 100 * max(1.0, hi[order[0]]), ("penalty mode not separated", hi)
    return lo[0], vlo[:, 0], hi[order[0]], vhi[:, order[0]]

def run(beta, L, n):
    t0 = time.time()
    h, X, Y, V, Lap = build(beta, L, n)
    u, lam1, its, res = ground_state(beta, h, X, Y, V, Lap)
    bu2 = beta * u**2
    Ab = (Lap + sp.diags(V + bu2)).tocsc()
    A3 = (Lap + sp.diags(V + 3 * bu2)).tocsc()
    A0 = (Lap + sp.diags(V)).tocsc()
    e_b = smallest_eigs(Ab, 2); e_3 = smallest_eigs(A3, 1); e_0 = smallest_eigs(A0, 1)
    lam2 = e_b[1]; lam1_A3 = e_3[0]; lam1_A0 = e_0[0]
    Nmat = (A3 - lam1 * sp.identity(V.size)).tocsc()     # constrained Hessian (M = h^2 I cancels in ratios)
    m = u / np.linalg.norm(u)                              # Euclidean unit vector along u (h-orthogonality = Euclidean)
    c = 1e4 * (lam1_A3 + 4 * 2 / h**2)                     # penalty >> spectral radius
    out = dict(beta=beta, L=L, n=n, h=h, lam1=lam1, lam2=lam2, lam1_A3=lam1_A3, lam1_A0=lam1_A0,
               bu2max=bu2.max(), R_TF=np.sqrt(lam1), gd_iters=its, resid=res, metrics={})
    for name, D in (('a_u', Ab), ('a_0', A0)):
        rmin, vmin, rmax, vmax = pencil_extremes(Nmat, D, m, c)
        r = np.sqrt(X**2 + Y**2)
        out['metrics'][name] = dict(rmin=rmin, rmax=rmax, r_eta_min=r[np.argmax(np.abs(vmin))],
                                    r_eta_max=r[np.argmax(np.abs(vmax))], orth=abs(m @ vmin) / np.linalg.norm(vmin))
    out['time'] = time.time() - t0
    return out

def report(o):
    lam1, lam2, lam1_A3 = o['lam1'], o['lam2'], o['lam1_A3']
    print(f"\n=== 2D beta = {o['beta']:g}  (L = {o['L']:.2f}, n = {o['n']}^2, h = {o['h']:.4f}, GD iters {o['gd_iters']}, resid {o['resid']:.1e}, time {o['time']:.0f}s) ===")
    print(f"  lam1 = {lam1:.5g}  lam2 = {lam2:.5g}  gap = {lam2-lam1:.4g}  R_TF = {o['R_TF']:.3f}   lam1(A_3b) - lam1 = {lam1_A3-lam1:.4g}   lam1(A_0) = {o['lam1_A0']:.4f}  beta max u^2 = {o['bu2max']:.4g}")
    for name, m in o['metrics'].items():
        print(f"  [{name}] rho_min = {m['rmin']:.5g}  rho_max = {m['rmax']:.5g}  kappa = {m['rmax']/m['rmin']:.4g}  | |eta_min| peaks at r = {m['r_eta_min']:.2f}, |eta_max| peaks at r = {m['r_eta_max']:.2f}, |<m,eta_min>| = {m['orth']:.1e}")
    print(f"      lower bounds: paper 1-lam1/lam2 = {1-lam1/lam2:.3e}   new 1-lam1/lam1(A_3b) = {1-lam1/lam1_A3:.4g}")
    print(f"      upper bounds a_u: 1+lam1/lam2 = {1+lam1/lam2:.4g};  a_0: 1+2lam1/lam1(A_0) = {1+2*lam1/o['lam1_A0']:.4g}")

if __name__ == '__main__':
    betas = [float(b) for b in sys.argv[1:]] if len(sys.argv) > 1 else [10, 100, 1000, 10000]
    n = int(os.environ.get('RQ2D_N', '200'))
    res = []
    for beta in betas:
        R = (2 * beta / np.pi) ** 0.25
        L = max(6.0, R + 3.5)
        o = run(beta, L, n)
        report(o); res.append(o)
        sys.stdout.flush()
    print("\n=== 2D scaling summary (a_u) ===")
    print(" beta     lam1     rho_min     rho_max   kappa    paper_LB    new_LB    rho_min*lam1^(2/3)  kappa*beta^(-1/3)")
    for o in res:
        m = o['metrics']['a_u']
        print(f" {o['beta']:7.0f}  {o['lam1']:8.3f}  {m['rmin']:.4e}  {m['rmax']:.4f}  {m['rmax']/m['rmin']:7.3f}  {1-o['lam1']/o['lam2']:.3e}  {1-o['lam1']/o['lam1_A3']:.3e}  {m['rmin']*o['lam1']**(2/3):.4f}  {m['rmax']/m['rmin']*o['beta']**(-1/3):.4f}")
