"""Smallest eigenvalue mu_1 of M0 = -4 d^2/dy^2 + 3 nu0(y)^2 - y, where nu0 is the Hastings-McLeod solution of
4 nu'' + y nu - nu^3 = 0 with nu ~ sqrt(y) (y -> +inf), nu -> 0 (y -> -inf)  [Gallo-Pelinovsky scaling].
Prediction: lambda_1(A_{3beta}) - lambda_1(A_beta) ~ mu_1 * lambda_1(A_beta)^{1/3} for the 1D harmonic trap."""
import numpy as np
from scipy.integrate import solve_bvp
import scipy.linalg as sla

Lm, Lp = 14.0, 40.0
def rhs(y, s):            # s = [nu, nu']
    return np.vstack([s[1], (s[0]**3 - y * s[0]) / 4.0])
def bc(sa, sb):
    # y -> -inf: nu -> 0 ; y -> +inf: nu ~ sqrt(y) (1 - 1/(2 y^3) ... ) : use nu = sqrt(y) at the right end
    return np.array([sa[0], sb[0] - np.sqrt(Lp) * (1 - 1.0 / (2 * Lp**3))])
y0 = np.linspace(-Lm, Lp, 4000)
guess = np.vstack([np.sqrt(np.maximum(y0, 0.0)) + 0.3 * np.exp(-(y0**2) / 4), 0.5 / np.sqrt(np.maximum(y0, 1.0))])
sol = solve_bvp(rhs, bc, y0, guess, tol=1e-10, max_nodes=200000)
print("bvp status:", sol.status, sol.message)
# fine grid for the eigenvalue problem, Dirichlet at both ends (potential -> +inf on both sides)
N = 20000; y = np.linspace(-Lm, Lp, N + 2)[1:-1]; h = y[1] - y[0]
nu = sol.sol(y)[0]
W = 3 * nu**2 - y
d = 4 * 2.0 / h**2 + W; o = np.full(N - 1, -4.0 / h**2)
ev = sla.eigh_tridiagonal(d, o, eigvals_only=True, select='i', select_range=(0, 3))
print("inf W0 =", W.min(), " (Gallo-Pelinovsky Lemma 2.2: > 0)")
print("first eigenvalues mu_k of M0 = -4 d2 + 3 nu0^2 - y :", ev)
print("check nu0 at y = 0:", sol.sol(0.0)[0], " residual of ODE (max):", np.max(np.abs(4*np.gradient(np.gradient(nu, h), h)[5:-5] + (y*nu - nu**3)[5:-5])))
print("measured (lam1(A_3b)-lam1)/lam1^(1/3) in 1D at beta=3e5: 2.408 ; at 2D beta=1e4: 2.35")
