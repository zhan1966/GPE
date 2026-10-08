"""Hastings-McLeod solution of the Painleve-II equation in the Gallo-Pelinovsky scaling of Corollary 3.11,

    4 nu'' + y nu - nu^3 = 0,    nu(y) ~ sqrt(y)  (y -> +inf),    nu(y) -> 0  (y -> -inf),

and the boundary-layer potential W0(y) = 3 nu0(y)^2 - y of the operator M0 = -4 d^2/dy^2 + W0 (Corollary 3.11)
and of the Painleve-profile metric (a development variant that the paper does not use).  Used by tf_table.py;
painleve_mu1.py keeps its own copy of the boundary value solve so that its logged output stays reproducible on
its own.  The tabulation: solve_bvp on [left, right], nu0 = 0 to the left of the window and the large-y expansion
sqrt(y) (1 - 1/(2 y^3) - 73/(8 y^6)) to the right of it.  Not a certified tabulation; the numbers that depend on
it (C_0 = nu0(0)^2 = 0.4278, quoted as 0.43 in Section 3.5 of the paper, and inf W0 = 1.2334 and the Painleve
column of tf_table_1d.log, which the paper does not quote) are stable to the digits printed under refinement of
the grid and the window."""
import numpy as np
from scipy.integrate import solve_bvp


def hastings_mcleod(left=-20.0, right=60.0, nodes=2400, tol=1e-9):
    """Return (nu0, W0): callables evaluating the profile and W0(y) = 3 nu0(y)^2 - y on the whole line."""
    def rhs(y, z):
        return np.vstack((z[1], (z[0]**3 - y * z[0]) / 4.0))

    def tail(t):
        return np.sqrt(t) * (1.0 - 0.5 / t**3 - 73.0 / (8.0 * t**6))

    def bc(a, b):
        return np.array((a[0], b[0] - tail(right)))

    y = np.linspace(left, right, nodes)
    guess = np.vstack((np.sqrt(np.maximum(y, 0.0)) + 0.3 * np.exp(-y**2 / 4.0),
                       0.5 / np.sqrt(np.maximum(y, 1.0))))
    sol = solve_bvp(rhs, bc, y, guess, tol=tol, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)

    def nu0(y):
        y = np.asarray(y, dtype=float)
        out = sol.sol(np.clip(y, left, right))[0]
        out = np.where(y < left, 0.0, out)
        big = y > right
        if np.any(big):
            out = np.where(big, tail(np.where(big, y, right)), out)
        return out

    def W0(y):
        y = np.asarray(y, dtype=float)
        return 3.0 * nu0(y)**2 - y

    return nu0, W0


def hastings_mcleod_profile(**kw):
    """W0 only, the interface the tf_*.py scripts use."""
    return hastings_mcleod(**kw)[1]


if __name__ == '__main__':
    nu0, W0 = hastings_mcleod()
    y = np.linspace(-20, 60, 200001)
    print('nu0(0)^2 = C_0 =', float(nu0(0.0)**2), '  inf W0 =', float(W0(y).min()), 'at y =', float(y[np.argmin(W0(y))]))
