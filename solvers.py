"""
solvers.py -- the methods compared in Section 6 of Li, Zheng and Zhang,
"Global convergence of an efficient splitting method for the defocusing
Gross-Pitaevskii ground state problem", arXiv:2609.13509 (2026).  This
repository uses the Davis-Yin scheme for the reference solutions, the
Thomas-Fermi initialization and the recording helpers.

Python ports of earlier MATLAB implementations:

    riemannian_gf      discrete H^1 gradient flow, exact line search
                       ("Riemannian GF"); with dt fixed it is
                       ("Riemannian GF constant stepsize")
    riemannian_cg      Danaila-Protas RCG, Fletcher-Reeves, Armijo
    preconditioned_gf  Antoine-Levitt-Tang PG
    preconditioned_cg  Antoine-Levitt-Tang PCG
    davis_yin          the Davis-Yin splitting scheme (dys)
    au_flow            the a_u flow of Henning-Peterseim, with the metric
                       g_u(w,z) = (grad w, grad z) + (w,Vz) + beta (w, u^2 z),
                       whose solve is done by PCG preconditioned with the
                       separable (-Lap_h + V1)^{-1}

Initializations: init_const, init_tf, init_linear.

Each returns a dict of equal-length arrays: iter, time, error, residue,
linsys, energy, step.  History is recorded after each step, on the new iterate,
as in the MATLAB implementations.

The MATLAB line searches call fminbnd on the full energy, which costs ~20
Laplacian applications per iteration.  Here the same one-dimensional function is
evaluated from its exact polynomial/trigonometric expansion (gpe3d.line_energy,
gpe3d.circle_energy), so a line search costs two applications instead.  The
minimizer is identical to roundoff; only the cost changes.
"""

import time as _time

import numpy as np
from scipy.optimize import minimize_scalar

EPS = float(np.finfo(np.float64).eps)

# scipy's bounded Brent (and MATLAB's fminbnd) default to a step tolerance of
# 1e-5 / 1e-4.  That is coarser than the step the preconditioned methods need
# once the gradient is small: the great-circle angle theta falls below the
# tolerance, the line search can no longer resolve it, and the iteration stalls
# around residue 1e-6 while oscillating.  Every evaluation here is an O(1)
# polynomial, so a tight tolerance costs nothing.
XATOL = {'xatol': 1e-14}


COLS = ('iter', 'time', 'error', 'lam_error', 'u_error', 'residue',
        'linsys', 'energy', 'eigenvalue', 'step')


class _Hist:
    def __init__(self):
        self.d = {k: [] for k in COLS}

    def add(self, **kw):
        for k, v in kw.items():
            self.d[k].append(v)

    def out(self):
        return {k: np.asarray(v) for k, v in self.d.items()}


def _ray_energy(P, u, d, t, mode, cf=None):
    """E(retraction(u, t d)).

    mode='closedform' -- from the closed-form expansion: two applications of A per
                      line search, however many t are probed.
    mode='fminbnd'    -- by building the iterate and evaluating the energy, which
                      is what fminbnd does in the MATLAB: one application of A
                      per probe.
    """
    if mode == 'closedform':
        return P['line_energy'](cf, t)
    P['counter'].evals += 1
    return float(P['ray_energy_at'](u, d, t))


def _line_search(P, u, d, mode, cf=None):
    """Minimize E along the retracted ray; returns the step size."""
    opts = XATOL if mode == 'closedform' else {'xatol': 1e-4}
    return float(minimize_scalar(
        lambda t: _ray_energy(P, u, d, t, mode, cf),
        bounds=(EPS, 5.0), method='bounded', options=opts).x)


def _rgrad(P, u, g):
    """Riemannian gradient g_alpha(u) for the shifted H^1 metric, and G_alpha u.

    proj_t(Egrad_X(u), u); costs two metric solves.
    """
    Gu = P['solve_G'](u)
    Ge = P['solve_G'](g)
    P['counter'].n += 2
    return P['proj_t'](Ge, u, Gu), Gu


def _record_vals(P, hist, kk, u, ref, t0, step, E, lam, res):
    """Record an iterate whose diagnostics a fused kernel already returned."""
    E, lam, res = float(E), float(lam), float(res)
    if ref.get('uerr') is not None:
        uerr = ref['uerr'](u)
    elif ref.get('u') is not None:
        uerr = float(P['norm_M'](u - ref['u']))
    else:
        uerr = np.nan
    hist.add(iter=kk, time=_time.time() - t0, error=abs(ref['E'] - E),
             lam_error=abs(ref['lam'] - lam), u_error=uerr, residue=res,
             linsys=P['counter'].n, energy=E, eigenvalue=lam, step=step)
    return E, res


def _record(P, hist, kk, u, ref, t0, step):
    """Record one iterate.  ref carries E*, lambda* and (optionally) u*.

    The eigenvalue is the Rayleigh quotient of the nonlinear operator,
    lambda(u) = <u, A_u u>_M = h^3 sum u . E'(u), which is free once E'(u) is
    in hand.  The eigenvector error is measured in the same discrete L^2 norm
    the iterates are normalized in, ||u - u*||_M.
    """
    E, g = P['energy_and_grad'](u)
    res = float(P['residue_from_grad'](u, g))
    E = float(E)
    lam = float(P['h3'] * P['xp'].sum(u * g))
    if ref.get('uerr') is not None:          # u* kept on the host
        uerr = ref['uerr'](u)
    elif ref.get('u') is not None:
        uerr = float(P['norm_M'](u - ref['u']))
    else:
        uerr = np.nan
    hist.add(iter=kk, time=_time.time() - t0, error=abs(ref['E'] - E),
             lam_error=abs(ref['lam'] - lam), u_error=uerr, residue=res,
             linsys=P['counter'].n, energy=E, eigenvalue=lam, step=step)
    return E, g, res


def _converged(err, res, tols):
    return err < tols['error_tol'] or res < tols['res_tol']


def _tols(error_tol=1e-12, res_tol=1e-12):
    return dict(error_tol=error_tol, res_tol=res_tol)


# --------------------------------------------------------------------------
def riemannian_gf(P, u0, maxit, ref, dt=None, tols=None, verbose=0,
                  linesearch='closedform'):
    """H^1 gradient flow.  dt=None -> line search; else the constant step size.

    With a constant step the whole iteration is one fused kernel.  With a line
    search the Riemannian gradient and the diagnostics are one fused kernel and
    the search runs on the host.
    """
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    t0 = _time.time()
    u = u0

    if dt is not None:
        for kk in range(1, maxit + 1):
            u, E, lam, res = P['h1_step_fused'](u, dt)
            P['counter'].n += 2
            E, res = _record_vals(P, hist, kk, u, ref, t0, float(dt),
                                  E, lam, res)
            if verbose and (kk % verbose == 0):
                print(f"  RGF   {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                      f"res={res:.3e}")
            if _converged(abs(ref['E'] - E), res, tols):
                break
        return hist.out()

    gc, _, E, lam, res = P['rgrad_fused'](u)
    P['counter'].n += 2
    direction = -gc
    gc = None                            # direction holds it; do not pin both
    for kk in range(1, maxit + 1):
        cf = P['ls_coeffs_fused'](u, direction) if linesearch == 'closedform' else None
        step = _line_search(P, u, direction, linesearch, cf)
        u = P['normalize'](u + step * direction)
        gc, _, E, lam, res = P['rgrad_fused'](u)
        P['counter'].n += 2
        E, res = _record_vals(P, hist, kk, u, ref, t0, step, E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  RGF   {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}")
        if _converged(abs(ref['E'] - E), res, tols):
            break
        direction = -gc
        gc = None
    return hist.out()


# --------------------------------------------------------------------------
def riemannian_cg(P, u0, maxit, ref, tols=None, verbose=0, beta_cg_type='FR',
                  linesearch='closedform'):
    """Riemannian CG -- Fletcher-Reeves in <,>_alpha, Armijo from the 1-D minimizer."""
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    ip_alpha = P['ip_alpha']
    u = u0
    gc, _Gu, E, lam, res = P['rgrad_fused'](u)
    P['counter'].n += 2
    direction = -gc
    ip_gc_old = float(ip_alpha(gc, gc))
    initslope = -ip_gc_old
    gc = _Gu = None                      # neither is read again

    t0 = _time.time()
    for kk in range(1, maxit + 1):
        cf = P['ls_coeffs_fused'](u, direction) if linesearch == 'closedform' else None
        init = _line_search(P, u, direction, linesearch, cf)
        Epre = _ray_energy(P, u, direction, 0.0, linesearch, cf)
        step = init
        while (Epre - _ray_energy(P, u, direction, step, linesearch, cf)
               < -1e-4 * initslope * step):
            step *= 0.5
            if step < EPS:
                step = EPS
                break

        u_new = P['normalize'](u + step * direction)
        u = None                         # freed before the next vectors form
        gc_new, Gu_new, E, lam, res = P['rgrad_fused'](u_new)
        P['counter'].n += 2
        E, res = _record_vals(P, hist, kk, u_new, ref, t0, step, E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  RCG   {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}")
        if _converged(abs(ref['E'] - E), res, tols):
            break

        ip_gc_new = float(ip_alpha(gc_new, gc_new))
        beta_cg = ip_gc_new / ip_gc_old          # Fletcher-Reeves
        direction, slope = P['rcg_dir_fused'](gc_new, direction, u_new,
                                              Gu_new, beta_cg)
        Gu_new = None
        if float(slope) >= 0.0:                  # not a descent direction
            direction = -gc_new
        gc_new = None
        initslope = float(slope) if float(slope) < 0.0 else -ip_gc_new
        u, ip_gc_old = u_new, ip_gc_new
        u_new = None
    return hist.out()


# --------------------------------------------------------------------------
def _great_circle(P, u, d, linesearch='closedform'):
    """Minimize E along the great circle through u in the direction d.

    The tangent projection and its normalization happen inside the fused
    kernels, so neither is ever a live Python-level n^3 buffer.
    """
    if linesearch == 'closedform':
        cf = P['gc_coeffs_fused'](u, d)
        theta = float(minimize_scalar(
            lambda t: P['circle_energy'](cf, t),
            bounds=(-np.pi, np.pi), method='bounded', options=XATOL).x)
    else:
        def f(t):
            P['counter'].evals += 1
            return float(P['gc_energy_at'](u, d, t))
        theta = float(minimize_scalar(
            f, bounds=(-np.pi, np.pi), method='bounded',
            options={'xatol': 1e-4}).x)
    return P['gc_step_fused'](u, d, theta), theta


def preconditioned_gf(P, u0, maxit, ref, tols=None, verbose=0,
                      linesearch='closedform'):
    """Antoine-Levitt-Tang PG, one metric solve per iteration.

    The evaluation after the step and the one before the next step are the same
    quantity at the same iterate, so pc_fused is called once per iteration and
    its result carried.  Calling it at both ends, which an earlier version did,
    doubles the elliptic solves the method actually needs and makes it slower
    than the H^1 flow despite doing half the work per iteration.
    """
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    u = u0
    t0 = _time.time()
    _, Pr, E, lam, res = P['pc_fused'](u)
    P['counter'].n += 1
    for kk in range(1, maxit + 1):
        u, theta = _great_circle(P, u, -Pr, linesearch)
        Pr = None                        # freed before its replacement forms
        _, Pr, E, lam, res = P['pc_fused'](u)
        P['counter'].n += 1
        E, res = _record_vals(P, hist, kk, u, ref, t0, theta, E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  PGF   {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}")
        if _converged(abs(ref['E'] - E), res, tols):
            break
    return hist.out()


def preconditioned_cg(P, u0, maxit, ref, tols=None, verbose=0,
                      beta_cg_type='FR', linesearch='closedform'):
    """Antoine-Levitt-Tang PCG.

    This is the method that does not reach n=999 on one GPU, and the reason is
    not the steady-state working set: that is 10 n^3, the same as the H^1 flow,
    which does run there.  Two things the count does not show:

      * it carries four n^3 vectors across the iteration boundary, against two
        for the flow, so its line search asks for the contiguous 3 n^3 that
        apply_A needs while twice as much is pinned;
      * the loop below forms u_new, r_new and Pr_new before rebinding over the
        old ones, so it transiently holds seven or eight n^3 vectors.

    An earlier version of this loop rebound u, r, Pr and d in a single tuple
    assignment at the end, so the old four were still live while the new four
    were built -- eight n^3 vectors at the worst moment, and the great-circle
    search asking for its contiguous 3 n^3 with all four old ones pinned.  It
    completed three iterations at n=999 and died on the fourth.  The loop below
    releases each vector as soon as it is dead and never materializes the
    residual at all under Fletcher-Reeves, which needs only the scalar
    <r, Pr>.  The carried state is then u and d, the same two the H^1 flow
    carries.
    """
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    ip_M = P['ip_M']
    if beta_cg_type not in ('FR', 'PrPlus'):
        raise ValueError(beta_cg_type)
    keep_r = beta_cg_type == 'PrPlus'   # only that rule needs the old residual
    u = u0
    r, Pr, E, lam, res = P['pc_fused'](u)
    P['counter'].n += 1
    d = -Pr
    denom = float(ip_M(r, Pr))
    Pr = None                           # d carries it from here on
    if not keep_r:
        r = None
    t0 = _time.time()
    for kk in range(1, maxit + 1):
        u_new, theta = _great_circle(P, u, d, linesearch)
        u = None                        # freed before the next vectors form
        r_new, Pr_new, E, lam, res = P['pc_fused'](u_new)
        P['counter'].n += 1
        num = float(ip_M(r_new, Pr_new))
        if keep_r:
            beta_cg = max(0.0, float(ip_M(r_new - r, Pr_new)) / denom)
            r = r_new
        else:
            beta_cg = num / denom
        r_new = None
        d = P['pcg_dir_fused'](Pr_new, d, beta_cg)
        Pr_new = None
        u, denom = u_new, num
        u_new = None
        E, res = _record_vals(P, hist, kk, u, ref, t0, theta, E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  PCG   {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}")
        if _converged(abs(ref['E'] - E), res, tols):
            break
    return hist.out()


# --------------------------------------------------------------------------
def davis_yin(P, u0, maxit, ref, tols=None, verbose=0,
              stepsize='standard', c_scale=1.5, lyap_every=1,
              c_init=2.0, c_max=3.0, grow=1.02, shrink=0.5):
    """Davis-Yin splitting (dys); one fused kernel per iteration.

    The step size enters the resolvent, not a direction, so there is no closed
    form for it and no cheap line search.  Four rules:

    'standard'  gamma_k = 1/L_k with L_k = max_i(V_i + 3 beta (u^k_i)^2), the
                Lipschitz constant of grad h at the iterate.  This is what the
                scheme is analyzed with, and it is conservative by about a
                factor of two.

    'scaled'    gamma_k = c/L_k for a fixed c (default 1.5).  The iteration
                count falls monotonically in c well past 1, but the value at
                which it stops falling is problem-dependent -- c=2 is the best
                tried on the stirrer and diverges on the lattice from the
                constant start -- so c=1.5, the largest stable everywhere
                tried, is the default.

    'lyap'      gamma_k = argmin_gamma L(z^{k+1}(gamma)), a line search on the
                Lyapunov function L = E_h(u) + R(z)^2/gamma of the theorem.
                This is the quantity the scheme decreases; the energy is not
                monotone along the iteration, and searching or backtracking on
                it misbehaves.  The search recovers the problem-dependent ratio
                gamma_k L_k on its own.

                A trial gamma costs a whole step, since gamma sits inside the
                resolvent and there is no polynomial in it to expand -- unlike
                the line searches of the gradient flows, where tau multiplies a
                direction.  Searching at every iteration is therefore about
                twelve times the cost of a step, and almost all of it is waste:
                the minimizing gamma moves by around 5e-5 in relative terms
                from one iteration to the next.  So the search runs only every
                `lyap_every` iterations and, after the first, over a narrow
                bracket around the previous value.

    'lyap_safe' gamma_k = c_k/L_k with c_k carried between iterations, halved
                on any step that increases L and grown slowly otherwise.  Costs
                one extra step only on the iterations that backtrack, and never
                failed on any problem tried.
    """
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    z = u0
    c_k = None
    gamma_prev, last_search, c_lyap = None, -10 ** 9, None
    t0 = _time.time()
    for kk in range(1, maxit + 1):
        if stepsize == 'standard':
            z, u, E, lam, res, gamma = P['dys_fused'](z)
            P['counter'].n += 1
        else:
            Lk = float(P['dys_L'](P['normalize'](z)))
            if stepsize == 'scaled':
                gamma = c_scale / Lk
            elif stepsize == 'lyap':
                due = (gamma_prev is None
                       or (kk - last_search) >= lyap_every)
                if due:
                    zc = z
                    nev = [0]

                    def obj(g):
                        nev[0] += 1
                        return float(P['dys_lyapunov_at'](zc, g))
                    # Always the full bracket.  Warm-starting the search in a
                    # narrow window around the previous gamma was tried and is
                    # worse, not better: minimizing L for a single step favours
                    # an overshooting gamma, so a myopic search drifts upward
                    # from wherever it starts and eventually destabilizes.  The
                    # full bracket is re-solved from scratch each time.
                    lo, hi, tol = 1e-3 / Lk, 4.0 / Lk, 1e-4 / Lk
                    gamma = float(minimize_scalar(
                        obj, bounds=(lo, hi), method='bounded',
                        options={'xatol': tol}).x)
                    P['counter'].n += nev[0]    # every trial is a whole step
                    gamma_prev, last_search = gamma, kk
                    c_lyap = gamma * Lk         # keep the RATIO, not gamma
                else:
                    # Between searches the ratio is held and the step size
                    # still tracks L_k.  Holding gamma itself would be the
                    # frozen rule, which fails outright from a far initial
                    # guess.
                    gamma = c_lyap / Lk
            elif stepsize == 'lyap_safe':
                if c_k is None:
                    c_k = c_init
                zc = z
                while True:
                    gamma = c_k / Lk
                    if (float(P['dys_lyapunov_at'](zc, gamma))
                            <= float(P['dys_lyapunov'](zc, gamma))
                            or c_k <= 0.05):
                        c_k = min(c_k * grow, c_max)
                        break
                    P['counter'].n += 1       # the rejected trial
                    c_k *= shrink
            else:
                raise ValueError(f"unknown stepsize {stepsize!r}")
            z, u, E, lam, res = P['dys_fused_gamma'](z, gamma)
            P['counter'].n += 1
        E, res = _record_vals(P, hist, kk, u, ref, t0, float(gamma),
                              E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  DYS   {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}  gamma*L={float(gamma)*Lk if stepsize!='standard' else 1.0:.3f}")
        if _converged(abs(ref['E'] - E), res, tols):
            break
    out = hist.out()
    out['c_eff'] = c_k
    return out


# --------------------------------------------------------------------------
def davis_yin2(P, u0, maxit, ref, tols=None, verbose=0, c_scale=1.0):
    """DYS II: the separable part of the potential is treated implicitly.

    The step is (dys2): the resolvent is (I - gamma Lap_h + gamma V1)^{-1},
    which costs the same single per-axis solve as DYS I, and only V2 and the
    cubic are explicit.  The step size is gamma_k = c / L_k with
    L_k = max_i(V2_i + 3 beta (u_i^k)^2), so on a fully separable potential --
    the optical lattice, where V1 = V and V2 = 0 -- the potential drops out of
    the step size entirely and L_k is set by the iterate alone.

    That is the whole point: for DYS I, max_i V_i is attained at the corner of
    the box where the ground state is zero, so the step size is limited by a
    region that contributes nothing.
    """
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    z = u0
    t0 = _time.time()
    for kk in range(1, maxit + 1):
        Lk = float(P['dys2_L'](P['normalize'](z)))
        gamma = c_scale / Lk
        z, u, E, lam, res = P['dys2_fused_gamma'](z, gamma)
        P['counter'].n += 1
        E, res = _record_vals(P, hist, kk, u, ref, t0, float(gamma),
                              E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  DYS2  {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}  gamma*L={float(gamma)*Lk:.3f}")
        if _converged(abs(ref['E'] - E), res, tols):
            break
    return hist.out()


# --------------------------------------------------------------------------
def reference_solution(P, method='dys', maxit=6000, stall_below=1e-11,
                       e_stall=1e-15, e_patience=200, verbose=0):
    """E*, lambda* and u* on this grid, from a Thomas-Fermi start.

    Section 6 of Li, Zheng and Zhang (2026) takes the reference from a flow
    run to residual stall.  Davis-Yin
    is the default because it is the one method proved to converge from any
    positive start, so the reference never depends on a step size that might
    not have been safe; `method='rgf'` runs the H^1 flow with its exact line
    search instead.

    The residue is not monotone in the transient, so "stopped decreasing" is
    only accepted once the residue is already below `stall_below`.  Run this
    well past where the comparisons stop: the reference has to be accurate
    enough that no plotted curve bottoms out on it.
    """
    u0, lam_tf = init_tf(P)
    P['counter'].reset()
    u = u0
    z = u0
    if method != 'dys':
        gc, _, _, _, _ = P['rgrad_fused'](u)
        P['counter'].n += 2
        direction = -gc

    old_res, E, lam = np.inf, np.nan, np.nan
    E_prev, e_flat = None, 0
    for kk in range(1, maxit + 1):
        if method == 'dys':
            z, u, E, lam, res, _ = P['dys_fused'](z)
            P['counter'].n += 1
        else:
            cf = P['ls_coeffs_fused'](u, direction)
            step = float(minimize_scalar(
                lambda t: P['line_energy'](cf, t),
                bounds=(EPS, 5.0), method='bounded', options=XATOL).x)
            u = P['normalize'](u + step * direction)
            gc, _, E, lam, res = P['rgrad_fused'](u)
            P['counter'].n += 2
            direction = -gc
        E, lam, res = float(E), float(lam), float(res)
        if verbose and kk % verbose == 0:
            print(f"  ref {kk:5d}  E={E:.15f}  lam={lam:.12f}  res={res:.3e}")
        # Two stopping tests.  The residual one is the primary; the energy one
        # is a backstop, because the residual can settle onto a floor slightly
        # above stall_below and then never trigger, leaving the loop to run out
        # its iteration cap long after the energy has stopped moving.  Observed
        # at n=999, where the residual floors at 1.66e-11.
        if E_prev is not None and abs(E - E_prev) <= e_stall * max(abs(E), 1.0):
            e_flat += 1
        else:
            e_flat = 0
        E_prev = E
        if e_flat >= e_patience:
            if verbose:
                print(f"  reference: energy flat to {e_stall:.0e} for "
                      f"{e_patience} iterations, stopping at {kk} "
                      f"(res={res:.3e})")
            break
        if res < 1e-14 or (res > old_res and res < stall_below):
            if verbose:
                print(f"  reference: residue stalled at iteration {kk} "
                      f"(res={res:.3e})")
            break
        old_res = min(old_res, res)
    return dict(E=E, lam=lam, u=u), kk, lam_tf


# --------------------------------------------------------------------------
def _au_f_tau(cf, t):
    """Energy along the a_u update ray, Remark 4.3 of Henning and Peterseim,
    SIAM J. Numer. Anal. 58 (2020).

    cf = (alpha0, alpha1, alpha2, zeta0, zeta1, zeta2, beta0..beta4).  With
    s(t) = ||(1-t)p + t q||^{-1}, the energy of the normalized update is
        f(t) = 1/2 ( s^2 A(t) + s^4 B(t) ),
    A and B the quadratic and quartic forms whose coefficients these are.  No
    n^3 work here: every evaluation is a handful of scalar operations, which is
    what makes the search free.
    """
    a0, a1, a2, z0, z1, z2, b0, b1, b2, b3, b4 = cf
    o = 1.0 - t
    A = o * o * a0 + o * t * a1 + t * t * a2
    Z = o * o * z0 + o * t * z1 + t * t * z2
    B = (o ** 4 * b0 + o ** 3 * t * b1 + o * o * t * t * b2
         + o * t ** 3 * b3 + t ** 4 * b4)
    return 0.5 * (A / Z + B / (Z * Z))


def au_flow(P, u0, maxit, ref, tols=None, verbose=0, tau=1.0,
            pcg_tol=1e-8, pcg_max=200, P32=None, au_refine=1, au_tol32=1e-6):
    """a_u flow of Henning and Peterseim (2020).

    Each step solves (-Lap_h + V + beta u^2) w = u by PCG preconditioned with
    the separable (-Lap_h + V1)^{-1}, then moves along dir = gamma w - u with
    gamma = <u,u>_M/<u,w>_M and retracts by normalization.  At tau = 1 this is
    a normalized inverse iteration in the a_u metric.

    P32, when supplied, runs the whole inner solve in single precision.  BE
    CAREFUL WITH IT: at tau = 1 the outer step is u <- gamma w / ||gamma w||,
    so the inner solution *is* the next iterate, and its accuracy is inherited
    directly.  An fp32 Krylov recurrence drives only the fp32 residual to zero,
    so w is accurate to about 1e-7 however tight pcg_tol is set, the ground
    state error floors near 1e-7 and the energy error near its square.
    Measured at n=399: fp64 reaches 9.9e-13 in 76 outer steps and 28.9s, fp32
    tracks it until 1.2e-7 and then floors at 6.9e-10, still there after 150
    steps and 43.2s -- slower as well as wrong, because it never converges and
    spends the whole budget.

    An earlier version of this docstring justified the fp32 path by saying that
    preconditioned CG converges on the true residual whatever precision the
    preconditioner is computed in.  That is a true statement about a different
    algorithm: it holds when only the preconditioner is fp32 and the residual
    and the Krylov vectors are fp64.  Here everything is fp32, so it does not
    apply.  Use fp32 only where fp64 does not fit, and prefer iterative
    refinement -- fp64 residual, fp32 correction -- if that case arises.
    """
    tols = tols or _tols()
    hist = _Hist()
    P['counter'].reset()
    xp = P['xp']
    u = u0
    t0 = _time.time()
    for kk in range(1, maxit + 1):
        if P32 is None:
            w = P['solve_V1'](u)                   # warm start
            P['counter'].n += 1
            r = u - P['apply_Au'](u, w)
            z = P['solve_V1'](r)
            P['counter'].n += 1
            p = z
            rz = xp.sum(r * z)
            rhs_norm = float(xp.sqrt(xp.sum(u * u)))
            its = 0
            for j in range(1, pcg_max + 1):
                w, r, p, rz, rn = P['au_cg_body'](u, w, r, p, rz)
                P['counter'].n += 1
                its = j
                if float(rn) / rhs_norm < pcg_tol:
                    break
        else:
            # Mixed precision by iterative refinement.  The Krylov work stays
            # fp32, which is what makes the method fit on one card at n=999,
            # but the residual is formed and the solution accumulated in fp64.
            # Forming u - A_u w cancels about seven digits, so that subtraction
            # is the one step that cannot be done in fp32: doing it there is
            # what floored the old version near 1e-7.
            f32, f64 = P32['dtype'], P['dtype']
            u32 = u.astype(f32)

            def solve32(rhs32, tol):
                """fp32 PCG for A_u x = rhs32, from the separable warm start."""
                x = P32['solve_V1'](rhs32)
                P['counter'].n += 1
                rr = rhs32 - P32['apply_Au'](u32, x)
                zz = P32['solve_V1'](rr)
                P['counter'].n += 1
                pp = zz
                rzl = xp.sum(rr * zz)
                nrm = float(xp.sqrt(xp.sum(rhs32 * rhs32)))
                nit = 0
                for j in range(1, pcg_max + 1):
                    x, rr, pp, rzl, rn = P32['au_cg_body'](u32, x, rr, pp, rzl)
                    P['counter'].n += 1
                    nit = j
                    if nrm == 0.0 or float(rn) / nrm < tol:
                        break
                return x, nit

            # How tight to drive the fp32 solve.  Refinement squares the
            # accuracy per pass, so eps^(refine+1) has to clear fp64: 1e-6 with
            # one pass, 1e-4 with two.  Asking for less than fp32 can deliver
            # is the trap -- on a potential whose separable part is a poor
            # preconditioner (the stirrer, whose Gaussian lives entirely in
            # V_2) the fp32 residual plateaus just above 1e-6 and every solve
            # then runs to pcg_max for nothing.  Prefer a loose tolerance and
            # another refinement pass.
            tol32 = au_tol32
            x32, its = solve32(u32, tol32)
            w = x32.astype(f64)
            del x32

            rhs64 = float(xp.sqrt(xp.sum(u * u)))
            for _ in range(au_refine):
                r64 = u - P['apply_Au'](u, w)          # fp64: the crucial step
                P['counter'].n += 1
                if float(xp.sqrt(xp.sum(r64 * r64))) / rhs64 < pcg_tol:
                    del r64
                    break
                r32 = r64.astype(f32)
                del r64
                d32, nit = solve32(r32, tol32)
                del r32
                w = w + d32.astype(f64)                # fp64 accumulation
                del d32
                its += nit
            del u32
        if tau == 1.0:
            u, E, lam, res, gamma = P['au_update'](u, w)
        else:
            gamma = P['ip_M'](u, u) / P['ip_M'](u, w)
            q = gamma * w
            if tau == 'adaptive':
                # Remark 4.3 of Henning and Peterseim: twelve inner
                # products, then a scalar search.
                # No extra linear solve, which is why the authors recommend it.
                import numpy as _np
                ap = _np.asarray(P['au_tau_alpha_p'](u, q), dtype=_np.float64)
                aq = float(P['au_tau_alpha_q'](q))
                mo = _np.asarray(P['au_tau_moments'](u, q), dtype=_np.float64)
                cf = (ap[0], ap[1], aq,
                      mo[0], mo[1], mo[2],
                      mo[3], mo[4], mo[5], mo[6], mo[7])
                t = float(minimize_scalar(lambda x: _au_f_tau(cf, x),
                                          bounds=(1e-3, 2.0 - 1e-3),
                                          method='bounded',
                                          options=XATOL).x)
            else:
                t = tau
            u, E, lam, res = P['au_step_tau'](u, q, t)
            gamma = t
        E, res = _record_vals(P, hist, kk, u, ref, t0, float(gamma),
                              E, lam, res)
        if verbose and (kk % verbose == 0):
            print(f"  AU    {kk:5d}  E={E:.12f}  err={abs(ref['E']-E):.3e}  "
                  f"res={res:.3e}  pcg={its}")
        if _converged(abs(ref['E'] - E), res, tols):
            break
    return hist.out()


# --------------------------------------------------------------------------
# Initializations
# --------------------------------------------------------------------------
def init_const(P):
    """The normalized constant vector; positive, as Theorem 3.4 of Li, Zheng
    and Zhang (2026) needs."""
    return P['u_const']()


def init_tf(P, bisect=100):
    """Thomas-Fermi profile.

    u_TF = sqrt((lam_TF - V)_+ / beta) with lam_TF the root of
    Phi(lam) = beta^{-1} int (lam - V)_+ - 1.  Phi is increasing, so a
    bisection on [min V, max V] is unconditionally safe.  No PDE is solved.
    """
    lo, hi = P['Vmin'], P['Vmax']
    for _ in range(bisect):
        mid = 0.5 * (lo + hi)
        if float(P['tf_mass'](mid)) < 1.0:
            lo = mid
        else:
            hi = mid
    lam_tf = 0.5 * (lo + hi)
    return P['tf_profile'](lam_tf), lam_tf


def init_linear(P, max_outer=50, tol=1e-10, pcg_tol=None, pcg_max=100,
                verbose=0):
    """Linear (beta=0) ground state: the lowest eigenpair of -Lap_h + V.

    Shifted inverse iteration, sigma = 0.9 lambda_min(-Lap_h + V1), each solve
    by PCG preconditioned with the separable (-Lap_h + V1 - sigma)^{-1}.  For
    the lattice potential V = V1 the preconditioner is exact and PCG stops
    after one iteration; for the stirrer it is the preconditioner of Liu and
    Zhang (2026).
    """
    xp = P['xp']
    ip_M, norm_M = P['ip_M'], P['norm_M']
    if pcg_tol is None:
        pcg_tol = 1e-10 if P['precision'] == 'fp64' else 1e-6
    sigma = 0.9 * P['mu_min']

    def shifted(q):
        return P['lap'](q) + P['apply_V'](q) - sigma * q

    def precond(r):
        P['counter'].n += 1
        return P['solve_V1_shifted'](r, sigma)

    def pcg(f):
        fn = xp.sqrt(xp.sum(f * f))
        w = xp.zeros_like(f)
        r = f
        z = precond(r)
        p = z
        rz = xp.sum(r * z)
        for k in range(1, pcg_max + 1):
            Ap = shifted(p)
            a_k = rz / xp.sum(p * Ap)
            w = w + a_k * p
            r = r - a_k * Ap
            if float(xp.sqrt(xp.sum(r * r)) / fn) < pcg_tol:
                return w, k
            z = precond(r)
            rz_new = xp.sum(r * z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return w, pcg_max

    u = P['u_const']()
    lam_old, its = None, 0
    for it in range(1, max_outer + 1):
        w, k = pcg(u)
        its += k
        lam = sigma + 1.0 / float(ip_M(u, w))
        u = w / norm_M(w)
        if verbose:
            print(f"  linear init {it:3d}: lambda = {lam:.12f}  pcg={k}")
        if lam_old is not None and abs(lam - lam_old) / abs(lam) < tol:
            break
        lam_old = lam
    return u, lam, it, its


def make_init(P, name, verbose=0):
    """Dispatch, returning (u0, description)."""
    if name == 'const':
        return init_const(P), 'constant one'
    if name == 'tf':
        u, lam_tf = init_tf(P)
        return u, f'Thomas-Fermi (lambda_TF = {lam_tf:.6f})'
    if name == 'linear':
        u, lam, outer, its = init_linear(P, verbose=verbose)
        return u, (f'linear (beta=0) ground state '
                   f'(lambda = {lam:.9f}, {outer} inverse iterations, '
                   f'{its} PCG)')
    raise ValueError(f"unknown initialization {name!r}")
