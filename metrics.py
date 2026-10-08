#!/usr/bin/env python3
"""Riemannian gradient descent and CG on M under seven Sobolev metrics.

The paper equips R^N with a metric g_u(xi, eta) = xi^T M G(u)^{-1} eta, one per
choice of discrete Green operator G(u):

    h1     G = (-Lap_h + alpha I)^{-1}          modified H^1, alpha >= 0
    a0     G = (-Lap_h + V)^{-1}                a_0
    a0mod  G = (-Lap_h + V_1)^{-1}              modified a_0, V_1 separable
    au     G = (-Lap_h + V + beta diag(u^2))^{-1}      energy adaptive, c=1
    a2u    G = (-Lap_h + V + 2 beta diag(u^2))^{-1}    modified a_u, c=2
    a3u    G = (-Lap_h + V + 3 beta diag(u^2))^{-1}    modified a_u, c=3
    tf     G = B_TF(u)^{-1}                             Thomas-Fermi, eq. (11)

Everything else is metric independent, which is the point of the framework:

    ambient gradient     grad_X E(u) = G(u) (-Lap_h u + V u + beta u^3)
    tangent projection   P^TS_u(w)   = w - (<w,u>_h / <G u,u>_h) G u
    Riemannian gradient  grad E(u)   = P^TS_u(grad_X E(u))
    retraction           R_u(xi)     = (u + xi) / ||u + xi||_h

so one implementation of gradient descent and one of conjugate gradients serve
all of them, and the only thing that changes is how G is applied. Two of the
seven are exact tensor-product solves (h1, a0mod) and five need an inner PCG
(a0, au, a2u, a3u, tf), preconditioned in every case by the separable
(-Lap_h + V_1)^{-1}.

For the a_u metric grad_X E(u) = u exactly, since G(u) grad E_h(u) = A_u^{-1}
A_u u, so its gradient costs one solve rather than two and the descent step
reduces to the normalized inverse iteration of Henning and Peterseim.

The modified a_u metric of Section 3.4 of the paper takes the same operator
with 3 beta in place of beta, that is A_3u = -Lap_h + V + 3 beta diag(u^2),
which is the derivative of u -> A_u u and therefore the metric closest to the
constrained Hessian among those positive definite at every u.  Its Riemannian
Hessian at the ground state is 1 - lambda_1(A_beta)/mu^T, so its condition
number is asymptotically half of the one of the a_u metric (Proposition 3.8).
The shortcut above is lost --
grad_X E(u) = A_3u^{-1} A_beta u = u - 2 beta A_3u^{-1} u^3 is no longer u --
so one iteration needs two solves with A_3u, one for the ambient gradient and
one for G(u) u in the tangent projection.

Both belong to the one-parameter family B_c = -Lap_h + V + c beta diag(u^2)
of eq. (24) of the paper: c=1 is the a_u metric and c=3 the modified one. For
a monotone scheme (Assumption 2.1) every c in [2,3] carries the same halved
condition number bound, because the maximum principle gives
-lambda I <= H - B_c <= 0 there (Remark 3.9). c=3 is singled out by having a
scalar defect, and c=2 by being the
closest of the interval to A_beta, hence the easiest to precondition with
(-Lap_h + V_1)^{-1}. Whether that shows up as fewer inner Krylov iterations is
an empirical question, which is why CFAM below is a table and not a constant.
"""
import time as _time

import numpy as np

import solvers as _S

EPS = _S.EPS
XATOL = _S.XATOL

# The u-dependent family G(u) = (-Lap_h + V + c beta diag(u^2))^{-1}, by metric
# key. 'au' (c=1) is not here: it is the one member whose ambient gradient
# collapses to u, so it takes a different, cheaper path in _grad below.
CFAM = {'a2u': 2.0, 'a3u': 3.0}

# The Thomas-Fermi metric of Section 3.5 of the paper, eq. (11),
#     B_TF(u) = -Lap_h + 2 beta diag(u^2) + diag((V - ell)_+) + f ell^(1/3),
# is the first metric here that is not uniformly equivalent to A_beta, which is
# what lets its condition number stay bounded as beta grows. Two scalars: the
# floor parameter f, default 1/2 (the paper's ell), and ell (the paper's
# lambda(u)), an approximation of lambda_1(A_beta)
# good to the layer scale lambda_1^(1/3). ell is taken to be the Rayleigh
# quotient lambda(u) = 2E(u) + (beta/2)<u^2,u^2>_h of the current iterate,
# recomputed each outer iteration unless frozen, which costs one energy
# evaluation and no solve.
TF_F = 0.5


# --------------------------------------------------------------------------
# the seven metrics
# --------------------------------------------------------------------------
def make_metric(P, name, pcg_tol=1e-8, pcg_max=200, verbose=0, P32=None,
                lean=False, tf_f=TF_F, tf_freeze=False, n_refine=1):
    """Return {apply_G, apply_Ginv, exact, inner_its} for one metric.

    apply_G(u, w)    solves G(u) x = w, i.e. applies the Green operator.
    apply_Ginv(u, w) applies its inverse, a matrix-vector product, which is
                     what the metric inner product g_u(a,b) = <a, G^{-1}b>_h
                     needs and which costs no solve.
    """
    xp, jit = P['xp'], P['jit']
    counter = P['counter']

    if name == 'h1':
        def apply_G(u, w):
            counter.n += 1
            return P['solve_G'](w)

        def apply_Ginv(u, w):
            return P['apply_Ginv'](w)

        return dict(name=name, apply_G=apply_G, apply_Ginv=apply_Ginv,
                    exact=True, inexact=False,
                    label=r'modified $H^1$', inner=lambda: 0,
                    stats={'its': 0, 'calls': 0, 'last': 0})

    if name == 'a0mod':
        @jit
        def apply_A1(w):
            return P['lap'](w) + P['apply_V1'](w)

        def apply_G(u, w):
            counter.n += 1
            return P['solve_V1'](w)

        def apply_Ginv(u, w):
            return apply_A1(w)

        return dict(name=name, apply_G=apply_G, apply_Ginv=apply_Ginv,
                    exact=True, inexact=False,
                    label=r'modified $a_0$', inner=lambda: 0,
                    stats={'its': 0, 'calls': 0, 'last': 0})

    # ---- the three metrics whose operator is not separable ----------------
    # A_0 = -Lap_h + V, A_u = -Lap_h + V + beta u^2 and its modification
    # A_3u = -Lap_h + V + 3 beta u^2 are inverted by conjugate gradients
    # preconditioned with the separable (-Lap_h+V_1)^{-1}.
    # One fused kernel per inner iteration, as in the a_u flow of solvers.py;
    # the preconditioner is exact for the lattice potential and is the
    # preconditioner of Liu and Zhang (2026) for the stirrer.
    if name not in ('a0', 'au', 'tf') and name not in CFAM:
        raise ValueError(f'unknown metric {name!r}')
    label = {'a0': r'$a_0$', 'au': r'$a_u$',
             'a2u': r'modified $a_u$, $c=2$',
             'a3u': r'modified $a_u$, $c=3$',
             'tf': r'Thomas--Fermi'}[name]
    cfam = CFAM.get(name)
    # ell lives in a one-element box so that the jitted kernels take it as a
    # traced argument -- a closure over a Python float would recompile every
    # time it changed.
    tf_state = {'ell': None, 'f': tf_f, 'frozen': bool(tf_freeze)}

    def _tf_pre(u):
        if tf_state['ell'] is not None and tf_state['frozen']:
            return
        tf_state['ell'] = float(P['lam_rq'](u))

    def _make_op(Pd):
        """G(u)^{-1} as a kernel in the precision of Pd.

        The Thomas-Fermi case is deliberately NOT wrapped in another jit:
        apply_Btf is already jitted and takes ell as an argument, so JAX traces
        it. A closure over the Python float would bake it in and recompile the
        kernel every outer iteration, since ell moves with the iterate.
        """
        if name == 'tf':
            def g(u, w):
                return Pd['apply_Btf'](u, w, tf_state['ell'], tf_state['f'])
            return g
        if name == 'a0':
            def f(u, w):
                return Pd['apply_A'](w)
        elif name == 'au':
            def f(u, w):
                return Pd['apply_Au'](u, w)
        else:
            def f(u, w):
                return Pd['apply_A'](w) + cfam * Pd['beta'] * (u * u) * w
        return Pd['jit'](f)

    def _make_cg_body(Pd, opd, acc=None):
        """One preconditioned CG iteration in the precision of Pd.

        `acc` is the accumulator dtype for the three reductions. In the fp32
        twin the vectors are single -- that is what makes the inner solve fit
        at n=999 -- while the inner products are sums over up to 10^9 terms,
        so accumulating them in double is free insurance: three scalars, no
        memory, and they are cast back before touching a vector, so no fp64
        temporary of size n^3 is ever formed. Kept because it is correct and
        costs nothing.

        It is NOT, however, the fix for the Thomas--Fermi metric's stall at
        n >= 599, which is what it was written for. With and without it that
        metric stalls at |E-E*| = 2.22e-10 at n=599, agreeing to three digits,
        so the reductions were not the problem. The cause is the recursively
        updated residual of the fp32 recurrence drifting from the true one over
        the ~75 inner iterations B_TF needs against ~29 for A_u. The remedy is
        not a better preconditioner and not more precision: it is to recompute
        the true residual more often. See the note on --pcg-max and --refine in
        README.md; short bursts fix it outright.
        """
        xpd, dt = Pd['xp'], Pd['dtype']

        def rsum(a, b):
            return xpd.sum(a * b, dtype=acc) if acc is not None \
                else xpd.sum(a * b)

        def f(u, x, r, p, rz):
            Ap = opd(u, p)
            a = (rz / rsum(p, Ap)).astype(dt)
            x = x + a * p
            r = r - a * Ap
            z = Pd['solve_V1'](r)
            rz_new = rsum(r, z)
            p = z + (rz_new / rz).astype(dt) * p
            return x, r, p, rz_new, xpd.sqrt(rsum(r, r))
        return Pd['jit'](f)

    op = _make_op(P)
    cg_body = _make_cg_body(P, op)

    stats = {'its': 0, 'calls': 0, 'last': 0}

    def _pcg(u, w, tol, cgb, Pp, opp, acc=None):
        """PCG for the metric operator from the separable warm start.

        rz is seeded in the same accumulator dtype the body uses, so that the
        Fletcher-Reeves-style ratio rz_new/rz is formed consistently.
        """
        x = Pp['solve_V1'](w)
        counter.n += 1
        r = w - opp(u, x)
        z = Pp['solve_V1'](r)
        counter.n += 1
        p = z
        rz = (Pp['xp'].sum(r * z, dtype=acc) if acc is not None
              else Pp['xp'].sum(r * z))
        nrm = float(Pp['xp'].sqrt(Pp['xp'].sum(w * w)))
        its = 0
        for j in range(1, pcg_max + 1):
            x, r, p, rz, rn = cgb(u, x, r, p, rz)
            counter.n += 1
            its = j
            if nrm == 0.0 or float(rn) / nrm < tol:
                break
        return x, its

    if P32 is None:
        def apply_G(u, w):
            x, its = _pcg(u, w, pcg_tol, cg_body, P, op)
            stats['its'] += its
            stats['calls'] += 1
            stats['last'] = its
            return x
    else:
        # Mixed precision, as in the a_u flow of solvers.py: the
        # Krylov vectors are fp32, which is what makes the inner solve fit at
        # n=999, while the residual is formed and the correction accumulated in
        # fp64.  Forming w - G^{-1}x cancels about half the fp32 digits, so
        # that subtraction is the one step that cannot be single.
        xp32, f32, f64 = P32['xp'], P32['dtype'], P['dtype']
        op32 = _make_op(P32)
        cg_body32 = _make_cg_body(P32, op32, acc=f64)

        # The right-hand side is never materialized in double precision.
        # It is built in the fp32 twin, and the one place the fp64 value is
        # needed -- the refinement residual w - G^{-1}x -- is a fused kernel
        # that forms it from u and x directly:
        #     kind='u'      w = u          r = u - G^{-1}x
        #     kind='gradE'  w = grad E_h   r = A(u-x) + beta u^3       (a_0)
        #                                  r = the same - c beta u^2 x (B_c)
        # Each returns the residual already in fp32; see below.
        # That saves two n^3 vectors, which at n=999 is 16 GB.  Both forms
        # apply the stencil once, not twice: writing grad E_h(u) - G^{-1}x
        # literally would cost a second application of -Lap_h + V.
        # The cast to fp32 is inside the kernel, not after it. The residual has
        # to be FORMED in fp64, being a cancellation, but it is CONSUMED in
        # fp32 by the correction solve, and written as two statements --
        # r64 = resid(u, x) then r64.astype(f32) -- both exist at once: 7.4 GiB
        # and 3.7 GiB side by side at n=999, with the fp64 one alive across the
        # whole inner solve. Fused, only the fp32 output is ever a buffer and
        # the fp64 difference stays in registers. The arithmetic is identical,
        # so iteration counts do not move; what moves is 7.4 GiB of peak, which
        # is what kept conjugate gradients in the two-solve metrics off the
        # 999^3 grid.
        # The stencil is kept OUT of the fused elementwise chain. Written as
        # one kernel -- apply_A(u-x) plus the cubic terms, cast to fp32 -- XLA
        # asks for a single 22.3 GiB allocation at n=999 and fails, the same
        # way fusing two applications of A asks for 29.75 GiB. Split into a stencil call and an elementwise tail, no single
        # allocation exceeds one n^3 vector, and only the 3.7 GiB fp32 result
        # survives into the correction solve.
        beta = P['beta']
        cu = 0.0 if name == 'a0' else (1.0 if name in ('au', 'tf') else cfam)
        sub = P['jit'](lambda a, b_: a - b_)

        def _tail_u(u, x, Ax):
            return (u - Ax - cu * beta * (u * u) * x).astype(f32)

        if cfam is not None:
            def _tail_gradE(u, x, At):
                return (At + beta * (u * u) * (u - cfam * x)).astype(f32)
        else:
            def _tail_gradE(u, x, At):
                return (At + beta * u ** 3).astype(f32)

        tail_u = P['jit'](_tail_u)
        tail_gradE = P['jit'](_tail_gradE)

        def resid_u(u, x):
            Ax = P['apply_A'](x)
            r = tail_u(u, x, Ax)
            del Ax
            return r

        def resid_gradE(u, x):
            t = sub(u, x)
            At = P['apply_A'](t)
            del t
            r = tail_gradE(u, x, At)
            del At
            return r

        if name == 'tf':
            # r = u - B_TF x            and
            # r = grad E_h(u) - B_TF x = Lap(u-x) + V u - (V-ell)_+ x
            #                            + beta u^3 - 2 beta u^2 x
            #                            - f ell^(1/3) x
            # One stencil each, ell and f as traced arguments, fp32 out.
            V3 = P['V3']

            def _tail_u_tf(u, x, Lx, ell, fp):
                return (u - Lx - xp.maximum(V3() - ell, 0.0) * x
                        - 2.0 * beta * (u * u) * x
                        - fp * ell ** (1.0 / 3.0) * x).astype(f32)

            def _tail_gradE_tf(u, x, Lt, ell, fp):
                V = V3()
                return (Lt + V * u - xp.maximum(V - ell, 0.0) * x
                        + beta * u ** 3 - 2.0 * beta * (u * u) * x
                        - fp * ell ** (1.0 / 3.0) * x).astype(f32)

            tail_u_tf = P['jit'](_tail_u_tf)
            tail_gradE_tf = P['jit'](_tail_gradE_tf)

            def resid_u(u, x):
                Lx = P['lap'](x)
                r = tail_u_tf(u, x, Lx, tf_state['ell'], tf_state['f'])
                del Lx
                return r

            def resid_gradE(u, x):
                t = sub(u, x)
                Lt = P['lap'](t)
                del t
                r = tail_gradE_tf(u, x, Lt, tf_state['ell'], tf_state['f'])
                del Lt
                return r

        # the stopping test only compares a ratio against pcg_tol, so the norm
        # of the already-rounded fp32 residual is as good as the fp64 one
        def _rnorm32(r):
            return xp32.sqrt(xp32.sum(r * r))

        rnorm32 = P32['jit'](_rnorm32)

        def apply_G(u, w=None, kind='u', tol32=1e-6, refine=n_refine):
            u32 = u.astype(f32)
            if kind == 'u':
                rhs32, resid = u32, resid_u
            else:
                rhs32 = P32['grad_E'](u32)
                resid = resid_gradE
            x32, its = _pcg(u32, rhs32, tol32, cg_body32, P32, op32, f64)
            if kind != 'u':
                del rhs32
            if lean:
                # The metric is applied in single precision and kept there:
                # no refinement, no fp64 copy.  What survives is a direction,
                # and the energy error is quadratic in the state error, so a
                # direction good to 1e-7 still supports an energy error near
                # 1e-13.  This is what makes the a_0 metric fit at n=999.
                del u32
                stats['its'] += its
                stats['calls'] += 1
                stats['last'] = its
                return x32
            x = x32.astype(f64)
            del x32
            for _ in range(refine):
                r32 = resid(u, x)      # fp64 arithmetic, fp32 result
                counter.n += 1
                rn = float(rnorm32(r32))
                nrm = float(xp.sqrt(xp.sum(x * x))) or 1.0
                if rn / nrm < pcg_tol:
                    del r32
                    break
                d32, nit = _pcg(u32, r32, tol32, cg_body32, P32, op32, f64)
                del r32
                x = x + d32.astype(f64)
                del d32
                its += nit
            del u32
            stats['its'] += its
            stats['calls'] += 1
            stats['last'] = its
            return x

    def apply_Ginv(u, w):
        return op(u, w)

    return dict(name=name, apply_G=apply_G, apply_Ginv=apply_Ginv,
                exact=False, inexact=(P32 is not None), label=label,
                inner=lambda: stats['its'], stats=stats,
                pre=(_tf_pre if name == 'tf' else None), tf_state=tf_state)


# --------------------------------------------------------------------------
# metric-generic pieces
# --------------------------------------------------------------------------
def _proj(P, w, u, Gu):
    """w - (<w,u>/<Gu,u>) Gu, at the precision of w and Gu.

    In lean mode w and Gu are single precision and u is cast to match, so the
    projection never materializes a double-precision temporary of size n^3.
    """
    if w.dtype == u.dtype:
        return P['proj_t'](w, u, Gu)
    f = P.get('_proj_lean')
    if f is None:
        xp = P['xp']

        def _f(w, u, Gu):
            return w - (xp.sum(w * u) / xp.sum(Gu * u)) * Gu

        f = P['jit'](_f)
        P['_proj_lean'] = f
    return f(w, u.astype(w.dtype), Gu)


def _grad(P, M, u):
    """Riemannian gradient of E at u, and G u.

    Two applications of G in general; one for the a_u metric, where the
    ambient gradient of the energy is u itself, since
    G(u) grad E_h(u) = A_u^{-1} A_u u.  That is a property of the metric, not
    of the implementation: every other metric needs G grad E_h and G u
    separately, because the projection onto the tangent space is taken in the
    metric and therefore needs G u.  The modified a_u metrics are the case where
    this costs the most, both solves being inner PCG solves with B_c.
    """
    if M.get('pre') is not None:
        M['pre'](u)                # refresh ell before G is applied at all
    Gu = M['apply_G'](u, u, 'u') if M['inexact'] else M['apply_G'](u, u)
    if M['name'] == 'au':
        GE = u if Gu.dtype == u.dtype else u.astype(Gu.dtype)
    elif M['inexact']:
        GE = M['apply_G'](u, None, 'gradE')      # rhs never leaves fp32
    else:
        GE = M['apply_G'](u, P['grad_E'](u))
    return _proj(P, GE, u, Gu), Gu


def _g_ip(P, M, u, a, b):
    """The metric inner product g_u(a,b) = <a, G(u)^{-1} b>_h: no solve.

    The stencil is applied on its own and the reduction is a second kernel.
    Fusing the two looks like the economical choice -- G^{-1}b is consumed by
    the reduction and never has to be a buffer -- but G^{-1} contains
    -Lap_h + V, and asking XLA to fuse a stencil into an elementwise chain and
    a reduction makes it materialize the intermediates as one block: a single
    22.3 GiB request at n=999, three n^3 fp64 vectors, which fails on a 95.6 GB
    card. That is the whole reason the conjugate gradient variants of the
    two-solve metrics did not fit, and it was never about carrying one vector
    more than the flow: _g_ip is called from rcg and never from rgd, so the
    flows never met it. Split, the only new buffer is G^{-1}b itself, 7.4 GiB.

    Same lesson as the fused residual above: fuse sequential elementwise work,
    never a stencil.
    """
    f = P.get('_dot_h3')
    if f is None:
        xp, h3 = P['xp'], P['h3']

        def _f(a, q):
            return h3 * xp.sum(a * q)

        f = P['jit'](_f)
        P['_dot_h3'] = f
    q = M['apply_Ginv'](u, b)
    v = float(f(a, q))
    del q
    return v


def _retract(P, u, d, t):
    """(u + t d) / ||u + t d||_h, in one kernel.

    Fused because the unfused form materializes u + t d as a separate live
    buffer, and at n=999 that buffer is 8 GB.
    """
    f = P.get('_retract_fused')
    if f is None:
        norm = P['normalize']          # M-normalization: h^3 I for finite
                                       # differences, the Gauss-Lobatto
                                       # weights for spectral elements
        def _f(u, d, t):
            return norm(u + t * d)

        f = P['jit'](_f)               # not a decorator: Python 3.8 rejects
        P['_retract_fused'] = f        # an arbitrary expression there
    return f(u, d, t)


def _cg_dir(P, d, u, Gu, gc, beta_fr):
    """beta_fr * P_u(d) - gc, in one kernel."""
    f = P.get('_cg_dir_fused')
    if f is None:
        xp = P['xp']

        def _f(d, u, Gu, gc, b):
            return b * (d - (xp.sum(d * u) / xp.sum(Gu * u)) * Gu) - gc

        f = P['jit'](_f)
        P['_cg_dir_fused'] = f
    return f(d, u, Gu, gc, beta_fr)


def _step(P, u, d, linesearch='closedform', tmax=5.0):
    """Exact line search of E along the retracted ray, then the retraction."""
    cf = P['ls_coeffs_fused'](u, d) if linesearch == 'closedform' else None
    from scipy.optimize import minimize_scalar
    t = float(minimize_scalar(
        lambda s: _S._ray_energy(P, u, d, s, linesearch, cf),
        bounds=(EPS, tmax), method='bounded', options=XATOL).x)
    return _retract(P, u, d, t), t


# --------------------------------------------------------------------------
def rgd(P, u0, maxit, ref, metric='h1', tols=None, verbose=0,
        linesearch='closedform', pcg_tol=1e-8, pcg_max=200, tmax=5.0,
        P32=None, lean=False, tf_f=TF_F, tf_freeze=False, n_refine=1):
    """Riemannian gradient descent under one metric, exact line search."""
    tols = tols or _S._tols()
    M = make_metric(P, metric, pcg_tol, pcg_max, verbose, P32, lean,
                    tf_f=tf_f, tf_freeze=tf_freeze, n_refine=n_refine)
    hist = _S._Hist()
    P['counter'].reset()
    u = u0
    t0 = _time.time()
    for kk in range(1, maxit + 1):
        gc, Gu = _grad(P, M, u)
        Gu = None
        d = -gc
        gc = None                      # freed before the ray is built
        u, step = _step(P, u, d, linesearch, tmax)
        d = None
        E, g, res = _S._record(P, hist, kk, u, ref, t0, step)
        g = None                       # the recorder's gradient is not reused
        if verbose and kk % verbose == 0:
            inner = '' if M['exact'] else f"  pcg={M['stats']['last']}"
            print(f"  {metric:6s} GD {kk:5d}  E={E:.12f}  "
                  f"err={abs(ref['E']-E):.3e}  res={res:.3e}  tau={step:.4f}  "
                  f"solves={P['counter'].n}{inner}", flush=True)
        if _S._converged(abs(ref['E'] - E), res, tols):
            break
    hist.d['_inner'] = M['stats']['its']
    return hist.out()


def rcg(P, u0, maxit, ref, metric='h1', tols=None, verbose=0,
        linesearch='closedform', pcg_tol=1e-8, pcg_max=200, tmax=5.0,
        P32=None, lean=False, return_u=False, dir32=False,
        tf_f=TF_F, tf_freeze=False, n_refine=1):
    """Riemannian CG under one metric.

    Fletcher-Reeves in the metric of the current point, the previous direction
    transported by the tangent projection at the new point, and a restart to
    steepest descent whenever the result is not a descent direction. With
    G(u) fixed this is the classical Riemannian CG; with the a_u metric the
    inner product itself moves with the iterate, which is why the FR ratio is
    formed from each gradient in the metric at its own point.
    """
    tols = tols or _S._tols()
    M = make_metric(P, metric, pcg_tol, pcg_max, verbose, P32, lean,
                    tf_f=tf_f, tf_freeze=tf_freeze, n_refine=n_refine)
    hist = _S._Hist()
    P['counter'].reset()
    u = u0
    gc, Gu = _grad(P, M, u)
    d = -gc
    if dir32 and P32 is not None:
        d = d.astype(P32['dtype'])
    gg_old = _g_ip(P, M, u, gc, gc)
    gc = Gu = None
    u_new = None

    t0 = _time.time()
    for kk in range(1, maxit + 1):
        u_new, step = _step(P, u, d, linesearch, tmax)
        u = None                       # the old iterate dies with the step
        E, g, res = _S._record(P, hist, kk, u_new, ref, t0, step)
        g = None
        if verbose and kk % verbose == 0:
            inner = '' if M['exact'] else f"  pcg={M['stats']['last']}"
            print(f"  {metric:6s} CG {kk:5d}  E={E:.12f}  "
                  f"err={abs(ref['E']-E):.3e}  res={res:.3e}  tau={step:.4f}  "
                  f"solves={P['counter'].n}{inner}", flush=True)
        if _S._converged(abs(ref['E'] - E), res, tols):
            break
        gc_new, Gu_new = _grad(P, M, u_new)
        gg_new = _g_ip(P, M, u_new, gc_new, gc_new)
        # Transport the old direction to the new tangent space, then
        # Fletcher-Reeves.  Each vector is released as it dies: at n=999 one
        # of them is 8 GB, and holding the old direction while the new one is
        # built is the difference between fitting on one card and not.
        # transport, scale and add in one kernel: the transported direction
        # is never a separate live buffer
        d = _cg_dir(P, d.astype(Gu_new.dtype), u_new, Gu_new, gc_new,
                    gg_new / gg_old)
        Gu_new = None
        if dir32 and P32 is not None:
            # The search direction is stored in single precision while the
            # inner solves of the next gradient run.  Unlike the gradient,
            # which is the difference of two nearly equal quantities, the
            # direction is stored as itself, so this costs 1e-7 of relative
            # accuracy in a ray the line search then minimizes along -- and it
            # is 4 GB at n=999 that the inner solve can use instead.
            d = d.astype(P32['dtype'])
        if _g_ip(P, M, u_new, d, gc_new) >= 0.0:      # not a descent direction
            d = -gc_new
            if dir32 and P32 is not None:
                d = d.astype(P32['dtype'])
        gc_new = None
        u, gg_old = u_new, gg_new
        u_new = None
    hist.d['_inner'] = M['stats']['its']
    # on an early break u_new is the newest iterate; after a full pass the
    # loop has already moved it into u
    best = u_new if u_new is not None else u
    return (hist.out(), best) if return_u else hist.out()
