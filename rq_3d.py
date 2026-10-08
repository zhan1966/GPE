#!/usr/bin/env python3
"""Exact extreme eigenvalues of the Riemannian Hessian in three dimensions.

This is the 3D counterpart of rq_1d.py and rq_2d.py, which produce the d = 1
and d = 2 rows of Table A.1 of the paper. The quantity is the same one: by
Section 3, the eigenvalues of
Hess E(u) at the ground state are those of the pencil

    (M (A_{3beta}(u) - lambda_1(A_beta) I),  M G(u)^{-1})

restricted to the tangent space T_u = {eta : <eta, u>_h = 0}, and the condition
number kappa = max rho / min rho is what controls the linear rate of the
Riemannian flow. Writing N = A_{3beta}(u) - lambda_1 and B = G(u)^{-1} for the
metric operator, both symmetric positive definite (N is PD because
lambda_1(A_{3beta}) > lambda_1(A_beta) whenever beta u^2 does not vanish), the
task is

    min / max  <N eta, eta>_h / <B eta, eta>_h   over   eta perp_h u.

In one dimension rq_1d.py forms an orthonormal basis of T_u and calls a dense
solver; in two dimensions rq_2d.py uses shift-invert Lanczos on a sparse
factorization. Neither is available at 10^7-10^8 unknowns: there is no matrix
to factor. The tangent constraint, however, is not what makes 3D hard. It is a
single vector, so the projection

    Pi(w) = w - <w, u>_h u                (u is h-normalized)

is exact and costs one reduction, and a Krylov method that applies Pi to every
basis vector it produces stays in T_u to rounding error. What is needed instead
of a factorization is a preconditioned eigensolver, and that is LOBPCG:

    min rho:  LOBPCG on the pencil (N, B) with a preconditioner for N
    max rho:  LOBPCG on the reversed pencil (B, N) with a preconditioner for B,
              and rho_max = 1 / min sigma

Both preconditioners are built from the machinery already in gpe3d.py: the
separable solve (-Lap_h + V_1)^{-1}, applied exactly by the tensor-product
eigen-decomposition, wrapped in a short inner PCG. That is the same
preconditioner the metric solves of metrics.py use, so nothing new has to be
tuned.

The ground state itself is computed here rather than read from disk -- only
scalars are ever written -- by the a_u flow with a line search, followed by
Newton on the bordered system

    F(u, lambda) = A_beta(u) u - lambda u = 0,   (<u,u>_h - 1)/2 = 0,

whose Jacobian block is J = A_{3beta}(u) - lambda, the same operator as N. Two
PCG solves per Newton step take the residual to rounding error, which matters:
min rho is a difference of nearby quantities, and a ground state converged only
to 10^-6 would corrupt its leading digits.

Usage
-----
    python3 rq_3d.py --n 199 --beta 1e4 --potential harmonic
    python3 rq_3d.py --n 499 --beta 1600 --potential stirrer --L 8 \
                     --metrics h1,a0,a0mod,au,a2u,a3u,tf --out out.json

Everything is fp64. Memory is 9 vectors per LOBPCG block vector plus the inner
PCG workspace, so n = 499 (1 GB per fp64 vector) needs about 25 GB with block
size 2 and fits on one GH200; n = 999 does not, which is why the mesh study
stops at 499 and the argument for the 999^3 runs of Section 5.2 is the
iteration counts there rather than a spectrum computed here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
# small helpers on device vectors
# --------------------------------------------------------------------------
def make_helpers(P):
    xp, ip = P['xp'], P['ip_M']

    def nrm(a):
        return float(np.sqrt(max(float(ip(a, a)), 0.0)))

    def proj(w, u):
        """Pi w = w - <w,u>_h u, exact for h-normalized u."""
        return w - float(ip(w, u)) * u

    return xp, ip, nrm, proj


def pcg(P, op, prec, b, tol=1e-8, maxit=200, x0=None):
    """Preconditioned CG for a symmetric positive definite op.

    Plain and unfused on purpose: op contains the Laplacian stencil, and fusing
    a stencil into the reductions of a CG body is what made XLA ask for one
    22 GiB block at n = 999. Here every apply is its own kernel.
    """
    xp, ip = P['xp'], P['ip_M']
    nb = float(np.sqrt(max(float(ip(b, b)), 0.0)))
    if nb == 0.0:
        return b * 0.0, 0
    if x0 is None:
        x = b * 0.0
        r = b
    else:
        x = x0
        r = b - op(x)
    z = prec(r)
    p = z
    rz = float(ip(r, z))
    for k in range(1, maxit + 1):
        Ap = op(p)
        denom = float(ip(p, Ap))
        if denom <= 0.0:
            break
        a = rz / denom
        x = x + a * p
        r = r - a * Ap
        if float(np.sqrt(max(float(ip(r, r)), 0.0))) <= tol * nb:
            return x, k
        z = prec(r)
        rz_new = float(ip(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit


# --------------------------------------------------------------------------
# operators: the constrained Hessian numerator N and the seven metrics B
# --------------------------------------------------------------------------
def make_ops(P, u, lam1, ell=None, tf_f=0.5, inner=40, inner_tol=1e-4):
    """Return (N, metric_factory) for the current ground state.

    N w = (-Lap_h + V + 3 beta u^2 - lambda_1) w  is the metric-independent
    numerator; metric_factory(name) returns (B, Binv_approx, exact) for the
    denominator of the Rayleigh quotient.
    """
    xp = P['xp']
    beta, alpha = P['beta'], P['alpha']
    lap, apply_V, apply_V1 = P['lap'], P['apply_V'], P['apply_V1']
    solve_V1, solve_G = P['solve_V1'], P['solve_G']
    u2 = u * u

    def N(w):
        return lap(w) + apply_V(w) + (3.0 * beta) * (u2 * w) - lam1 * w

    def A_beta(w):
        return lap(w) + apply_V(w) + beta * (u2 * w)

    def A_0(w):
        return lap(w) + apply_V(w)

    def A_3beta(w):
        return lap(w) + apply_V(w) + (3.0 * beta) * (u2 * w)

    def Bc(c):
        def f(w):
            return lap(w) + apply_V(w) + (c * beta) * (u2 * w)
        return f

    def B_tf(w):
        return P['apply_Btf'](u, w, ell, tf_f)

    def prec_V1(w):
        return solve_V1(w)

    def metric(name):
        """(apply B, approximate B^{-1}, exact flag)"""
        if name == 'h1':
            return (lambda w: lap(w) + alpha * w), solve_G, True
        if name == 'a0mod':
            return (lambda w: lap(w) + apply_V1(w)), solve_V1, True
        if name == 'a0':
            B = A_0
        elif name == 'au':
            B = Bc(1.0)
        elif name == 'a2u':
            B = Bc(2.0)
        elif name == 'a3u':
            B = Bc(3.0)
        elif name == 'tf':
            B = B_tf
        else:
            raise ValueError(f'unknown metric {name!r}')

        def Binv(w, _B=B):
            x, _ = pcg(P, _B, prec_V1, w, tol=inner_tol, maxit=inner)
            return x
        return B, Binv, False

    return dict(N=N, A_beta=A_beta, A_0=A_0, A_3beta=A_3beta, metric=metric,
                prec_V1=prec_V1)


# --------------------------------------------------------------------------
# ground state: a_u flow with line search, then Newton on the bordered system
# --------------------------------------------------------------------------
def ground_state(P, tol=1e-12, maxit=400, verbose=True, pcg_tol=1e-10):
    xp, ip, nrm, proj = make_helpers(P)
    beta = P['beta']
    lap, apply_V, solve_V1 = P['lap'], P['apply_V'], P['solve_V1']
    energy, normalize = P['energy'], P['normalize']
    n, L, h = P['n'], P['L'], P['h']

    # Thomas-Fermi start: u^2 = (mu - V)_+/beta with mu from the 3D TF mass
    # integral (8 pi/15) mu^{5/2}/beta = 1 for the isotropic harmonic trap;
    # for any other trap this is only a shape, and the flow fixes it.
    V3 = P['V3']()
    mu_tf = (15.0 * beta / (8.0 * np.pi)) ** 0.4
    u = xp.sqrt(xp.maximum(mu_tf - V3, 0.0) / beta)
    s = float(np.sqrt(max(float(ip(u, u)), 0.0)))
    if not np.isfinite(s) or s < 1e-8:
        x = P['to_dev'](xp.asarray(P['x'], dtype=P['dtype']))
        u = xp.exp(-(x[:, None, None] ** 2 + x[None, :, None] ** 2
                     + x[None, None, :] ** 2) / 2.0)
    u = normalize(u)
    del V3

    def A_u(uc, w):
        return lap(w) + apply_V(w) + beta * (uc * uc) * w

    t0 = time.time()
    lam = float(ip(u, A_u(u, u)))
    for it in range(1, maxit + 1):
        Gu, _ = pcg(P, lambda w: A_u(u, w), solve_V1, u, tol=pcg_tol, maxit=300)
        lam = 1.0 / float(ip(u, Gu))
        g = u - lam * Gu                       # a_u Riemannian gradient
        res = float(np.sqrt(max(float(ip(g, A_u(u, g))), 0.0)))
        if res < 1e-9:
            break
        # line search on E(normalize(u - t g)) by golden section, cheap:
        # each evaluation is one Laplacian apply plus reductions.
        def phi(t):
            return float(energy(normalize(u - t * g)))
        a_, b_ = 0.0, 2.0
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        fc, fd = phi(c_), phi(d_)
        for _ in range(20):
            if fc < fd:
                b_, d_, fd = d_, c_, fc
                c_ = b_ - gr * (b_ - a_)
                fc = phi(c_)
            else:
                a_, c_, fc = c_, d_, fd
                d_ = a_ + gr * (b_ - a_)
                fd = phi(d_)
            if b_ - a_ < 1e-5:
                break
        t = 0.5 * (a_ + b_)
        u = normalize(u - t * g)
        if verbose and (it % 10 == 0 or it == 1):
            print(f'  [gs] it {it:4d}  E {float(energy(u)):.12f}  '
                  f'|grad|_au {res:.3e}  t {t:.3f}', flush=True)
    gd_its, gd_res = it, res

    # Newton on F(u,lam) = A_beta(u) u - lam u, (<u,u>_h - 1)/2 = 0.
    # J = A_{3beta}(u) - lam is SPD, and is the same operator as N below.
    for k in range(8):
        F = A_u(u, u) - lam * u
        c = 0.5 * (float(ip(u, u)) - 1.0)
        fn = nrm(F)
        if verbose:
            print(f'  [newton] {k}  |F|_h {fn:.3e}  lam {lam:.12f}', flush=True)
        if fn < tol:
            break

        def J(w, uc=u, lc=lam):
            return lap(w) + apply_V(w) + (3.0 * beta) * (uc * uc) * w - lc * w

        y1, i1 = pcg(P, J, solve_V1, F, tol=1e-12, maxit=400)
        y2, i2 = pcg(P, J, solve_V1, u, tol=1e-12, maxit=400)
        dlam = (float(ip(u, y1)) - c) / float(ip(u, y2))
        u = u - y1 + dlam * y2
        lam = lam + dlam
    u = P['normalize'](u)
    lam1 = float(ip(u, A_u(u, u)))
    res_f = nrm(A_u(u, u) - lam1 * u)
    if verbose:
        print(f'  [gs] done in {time.time()-t0:.1f}s: E {float(P["energy"](u)):.12f}'
              f'  lam1 {lam1:.10f}  |F|_h {res_f:.2e}  min u {float(u.min()):.2e}',
              flush=True)
    return u, lam1, dict(gd_iters=gd_its, gd_res=gd_res, newton_res=res_f,
                         energy=float(P['energy'](u)), time=time.time() - t0)


# --------------------------------------------------------------------------
# LOBPCG for the smallest eigenvalue of a symmetric definite pencil,
# on the tangent space of a single constraint vector
# --------------------------------------------------------------------------
def lobpcg_min(P, applyA, applyB, prec, k=2, constraint=None, tol=1e-7,
               maxit=300, seed=0, verbose=0, label='', osc=False,
               seed_bump=None, eig_tol=1e-10, settle_res=1e-3,
               max_restarts=2):
    """Smallest k eigenvalues of A x = theta B x with x perp_h constraint.

    Locally optimal block preconditioned CG. The search space each iteration is
    span{X, W, Pold} with W = prec(A X - theta B X); the small (3k x 3k)
    Rayleigh-Ritz problem is solved on the host, after dropping the directions
    that a B-Gram eigendecomposition finds numerically dependent. Every block
    vector is projected onto the constraint's orthogonal complement as soon as
    it is produced, which is what keeps the iteration inside T_u.
    """
    xp, ip, nrm, proj = make_helpers(P)
    rng = np.random.default_rng(seed)
    shape = (P['n'],) * 3

    def PI(w):
        return w if constraint is None else proj(w, constraint)

    def block_ip(Xs, Ys):
        m, p = len(Xs), len(Ys)
        G = np.zeros((m, p))
        for i in range(m):
            for j in range(p):
                G[i, j] = float(ip(Xs[i], Ys[j]))
        return G

    # initial block: random, projected, B-orthonormalized
    X = []
    for j in range(k):
        v = rng.standard_normal(shape)
        w = PI(P['to_dev'](xp.asarray(v, dtype=P['dtype'])))
        del v
        if osc and j == 0:
            # max rho lives in one of two places, depending on the sign of the
            # defect Z = N - B. Where Z <= 0 (the modified a_u metric, whose
            # defect is the scalar -lambda_1) the maximizing direction is the
            # one on which B is largest, that is, the checkerboard mode of the
            # FD Laplacian.
            i0 = np.arange(shape[0])
            sgn = (-1.0) ** (i0[:, None, None] + i0[None, :, None]
                             + i0[None, None, :])
            w = PI(P['to_dev'](xp.asarray(sgn, dtype=P['dtype'])) * (1.0 + 0.0 * w))
            del sgn
        elif osc and j == 1 and seed_bump is not None:
            # Where Z has a large positive part -- the Euclidean metrics, which
            # omit 3 beta u^2 -- it is instead centred on the trap, where the
            # density is largest. Both 1D and 2D find it at r ~ 0.
            w = PI(seed_bump)
        X.append(w)
    Pblk = []
    theta = np.zeros(k)
    hist = []
    reason = 'maxit'
    restarts = 0
    t0 = time.time()
    for it in range(1, maxit + 1):
        AX = [applyA(x) for x in X]
        BX = [applyB(x) for x in X]
        GA, GB = block_ip(X, AX), block_ip(X, BX)
        GA, GB = 0.5 * (GA + GA.T), 0.5 * (GB + GB.T)
        w_, Q = np.linalg.eigh(GB)
        keep = w_ > 1e-13 * max(w_.max(), 1e-300)
        T = Q[:, keep] / np.sqrt(w_[keep])
        Ar = T.T @ GA @ T
        th, C = np.linalg.eigh(0.5 * (Ar + Ar.T))
        Z = T @ C[:, :k]
        nk = Z.shape[1]
        Xn = [PI(sum(Z[i, j] * X[i] for i in range(len(X)))) for j in range(nk)]
        AXn = [sum(Z[i, j] * AX[i] for i in range(len(AX))) for j in range(nk)]
        BXn = [sum(Z[i, j] * BX[i] for i in range(len(BX))) for j in range(nk)]
        X, AX, BX, theta = Xn, AXn, BXn, th[:nk]

        # residuals and convergence
        R, rel = [], []
        for j in range(len(X)):
            r = AX[j] - theta[j] * BX[j]
            R.append(PI(r))
            scale = nrm(AX[j]) + abs(theta[j]) * nrm(BX[j]) + 1e-300
            rel.append(nrm(r) / scale)
        # rel[0] belongs to the Ritz pair this routine returns; the rest of
        # the block is there to keep the search space rich and its residual
        # says nothing about the eigenvalue reported.
        hist.append((float(theta[0]), float(rel[0])))
        if verbose and (it % 10 == 0 or it == 1):
            print(f'    [lobpcg {label}] it {it:3d}  theta {theta[0]:.10e}  '
                  f'res {rel[0]:.2e} (block {max(rel):.1e})  '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if rel[0] < tol:
            reason = 'residual'
            break
        if len(hist) > 12:
            past = [x[0] for x in hist[-12:]]
            if (max(past) - min(past)) <= eig_tol * max(abs(past[-1]), 1e-300):
                if rel[0] < settle_res or restarts >= max_restarts:
                    reason = 'settled'
                    break
                # The Ritz value has stopped moving while the residual is still
                # large: the search space has gone B-degenerate rather than
                # converged. Drop the history and mix in a fresh direction.
                restarts += 1
                v = rng.standard_normal(shape)
                X = list(X) + [PI(P['to_dev'](xp.asarray(v, dtype=P['dtype'])))]
                del v
                Pblk, hist = [], []
                if verbose:
                    print(f'    [lobpcg {label}] restart {restarts} at '
                          f'res {rel[0]:.1e}', flush=True)
                continue

        W = [PI(prec(r)) for r in R]
        S = list(X) + W + list(Pblk)
        AS = list(AX) + [applyA(w) for w in W] + [applyA(p) for p in Pblk]
        BS = list(BX) + [applyB(w) for w in W] + [applyB(p) for p in Pblk]
        GA, GB = block_ip(S, AS), block_ip(S, BS)
        GA, GB = 0.5 * (GA + GA.T), 0.5 * (GB + GB.T)
        w_, Q = np.linalg.eigh(GB)
        keep = w_ > 1e-12 * max(w_.max(), 1e-300)
        T = Q[:, keep] / np.sqrt(w_[keep])
        Ar = T.T @ GA @ T
        th, C = np.linalg.eigh(0.5 * (Ar + Ar.T))
        m = min(k, C.shape[1])
        Z = T @ C[:, :m]
        nx = len(X)
        Xn = [PI(sum(Z[i, j] * S[i] for i in range(len(S)))) for j in range(m)]
        # P block: the same combination with the X part removed, which is the
        # locally optimal direction that gives LOBPCG its name
        Pn = [PI(sum(Z[i, j] * S[i] for i in range(nx, len(S)))) for j in range(m)]
        X, Pblk = Xn, Pn
        del S, AS, BS, W, R

    drift = 0.0
    if len(hist) > 12:
        past = [x[0] for x in hist[-12:]]
        drift = (max(past) - min(past)) / max(abs(past[-1]), 1e-300)
    return np.asarray(theta), X, dict(iters=it, res=float(rel[0]),
                                      res_block=float(max(rel)),
                                      why=reason, drift=drift,
                                      restarts=restarts,
                                      time=time.time() - t0, hist=hist)


def defect_range(P, name, u, lam1, ell, tf_f):
    """Extreme entries of the diagonal defect Z = N - B of Theorem 3.4.

    N - B is diagonal for all seven metrics, which is the structural fact the
    whole of Section 3 rests on:

        a_u     Z = 2 beta u^2 - lambda_1
        B_c     Z = (3 - c) beta u^2 - lambda_1
        a_0     Z = 3 beta u^2 - lambda_1
        a_0 mod Z = V_2 + 3 beta u^2 - lambda_1
        H^1     Z = V + 3 beta u^2 - lambda_1 - alpha
        TF      Z = V - (V - ell)_+ + beta u^2 - lambda_1 - f ell^{1/3}
    """
    xp = P['xp']
    beta, alpha = P['beta'], P['alpha']
    bu2 = beta * u * u
    V3 = P['V3']()
    if name == 'au':
        Z = 2.0 * bu2 - lam1
    elif name == 'a2u':
        Z = bu2 - lam1
    elif name == 'a3u':
        Z = 0.0 * bu2 - lam1
    elif name == 'a0':
        Z = 3.0 * bu2 - lam1
    elif name == 'a0mod':
        Z = (V3 - P['apply_V1'](xp.ones_like(u))) + 3.0 * bu2 - lam1
    elif name == 'h1':
        Z = V3 + 3.0 * bu2 - lam1 - alpha
    elif name == 'tf':
        Z = (V3 - xp.maximum(V3 - ell, 0.0) + bu2 - lam1
             - tf_f * ell ** (1.0 / 3.0))
    else:
        raise ValueError(name)
    zmin, zmax = float(Z.min()), float(Z.max())
    del Z, V3, bu2
    return zmin, zmax


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
METRICS = ['h1', 'a0', 'a0mod', 'au', 'a2u', 'a3u', 'tf']


def run(a):
    import gpe3d
    P = gpe3d.build(a.n, L=a.L, beta=a.beta, alpha=a.alpha,
                    potential=a.potential, backend=a.backend, solver=a.solver,
                    precision='fp64', disc='fd', w0=a.w0, delta=a.delta,
                    r0=a.r0, gz=a.gz)
    xp, ip, nrm, proj = make_helpers(P)
    print(f'== 3D Riemannian Hessian spectrum: n = {a.n}^3, L = {a.L}, '
          f'beta = {a.beta:g}, V = {a.potential}, h = {P["h"]:.4f} ==', flush=True)

    u, lam1, gs = ground_state(P, verbose=True, maxit=a.gs_maxit)
    ell = float(P['lam_rq'](u))
    ops = make_ops(P, u, lam1, ell=lam1, tf_f=a.tf_f, inner=a.inner,
                   inner_tol=a.inner_tol)
    N, prec_V1 = ops['N'], ops['prec_V1']

    def prec_N(w):
        x, _ = pcg(P, N, prec_V1, w, tol=a.inner_tol, maxit=a.inner)
        return x

    # the trap-centred seed for the rho_max problem: u^2 is the density, which
    # is where the positive part of the defect of a Euclidean metric lives
    bump = u * u

    out = dict(n=a.n, L=a.L, beta=a.beta, alpha=a.alpha, potential=a.potential,
               h=P['h'], lam1=lam1, ell=ell, tf_f=a.tf_f, ground_state=gs,
               block=a.block, tol=a.tol, metrics={})

    # linear spectral quantities the bounds are stated in terms of
    ident = lambda w: w
    t = time.time()
    th, _, inf2 = lobpcg_min(P, ops['A_beta'], ident, prec_V1, k=a.block,
                             constraint=u, tol=a.tol, maxit=a.maxit,
                             verbose=a.verbose, label='lam2(A_beta)')
    out['lam2'] = float(th[0])
    th, _, _ = lobpcg_min(P, ops['A_3beta'], ident, prec_V1, k=a.block,
                          constraint=None, tol=a.tol, maxit=a.maxit,
                          verbose=a.verbose, label='lam1(A_3beta)')
    out['lam1_A3'] = float(th[0])
    th, _, _ = lobpcg_min(P, ops['A_0'], ident, prec_V1, k=a.block,
                          constraint=None, tol=a.tol, maxit=a.maxit,
                          verbose=a.verbose, label='lam1(A_0)')
    out['lam1_A0'] = float(th[0])
    print(f'  lam1 = {lam1:.8f}  lam2 = {out["lam2"]:.8f}  '
          f'lam1(A_3b) = {out["lam1_A3"]:.8f}  lam1(A_0) = {out["lam1_A0"]:.8f}'
          f'   ({time.time()-t:.0f}s)', flush=True)

    # sigma_T: the coercivity constant of the constrained Hessian, min over
    # the tangent space of <N eta,eta>_h/<eta,eta>_h. This is the mu_1 of
    # rq_1d.py and the sigma_T of Theorem 3.4, and Section 3.5 predicts
    # sigma_T = mu_1 lambda_1^{1/3}(1+o(1)) with mu_1 = 2.4105.
    th, _, _ = lobpcg_min(P, N, ident, prec_N, k=a.block, constraint=u,
                          tol=a.tol, maxit=a.maxit, verbose=a.verbose,
                          label='sigma_T')
    out['sigma_T'] = float(th[0])
    print(f'  sigma_T = {out["sigma_T"]:.8f}   '
          f'sigma_T/lam1^(1/3) = {out["sigma_T"] / lam1 ** (1 / 3):.4f}',
          flush=True)

    for name in a.metrics.split(','):
        name = name.strip()
        if not name:
            continue
        B, Binv, exact = ops['metric'](name)
        # the defect Z = N - B is diagonal for every metric here, so its
        # extreme entries are the zeta_- and zeta_+ of Theorem 3.4 and cost
        # one pass over the grid
        zmin, zmax = defect_range(P, name, u, lam1, ell, a.tf_f)
        t = time.time()
        th, Xmin, i1 = lobpcg_min(P, N, B, prec_N, k=a.block, constraint=u,
                                  tol=a.tol, maxit=a.maxit, verbose=a.verbose,
                                  label=f'{name} min')
        rmin = float(th[0])
        th2, Xmax, i2 = lobpcg_min(P, B, N, Binv, k=max(a.block, a.block_max),
                                   constraint=u, tol=a.tol_max,
                                   maxit=a.maxit_max, verbose=a.verbose,
                                   label=f'{name} max', osc=True,
                                   seed_bump=bump)
        rmax = 1.0 / float(th2[0])
        # where the extreme directions live: the mean of |x| weighted by
        # eta^2, for the minimizing and the maximizing direction. The one- and
        # two-dimensional runs of rq_1d.py and rq_2d.py report the same quantity.
        r2 = rmx = None
        try:
            x1 = P['to_dev'](xp.asarray(P['x'], dtype=P['dtype']))
            rr = xp.sqrt(x1[:, None, None] ** 2 + x1[None, :, None] ** 2
                         + x1[None, None, :] ** 2)
            one = xp.ones_like(rr)
            e2 = Xmin[0] * Xmin[0]
            r2 = float(ip(rr, e2) / ip(e2, one))
            e2 = Xmax[0] * Xmax[0]
            rmx = float(ip(rr, e2) / ip(e2, one))
            del rr, e2, x1, one
        except Exception:
            pass
        bT = None
        if a.bT:
            thb, _, _ = lobpcg_min(P, B, ident, Binv, k=a.block, constraint=u,
                                   tol=1e-6, maxit=a.maxit,
                                   verbose=a.verbose, label=f'{name} b_T')
            bT = float(thb[0])
        sT = out['sigma_T']
        lb = sT / (sT + max(0.0, -zmin))            # Theorem 3.4 lower bound
        ub = None if bT is None else 1.0 + max(0.0, zmax) / bT
        out['metrics'][name] = dict(rmin=rmin, rmax=rmax, kappa=rmax / rmin,
                                    r_eta_min=r2, r_eta_max=rmx,
                                    its_min=i1['iters'],
                                    its_max=i2['iters'], res_min=i1['res'],
                                    res_max=i2['res'], why_min=i1['why'],
                                    why_max=i2['why'], drift_min=i1['drift'],
                                    drift_max=i2['drift'], zeta_min=zmin,
                                    zeta_max=zmax, b_T=bT, lb=lb, ub=ub,
                                    kappa_bound=None if ub is None else ub / lb,
                                    time=time.time() - t)
        print(f'  [{name:6s}] rho_min = {rmin:.6e}  rho_max = {rmax:.6f}  '
              f'kappa = {rmax/rmin:.4g}   (lobpcg {i1["iters"]}/{i2["iters"]} '
              f'its {i1["why"]}/{i2["why"]}, res {i1["res"]:.1e}/{i2["res"]:.1e}, '
              f'drift {i1["drift"]:.1e}/{i2["drift"]:.1e}, {time.time()-t:.0f}s)',
              flush=True)
        print(f'           |eta| weighted mean radius: min dir '
              f'{"--" if r2 is None else format(r2, ".2f")}, max dir '
              f'{"--" if rmx is None else format(rmx, ".2f")}', flush=True)
        print(f'           zeta_- = {max(0.0, -zmin):.4f}  zeta_+ = '
              f'{max(0.0, zmax):.4f}  bound: rho_min >= {lb:.4e}'
              + ('' if ub is None else f'  rho_max <= {ub:.4f}  '
                 f'kappa <= {ub / lb:.4g}  (b_T = {bT:.4f})'), flush=True)
        if a.out:
            with open(a.out, 'w') as f:
                json.dump(out, f, indent=1)

    lb_gap = 1.0 - lam1 / out['lam2']
    lb_A3 = 1.0 - lam1 / out['lam1_A3']
    print(f'  lower bounds: 1 - lam1/lam2 = {lb_gap:.4e}   '
          f'1 - lam1/lam1(A_3b) = {lb_A3:.4e}   '
          f'(lam1(A_3b) - lam1)/lam1^(1/3) = '
          f'{(out["lam1_A3"] - lam1) / lam1 ** (1 / 3):.4f}', flush=True)
    out['lb_gap'] = lb_gap
    out['lb_A3'] = lb_A3
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=99)
    p.add_argument('--L', type=float, default=8.0)
    p.add_argument('--beta', type=float, default=1600.0)
    p.add_argument('--alpha', type=float, default=20.0)
    p.add_argument('--potential', default='harmonic',
                   choices=['harmonic', 'stirrer', 'lattice'])
    p.add_argument('--w0', type=float, default=30.0)
    p.add_argument('--delta', type=float, default=1.0)
    p.add_argument('--r0', type=float, default=1.0)
    p.add_argument('--gz', type=float, default=2.0)
    p.add_argument('--metrics', default=','.join(METRICS))
    p.add_argument('--block', type=int, default=2)
    p.add_argument('--tol', type=float, default=1e-7)
    p.add_argument('--maxit', type=int, default=300)
    p.add_argument('--maxit-max', type=int, default=200,
                   help='iteration cap for the rho_max problem, whose extreme '
                        'eigenvalue sits in a grid-scale cluster')
    p.add_argument('--tol-max', type=float, default=1e-5)
    p.add_argument('--block-max', type=int, default=0)
    p.add_argument('--bT', type=int, default=1,
                   help='also compute b_T = min spec(B|_T) for the bounds')
    p.add_argument('--inner', type=int, default=40,
                   help='inner PCG iterations inside the LOBPCG preconditioner')
    p.add_argument('--inner-tol', type=float, default=1e-4)
    p.add_argument('--gs-maxit', type=int, default=400)
    p.add_argument('--tf-f', type=float, default=0.5)
    p.add_argument('--backend', default='jax', choices=['jax', 'numpy'])
    p.add_argument('--solver', default='dense', choices=['dense', 'fft'])
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--out', default='')
    p.add_argument('--auto-L', action='store_true',
                   help='L = 1.25 R_TF + 4 with the 3D Thomas-Fermi radius')
    a = p.parse_args()
    if a.auto_L:
        a.L = float(1.25 * (15.0 * a.beta / (8.0 * np.pi)) ** 0.2 + 4.0)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    run(a)
    print(f'total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
