"""
gpe3d.py -- discrete 3D Gross-Pitaevskii ground state problem and its operators.

Second-order finite differences on Omega = (-L,L)^3 with homogeneous Dirichlet
conditions, h = 2L/(n+1), interior nodes only, so the mass matrix is M = h^3 I.

Three potentials:

  'lattice'  V = x1^2 + x2^2 + x3^2 + 100 sum_j sin^2(pi x_j/4)
             the optical-lattice trap of Liu and Zhang (2026); fully
             separable, so V1 = V and V2 = 0.

  'harmonic' V = |x|^2, the isotropic trap of the spectra in Appendix A of
             the paper
  'stirrer'  V = x1^2 + x2^2 + gz^2 x3^2 + 2 w0 exp(-delta((x1-r0)^2 + x2^2))
             the 3D stirrer of Zhang and Zhou, with
             w0 = 30, delta = 1, r0 = 1, gz = 2.  Non-separable: the Gaussian
             couples x1 and x2.  It is a rank-one outer product in (x1,x2) and
             constant in x3, so V is applied by broadcasting and the n^3 array
             is never stored -- at n = 999 in fp64 that array alone is 8 GB and
             XLA would additionally capture it as a jit constant.

Conventions:
    normalization   <u,u>_M = h^3 sum(u^2) = 1
    energy          E(u) = h^3 [ 1/2 sum u.(A u) + (beta/4) sum u^4 ],
                    A = -Lap_h + V
    gradient        E'(u) = A u + beta u^3
    eigenvalue      lambda(u) = <u, A_u u>_M = h^3 sum u . E'(u)
    residue         v = |E'(u)|/(||E'(u)||_2 h^{3/2});  res = ||u-v||_2/||u||_2

Three linear solves, all separable, all by per-axis eigendecomposition:
    solve_G     (alpha I - Lap_h)^{-1}          H^1 metric
    solve_dys   (I - gamma Lap_h)^{-1}          Davis-Yin
    solve_V1    (-Lap_h + V1 - sigma)^{-1}      a_u preconditioner, linear init
"""

import os


def setup_backend(name, precision):
    """Return (xp, jit, device_put, sync, dtype). Call before importing jax."""
    if name == 'jax':
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.99')
        os.environ.setdefault('XLA_FLAGS',
                              '--xla_gpu_force_compilation_parallelism=1')
        os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
        os.environ.setdefault('JAX_ENABLE_X64', 'True')
        os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'float32')
        import jax
        import jax.numpy as jnp
        from jax import config
        config.update('jax_enable_x64', True)
        dtype = jnp.float64 if precision == 'fp64' else jnp.float32

        def sync(a):
            try:
                a.block_until_ready()
            except AttributeError:
                pass
            return a
        return jnp, jax.jit, jax.device_put, sync, dtype

    import numpy as np
    dtype = np.float64 if precision == 'fp64' else np.float32
    return np, (lambda f, **kw: f), (lambda a: a), (lambda a: a), dtype


class Counter:
    """Counts linear solves."""

    def __init__(self):
        self.n = 0
        self.evals = 0        # energy evaluations inside line searches

    def reset(self):
        self.n = 0
        self.evals = 0


def potential_1d(potential, x, w0=30.0, delta=1.0, r0=1.0, gz=2.0):
    """Separable part V1 per axis, and the rank-one factors of V2.

    Returns (v_ax, st_a, st_b) with
        V1(x) = v_ax[0](x1) + v_ax[1](x2) + v_ax[2](x3)
        V2(x) = st_a(x1) * st_b(x2)          (None when V2 = 0)
    """
    import numpy as np
    if potential == 'lattice':
        v = x ** 2 + 100.0 * np.sin(np.pi * x / 4.0) ** 2
        return [v, v, v], None, None
    if potential == 'harmonic':
        return [x ** 2, x ** 2, x ** 2], None, None
    if potential == 'stirrer':
        v_ax = [x ** 2, x ** 2, (gz ** 2) * x ** 2]
        st_a = 2.0 * w0 * np.exp(-delta * (x - r0) ** 2)
        st_b = np.exp(-delta * x ** 2)
        return v_ax, st_a, st_b
    raise ValueError(f"unknown potential {potential!r}")


def _make_uerr_host(u_star_np, chunk, n, h3, xp, dtype):
    """||u - u*||_M with u* kept in HOST memory.

    At n = 999 an fp64 u* is 8 GB on the device, which the working set of the
    line-search methods cannot spare.  u* is a diagnostic, not part of any
    iteration, so it stays on the host: u^k is pulled across in fp32 and the
    norm is accumulated in slabs.  This floors the diagnostic near 1e-7, well
    below where the curves stop.
    """
    import numpy as np
    us = np.asarray(u_star_np, dtype=np.float32)

    def uerr(u):
        # Cast to fp32 ON THE DEVICE before transferring.  np.asarray(u,
        # dtype=np.float32) on an fp64 device array copies the full fp64 buffer
        # across first, and JAX stages that through the device: at n=999 that
        # is a 15 GB device allocation which OOMs whenever the solver's own
        # working set is large (it killed Riemannian CG and the a_u flow while
        # Davis-Yin, with a smaller working set, survived it).
        uh = np.asarray(u.astype('float32'))
        tot = np.float64(0.0)
        for s0 in range(0, n, chunk):
            d = uh[s0:s0 + chunk] - us[s0:s0 + chunk]
            tot += np.float64(np.sum(d.astype(np.float64) ** 2))
        return float(np.sqrt(h3 * tot))
    return uerr


def build(n, L=8.0, beta=1600.0, alpha=20.0, potential='lattice', order=2,
          backend='jax', solver='dense', precision='fp64',
          w0=30.0, delta=1.0, r0=1.0, gz=2.0, disc='fd', q_order=20):
    """Build the problem. Returns a dict of arrays and (jitted) callables.

    disc='fd'   second-order finite differences on a uniform grid, mass
                matrix M = h^3 I, spectrum of the 1D Laplacian in closed form
                and eigenvectors the DST-I.
    disc='sem'  Q^q spectral elements with (q+1)-point Gauss-Lobatto
                quadrature, n = q*Ncell - 1.  The mass matrix is the diagonal
                of GLL weights, M = h^3 m_i m_j m_k, so every inner product
                carries that weight; the 1D operators are diagonalized
                numerically by SEM.cal_matrix_V1_fast, whose eigenvector matrix
                is M^{-1/2}-orthogonal rather than orthogonal, so the forward
                and backward transforms are invT and T and not transposes of
                one another.
    """
    import numpy as np
    xp, jit, to_dev, sync, dtype = setup_backend(backend, precision)

    if disc == 'sem':
        return _build_sem(n, L, beta, alpha, potential, backend, solver,
                          precision, w0, delta, r0, gz, q_order,
                          xp, jit, to_dev, sync, dtype)

    h = 2.0 * L / (n + 1)
    j = np.arange(1, n + 1)
    x = -L + h * j
    h3 = h ** 3

    # ---- spectrum of the 1D Laplacian (DST-I) ------------------------------
    lam = (2.0 - 2.0 * np.cos(np.pi * j / (n + 1))) / h ** 2
    if order == 4:
        lam = lam / (5.0 / 6.0 + np.cos(np.pi * j / (n + 1)) / 6.0)
    T = np.sin(np.pi * np.outer(j, j) / (n + 1))
    T = T / np.linalg.norm(T, axis=0, keepdims=True)

    # ---- spectrum of the 1D operators K + diag(V1_k) -----------------------
    # K + diag(v) is symmetric, so eigh gives an orthogonal basis per axis and
    # (-Lap_h + V1) is diagonalized by the tensor product of the three.
    v_ax_np, st_a_np, st_b_np = potential_1d(potential, x, w0, delta, r0, gz)
    K1 = (np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1)
          + np.diag(-np.ones(n - 1), -1)) / h ** 2
    if order == 4:
        K1 = (T * lam) @ T.T
    TV, muV = [], []
    for v in v_ax_np:
        mu_k, T_k = np.linalg.eigh(K1 + np.diag(v))
        TV.append(T_k)
        muV.append(mu_k)

    d_ = dict(dtype=dtype)
    lam_d = to_dev(xp.asarray(lam, **d_))
    T_d = to_dev(xp.asarray(T, **d_))
    Tt_d = to_dev(xp.asarray(T.T, **d_))
    v_ax = [to_dev(xp.asarray(v, **d_)) for v in v_ax_np]
    TV_d = [to_dev(xp.asarray(t, **d_)) for t in TV]
    TVt_d = [to_dev(xp.asarray(t.T, **d_)) for t in TV]
    muV_d = [to_dev(xp.asarray(m, **d_)) for m in muV]
    st_a = None if st_a_np is None else to_dev(xp.asarray(st_a_np, **d_))
    st_b = None if st_b_np is None else to_dev(xp.asarray(st_b_np, **d_))

    counter = Counter()

    def lam3(shift):
        return (lam_d[:, None, None] + lam_d[None, :, None]
                + lam_d[None, None, :] + shift)

    # The n^3 eigenvalue tensor is NEVER materialized: lam3() is called inside
    # each jitted kernel so XLA fuses it into the elementwise multiply.  Held
    # as a module-level array instead it is 8 GB at n=999 in fp64, and -- worse
    # -- XLA captures it as a jit constant in every kernel that closes over it,
    # which OOM-kills the host during lowering long before the GPU is touched.

    def muV3(shift):
        return (muV_d[0][:, None, None] + muV_d[1][None, :, None]
                + muV_d[2][None, None, :] + shift)

    # ---- per-axis contraction ---------------------------------------------
    if backend == 'jax':
        def apply_axis(u, M, axis):
            if axis == 0:
                return xp.einsum('ijk,il->ljk', u, M)
            if axis == 1:
                return xp.einsum('ijk,jl->ilk', u, M)
            return xp.einsum('ijk,kl->ijl', u, M)
    else:
        def apply_axis(u, M, axis):
            if axis == 0:
                return (M.T @ u.reshape(n, -1)).reshape(n, n, n)
            if axis == 1:
                return (M.T @ u.transpose(1, 0, 2).reshape(n, -1)
                        ).reshape(n, n, n).transpose(1, 0, 2)
            return (M.T @ u.transpose(2, 0, 1).reshape(n, -1)
                    ).reshape(n, n, n).transpose(1, 2, 0)

    # NOTE apply_axis(u, M, ax) computes sum_b u[b] M[b,a], i.e. it applies
    # M^T.  So to evaluate B = T diag(mu) T^T the forward sweep is handed T and
    # the backward sweep T^T.  The DST matrix is symmetric so the distinction is
    # invisible there, but the eigh bases of K + diag(V1) are not symmetric and
    # swapping the two gives a silently wrong solve.
    def tsf(u, mats):
        w = apply_axis(u, mats[0], 0)
        w = apply_axis(w, mats[1], 1)
        return apply_axis(w, mats[2], 2)

    # ---- Laplacian transforms: closed-form DST, or an FFT ------------------
    c = 0.5 * np.sqrt(2.0 / (n + 1))
    if backend == 'numpy':
        from scipy.fft import dstn as _dstn

        def dst3(u):
            return _dstn(u, type=1, axes=(0, 1, 2))
    else:
        def _dst1(u, axis):
            u = xp.moveaxis(u, axis, -1)
            zero = xp.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
            y = xp.concatenate([zero, u, zero, -xp.flip(u, axis=-1)], axis=-1)
            out = -xp.imag(xp.fft.fft(y, axis=-1))[..., 1:n + 1]
            return xp.moveaxis(out, -1, axis)

        def dst3(u):
            return _dst1(_dst1(_dst1(u, 0), 1), 2)

    if solver == 'dense':
        _fwd = lambda u: tsf(u, [T_d] * 3)     # T^T
        _bwd = lambda u: tsf(u, [Tt_d] * 3)    # T
        _scale = 1.0
    else:
        _fwd = _bwd = dst3
        _scale = c ** 6

    # ---- operators ---------------------------------------------------------
    @jit
    def lap(u):
        return _bwd(_fwd(u) * (_scale * lam3(0.0)))

    @jit
    def solve_G(w):
        """G_alpha w = (alpha I - Lap_h)^{-1} w"""
        return _bwd(_fwd(w) * (_scale / lam3(alpha)))

    @jit
    def apply_Ginv(w):
        return _bwd(_fwd(w) * (_scale * lam3(alpha)))

    @jit
    def solve_dys(w, gamma):
        return _bwd(_fwd(w) * (_scale / (gamma * lam3(0.0) + 1.0)))

    @jit
    def apply_V1(u):
        return (v_ax[0][:, None, None] + v_ax[1][None, :, None]
                + v_ax[2][None, None, :]) * u

    if st_a is None:
        apply_V = apply_V1

        @jit
        def V3():
            return (v_ax[0][:, None, None] + v_ax[1][None, :, None]
                    + v_ax[2][None, None, :]) * xp.ones((1, 1, n), dtype=dtype)
    else:
        @jit
        def apply_V(u):
            return apply_V1(u) + (st_a[:, None, None] * st_b[None, :, None]) * u

        @jit
        def V3():
            return ((v_ax[0][:, None, None] + v_ax[1][None, :, None]
                     + v_ax[2][None, None, :])
                    + st_a[:, None, None] * st_b[None, :, None])

    @jit
    def solve_V1_shifted(w, sigma):
        """(-Lap_h + V1 - sigma)^{-1} w -- the preconditioner of Liu and Zhang."""
        z = tsf(w, TV_d)                       # T^T
        z = z / muV3(-sigma)
        return tsf(z, TVt_d)                   # T

    @jit
    def solve_V1(w):
        z = tsf(w, TV_d)                       # T^T
        z = z / muV3(0.0)
        return tsf(z, TVt_d)                   # T

    @jit
    def apply_A(u):
        """A u = (-Lap_h + V) u"""
        return lap(u) + apply_V(u)

    @jit
    def apply_Au(u_cur, w):
        """A_u w = (-Lap_h + V + beta u_cur^2) w"""
        return lap(w) + apply_V(w) + beta * (u_cur * u_cur) * w

    @jit
    def grad_E(u):
        return apply_A(u) + beta * u ** 3

    @jit
    def energy(u):
        return h3 * (0.5 * xp.sum(u * apply_A(u))
                     + 0.25 * beta * xp.sum(u ** 4))

    @jit
    def energy_and_grad(u):
        g = grad_E(u)
        E = h3 * (0.5 * xp.sum(u * g) - 0.25 * beta * xp.sum(u ** 4))
        return E, g

    @jit
    def residue_from_grad(u, g):
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        return xp.sqrt(xp.sum((u - v) ** 2)) / xp.sqrt(xp.sum(u * u))

    @jit
    def ip_M(a, b):
        return h3 * xp.sum(a * b)

    @jit
    def norm_M(a):
        return xp.sqrt(h3 * xp.sum(a * a))

    @jit
    def apply_Btf(u_cur, w, ell, fpar):
        """B_TF w, the Thomas-Fermi metric operator of eq. (11) of the paper:

            B_TF = -Lap_h + 2 beta diag(u^2) + diag((V - ell)_+) + f ell^(1/3),

        where ell is the paper's lambda(u) and f is the paper's ell.

        The potential is clipped rather than shifted, which is what keeps the
        operator positive definite everywhere while still subtracting the
        chemical potential inside the Thomas-Fermi region, where V < ell. V is
        a broadcast of the separable axes plus the rank-one stirrer, so it is
        computed tile by tile here and never stored as an n^3 array.
        """
        return (lap(w) + xp.maximum(V3() - ell, 0.0) * w
                + 2.0 * beta * (u_cur * u_cur) * w
                + fpar * ell ** (1.0 / 3.0) * w)

    @jit
    def lam_rq(u):
        """lambda(u) = <A_beta(u) u, u>_h = 2 E(u) + (beta/2) <u^2, u^2>_h.

        The scalar ell of the Thomas-Fermi metric. It is not an eigenvalue to
        be computed: it only has to approximate lambda_1(A_beta) to the layer
        scale, and this Rayleigh quotient of the current iterate does, at the
        cost of the energy evaluation plus one quartic reduction.
        """
        return 2.0 * energy(u) + 0.5 * beta * ip_M(u * u, u * u)

    @jit
    def normalize(a):
        return a / xp.sqrt(h3 * xp.sum(a * a))

    @jit
    def ip_alpha(a, b):
        """<a,b>_alpha = h^3 <a, (alpha I - Lap_h) b>"""
        return h3 * xp.sum(a * apply_Ginv(b))

    @jit
    def _proj_t(w, u, Gu):
        return w - (xp.sum(w * u) / xp.sum(Gu * u)) * Gu

    # ---- fused single-step kernels ----------------------------------------
    # One @jit per iteration, not one per operator.  Calling solve_G, grad_E and
    # normalize as separate jitted functions from Python forces every
    # intermediate to be materialized as a live device buffer, because XLA
    # cannot do liveness analysis across jit boundaries.  Fusing the step lets
    # it reuse buffers, which is the difference between fitting and not fitting
    # at n=999 in fp64.  The arithmetic is identical either way.
    @jit
    def h1_step_fused(u, dt):
        """One Riemannian H^1 gradient-flow step at constant step size."""
        Proj = solve_G(u)
        Grad_raw = apply_A(u) + beta * u ** 3
        Grad = solve_G(Grad_raw)
        u_new = u - dt * Grad + dt * (xp.sum(u * Grad) / xp.sum(u * Proj)) * Proj
        u_new = u_new / xp.sqrt(h3 * xp.sum(u_new * u_new))
        lam = h3 * xp.sum(u * Grad_raw)
        E = lam - 0.5 * h3 * xp.sum(u * Grad_raw) - 0.25 * beta * h3 * xp.sum(u ** 4)
        v = xp.abs(Grad_raw) / xp.sqrt(xp.sum(Grad_raw ** 2)) / h ** 1.5
        res = xp.sqrt(xp.sum((u - v) ** 2)) / xp.sqrt(xp.sum(u * u))
        return u_new, E, lam, res

    @jit
    def diag_fused(u):
        """E, lambda and the residue in one kernel."""
        g = apply_A(u) + beta * u ** 3
        lam = h3 * xp.sum(u * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u - v) ** 2)) / xp.sqrt(xp.sum(u * u))
        return E, lam, res

    @jit
    def rgrad_fused(u):
        """Riemannian gradient g_alpha(u) and G_alpha u, with the diagnostics."""
        g = apply_A(u) + beta * u ** 3
        Gu = solve_G(u)
        Ge = solve_G(g)
        gc = Ge - (xp.sum(Ge * u) / xp.sum(Gu * u)) * Gu
        lam = h3 * xp.sum(u * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u - v) ** 2)) / xp.sqrt(xp.sum(u * u))
        return gc, Gu, E, lam, res

    # ---- fused kernels for the remaining methods ---------------------------
    @jit
    def pc_fused(u):
        """Projected gradient r = E'(u) - <u,E'(u)>_M u and G_alpha r, with the
        diagnostics.  One kernel for the preconditioned methods."""
        g = apply_A(u) + beta * u ** 3
        lam = h3 * xp.sum(u * g)
        r = g - lam * u
        Pr = solve_G(r)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u - v) ** 2)) / xp.sqrt(xp.sum(u * u))
        return r, Pr, E, lam, res

    @jit
    def dys_L(u):
        """L(u) = max_i (V_i + 3 beta u_i^2), the Lipschitz constant of the
        gradient of the smooth part of the splitting at u."""
        V3v = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
               + v_ax[2][None, None, :])
        if st_a is not None:
            V3v = V3v + st_a[:, None, None] * st_b[None, :, None]
        return xp.max(V3v + 3.0 * beta * u * u)

    @jit
    def dys_fused_gamma(z, gamma):
        """One Davis-Yin step at a prescribed step size, with diagnostics.

        Identical to dys_fused except that gamma is supplied rather than
        computed from the iterate; used by the step-size ablation.
        """
        u = z / xp.sqrt(h3 * xp.sum(z * z))
        V3v = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
               + v_ax[2][None, None, :])
        if st_a is not None:
            V3v = V3v + st_a[:, None, None] * st_b[None, :, None]
        w = 2.0 * u - z - gamma * ((V3v + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        u_new = z_new / xp.sqrt(h3 * xp.sum(z_new * z_new))
        g = apply_A(u_new) + beta * u_new ** 3
        lam = h3 * xp.sum(u_new * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u_new ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u_new - v) ** 2)) / xp.sqrt(xp.sum(u_new * u_new))
        return z_new, u_new, E, lam, res

    # ---- DYS II: the separable part of the potential goes implicit ---------
    # (I - gamma Lap_h + gamma V1)^{-1} is the same per-axis eigendecomposition
    # solve_V1 uses, with the multiplier 1/(1 + gamma mu) in place of 1/mu, so
    # one step of DYS II costs exactly what one step of DYS I costs.  Only V2,
    # the non-separable remainder, stays explicit, and the Lipschitz constant
    # of the explicit part falls with it: on the lattice V2 = 0, and the
    # potential stops limiting the step size at all.
    @jit
    def solve_dys2(w, gamma):
        z = tsf(w, TV_d)                       # T^T
        z = z / (1.0 + gamma * muV3(0.0))
        return tsf(z, TVt_d)                   # T

    @jit
    def dys2_L(u):
        """L(u) = max_i (V2_i + 3 beta u_i^2) -- the explicit part only."""
        if st_a is None:
            return xp.max(3.0 * beta * u * u)
        return xp.max(st_a[:, None, None] * st_b[None, :, None]
                      + 3.0 * beta * u * u)

    @jit
    def dys2_fused_gamma(z, gamma):
        """One DYS II step at a prescribed step size, with diagnostics."""
        u = z / xp.sqrt(h3 * xp.sum(z * z))
        if st_a is None:
            expl = beta * u * u * u
        else:
            expl = (st_a[:, None, None] * st_b[None, :, None]
                    + beta * u * u) * u
        w = 2.0 * u - z - gamma * expl
        z_new = z + solve_dys2(w, gamma) - u
        u_new = z_new / xp.sqrt(h3 * xp.sum(z_new * z_new))
        g = apply_A(u_new) + beta * u_new ** 3
        lam = h3 * xp.sum(u_new * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u_new ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u_new - v) ** 2)) / xp.sqrt(xp.sum(u_new * u_new))
        return z_new, u_new, E, lam, res

    @jit
    def dys_grad_h(u):
        """grad h(u) = (V + beta u^2) u, the smooth part of the splitting."""
        V3v = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
               + v_ax[2][None, None, :])
        if st_a is not None:
            V3v = V3v + st_a[:, None, None] * st_b[None, :, None]
        return (V3v + beta * u * u) * u

    @jit
    def dys_energy_at(z, gamma):
        """Energy of the iterate one Davis-Yin step at step size gamma would
        produce.  Used to search over gamma; one step per trial."""
        u = z / xp.sqrt(h3 * xp.sum(z * z))
        V3v = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
               + v_ax[2][None, None, :])
        if st_a is not None:
            V3v = V3v + st_a[:, None, None] * st_b[None, :, None]
        w = 2.0 * u - z - gamma * ((V3v + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        un = z_new / xp.sqrt(h3 * xp.sum(z_new * z_new))
        return h3 * (0.5 * xp.sum(un * (lap(un) + apply_V(un)))
                     + 0.25 * beta * xp.sum(un ** 4))

    @jit
    def dys_lyapunov(z, gamma):
        """The Lyapunov function of Theorem 3.4 of Li, Zheng and Zhang (2026)
        at z, for step size gamma.

            R(z) = ||z||_2 - 1 + gamma lambda(u),   u = z/||z||_2
            L(z) = E_h(u) + R(z)^2 / gamma

        The scheme decreases this, not the energy: the energy alone is not
        monotone along the iteration, which is why choosing gamma to minimize
        the energy of the next iterate, or backtracking on it, both misbehave.
        """
        nz = xp.sqrt(h3 * xp.sum(z * z))
        u = z / nz
        g = lap(u) + apply_V(u) + beta * u ** 3
        lam = h3 * xp.sum(u * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u ** 4)
        R = nz - 1.0 + gamma * lam
        return E + R * R / gamma

    @jit
    def dys_lyapunov_at(z, gamma):
        """The Lyapunov value one Davis-Yin step at gamma would produce."""
        u = z / xp.sqrt(h3 * xp.sum(z * z))
        V3v = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
               + v_ax[2][None, None, :])
        if st_a is not None:
            V3v = V3v + st_a[:, None, None] * st_b[None, :, None]
        w = 2.0 * u - z - gamma * ((V3v + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        nz = xp.sqrt(h3 * xp.sum(z_new * z_new))
        un = z_new / nz
        g = lap(un) + apply_V(un) + beta * un ** 3
        lam = h3 * xp.sum(un * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(un ** 4)
        R = nz - 1.0 + gamma * lam
        return E + R * R / gamma

    @jit
    def dys_fused(z):
        """One Davis-Yin step with the diagnostics of the new iterate."""
        u = z / xp.sqrt(h3 * xp.sum(z * z))
        V3v = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
               + v_ax[2][None, None, :])
        if st_a is not None:
            V3v = V3v + st_a[:, None, None] * st_b[None, :, None]
        gamma = 1.0 / xp.max(V3v + 3.0 * beta * u * u)
        w = 2.0 * u - z - gamma * ((V3v + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        u_new = z_new / xp.sqrt(h3 * xp.sum(z_new * z_new))
        g = apply_A(u_new) + beta * u_new ** 3
        lam = h3 * xp.sum(u_new * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u_new ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u_new - v) ** 2)) / xp.sqrt(xp.sum(u_new * u_new))
        return z_new, u_new, E, lam, res, gamma

    @jit
    def au_cg_body(u_cur, w, r, p, rz):
        """One preconditioned CG iteration for A_u w = rhs, fused."""
        Ap = lap(p) + apply_V(p) + beta * (u_cur * u_cur) * p
        a_k = rz / xp.sum(p * Ap)
        w = w + a_k * p
        r = r - a_k * Ap
        z = solve_V1(r)
        rz_new = xp.sum(r * z)
        p = z + (rz_new / rz) * p
        return w, r, p, rz_new, xp.sqrt(xp.sum(r * r))

    # ---- adaptive step size for the a_u flow, Remark 4.3 of Henning-Peterseim
    # With p = z^n and q = gamma^n G^n z^n, the energy along the update
    #     v(tau) = ((1-tau) p + tau q) / ||(1-tau) p + tau q||
    # is a ratio of polynomials in tau whose twelve coefficients are inner
    # products of p and q.  Minimizing it therefore costs no extra solve, which
    # is the point of the remark: alpha and the zeta/beta moments are gathered
    # once per step and f(tau) is then a scalar evaluation.
    #
    # Two kernels for the alphas so that neither holds two applications of A at
    # once, and a third for the moments; a single kernel holding Ap, Aq, p^2,
    # q^2 and pq at n=999 asks for an allocation larger than it saves.
    @jit
    def au_tau_alpha_p(p, q):
        Ap = apply_A(p)
        return xp.array([h3 * xp.sum(p * Ap), 2.0 * h3 * xp.sum(q * Ap)])

    @jit
    def au_tau_alpha_q(q):
        return h3 * xp.sum(q * apply_A(q))

    @jit
    def au_tau_moments(p, q):
        return xp.array([
            h3 * xp.sum(p * p), 2.0 * h3 * xp.sum(p * q), h3 * xp.sum(q * q),
            0.5 * beta * h3 * xp.sum(p ** 4),
            2.0 * beta * h3 * xp.sum(p ** 3 * q),
            3.0 * beta * h3 * xp.sum(p ** 2 * q ** 2),
            2.0 * beta * h3 * xp.sum(p * q ** 3),
            0.5 * beta * h3 * xp.sum(q ** 4)])

    @jit
    def au_step_tau(u, q, tau):
        """u <- ((1-tau) u + tau q) / ||.||, with the diagnostics."""
        u_new = (1.0 - tau) * u + tau * q
        u_new = u_new / xp.sqrt(h3 * xp.sum(u_new * u_new))
        g = apply_A(u_new) + beta * u_new ** 3
        lam = h3 * xp.sum(u_new * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u_new ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u_new - v) ** 2)) / xp.sqrt(xp.sum(u_new * u_new))
        return u_new, E, lam, res

    @jit
    def au_update(u, w):
        """gamma = <u,u>/<u,w>, then u <- (u + tau(gamma w - u))/||.||, tau=1."""
        gamma = xp.sum(u * u) / xp.sum(u * w)
        u_new = gamma * w
        u_new = u_new / xp.sqrt(h3 * xp.sum(u_new * u_new))
        g = apply_A(u_new) + beta * u_new ** 3
        lam = h3 * xp.sum(u_new * g)
        E = 0.5 * lam - 0.25 * beta * h3 * xp.sum(u_new ** 4)
        v = xp.abs(g) / xp.sqrt(xp.sum(g * g)) / h ** 1.5
        res = xp.sqrt(xp.sum((u_new - v) ** 2)) / xp.sqrt(xp.sum(u_new * u_new))
        return u_new, E, lam, res, gamma

    # ---- fused great-circle and CG-direction kernels -----------------------
    # p, q and the tangent projection are computed inside the kernel and never
    # returned, so they never become live Python-level buffers.
    # The coefficient kernels are split so that no kernel holds two
    # applications of A at once.  Fusing them into one is tempting and does
    # reduce Python-level buffers, but it makes XLA hold u, d, p, q, Au and Aq
    # simultaneously -- a 29.75 GiB allocation at n=999, which is worse than
    # the buffer it saves.
    @jit
    def gc_q(u, d):
        """The normalized tangent part of d at u."""
        p = d - (h3 * xp.sum(d * u)) * u
        return p / xp.sqrt(h3 * xp.sum(p * p))

    @jit
    def _gcq_a(u, q):
        Au = apply_A(u)
        return xp.array([xp.sum(u * Au), xp.sum(q * Au)])

    @jit
    def _gcq_b(u, q):
        Aq = apply_A(q)
        return xp.array([
            xp.sum(q * Aq),
            xp.sum(u ** 4), xp.sum(u ** 3 * q), xp.sum(u ** 2 * q ** 2),
            xp.sum(u * q ** 3), xp.sum(q ** 4)])

    def gc_coeffs_fused(u, d):
        import numpy as _np
        q = gc_q(u, d)
        a = _np.asarray(_gcq_a(u, q), dtype=_np.float64)
        b = _np.asarray(_gcq_b(u, q), dtype=_np.float64)
        del q
        return _np.concatenate([a, b])

    @jit
    def gc_step_fused(u, d, theta):
        """u_new = cos(theta) u + sin(theta) q, with q recomputed in-kernel."""
        p = d - (h3 * xp.sum(d * u)) * u
        q = p / xp.sqrt(h3 * xp.sum(p * p))
        return xp.cos(theta) * u + xp.sin(theta) * q

    @jit
    def gc_energy_at(u, d, theta):
        """E at angle theta along the great circle, built explicitly.  This is
        what the numeric line search calls, and it is one kernel per probe."""
        p = d - (h3 * xp.sum(d * u)) * u
        q = p / xp.sqrt(h3 * xp.sum(p * p))
        v = xp.cos(theta) * u + xp.sin(theta) * q
        return h3 * (0.5 * xp.sum(v * apply_A(v))
                     + 0.25 * beta * xp.sum(v ** 4))

    @jit
    def pcg_dir_fused(Pr_new, p, beta_cg):
        """d = -G_alpha r_new + beta_cg p."""
        return -Pr_new + beta_cg * p

    @jit
    def rcg_dir_fused(gc_new, direction, u_new, Gu_new, beta_cg):
        """Transport the previous direction and form the new CG direction.

        The vector transport is the projection proj_t applied at the new
        point, so the transport and the combination fuse into one kernel and
        the transported direction is never a separate buffer.
        """
        c = xp.sum(direction * u_new) / xp.sum(Gu_new * u_new)
        dt_ = direction - c * Gu_new
        d_new = -gc_new + beta_cg * dt_
        slope = h3 * xp.sum(gc_new * apply_Ginv(d_new))
        return d_new, slope

    @jit
    def _lsf_a(u, d):
        Au = apply_A(u)
        return xp.array([xp.sum(u * Au), xp.sum(d * Au)])

    @jit
    def _lsf_b(u, d):
        Ad = apply_A(d)
        return xp.array([
            xp.sum(d * Ad),
            xp.sum(u * u), xp.sum(u * d), xp.sum(d * d),
            xp.sum(u ** 4), xp.sum(u ** 3 * d), xp.sum(u ** 2 * d ** 2),
            xp.sum(u * d ** 3), xp.sum(d ** 4)])

    def ls_coeffs_fused(u, d):
        """Line-search coefficients for E((u + t d)/||u + t d||_M), in two
        kernels so neither holds two applications of A."""
        import numpy as _np
        a = _np.asarray(_lsf_a(u, d), dtype=_np.float64)
        b = _np.asarray(_lsf_b(u, d), dtype=_np.float64)
        return _np.concatenate([a, b])

    @jit
    def ray_energy_at(u, d, t):
        """E(retraction(u, t d)), built explicitly: one kernel per probe."""
        v = u + t * d
        v = v / xp.sqrt(h3 * xp.sum(v * v))
        return h3 * (0.5 * xp.sum(v * apply_A(v))
                     + 0.25 * beta * xp.sum(v ** 4))

    # ---- Davis-Yin step ----------------------------------------------------
    @jit
    def dys_step(z):
        u = z / xp.sqrt(h3 * xp.sum(z * z))
        Vu = apply_V(u)
        gamma = 1.0 / xp.max(V3() + 3.0 * beta * u * u)
        w = 2.0 * u - z - gamma * (Vu + beta * u * u * u)
        xi = solve_dys(w, gamma)
        return z + xi - u, gamma

    # ---- exact line-search coefficients ------------------------------------
    # Two kernels rather than one: with a single kernel XLA keeps A u and A d
    # live at the same time, and at n = 999 in fp64 each is 8 GB.
    @jit
    def _ls_a(u, d):
        Au = apply_A(u)
        return xp.array([xp.sum(u * Au), xp.sum(d * Au)])

    @jit
    def _ls_b(u, d):
        Ad = apply_A(d)
        return xp.array([
            xp.sum(d * Ad),
            xp.sum(u * u), xp.sum(u * d), xp.sum(d * d),
            xp.sum(u ** 4), xp.sum(u ** 3 * d), xp.sum(u ** 2 * d ** 2),
            xp.sum(u * d ** 3), xp.sum(d ** 4)])

    def line_coeffs(u, d):
        import numpy as _np
        a = _np.asarray(_ls_a(u, d), dtype=_np.float64)
        b = _np.asarray(_ls_b(u, d), dtype=_np.float64)
        return _np.concatenate([a, b])

    def line_energy(cf, t):
        q0, q1, q2, n0, n1, n2, p0, p1, p2, p3, p4 = cf
        Q = q0 + 2 * q1 * t + q2 * t * t
        N = n0 + 2 * n1 * t + n2 * t * t
        P = (p0 + 4 * p1 * t + 6 * p2 * t ** 2
             + 4 * p3 * t ** 3 + p4 * t ** 4)
        return 0.5 * Q / N + 0.25 * beta * P / (h3 * N * N)

    @jit
    def _gc_a(u, q):
        Au = apply_A(u)
        return xp.array([xp.sum(u * Au), xp.sum(q * Au)])

    @jit
    def _gc_b(u, q):
        Aq = apply_A(q)
        return xp.array([
            xp.sum(q * Aq),
            xp.sum(u ** 4), xp.sum(u ** 3 * q), xp.sum(u ** 2 * q ** 2),
            xp.sum(u * q ** 3), xp.sum(q ** 4)])

    def circle_coeffs(u, q):
        import numpy as _np
        a = _np.asarray(_gc_a(u, q), dtype=_np.float64)
        b = _np.asarray(_gc_b(u, q), dtype=_np.float64)
        return _np.concatenate([a, b])

    def circle_energy(cf, th):
        import numpy as _np
        a_uu, a_uq, a_qq, p0, p1, p2, p3, p4 = cf
        c_, s_ = _np.cos(th), _np.sin(th)
        quad = c_ * c_ * a_uu + 2 * c_ * s_ * a_uq + s_ * s_ * a_qq
        quar = (c_ ** 4 * p0 + 4 * c_ ** 3 * s_ * p1
                + 6 * c_ ** 2 * s_ ** 2 * p2 + 4 * c_ * s_ ** 3 * p3
                + s_ ** 4 * p4)
        return h3 * (0.5 * quad + 0.25 * beta * quar)

    # ---- Thomas-Fermi helper ----------------------------------------------
    @jit
    def tf_mass(lam_tf):
        """h^3 sum (lam - V)_+ / beta -- the discrete Phi(lam) + 1."""
        return h3 * xp.sum(xp.maximum(lam_tf - V3(), 0.0)) / beta

    @jit
    def tf_profile(lam_tf):
        u = xp.sqrt(xp.maximum(lam_tf - V3(), 0.0) / beta)
        return u / xp.sqrt(h3 * xp.sum(u * u))

    Vmin = float(xp.min(V3()))
    Vmax = float(xp.max(V3()))
    mu_min = float(sum(float(xp.min(m)) for m in muV_d))

    return dict(
        n=n, L=L, h=h, h3=h3, beta=beta, alpha=alpha, order=order, x=x,
        disc='fd', potential=potential, backend=backend, solver=solver,
        precision=precision, Vmin=Vmin, Vmax=Vmax, mu_min=mu_min,
        xp=xp, jit=jit, to_dev=to_dev, sync=sync, dtype=dtype, counter=counter,
        lap=lap, solve_G=solve_G, apply_Ginv=apply_Ginv, solve_dys=solve_dys,
        solve_V1=solve_V1, solve_V1_shifted=solve_V1_shifted,
        au_tau_alpha_p=au_tau_alpha_p, au_tau_alpha_q=au_tau_alpha_q,
        au_tau_moments=au_tau_moments, au_step_tau=au_step_tau,
        solve_dys2=solve_dys2, dys2_L=dys2_L,
        dys2_fused_gamma=dys2_fused_gamma,
        apply_V=apply_V, apply_V1=apply_V1, apply_A=apply_A, apply_Au=apply_Au,
        apply_Btf=apply_Btf, lam_rq=lam_rq, V3=V3,
        grad_E=grad_E, energy=energy, energy_and_grad=energy_and_grad,
        residue_from_grad=residue_from_grad, ip_M=ip_M, norm_M=norm_M,
        normalize=normalize, ip_alpha=ip_alpha, proj_t=_proj_t,
        transport=_proj_t,   # vector transport by projection at the new
                             # point
        dys_step=dys_step, tf_mass=tf_mass, tf_profile=tf_profile,
        dys_L=dys_L, dys_fused_gamma=dys_fused_gamma,
        dys_grad_h=dys_grad_h, dys_energy_at=dys_energy_at,
        dys_lyapunov=dys_lyapunov, dys_lyapunov_at=dys_lyapunov_at,
        h1_step_fused=h1_step_fused, diag_fused=diag_fused,
        pc_fused=pc_fused, dys_fused=dys_fused,
        gc_coeffs_fused=gc_coeffs_fused, gc_step_fused=gc_step_fused,
        gc_q=gc_q,
        gc_energy_at=gc_energy_at, pcg_dir_fused=pcg_dir_fused,
        rcg_dir_fused=rcg_dir_fused, ls_coeffs_fused=ls_coeffs_fused,
        ray_energy_at=ray_energy_at,
        au_cg_body=au_cg_body, au_update=au_update,
        rgrad_fused=rgrad_fused,
        line_coeffs=line_coeffs, line_energy=line_energy,
        circle_coeffs=circle_coeffs, circle_energy=circle_energy,
        u_const=lambda: normalize(to_dev(xp.ones((n, n, n), dtype=dtype))),
        make_uerr_host=lambda u_star_np, chunk=32: _make_uerr_host(
            u_star_np, chunk, n, h3, xp, dtype),
    )


# --------------------------------------------------------------------------
def _build_sem(n, L, beta, alpha, potential, backend, solver, precision,
               w0, delta, r0, gz, q_order, xp, jit, to_dev, sync, dtype):
    """Q^q spectral-element discretization; see build() for the contract.

    Two departures from the finite difference path, both of which touch every
    formula:

      * the mass matrix is the diagonal of GLL quadrature weights rather than
        h^3 I, so <a,b>_M = h^3 sum m_i m_j m_k a b and the normalization,
        energy and every projection carry that weight;

      * cal_matrix_V1_fast diagonalizes H = M^{-1}S + diag(V) as
        H = T diag(mu) T^{-1} with T only M^{-1/2}-orthogonal, so the forward
        transform is invT and the backward is T.  Applying the wrong one of the
        pair is silent on the finite difference side, where the DST matrix is
        symmetric, and wrong here.
    """
    import numpy as np
    import SEM

    Ncell = (n + 1) // q_order
    if q_order * Ncell - 1 != n:
        raise ValueError(f"n={n} is not q*Ncell-1 for q={q_order}")

    V_zero = lambda t: 0.0 * t
    v_ax_fn, st_a_fn, st_b_fn = _sem_potential(potential, w0, delta, r0, gz)

    # Laplacian alone, and (-Lap + V1), each per axis, always in float64
    h_sem, x_np, T_lap, invT_lap, eig_lap = SEM.cal_matrix_V1_fast(
        q_order, Ncell, 'dirichlet', L, V_zero)
    h = float(h_sem)
    x = np.asarray(x_np, dtype=np.float64)
    h3 = h ** 3

    TV, invTV, eigV = [], [], []
    for f in v_ax_fn:
        _, _, T_k, invT_k, mu_k = SEM.cal_matrix_V1_fast(
            q_order, Ncell, 'dirichlet', L, f)
        TV.append(np.asarray(T_k, dtype=np.float64))
        invTV.append(np.asarray(invT_k, dtype=np.float64))
        eigV.append(np.asarray(mu_k, dtype=np.float64))

    # diagonal mass: assembled GLL weights, Dirichlet-trimmed
    _, _, w_gll = SEM.LegendreD(q_order)
    w_gll = np.asarray(w_gll, dtype=np.float64)
    cell_idx = (np.arange(Ncell)[:, None] * q_order
                + np.arange(q_order + 1)[None, :])
    m_full = np.zeros(q_order * Ncell + 1)
    np.add.at(m_full, cell_idx.ravel(), np.tile(w_gll, Ncell))
    m_np = m_full[1:-1]

    d_ = dict(dtype=dtype)
    Tf_lap = to_dev(xp.asarray(np.asarray(invT_lap, dtype=np.float64).T, **d_))
    Tb_lap = to_dev(xp.asarray(np.asarray(T_lap, dtype=np.float64).T, **d_))
    eigL = to_dev(xp.asarray(np.asarray(eig_lap, dtype=np.float64), **d_))
    Tf_V = [to_dev(xp.asarray(a.T, **d_)) for a in invTV]
    Tb_V = [to_dev(xp.asarray(a.T, **d_)) for a in TV]
    muV = [to_dev(xp.asarray(a, **d_)) for a in eigV]
    m1 = to_dev(xp.asarray(m_np, **d_))
    v_ax = [to_dev(xp.asarray(f(x), **d_)) for f in v_ax_fn]
    st_a = None if st_a_fn is None else to_dev(xp.asarray(st_a_fn(x), **d_))
    st_b = None if st_b_fn is None else to_dev(xp.asarray(st_b_fn(x), **d_))

    counter = Counter()

    def lam3(shift):
        return (eigL[:, None, None] + eigL[None, :, None]
                + eigL[None, None, :] + shift)

    def muV3(shift):
        return (muV[0][:, None, None] + muV[1][None, :, None]
                + muV[2][None, None, :] + shift)

    if backend == 'jax':
        def apply_axis(u, M, axis):
            if axis == 0:
                return xp.einsum('ijk,il->ljk', u, M)
            if axis == 1:
                return xp.einsum('ijk,jl->ilk', u, M)
            return xp.einsum('ijk,kl->ijl', u, M)
    else:
        def apply_axis(u, M, axis):
            if axis == 0:
                return (M.T @ u.reshape(n, -1)).reshape(n, n, n)
            if axis == 1:
                return (M.T @ u.transpose(1, 0, 2).reshape(n, -1)
                        ).reshape(n, n, n).transpose(1, 0, 2)
            return (M.T @ u.transpose(2, 0, 1).reshape(n, -1)
                    ).reshape(n, n, n).transpose(1, 2, 0)

    def tsf(u, mats):
        w = apply_axis(u, mats[0], 0)
        w = apply_axis(w, mats[1], 1)
        return apply_axis(w, mats[2], 2)

    fwd_L = lambda u: tsf(u, [Tf_lap] * 3)
    bwd_L = lambda u: tsf(u, [Tb_lap] * 3)

    @jit
    def lap(u):
        return bwd_L(fwd_L(u) * lam3(0.0))

    @jit
    def solve_G(w):
        return bwd_L(fwd_L(w) / lam3(alpha))

    @jit
    def apply_Ginv(w):
        return bwd_L(fwd_L(w) * lam3(alpha))

    @jit
    def solve_dys(w, gamma):
        return bwd_L(fwd_L(w) / (gamma * lam3(0.0) + 1.0))

    @jit
    def solve_V1(w):
        return tsf(tsf(w, Tf_V) / muV3(0.0), Tb_V)

    @jit
    def solve_V1_shifted(w, sigma):
        return tsf(tsf(w, Tf_V) / muV3(-sigma), Tb_V)

    @jit
    def apply_V1(u):
        return (v_ax[0][:, None, None] + v_ax[1][None, :, None]
                + v_ax[2][None, None, :]) * u

    def V3v():
        V = (v_ax[0][:, None, None] + v_ax[1][None, :, None]
             + v_ax[2][None, None, :])
        if st_a is not None:
            V = V + st_a[:, None, None] * st_b[None, :, None]
        return V

    if st_a is None:
        apply_V = apply_V1
    else:
        @jit
        def apply_V(u):
            return apply_V1(u) + (st_a[:, None, None] * st_b[None, :, None]) * u

    @jit
    def V3():
        return V3v() * xp.ones((1, 1, n), dtype=dtype)

    # ---- mass-weighted inner product: h^3 sum m_i m_j m_k a b --------------
    @jit
    def ip_M(a, b):
        t = xp.einsum('ijk,k->ij', a * b, m1)
        t = xp.einsum('ij,j->i', t, m1)
        return h3 * xp.einsum('i,i->', t, m1)

    @jit
    def norm_M(a):
        return xp.sqrt(ip_M(a, a))

    @jit
    def apply_Btf(u_cur, w, ell, fpar):
        """B_TF w, the Thomas-Fermi metric operator of eq. (11) of the paper:

            B_TF = -Lap_h + 2 beta diag(u^2) + diag((V - ell)_+) + f ell^(1/3),

        where ell is the paper's lambda(u) and f is the paper's ell.

        The potential is clipped rather than shifted, which is what keeps the
        operator positive definite everywhere while still subtracting the
        chemical potential inside the Thomas-Fermi region, where V < ell. V is
        a broadcast of the separable axes plus the rank-one stirrer, so it is
        computed tile by tile here and never stored as an n^3 array.
        """
        return (lap(w) + xp.maximum(V3() - ell, 0.0) * w
                + 2.0 * beta * (u_cur * u_cur) * w
                + fpar * ell ** (1.0 / 3.0) * w)

    @jit
    def lam_rq(u):
        """lambda(u) = <A_beta(u) u, u>_h = 2 E(u) + (beta/2) <u^2, u^2>_h.

        The scalar ell of the Thomas-Fermi metric. It is not an eigenvalue to
        be computed: it only has to approximate lambda_1(A_beta) to the layer
        scale, and this Rayleigh quotient of the current iterate does, at the
        cost of the energy evaluation plus one quartic reduction.
        """
        return 2.0 * energy(u) + 0.5 * beta * ip_M(u * u, u * u)

    @jit
    def normalize(a):
        return a / xp.sqrt(ip_M(a, a))

    @jit
    def apply_A(u):
        return lap(u) + apply_V(u)

    @jit
    def apply_Au(u_cur, w):
        return lap(w) + apply_V(w) + beta * (u_cur * u_cur) * w

    @jit
    def grad_E(u):
        return apply_A(u) + beta * u ** 3

    @jit
    def energy(u):
        return 0.5 * ip_M(u, apply_A(u)) + 0.25 * beta * ip_M(u * u, u * u)

    @jit
    def energy_and_grad(u):
        g = grad_E(u)
        return 0.5 * ip_M(u, g) - 0.25 * beta * ip_M(u * u, u * u), g

    @jit
    def residue_from_grad(u, g):
        v = xp.abs(g) / norm_M(g)
        return norm_M(u - v) / norm_M(u)

    @jit
    def ip_alpha(a, b):
        return ip_M(a, apply_Ginv(b))

    @jit
    def _proj_t(w, u, Gu):
        return w - (ip_M(w, u) / ip_M(Gu, u)) * Gu

    @jit
    def diag_fused(u):
        g = apply_A(u) + beta * u ** 3
        u4 = ip_M(u * u, u * u)
        lam = ip_M(u, g)
        E = 0.5 * lam - 0.25 * beta * u4
        v = xp.abs(g) / norm_M(g)
        return E, lam, norm_M(u - v) / norm_M(u)

    @jit
    def h1_step_fused(u, dt):
        Proj = solve_G(u)
        Grad_raw = apply_A(u) + beta * u ** 3
        Grad = solve_G(Grad_raw)
        u_new = u - dt * Grad + dt * (ip_M(u, Grad) / ip_M(u, Proj)) * Proj
        u_new = normalize(u_new)
        lam = ip_M(u, Grad_raw)
        E = 0.5 * lam - 0.25 * beta * ip_M(u * u, u * u)
        v = xp.abs(Grad_raw) / norm_M(Grad_raw)
        return u_new, E, lam, norm_M(u - v) / norm_M(u)

    @jit
    def rgrad_fused(u):
        g = apply_A(u) + beta * u ** 3
        Gu = solve_G(u)
        Ge = solve_G(g)
        gc = Ge - (ip_M(Ge, u) / ip_M(Gu, u)) * Gu
        lam = ip_M(u, g)
        E = 0.5 * lam - 0.25 * beta * ip_M(u * u, u * u)
        v = xp.abs(g) / norm_M(g)
        return gc, Gu, E, lam, norm_M(u - v) / norm_M(u)

    @jit
    def pc_fused(u):
        g = apply_A(u) + beta * u ** 3
        lam = ip_M(u, g)
        r = g - lam * u
        Pr = solve_G(r)
        E = 0.5 * lam - 0.25 * beta * ip_M(u * u, u * u)
        v = xp.abs(g) / norm_M(g)
        return r, Pr, E, lam, norm_M(u - v) / norm_M(u)

    @jit
    def dys_L(u):
        return xp.max(V3v() + 3.0 * beta * u * u)

    # ---- DYS II, spectral element path (see the finite difference one) -----
    @jit
    def solve_dys2(w, gamma):
        return tsf(tsf(w, Tf_V) / (1.0 + gamma * muV3(0.0)), Tb_V)

    @jit
    def dys2_L(u):
        if st_a is None:
            return xp.max(3.0 * beta * u * u)
        return xp.max(st_a[:, None, None] * st_b[None, :, None]
                      + 3.0 * beta * u * u)

    @jit
    def dys2_fused_gamma(z, gamma):
        u = normalize(z)
        if st_a is None:
            expl = beta * u * u * u
        else:
            expl = (st_a[:, None, None] * st_b[None, :, None]
                    + beta * u * u) * u
        w = 2.0 * u - z - gamma * expl
        z_new = z + solve_dys2(w, gamma) - u
        u_new = normalize(z_new)
        E, lam, res = diag_fused(u_new)
        return z_new, u_new, E, lam, res

    @jit
    def dys_grad_h(u):
        """grad h(u) = (V + beta u^2) u."""
        return (V3v() + beta * u * u) * u

    @jit
    def dys_energy_at(z, gamma):
        """Energy after one Davis-Yin step at step size gamma."""
        u = normalize(z)
        V = V3v()
        w = 2.0 * u - z - gamma * ((V + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        un = normalize(z_new)
        return 0.5 * ip_M(un, apply_A(un)) + 0.25 * beta * ip_M(un * un, un * un)

    @jit
    def dys_lyapunov(z, gamma):
        """Lyapunov function of Theorem 3.4 of Li, Zheng and Zhang; see the finite difference
        path for why the scheme is safeguarded on this and not the energy."""
        nz = norm_M(z)
        u = z / nz
        g = apply_A(u) + beta * u ** 3
        lam = ip_M(u, g)
        E = 0.5 * lam - 0.25 * beta * ip_M(u * u, u * u)
        R = nz - 1.0 + gamma * lam
        return E + R * R / gamma

    @jit
    def dys_lyapunov_at(z, gamma):
        """Lyapunov value after one step at gamma."""
        u = normalize(z)
        V = V3v()
        w = 2.0 * u - z - gamma * ((V + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        nz = norm_M(z_new)
        un = z_new / nz
        g = apply_A(un) + beta * un ** 3
        lam = ip_M(un, g)
        E = 0.5 * lam - 0.25 * beta * ip_M(un * un, un * un)
        R = nz - 1.0 + gamma * lam
        return E + R * R / gamma

    @jit
    def dys_fused(z):
        u = normalize(z)
        V = V3v()
        gamma = 1.0 / xp.max(V + 3.0 * beta * u * u)
        w = 2.0 * u - z - gamma * ((V + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        u_new = normalize(z_new)
        E, lam, res = diag_fused(u_new)
        return z_new, u_new, E, lam, res, gamma

    @jit
    def dys_fused_gamma(z, gamma):
        u = normalize(z)
        V = V3v()
        w = 2.0 * u - z - gamma * ((V + beta * u * u) * u)
        z_new = z + solve_dys(w, gamma) - u
        u_new = normalize(z_new)
        E, lam, res = diag_fused(u_new)
        return z_new, u_new, E, lam, res

    @jit
    def au_cg_body(u_cur, w, r, p, rz):
        Ap = apply_Au(u_cur, p)
        a_k = rz / xp.sum(p * Ap)
        w = w + a_k * p
        r = r - a_k * Ap
        z = solve_V1(r)
        rz_new = xp.sum(r * z)
        return w, r, z + (rz_new / rz) * p, rz_new, xp.sqrt(xp.sum(r * r))

    # ---- adaptive step size for the a_u flow, Remark 4.3 of Henning-Peterseim
    @jit
    def au_tau_alpha_p(p, q):
        Ap = apply_A(p)
        return xp.array([ip_M(p, Ap), 2.0 * ip_M(q, Ap)])

    @jit
    def au_tau_alpha_q(q):
        return ip_M(q, apply_A(q))

    @jit
    def au_tau_moments(p, q):
        return xp.array([
            ip_M(p, p), 2.0 * ip_M(p, q), ip_M(q, q),
            0.5 * beta * ip_M(p * p, p * p),
            2.0 * beta * ip_M(p * p * p, q),
            3.0 * beta * ip_M(p * p, q * q),
            2.0 * beta * ip_M(p, q * q * q),
            0.5 * beta * ip_M(q * q, q * q)])

    @jit
    def au_step_tau(u, q, tau):
        u_new = normalize((1.0 - tau) * u + tau * q)
        E, lam, res = diag_fused(u_new)
        return u_new, E, lam, res

    @jit
    def au_update(u, w):
        gamma = ip_M(u, u) / ip_M(u, w)
        u_new = normalize(gamma * w)
        E, lam, res = diag_fused(u_new)
        return u_new, E, lam, res, gamma

    # ---- line-search coefficients, mass-weighted --------------------------
    @jit
    def _lsf_a(u, d):
        Au = apply_A(u)
        return xp.array([ip_M(u, Au), ip_M(d, Au)])

    @jit
    def _lsf_b(u, d):
        Ad = apply_A(d)
        return xp.array([
            ip_M(d, Ad), ip_M(u, u), ip_M(u, d), ip_M(d, d),
            ip_M(u * u, u * u), ip_M(u * u, u * d), ip_M(u * d, u * d),
            ip_M(u * d, d * d), ip_M(d * d, d * d)])

    def ls_coeffs_fused(u, d):
        import numpy as _np
        return _np.concatenate([_np.asarray(_lsf_a(u, d), dtype=_np.float64),
                                _np.asarray(_lsf_b(u, d), dtype=_np.float64)])

    def line_energy(cf, t):
        q0, q1, q2, n0, n1, n2, p0, p1, p2, p3, p4 = cf
        Q = q0 + 2 * q1 * t + q2 * t * t
        N = n0 + 2 * n1 * t + n2 * t * t
        P = (p0 + 4 * p1 * t + 6 * p2 * t ** 2
             + 4 * p3 * t ** 3 + p4 * t ** 4)
        return 0.5 * Q / N + 0.25 * beta * P / (N * N)

    @jit
    def gc_q(u, d):
        p = d - ip_M(d, u) * u
        return normalize(p)

    @jit
    def _gcq_a(u, q):
        Au = apply_A(u)
        return xp.array([ip_M(u, Au), ip_M(q, Au)])

    # Splitting this into one kernel per term was tried, to get preconditioned
    # CG running at n=999: it does not help.  The 22.3 GiB block that fails
    # there is the workspace of apply_A inside _gcq_a, not the products held
    # here, and splitting only costs five extra passes over u and q on every
    # great-circle iteration.
    @jit
    def _gcq_b(u, q):
        Aq = apply_A(q)
        return xp.array([
            ip_M(q, Aq), ip_M(u * u, u * u), ip_M(u * u, u * q),
            ip_M(u * q, u * q), ip_M(u * q, q * q), ip_M(q * q, q * q)])

    def circle_coeffs(u, q):
        import numpy as _np
        return _np.concatenate([_np.asarray(_gcq_a(u, q), dtype=_np.float64),
                                _np.asarray(_gcq_b(u, q), dtype=_np.float64)])

    def gc_coeffs_fused(u, d):
        q = gc_q(u, d)
        cf = circle_coeffs(u, q)
        del q
        return cf

    def circle_energy(cf, th):
        import numpy as _np
        a_uu, a_uq, a_qq, p0, p1, p2, p3, p4 = cf
        c_, s_ = _np.cos(th), _np.sin(th)
        quad = c_ * c_ * a_uu + 2 * c_ * s_ * a_uq + s_ * s_ * a_qq
        quar = (c_ ** 4 * p0 + 4 * c_ ** 3 * s_ * p1
                + 6 * c_ ** 2 * s_ ** 2 * p2 + 4 * c_ * s_ ** 3 * p3
                + s_ ** 4 * p4)
        return 0.5 * quad + 0.25 * beta * quar

    @jit
    def gc_step_fused(u, d, theta):
        q = gc_q(u, d)
        return xp.cos(theta) * u + xp.sin(theta) * q

    @jit
    def gc_energy_at(u, d, theta):
        q = gc_q(u, d)
        v = xp.cos(theta) * u + xp.sin(theta) * q
        return 0.5 * ip_M(v, apply_A(v)) + 0.25 * beta * ip_M(v * v, v * v)

    @jit
    def ray_energy_at(u, d, t):
        v = normalize(u + t * d)
        return 0.5 * ip_M(v, apply_A(v)) + 0.25 * beta * ip_M(v * v, v * v)

    @jit
    def pcg_dir_fused(Pr_new, p, beta_cg):
        return -Pr_new + beta_cg * p

    @jit
    def rcg_dir_fused(gc_new, direction, u_new, Gu_new, beta_cg):
        c = ip_M(direction, u_new) / ip_M(Gu_new, u_new)
        dt_ = direction - c * Gu_new
        d_new = -gc_new + beta_cg * dt_
        return d_new, ip_M(gc_new, apply_Ginv(d_new))

    @jit
    def tf_mass(lam_tf):
        u = xp.maximum(lam_tf - V3v(), 0.0) / beta
        return ip_M(u, xp.ones_like(u))

    @jit
    def tf_profile(lam_tf):
        return normalize(xp.sqrt(xp.maximum(lam_tf - V3v(), 0.0) / beta))

    Vmin = float(xp.min(V3v()))
    Vmax = float(xp.max(V3v()))
    mu_min = float(sum(float(xp.min(m)) for m in muV))

    return dict(
        n=n, L=L, h=h, h3=h3, beta=beta, alpha=alpha, order=q_order, x=x,
        disc='sem', q_order=q_order, n_cell=Ncell,
        potential=potential, backend=backend, solver=solver,
        precision=precision, Vmin=Vmin, Vmax=Vmax, mu_min=mu_min,
        xp=xp, jit=jit, to_dev=to_dev, sync=sync, dtype=dtype, counter=counter,
        lap=lap, solve_G=solve_G, apply_Ginv=apply_Ginv, solve_dys=solve_dys,
        solve_V1=solve_V1, solve_V1_shifted=solve_V1_shifted,
        au_tau_alpha_p=au_tau_alpha_p, au_tau_alpha_q=au_tau_alpha_q,
        au_tau_moments=au_tau_moments, au_step_tau=au_step_tau,
        solve_dys2=solve_dys2, dys2_L=dys2_L,
        dys2_fused_gamma=dys2_fused_gamma,
        apply_V=apply_V, apply_V1=apply_V1, apply_A=apply_A, apply_Au=apply_Au,
        apply_Btf=apply_Btf, lam_rq=lam_rq, V3=V3,
        grad_E=grad_E, energy=energy, energy_and_grad=energy_and_grad,
        residue_from_grad=residue_from_grad, ip_M=ip_M, norm_M=norm_M,
        normalize=normalize, ip_alpha=ip_alpha, proj_t=_proj_t,
        transport=_proj_t,
        dys_step=lambda z: (dys_fused(z)[0], dys_fused(z)[5]),
        dys_fused=dys_fused, dys_L=dys_L, dys_fused_gamma=dys_fused_gamma,
        dys_lyapunov=dys_lyapunov, dys_lyapunov_at=dys_lyapunov_at,
        dys_grad_h=dys_grad_h, dys_energy_at=dys_energy_at,
        h1_step_fused=h1_step_fused, diag_fused=diag_fused,
        rgrad_fused=rgrad_fused, pc_fused=pc_fused,
        au_cg_body=au_cg_body, au_update=au_update,
        gc_coeffs_fused=gc_coeffs_fused, gc_step_fused=gc_step_fused,
        gc_q=gc_q, gc_energy_at=gc_energy_at, pcg_dir_fused=pcg_dir_fused,
        rcg_dir_fused=rcg_dir_fused, ls_coeffs_fused=ls_coeffs_fused,
        ray_energy_at=ray_energy_at,
        line_coeffs=ls_coeffs_fused, line_energy=line_energy,
        circle_coeffs=circle_coeffs, circle_energy=circle_energy,
        tf_mass=tf_mass, tf_profile=tf_profile,
        make_uerr_host=lambda u_star_np, chunk=32: _make_uerr_host_sem(
            u_star_np, chunk, n, h3, m_np),
        u_const=lambda: normalize(to_dev(xp.ones((n, n, n), dtype=dtype))),
    )


def _sem_potential(potential, w0, delta, r0, gz):
    """Per-axis V1 as callables, plus the rank-one factors of V2."""
    import numpy as np
    if potential == 'lattice':
        f = lambda t: t ** 2 + 100.0 * np.sin(np.pi * t / 4.0) ** 2
        return [f, f, f], None, None
    if potential == 'stirrer':
        return ([lambda t: t ** 2, lambda t: t ** 2,
                 lambda t: (gz ** 2) * t ** 2],
                lambda t: 2.0 * w0 * np.exp(-delta * (t - r0) ** 2),
                lambda t: np.exp(-delta * t ** 2))
    raise ValueError(f"unknown potential {potential!r}")


def _make_uerr_host_sem(u_star_np, chunk, n, h3, m_np):
    """||u - u*||_M with u* on the host, mass-weighted for the SEM grid."""
    import numpy as np
    us = np.asarray(u_star_np, dtype=np.float32)
    m = np.asarray(m_np, dtype=np.float32)

    def uerr(u):
        uh = np.asarray(u.astype('float32'))
        tot = np.float64(0.0)
        for s0 in range(0, n, chunk):
            d = (uh[s0:s0 + chunk] - us[s0:s0 + chunk]).astype(np.float64) ** 2
            r = np.einsum('ijk,k->ij', d, m.astype(np.float64))
            r = np.einsum('ij,j->i', r, m.astype(np.float64))
            tot += np.float64(np.dot(r, m[s0:s0 + chunk].astype(np.float64)))
        return float(np.sqrt(h3 * tot))
    return uerr
