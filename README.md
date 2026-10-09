# Sobolev gradient flows for the Gross–Pitaevskii ground state

Code and data for

> Shixin Zheng and Xiangxiong Zhang,
> **Riemannian Hessian analysis of Sobolev gradient flows for the Gross–Pitaevskii ground state problem**, 2026.
> [paper.pdf](paper.pdf)

The code runs Riemannian gradient descent and Riemannian conjugate gradients on
the discrete sphere $\mathcal M=\{u:\langle u,u\rangle_h=1\}$ under seven
Sobolev metrics for the three-dimensional Gross–Pitaevskii energy, on one GPU
with JAX, and computes the exact spectrum of the Riemannian Hessian at the
ground state in one, two and three dimensions (Appendix A of the paper).

## Requirements

- Python 3.8 or later, with NumPy, SciPy and Matplotlib. This is all that the
  figure scripts and the one- and two-dimensional spectra need.
- JAX for the three-dimensional runs. The three-dimensional runs of the paper
  used Python 3.12 and a CUDA 12 build of JAX on one NVIDIA GH200 (96 GB):
  ```bash
  pip install --upgrade "jax[cuda12]"
  ```
  `run_metrics.py` and `rq_3d.py` also take `--backend numpy`, which runs
  small grids on a CPU without JAX.

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python3 make_figures.py              # Figures 1-6 from the data in results/, about 10 s
python3 plot_rayleigh_bounds_3d.py   # Figure A.1
python3 run_metrics.py --n 29 --backend numpy --methods all --outdir runs
                                     # all fourteen methods on a 29^3 grid, about 1 min on a CPU
```

The figures are written to `figures/` (set `FIG_DIR` to change this). The
first two commands need no GPU and no JAX.

## The seven metrics

A metric is a choice of the operator $B(u)$ in $g_u(\xi,\eta)=\langle B(u)\xi,\eta\rangle_h$,
applied through its inverse $\mathbb G(u)=B(u)^{-1}$.

| key | $B(u)$ | in the paper | applied by |
|---|---|---|---|
| `h1` | $-\Delta_h+\alpha\mathbb I$ | modified $H^1$, $\alpha=20$ | exact tensor-product solve |
| `a0` | $-\Delta_h+\mathbb V$ | $a_0$ | inner PCG |
| `a0mod` | $-\Delta_h+\mathbb V_1$ | modified $a_0$ (Section 3.3) | exact tensor-product solve |
| `au` | $-\Delta_h+\mathbb V+\beta\,\mathrm{diag}(u^2)$ | $a_u$ | inner PCG |
| `a3u` | $-\Delta_h+\mathbb V+3\beta\,\mathrm{diag}(u^2)$ | modified $a_u$ (Section 3.4) | inner PCG |
| `a2u` | $-\Delta_h+\mathbb V+2\beta\,\mathrm{diag}(u^2)$ | the member $c=2$ of the family (24) (Remark 3.9) | inner PCG |
| `tf` | $B_{\rm TF}(u)$ of eq. (11) | Thomas–Fermi (Section 3.5) | inner PCG |

$\mathbb V_1$ is the separable part of the potential. Every inner PCG is
preconditioned by $(-\Delta_h+\mathbb V_1)^{-1}$, which is applied exactly by
per-axis eigendecomposition. The `a2u` runs are in the data, and the figures
do not draw them.

**Names in the code.** In `metrics.py`, `gpe3d.py` and the one-dimensional
scripts, the variable `ell` is the paper's $\lambda(u)$ and `f` (option
`--tf-f`) is the paper's $\ell$, so that the Thomas–Fermi operator reads
`-Lap_h + 2 beta diag(u^2) + diag((V - ell)_+) + f ell^(1/3)`. The default
`--tf-f 0.5` is the paper's $\ell=1/2$.

## Files

| file | contents |
|---|---|
| `metrics.py` | the seven metrics, Riemannian gradient descent `rgd` and conjugate gradients `rcg` |
| `run_metrics.py` | driver: builds the problem, runs the methods, writes one CSV per method and a `<tag>_meta.json` |
| `gpe3d.py` | grids, operators, energy and gradient, fast diagonalization, fused kernels; second-order finite differences |
| `SEM.py` | $Q^k$ spectral elements, used with `--disc sem` |
| `solvers.py` | the Thomas–Fermi initialization, the Davis–Yin reference solve, the recording helpers |
| `make_figures.py` | Figures 1–6 from `results/data` |
| `rq_1d.py`, `rq_2d.py` | exact Hessian spectra for the harmonic trap in one and two dimensions (Table A.1, $d=1,2$) |
| `rq_3d.py` | the same in three dimensions, matrix free with LOBPCG on one GPU (Table A.1, $d=3$; Figure A.1; the stirrer spectrum of Section 5.2) |
| `rq_3d_table.py` | prints the tables of the three-dimensional runs in `results/spectrum3d` |
| `plot_rayleigh_bounds_3d.py` | Figure A.1 |
| `plot_rayleigh_bounds_1d.py` | the $d=1$ rows of Table A.1 from the logs of `rq_1d.py` |
| `tf_table.py` | Table A.2 |
| `painleve.py`, `painleve_mu1.py` | the Hastings–McLeod profile, $\mu_1=2.4105$ of Corollary 3.11 and $C_0=\nu_0(0)^2\approx0.43$ of Section 3.5 |
| `a3u_test.py` | the iteration counts of the modified $a_u$ metric quoted in Appendix A |
| `henning_check.py` | reproduces Section 7.2 of Henning (2023), quoted in Appendix A |
| `rq_2d_check.py`, `rq_3d_check.py`, `rq_3d_accuracy.py` | checks of the two- and three-dimensional eigensolvers against dense solvers |
| `paper.pdf` | the paper |
| `results/` | the data behind every figure and table, described below |

`gpe3d.py`, `SEM.py` and `solvers.py` come from the code of Li, Zheng and
Zhang (2026) and also contain its Davis–Yin splitting scheme and the other
methods compared there.

## Reproducing the figures and tables

| paper | script | data |
|---|---|---|
| Figure 1 | `make_figures.py` | `results/data/stirrer_tf_n499_*` |
| Figure 2 | `make_figures.py` | `results/data/stirrer_tf_n999_q20_*` |
| Figures 3, 4 | `make_figures.py` | `results/data/stirrer_tf_n499_L16_beta{1600,6400,102400}_*` |
| Figures 5, 6 | `make_figures.py` | `stirrer_tf_n499_*`, `stirrer_tf_n499_L16_beta1600_*`, `stirrer_tf_n499_L32_beta1600_*` |
| $\alpha$ sweep, Section 5.1 | -- | `results/data/stirrer_tf_n499_alpha{0,1,5,100}_meta.json` |
| $99^3$ check, Section 5.2 | -- | `results/data/stirrer_tf_n99_meta.json` |
| Table A.1, $d=1$ | `plot_rayleigh_bounds_1d.py` | `results/spectrum1d/run_small.log`, `run_large.log` |
| Table A.1, $d=2$ | `rq_2d.py` | `results/spectrum1d/run_2d.log` |
| Table A.1, $d=3$; Section 5.2 | `rq_3d_table.py` | `results/spectrum3d/*.json` |
| Table A.2 | `tf_table.py` | `results/spectrum1d/tf_table_1d.json` |
| Figure A.1 | `plot_rayleigh_bounds_3d.py` | `results/spectrum3d/harm*.json` |

Table A.1 prints five of the nine values of $\beta$ of the one-dimensional
sweep. Its bound column is $(1+\lambda_1/\lambda_2)/(1-\lambda_1/\Lambda)$
of Corollary 3.5, with $\Lambda=\max\{\lambda_2,\lambda_1(A_{3\beta})\}$.
`plot_rayleigh_bounds_1d.py` prints it for $d=1$. For $d=2$ and $d=3$ it
follows from the two lower bounds that `run_2d.log` prints and that the JSON
files record as `lb_gap` and `lb_A3`, since $1-\lambda_1/\Lambda$ is the
larger of the two.

### Recomputing the three-dimensional runs (GPU)

The test problem is the stirrer potential

```
V(x) = x1^2 + x2^2 + 4 x3^2 + 60 exp(-((x1-1)^2 + x2^2))
```

on $(-L,L)^3$ with $\beta=1600$, started from the Thomas–Fermi profile. Runs
stop at $|E-E^*|<10^{-12}$, where $E^*$ is read from the reference file
`results/data/ref_*.json` of the same grid. A missing reference is computed by
Davis–Yin splitting on the same grid, or by `--ref-method rcg_a0mod` on the
large boxes, where the Davis–Yin step size decreases like $L^{-2}$. The runs
below write to `results/data` and replace the files there.

```bash
# Figure 1, and the L = 8 row of Figures 5 and 6
python3 run_metrics.py --n 499 --methods all --maxit 400

# Figure 2: Q^20 spectral elements, 999^3 unknowns, one process per method
for m in h1_gd h1_cg a0_gd a0_cg a0mod_gd a0mod_cg au_gd au_cg a3u_gd a3u_cg a2u_gd a2u_cg; do
  python3 run_metrics.py --n 999 --disc sem --q-order 20 --error-tol 5e-12 \
      --fp32-inner --cg-dir-fp32 --maxit 300 --methods $m
done
python3 run_metrics.py --n 999 --disc sem --q-order 20 --error-tol 5e-12 \
    --fp32-inner --cg-dir-fp32 --pcg-max 25 --refine 6 --maxit 40 --methods tf_gd,tf_cg

# Figures 3 and 4; beta = 1600 is also the L = 16 row of Figures 5 and 6
python3 run_metrics.py --n 499 --L 16 --beta 1600 --methods all --maxit 600 \
    --tag stirrer_tf_n499_L16_beta1600
for b in 6400 102400; do
  python3 run_metrics.py --n 499 --L 16 --beta $b --methods all --maxit 600 \
      --ref-method rcg_a0mod --tag stirrer_tf_n499_L16_beta$b
done

# the L = 32 row of Figures 5 and 6
python3 run_metrics.py --n 499 --L 32 --beta 1600 --methods all --maxit 600 \
    --ref-method rcg_a0mod --tag stirrer_tf_n499_L32_beta1600

# the alpha sweep of Section 5.1
for al in 0 1 5 100; do
  python3 run_metrics.py --n 499 --alpha $al --methods h1_gd,h1_cg --maxit 600 \
      --tag stirrer_tf_n499_alpha$al
done

# the 99^3 check of Section 5.2
python3 run_metrics.py --n 99 --maxit 400 \
    --methods h1_gd,h1_cg,a0_gd,a0_cg,a0mod_gd,a0mod_cg,au_gd,au_cg,a3u_gd,a3u_cg
```

Each `<tag>_meta.json` records the settings of every method of a run. Three
flags matter at $999^3$, where one fp64 vector is 8 GB:

- `--fp32-inner` holds the Krylov vectors of the inner PCG in single precision,
  while the residual is formed and the correction accumulated in double
  precision. The metrics with an inner solve fit on one GH200 only this way.
- The Thomas–Fermi metric needs `--pcg-max 25 --refine 6`, many short
  single-precision solves with six recomputations of the true residual. Its
  inner solve takes about 75 iterations, and over that many the residual of the
  single-precision recurrence drifts from the true one. With one long solve the
  outer iteration stalls near $2\times10^{-10}$ with a collapsing step size.
- `--cg-dir-fp32` stores the search direction of the conjugate gradient
  methods in single precision.

Leave out `--cache-dir`. The runs measure the energy error only, and without
the flag the ground state, 8 GB at $999^3$, is neither read nor written.

### The exact Hessian spectra (Appendix A)

```bash
# one and two dimensions, on a CPU
RQ_N=3000 python3 rq_1d.py 1 10 100                 > results/spectrum1d/run_small.log   # 5 min
RQ_N=4000 python3 rq_1d.py 1e3 3e3 1e4 3e4 1e5 3e5  > results/spectrum1d/run_large.log   # 15 min
RQ2D_N=200 python3 rq_2d.py 10 100 1000 10000       > results/spectrum1d/run_2d.log      # 10 min
python3 plot_rayleigh_bounds_1d.py                  # d = 1 rows of Table A.1, slopes
python3 tf_table.py                                 > results/spectrum1d/tf_table_1d.log # 20 min
python3 a3u_test.py                                 > results/spectrum1d/a3u_metric_1d.log
python3 henning_check.py                            > results/spectrum1d/henning_check.log
python3 painleve_mu1.py                             > results/spectrum1d/painleve_mu1.log

# three dimensions, one GPU per run; the harmonic-trap sweep of Table A.1 and Figure A.1
python3 -u rq_3d.py --n 199 --beta 1e4 --potential harmonic --auto-L --block 2 \
    --tol 1e-8 --maxit 400 --tol-max 1e-6 --maxit-max 300 --inner 40 --inner-tol 1e-4 \
    --metrics h1,a0,au,a2u,a3u,tf --out results/spectrum3d/harm1e4.json
python3 rq_3d_table.py                              # the tables, from results/spectrum3d
python3 plot_rayleigh_bounds_3d.py                  # Figure A.1

# checks of the eigensolvers, on a CPU
python3 rq_3d.py --n 49 --beta 1e-6 --potential harmonic --L 8 --backend numpy --metrics a0
                                                    # beta -> 0: min rho must equal 1 - lam1/lam2
python3 rq_3d_check.py                              # rq_3d.py against a dense solver at n = 15
python3 rq_3d_accuracy.py                           # what limits the accuracy of rq_3d.py
python3 rq_2d_check.py                              # the rank-one penalty of rq_2d.py
```

The harmonic-trap sweep repeats the `rq_3d.py` command above for each
$\beta$, at $199^3$, with the mesh checks at $399^3$ ($\beta=10^5$) and
$299^3$ ($\beta=3\times10^5$). The stirrer runs `stir99`, `stir199` and
`stir499` use `--potential stirrer --L 8 --beta 1600` with all seven metrics.

## How the code works

Only the application of $\mathbb G(u)$ depends on the metric:

```
ambient gradient      grad_X E(u) = G(u) (-Lap_h u + V u + beta u^3)
tangent projection    P_u(w)      = w - (<w,u>_h / <G u, u>_h) G u
Riemannian gradient   grad E(u)   = P_u(grad_X E(u))
retraction            R_u(xi)     = (u + xi) / ||u + xi||_h
```

so `metrics.py` holds one gradient descent and one conjugate gradient method
for all seven metrics.

- **Solves per iteration.** Every metric needs $\mathbb G\nabla E_h(u)$ and
  $\mathbb Gu$, two solves, except the $a_u$ metric, for which
  $\mathbb G\nabla E_h(u)=A_u^{-1}A_uu=u$.
- **Line search.** The energy along the retracted ray $t\mapsto E(R_u(td))$
  is a quartic over a quadratic in $t$ whose coefficients are inner products
  of $u$ and $d$. Each method uses the exact minimizer over $t\in(0,5]$, at
  the cost of two applications of $-\Delta_h+\mathbb V$, so the differences
  between the curves are differences between the metrics.
- **Conjugate gradients.** Fletcher–Reeves in the metric of the current point,
  the previous direction transported by the tangent projection at the new
  point, and a restart to steepest descent whenever the result is not a
  descent direction.
- **Cost.** The `linsys` column counts applications of the separable solver:
  one per exact metric application, and one per inner PCG iteration plus two
  per inner solve.
- **Precision.** Single precision is safe for the Krylov vectors of the inner
  solves and never for the Riemannian gradient, which is a difference of two
  nearly equal vectors near convergence. `--lean`, which keeps the whole
  metric in single precision, is kept to show this: the $a_0$ flow stalls at
  $|E-E^*|=6\times10^{-9}$ on the $499^3$ grid.

## Data

`results/data` holds one CSV per run and method, named `<tag>_<method>.csv`,
with the columns

```
iter, time, error, lam_error, u_error, residue, linsys, energy, eigenvalue, step
```

where `error` is $|E-E^*|$, `time` is GPU wall clock in seconds and `u_error`
is not recorded (NaN). `<tag>_meta.json` records the settings and the summary
of every method of a run, and `ref_*.json` the reference values $E^*$ and
$\lambda^*$ of each grid.

`results/spectrum1d` holds the logs of the one- and two-dimensional scripts,
and `results/spectrum3d` one JSON and one log per run of `rq_3d.py`. The JSON
files written before 2026-09-09 report `res` as the largest residual over the
LOBPCG block rather than the residual of the returned eigenpair, so their
$a_0$ rows look unconverged. `a0conv` and `dense25.log` show that the
eigenvalues are converged. `harm1e5_n399.json` and `stir499.json` were run
with `--bT 0`, so their bound fields are null.

## Related papers

- J. Li, S. Zheng and X. Zhang, Global convergence of an efficient splitting
  method for the defocusing Gross–Pitaevskii ground state problem,
  arXiv:2609.13509, 2026.
- X. Liu and X. Zhang, A simple GPU-accelerated solver for the Schrödinger
  operator with applications to ground states and Hamiltonian simulation,
  arXiv:2605.20491, 2026.
- X. Zhang and H. Zhou, Computing the Gross–Pitaevskii ground state via
  Wasserstein gradient flow in diffeomorphism space.
- P. Henning and D. Peterseim, Sobolev gradient flow for the Gross–Pitaevskii
  eigenvalue problem: global convergence and computational efficiency, SIAM
  J. Numer. Anal. 58 (2020) 1744–1772.
- P. Henning, The dependency of spectral gaps on the convergence of the inverse
  iteration for a nonlinear eigenvector problem, Math. Models Methods Appl.
  Sci. 33 (2023) 1517–1544.
- C. Gallo and D. Pelinovsky, On the Thomas–Fermi ground state in a harmonic
  potential, Asymptot. Anal. 73 (2011) 53–96.

## License

MIT, see `LICENSE`.
