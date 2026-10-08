"""Table A.2 (tab:tf-spectrum) of the paper: exact Riemannian Hessian condition numbers under a_u, modified a_u, Thomas-Fermi
(f=1/2, l=lambda_1; f and l are the paper's ell and lambda(u)), Thomas-Fermi with beta u^2 replaced by the Thomas-Fermi profile
(TF_frozen) and Painleve metrics, 1D, grid setup of Table A.1 (N = 3000 for beta <= 100, 4000 otherwise,
L = 1.25 sqrt(lambda_TF) + 4). Table A.2 prints the a_u, mod_a_u and TF_f0.5 columns.
Harmonic trap V = x^2 and the trap with the stirrer bump 2 w0 exp(-delta (x - r0)^2), w0 = 30, delta = 1, r0 = 1.
The constant printed as 'C_TF = sup|nu0^2 - y_+|' is the paper's C_0."""
import sys, os, json, time, numpy as np, scipy.linalg as sla
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # run from anywhere; rq_1d.py and painleve.py sit beside this file
import rq_1d as R
from painleve import hastings_mcleod_profile
W0 = hastings_mcleod_profile()
OUT_JSON = os.path.join(HERE, 'results', 'spectrum1d', 'tf_table_1d.json')
yy = np.linspace(-20, 60, 200001)
nu0sq = (W0(yy) + yy)/3.0
consts = {'C_TF = sup|nu0^2 - y_+|': float(np.max(np.abs(nu0sq - np.maximum(yy, 0)))),
          'argmax y': float(yy[np.argmax(np.abs(nu0sq - np.maximum(yy, 0)))]),
          'm0 = inf W0': float(np.min(W0(yy))), 'argmin y': float(yy[np.argmin(W0(yy))]), 'nu0(0)^2': float(nu0sq[np.argmin(np.abs(yy))])}
print(json.dumps(consts), flush=True)

def ext(NT, DT):
    n = NT.shape[0]
    return (float(sla.eigh(NT, DT, subset_by_index=[0, 0], eigvals_only=True)[0]),
            float(sla.eigh(NT, DT, subset_by_index=[n-1, n-1], eigvals_only=True)[0]))

rows = []
for trap in ('harmonic', 'bump'):
    for beta in ([100.0, 1000.0, 10000.0] if trap == 'harmonic' else [100.0, 1600.0, 10000.0]):
        t0 = time.time()
        lTF = (3*beta/4)**(2/3); N = 3000 if beta <= 100 else 4000; L = 1.25*np.sqrt(lTF) + 4
        h, x, V, lap_d, lap_o = R.setup(beta, L, N)
        if trap == 'bump':
            V = x**2 + 60.0*np.exp(-(x - 1.0)**2)
        u, lam = R.ground_state(beta, h, x, V, lap_d, lap_o, verbose=False)
        bu2 = beta*u**2; s = lam**(1/3)
        Q = R.tangent_basis(u)
        NT = Q.T @ R.dense_tridiag(lap_d + V + 3*bu2 - lam, lap_o) @ Q; NT = 0.5*(NT+NT.T)
        sigT = float(sla.eigh(NT, subset_by_index=[0, 0], eigvals_only=True)[0])
        eTF = float(np.max(np.abs(bu2 - np.maximum(lam - V, 0.0))))
        row = {'trap': trap, 'beta': beta, 'N': N, 'L': L, 'lambda': float(lam), 'sigma_T': sigT, 'sigma_T/lam^(1/3)': sigT/s, 'e_TF/lam^(1/3)': eTF/s}
        for name, dg in (('a_u', lap_d + V + bu2), ('mod_a_u', lap_d + V + 3*bu2),
                         ('TF_f0.5', lap_d + 2*bu2 + np.maximum(V - lam, 0) + 0.5*s),
                         ('TF_frozen_f0.5', lap_d + 2*np.maximum(lam - V, 0) + np.maximum(V - lam, 0) + 0.5*s),
                         ('PI', lap_d + s*W0((lam - V)/s))):
            DT = Q.T @ R.dense_tridiag(dg, lap_o) @ Q; DT = 0.5*(DT+DT.T)
            lo, hi = ext(NT, DT)
            row[name] = {'min': lo, 'max': hi, 'kappa': hi/lo}
        row['seconds'] = time.time() - t0
        rows.append(row); print(json.dumps(row), flush=True)
json.dump({'consts': consts, 'rows': rows}, open(OUT_JSON, 'w'), indent=1)
print('done; JSON written to', OUT_JSON)
