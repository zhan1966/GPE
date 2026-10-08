"""What limits the accuracy of the three-dimensional answer.

rq_3d.py agrees with a dense solver to about 1e-11, not to machine zero, and
this pins down why. The same Rayleigh quotient is evaluated on the same vectors
in every combination of (dense matrices, matrix-free operators) x (dense
eigenvector, LOBPCG eigenvector).

The result: the matrix-free operators reproduce the assembled dense ones to
9e-16 on the same vector, so the elimination and the application are exact to
machine precision, and the whole of the 5e-12 gap is carried by the eigenvector
-- which differs from the dense one by about 1e-5 radians. The Rayleigh
quotient is stationary at an eigenvector, so the eigenvalue error is the square
of that angle, damped by the local gap. It is the vector that stalls, not the
value, which is why tightening the stopping rule buys nothing below 1e-12.
"""
import os, sys, numpy as np, scipy.linalg as sla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpe3d, rq_3d
n, L, beta = 15, 8.0, 1600.0
P = gpe3d.build(n, L=L, beta=beta, alpha=20.0, potential='stirrer',
                backend='numpy', solver='dense', precision='fp64', disc='fd')
u, lam1, gs = rq_3d.ground_state(P, verbose=False)
ops = rq_3d.make_ops(P, u, lam1, ell=lam1, tf_f=0.5, inner=120, inner_tol=1e-10)
N3 = n**3
def dense(op):
    A = np.zeros((N3, N3)); e = np.zeros((n,n,n))
    for i in range(N3):
        e.flat[i] = 1.0; A[:, i] = np.asarray(op(e)).ravel(); e.flat[i] = 0.0
    return 0.5*(A+A.T)
B, Binv, _ = ops['metric']('au')
Nd, Bd = dense(ops['N']), dense(B)
m = np.asarray(u).ravel(); m = m/np.linalg.norm(m)
w = m.copy(); w[0] -= 1.0 if m[0] >= 0 else -1.0; w /= np.linalg.norm(w)
Q = (np.eye(m.size) - 2.0*np.outer(w,w))[:, 1:]
NT = 0.5*((Q.T@Nd@Q)+(Q.T@Nd@Q).T); BT = 0.5*((Q.T@Bd@Q)+(Q.T@Bd@Q).T)
ev, evec = sla.eigh(NT, BT)
eta_d = Q @ evec[:, 0]                      # dense minimizer, lifted into T
h3 = P['h3']
def R_free(x):
    x3 = x.reshape(n,n,n)
    return float(P['ip_M'](ops['N'](x3), x3) / P['ip_M'](B(x3), x3))
def R_dense(x):
    return float((x @ (Nd @ x)) / (x @ (Bd @ x)))
th, X, info = rq_3d.lobpcg_min(P, ops['N'], B, 
    lambda z: rq_3d.pcg(P, ops['N'], ops['prec_V1'], z, tol=1e-10, maxit=120)[0],
    k=2, constraint=u, tol=1e-13, maxit=1500, eig_tol=1e-14, label='au')
eta_l = np.asarray(X[0]).ravel()
r = ev[0]
print(f'dense eigenvalue                              {ev[0]:.16f}')
print(f'dense R() of the dense eigenvector            {R_dense(eta_d):.16f}   '
      f'{abs(R_dense(eta_d)-r)/r:.2e}')
print(f'matrix-free R() of the DENSE eigenvector      {R_free(eta_d):.16f}   '
      f'{abs(R_free(eta_d)-r)/r:.2e}')
print(f'matrix-free R() of the LOBPCG eigenvector     {R_free(eta_l):.16f}   '
      f'{abs(R_free(eta_l)-r)/r:.2e}')
print(f'dense R() of the LOBPCG eigenvector           {R_dense(eta_l):.16f}   '
      f'{abs(R_dense(eta_l)-r)/r:.2e}')
print(f'LOBPCG Ritz value                             {th[0]:.16f}   '
      f'{abs(th[0]-r)/r:.2e}')
print(f'\nangle between the two eigenvectors: '
      f'{np.arccos(min(1,abs(eta_d@eta_l)/np.linalg.norm(eta_d)/np.linalg.norm(eta_l))):.3e} rad')
