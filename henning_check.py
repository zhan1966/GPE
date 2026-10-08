# Henning (M3AS 2023) Example 7.2: -1/2 u'' + 1/2 x^2 u + 400 u^3 = lam u on (-16,16).
# Multiply by 2:  -u'' + x^2 u + 800 u^3 = (2 lam) u  -> our beta = 800, our lam1 = 2*35.57746 = 71.15492,
# lam2 = 2*35.60994 = 71.21988.  His |mu_1| = 0.94192 with mu_1 < 0, i.e. rho_max = 1 + 0.94192 = 1.94192.
import numpy as np, rq_1d as R
out, x, u = R.run(800.0, 16.0, 3000, alphas=(0.0,), verbose=False)
m = out['metrics']['a_u']
print(f"lam1 = {out['lam1']:.5f} (Henning*2 = 71.15492)   lam2 = {out['lam2']:.5f} (Henning*2 = 71.21988)")
print(f"rho_max = {m['rmax']:.5f} (Henning: 1.94192)   rho_min = {m['rmin']:.5f}   kappa = {m['rmax']/m['rmin']:.2f}")
print(f"undamped inverse-iteration rate max(1-rho_min, rho_max-1) = {max(1-m['rmin'], m['rmax']-1):.5f} (Henning |mu_1| = 0.94192)")
print(f"paper lower bound 1-lam1/lam2 = {1-out['lam1']/out['lam2']:.2e};  new bound 1-lam1/lam1(A_3b) = {1-out['lam1']/out['lam1_A3']:.4f};  mu1/(mu1+lam1) = {out['mu1']/(out['mu1']+out['lam1']):.4f}")
print(f"optimal-step GD rate (kappa-1)/(kappa+1) = {(m['rmax']/m['rmin']-1)/(m['rmax']/m['rmin']+1):.4f}")
