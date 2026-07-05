"""
Benchmark data generation for the paper suite.

CAACT solvers (lab codebase `solvers/`): burgers, buckley, heat, ns.
Inline finite-volume solver: lwr (LWR traffic, Rusanov) -- flagged as NOT part of CAACT.

For each PDE we sweep one physical parameter over `n_traj` trajectories to create heterogeneity,
and build a HELD-OUT set at interleaved (unseen) parameter values for generalization testing.

Fields are resampled to `nx` points and normalized to zero mean / unit variance using TRAIN stats.
`generate(pde)` returns (U_train, U_heldout, mean, std), tensors of shape (n_traj, T, n_ch, nx).
"""
import os, sys
import numpy as np
import torch
_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path: sys.path.insert(0,_ROOT)

# (low, high, spacing) parameter sweep per benchmark
RANGES = {
    'burgers':  (0.002, 0.02,  'log'),   # viscosity nu
    'buckley':  (0.05,  0.085, 'lin'),   # initial-condition width
    'heat':     (0.005, 0.05,  'log'),   # diffusivity alpha
    'ns':       (0.006, 0.025, 'log'),   # viscosity nu (forced Burgers)
    'lwr':      (0.4,   0.85,  'lin'),   # jam amplitude (inline, not CAACT)
}


def _resample(u, nt, nx):
    return u[np.ix_(np.linspace(0, u.shape[0]-1, nt).astype(int),
                    np.linspace(0, u.shape[1]-1, nx).astype(int))]


def _lwr(param, nx=256, T=0.6, cfl=0.4):
    """LWR traffic rho_t + (rho(1-rho))_x = 0, Rusanov scheme; localized jam -> moving shock."""
    dx = 1.0/(nx-1); x = np.linspace(0, 1, nx)
    rho = np.clip(0.1 + param*np.exp(-((x-0.3)**2)/(2*0.05**2)), 0.01, 0.99)
    hist = []; t = 0.0; n = 0
    while t < T and n < 3000:
        a = float(np.max(np.abs(1 - 2*rho))); dt = cfl*dx/(a+1e-9)
        Fx = rho*(1-rho)
        Fr = 0.5*(Fx + np.roll(Fx, -1)) - 0.5*a*(np.roll(rho, -1) - rho); Fl = np.roll(Fr, 1)
        rho = rho - dt/dx*(Fr - Fl); rho[0] = rho[1]; rho[-1] = rho[-2]
        hist.append(rho.copy()); t += dt; n += 1
    return np.array(hist)


def _make_traj(pde, param, seed, i, nx, nt):
    """One (T, 1, nx) trajectory for the given parameter value."""
    if pde == 'burgers':
        from solvers.burgers_1d_solver import Burgers1DSolver
        s = Burgers1DSolver(nx=256, nu=param, domain_size=2.0, total_time=1.0, x_start=-1.0,
                            initial_condition='sine_fno', boundary_condition='dirichlet').solve()
    elif pde == 'buckley':
        from solvers.buckley_leverett_1d_solver import BuckleyLeverett1DSolver
        s = BuckleyLeverett1DSolver(nx=256, nt=200, ic_width=param).solve()
    elif pde == 'heat':
        from solvers.heat_1d_solver import Heat1DSolver
        s = Heat1DSolver(nx=256, alpha=param, domain_size=1.0, total_time=1.0, n_modes=8,
                         seed=int(seed*10+i), initial_condition='sine_canonical').solve()
    elif pde == 'ns':
        from solvers.ns_1d_solver import NS1DSolver
        s = NS1DSolver(nx=256, nu=param, domain_size=8.0, total_time=2.0, n_modes=4,
                       forcing_amplitude=0.3, seed=int(seed*10+i)).solve()
    elif pde == 'lwr':
        return _resample(_lwr(param), nt, nx)[:, None, :]
    else:
        raise ValueError(pde)
    return _resample(np.asarray(s['u_history']), nt, nx)[:, None, :]


def generate(pde, n_traj=12, nx=128, nt=55, seed=42):
    """Return (U_train, U_heldout, mean, std) normalized tensors of shape (B, nt, 1, nx)."""
    lo, hi, scl = RANGES[pde]
    sweep = (lambda n: np.logspace(np.log10(lo), np.log10(hi), n)) if scl == 'log' \
        else (lambda n: np.linspace(lo, hi, n))
    p_tr = sweep(n_traj)
    p_te = np.sqrt(p_tr[:-1]*p_tr[1:]) if scl == 'log' else (p_tr[:-1]+p_tr[1:])/2   # interleaved held-out
    D_tr = np.stack([_make_traj(pde, float(p), seed, i, nx, nt) for i, p in enumerate(p_tr)]).astype(np.float32)
    D_te = np.stack([_make_traj(pde, float(p), seed, i, nx, nt) for i, p in enumerate(p_te)]).astype(np.float32)
    m = D_tr.mean((0, 1, 3), keepdims=True); sd = D_tr.std((0, 1, 3), keepdims=True) + 1e-6
    return (torch.tensor((D_tr-m)/sd), torch.tensor((D_te-m)/sd), m, sd)
