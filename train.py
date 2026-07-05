"""Train a surrogate with the paper config and save weights + held-out data.

Usage:
    python train.py --pde burgers --mode moe     # E2-soft-R4 (default)
    python train.py --pde burgers --mode dense   # dense-R8 baseline
Saves to  saved/<pde>_<mode>.pt
"""
import os, sys, argparse
import numpy as np, torch, torch.nn as nn
_HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,_HERE)  # spikemoe package
sys.path.insert(0,os.path.dirname(os.path.dirname(_HERE)))  # lab repo root (models_transformer, solvers)  # lab repo root
from spikemoe import SpikingRollout, generate, PAPER, count_params

def train(pde, mode, cfg=PAPER, save_dir='saved'):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    U, U_te, m, sd = generate(pde, cfg.n_traj, cfg.nx, cfg.nt_long, cfg.seed); U = U.to(dev)
    n_ch = U.shape[2]
    ratio = cfg.dense_ratio if mode == 'dense' else cfg.ratio
    model = SpikingRollout(n_ch, cfg.C, mode, ratio, cfg.experts, cfg.top_k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    crit = nn.MSELoss()
    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad(); mem = None; loss = 0.0
        for t in range(cfg.nt_train - 1):
            p, mem, _ = model.step(U[:, t], mem); loss = loss + crit(p, U[:, t+1])
        (loss/(cfg.nt_train-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sch.step()
    rel = held_out_relL2(model, U_te.to(dev), cfg.nt_long)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'mean': m, 'std': sd, 'U_heldout': U_te,
                'pde': pde, 'mode': mode, 'n_ch': n_ch, 'params': count_params(model),
                'heldout_relL2': rel, 'config': cfg.__dict__},
               f'{save_dir}/{pde}_{mode}.pt')
    print(f'[{pde}/{mode}] params={count_params(model)} held-out relL2={rel:.4f}  -> saved')
    return rel

@torch.no_grad()
def held_out_relL2(model, U_te, nt):
    model.eval(); rel = []
    for b in range(U_te.shape[0]):
        preds = model.rollout(U_te[b, 0:1], nt-1).squeeze(1).cpu().numpy()
        true = U_te[b, 1:].cpu().numpy()
        rel.append(np.linalg.norm(preds-true)/(np.linalg.norm(true)+1e-8))
    return float(np.mean(rel))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pde', default='burgers', choices=list(__import__('spikemoe').RANGES))
    ap.add_argument('--mode', default='moe', choices=['moe', 'dense'])
    ap.parse_args(); a = ap.parse_args()
    train(a.pde, a.mode)
