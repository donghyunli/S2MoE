"""Held-out results table: MoE vs dense across benchmarks (trains if weights missing)."""
import os, sys, numpy as np, torch
_HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,_HERE)  # spikemoe package
sys.path.insert(0,os.path.dirname(os.path.dirname(_HERE)))  # lab repo root (models_transformer, solvers)
from spikemoe import RANGES
from train import train

def relL2(pde, mode):
    p = f'saved/{pde}_{mode}.pt'
    if os.path.exists(p): return torch.load(p, map_location='cpu')['heldout_relL2'], torch.load(p, map_location='cpu')['params']
    r = train(pde, mode); import torch as T; d = T.load(f'saved/{pde}_{mode}.pt', map_location='cpu'); return r, d['params']

if __name__ == '__main__':
    print(f"{'benchmark':12s}{'MoE':>10s}{'dense-R8':>10s}{'MoE params':>12s}{'dense params':>13s}")
    for pde in ['burgers', 'buckley', 'heat', 'ns', 'lwr']:
        rm, pm = relL2(pde, 'moe'); rd, pd = relL2(pde, 'dense')
        tag = 'WIN' if rd - rm > 0.01 else 'tie'
        print(f'{pde:12s}{rm:>10.3f}{rd:>10.3f}{pm:>12d}{pd:>13d}   {tag}')
