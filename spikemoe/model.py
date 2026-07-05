"""
Model definitions: a 1D convolutional autoregressive-rollout surrogate whose block is either a
spiking dense feed-forward network or a spiking *soft-gated* Mixture-of-Experts (MoE).

All nonlinearities are quadratic integrate-and-fire (QIF) spiking neurons (imported from the lab
codebase `models_transformer.spiking_layers`).

Blocks
------
- 'dense' : single QIF feed-forward expert of expansion ratio R.
- 'moe'   : E experts (each ratio R), a per-location softmax router, a shared QIF, one down-proj.
            With top_k == E (default) this is the classical *soft* MoE (all experts active,
            Jacobs et al. 1991). top_k < E gives sparse routing (not used for the paper result).

Reference config (paper): mode='moe', E=2, R=4  ==>  "E2-soft-R4".
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# ensure the lab repository root is importable (models_transformer, solvers)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from models_transformer.spiking_layers import QIFNeuronWithLearnableSurrogate


class SpikingRollout(nn.Module):
    """u^{t+1} = u^t + Conv( block( Conv(u^t) ) )  applied autoregressively.

    Args:
        n_ch:   number of physical channels of the field (1 for scalar PDEs).
        C:      hidden channel width.
        mode:   'dense' or 'moe'.
        ratio:  expert MLP expansion ratio R (hidden = R*C).
        experts: number of experts E (moe only).
        top_k:  active experts per token (moe only); top_k==experts => soft gating.
        kernel: spatial conv kernel size.
    """

    def __init__(self, n_ch=1, C=48, mode='moe', ratio=4, experts=2, top_k=None, kernel=5):
        super().__init__()
        self.mode, self.E, self.K = mode, experts, (top_k or experts)
        H = ratio * C
        pad = kernel // 2
        self.c1 = nn.Conv1d(n_ch, C, kernel, padding=pad)   # lift field -> C channels
        self.c2 = nn.Conv1d(C, n_ch, kernel, padding=pad)   # project residual back to field
        self.qif = QIFNeuronWithLearnableSurrogate()        # shared spiking nonlinearity
        if mode == 'dense':
            self.up = nn.Conv1d(C, H, 1)
            self.down = nn.Conv1d(H, C, 1)
        elif mode == 'moe':
            self.experts = nn.ModuleList([nn.Conv1d(C, H, 1) for _ in range(experts)])
            self.router = nn.Conv1d(C, experts, 1)
            self.down = nn.Conv1d(H, C, 1)
        else:
            raise ValueError(mode)

    def _block(self, h, mem):
        """Return (feature_C, new_mem, gate) where gate is (B,E,X) for moe else None."""
        if self.mode == 'dense':
            sp, mem = self.qif(self.up(h), mem)
            return self.down(sp), mem, None
        g = F.softmax(self.router(h), dim=1)                 # (B, E, X) per-location gate
        if self.K < self.E:                                  # optional sparse top-k
            idx = g.topk(self.K, dim=1).indices
            mask = torch.zeros_like(g).scatter(1, idx, 1.0)
            g = g * mask; g = g / (g.sum(1, keepdim=True) + 1e-9)
        mix = sum(g[:, e:e+1] * self.experts[e](h) for e in range(self.E))   # soft mixture in H-dim
        sp, mem = self.qif(mix, mem)
        return self.down(sp), mem, g

    def step(self, u, mem):
        """One rollout step. u: (B, n_ch, X). Returns (u_next, mem, gate)."""
        feat, mem, g = self._block(self.c1(u), mem)
        return u + self.c2(feat), mem, g

    @torch.no_grad()
    def rollout(self, u0, n_steps, return_gates=False):
        """Free rollout from initial field u0 (B, n_ch, X) for n_steps. Returns preds (n_steps, B, n_ch, X)."""
        self.eval(); mem = None; u = u0.clone(); preds, gates = [], []
        for _ in range(n_steps):
            u, mem, g = self.step(u, mem)
            preds.append(u)
            if return_gates and g is not None: gates.append(g)
        preds = torch.stack(preds)
        return (preds, torch.stack(gates)) if return_gates else preds


def count_params(model):
    return sum(p.numel() for p in model.parameters())
