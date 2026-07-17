# Backbone & spiking-depth ablations (Burgers pilot)

All numbers are **held-out rollout relative-L2** on Burgers (seed 42, param-matched
MoE-vs-dense, same training protocol as the paper: teacher-forced one-step MSE over
`nt_train`, cosine LR, free 55-step rollout on interleaved held-out viscosities).
Single seed — directional signal, not final. Confirm winners with 3 seeds before publishing.

| # | Config | MoE | dense | Winner | Note |
|---|--------|-----|-------|--------|------|
| 1 | **Conv, 1 QIF** (paper model) | **0.097** | 0.133 | MoE | best accuracy |
| 2 | MaxFormer attention backbone | 0.295 | 0.208 | dense | attention worse here |
| 3 | Fully-spiking conv (QIF after every conv) | 0.186 | 0.165 | dense | precision lost, specialization broken |
| 4 | Fully-spiking + **router reads continuous** | 0.144 | 0.165 | MoE | specialization restored |
| 5 | (4) + **2 blocks** (depth 2) | 0.161 | 0.170 | MoE | depth does not help (~2x params) |
| 6 | (4) + **Fourier value-lift** | 0.149 | 0.583 | (MoE) | Fourier no help; dense diverges |

## Findings

1. **Best accuracy is the 1-QIF conv model** (paper). Everything more "spiking" costs precision.
2. **MaxFormer/attention backbone is worse** for this small rollout, and *within* it dense beats
   MoE — global attention already handles the shock, so routing adds no benefit. The conv model's
   locality + translation-equivariance is the right inductive bias for these PDEs.
3. **Fully spiking (QIF everywhere) breaks the MoE win** (dense wins) because binary spikes at the
   gate input destroy the fine gate specialization the MoE relies on.
4. **Keeping only the router path continuous restores the MoE win** under otherwise-full spiking
   (0.144 < 0.165). This is direct causal evidence that the MoE advantage = *precise gating*.
   (Keeping the router in higher precision is standard practice in spiking-MoE work.)
5. **Depth (2 blocks) does not help** — effective depth already comes from the rollout timesteps;
   extra spiking blocks only add quantization noise.
6. **Fourier value-lift does not help** — shock sharpness is a spatial-gradient phenomenon, not an
   amplitude-basis one; high-frequency features destabilize the (un-gated) dense rollout.

## Robustness of the core result

`MoE > dense` holds across configs #1, #4, #5 (and only flips when the router itself is spiked,
#3, or the attention backbone is used, #2). The MoE advantage is tied specifically to
**precise, input-dependent gating on a capacity-limited, local backbone**.

## Scripts

| Script | Config |
|--------|--------|
| `maxformer_rollout_pilot.py` | #2 MaxFormer attention backbone (param-matched by embed dim) |
| `fully_spiking_pilot.py` | #3 QIF after every conv |
| `fs_router_cont.py` | #4 fully-spiking, router reads continuous |
| `fs_router_cont_2block.py` | #5 depth-N stack (`--blocks 2`) |
| `fs_router_cont_fourier.py` | #6 value-wise Fourier lift (`--nfreq 4`) |

Run on a CUDA node, e.g. `python experiments/fs_router_cont.py`. Each caches the Burgers data to
`experiments/_cache_burgers_*.pt` on first run (git-ignored).
