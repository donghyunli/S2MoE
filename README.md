# S2MoE: Spiking-based Mixture-of-Experts for Scientific Machine Learning

**S2MoE** (S²MoE) is a minimal, reproducible **spiking soft-gated Mixture-of-Experts** surrogate
for 1D PDE rollout. On advective *shock*-forming conservation laws (Burgers, Buckley--Leverett) it
beats a parameter-matched spiking dense baseline with ~24% fewer parameters; on smooth/forced PDEs
it ties. Experts specialize spatially to shock vs. smooth regions.

## Layout
```
spikemoe/model.py   # SpikingRollout: QIF neuron, dense block, soft-gated MoE block (E2-soft-R4)
spikemoe/data.py    # benchmark data + held-out split (CAACT solvers; LWR is an inline extra)
spikemoe/config.py  # Config; PAPER = exact reported setting (seed 42 canonical)
train.py            # train one (pde, mode), save weights + held-out data to saved/
evaluate.py         # held-out results table: MoE vs dense across benchmarks
make_figures.py     # per-token expert-assignment (separation) figure from saved weights
saved/              # pretrained MoE weights (seed 42)
paper/              # method_full.{tex,pdf}
```

## Requirements
- PyTorch, NumPy, Matplotlib (see `requirements.txt`).
- **The lab repository must be on `PYTHONPATH`** (this folder lives inside it) for
  `models_transformer.spiking_layers` (QIF neuron) and `solvers.*` (CAACT PDE solvers).

## Usage
```bash
# reproduce the held-out results table (trains any missing weights)
python evaluate.py

# train a single model
python train.py --pde burgers --mode moe     # E2-soft-R4
python train.py --pde burgers --mode dense   # dense-R8 baseline

# regenerate the separation figure from saved weights (fast)
python make_figures.py
```

## Reference configuration (paper)
`mode='moe', experts E=2, ratio R=4, top_k=E` (soft gating) --- "**E2-soft-R4**", 28,713 params.
Baseline: `mode='dense', dense_ratio=8` --- "dense-R8", 37,831 params.
Training: AdamW lr 1e-3 (cosine -> 1e-5), 1500 epochs, teacher-forced one-step MSE on a 45-step
horizon, evaluated by free rollout to 55 steps on held-out parameter values. Canonical seed 42.

## Benchmarks
Burgers, Buckley--Leverett, Heat, Navier--Stokes (forced Burgers) follow the CAACT lab-solver
convention. LWR traffic uses an inline Rusanov solver (flagged; not CAACT).

## Backbone and spiking-depth ablations

After finding that a convolutional backbone works, we asked *where* the MoE win comes from and
*how far* the model can be pushed toward being fully spiking. All numbers below are **held-out
rollout relative-L2 on Burgers** (seed 42, param-matched MoE-vs-dense, same training protocol).
Single seed --- directional signal; confirm winners with 3 seeds before publishing. Scripts and a
longer write-up are in [`experiments/`](experiments/RESULTS.md).

| # | Config | MoE | dense | Winner | Note |
|---|--------|-----|-------|--------|------|
| 1 | **Conv, 1 QIF** (paper model) | **0.097** | 0.133 | MoE | best accuracy |
| 2 | MaxFormer attention backbone | 0.295 | 0.208 | dense | attention worse here |
| 3 | Fully-spiking conv (QIF after every conv) | 0.186 | 0.165 | dense | precision lost, specialization broken |
| 4 | Fully-spiking + **router reads continuous** | 0.144 | 0.165 | MoE | specialization restored |
| 5 | (4) + **2 blocks** (depth 2) | 0.161 | 0.170 | MoE | depth does not help (~2x params) |
| 6 | (4) + **Fourier value-lift** | 0.149 | 0.583 | (MoE) | Fourier no help; dense diverges |

**Takeaways.**
1. Best accuracy is the **1-QIF conv model** (paper); every step toward "more spiking" costs precision.
2. The **MaxFormer / attention backbone is worse** here, and *within* it dense beats MoE --- global
   attention already handles the shock, so routing adds nothing. Conv locality + translation-
   equivariance is the right inductive bias for these PDEs.
3. Going **fully spiking breaks the MoE win** (dense wins): binary spikes at the gate input destroy
   the fine gate specialization the MoE relies on.
4. Keeping **only the router path continuous restores the MoE win** under otherwise-full spiking ---
   direct causal evidence that the advantage is *precise, input-dependent gating*.
5. **Depth (2 blocks) does not help**: effective depth already comes from the rollout timesteps.
6. **Fourier value-lift does not help**: shock sharpness is a spatial-gradient, not amplitude-basis,
   phenomenon, and its high-frequency features destabilize the un-gated dense rollout.

`MoE > dense` holds across #1, #4, #5, flipping only when the router itself is spiked (#3) or the
attention backbone is used (#2): the MoE advantage is tied to precise gating on a capacity-limited,
local backbone.

## Note on "MoE"
This is the **classical soft-gated** MoE (Jacobs et al., 1991): all experts are active with
input-dependent weights. Sparse top-k routing (set `top_k < experts`) did not help in our setting;
the efficiency claim is *parameter* efficiency, not conditional-compute sparsity.
