# S2MoE: Spiking-based Mixture-of-Experts for Scientific Machine Learning

**S2MoE** (S²MoE) is a minimal, reproducible **spiking soft-gated Mixture-of-Experts** surrogate
for 1D PDE rollout. On advective *shock*-forming conservation laws (Burgers, Buckley--Leverett) it
beats a parameter-matched spiking dense baseline with ~24% fewer parameters; on smooth/forced PDEs
it ties. Experts specialize spatially to shock vs. smooth regions.

rollout. On advective *shock*-forming conservation laws (Burgers, Buckley--Leverett) the MoE
beats a parameter-matched spiking dense baseline with ~24% fewer parameters; on smooth/forced
PDEs it ties. Experts specialize spatially to shock vs. smooth regions.

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

## Note on "MoE"
This is the **classical soft-gated** MoE (Jacobs et al., 1991): all experts are active with
input-dependent weights. Sparse top-k routing (set `top_k < experts`) did not help in our setting;
the efficiency claim is *parameter* efficiency, not conditional-compute sparsity.
