"""Central configuration. PAPER = the exact setting used for reported numbers (seed 42 canonical)."""
from dataclasses import dataclass, field

@dataclass
class Config:
    # architecture (E2-soft-R4)
    C: int = 48                # hidden width
    mode: str = 'moe'          # 'moe' or 'dense'
    experts: int = 2           # E
    ratio: int = 4             # R (expert hidden = R*C)
    top_k: int = 2             # ==experts -> soft gating
    dense_ratio: int = 8       # baseline dense-R8 expansion
    # data
    n_traj: int = 12
    nx: int = 128
    nt_train: int = 45         # teacher-forced horizon
    nt_long: int = 55          # free-rollout eval horizon
    # optimization
    lr: float = 1e-3
    lr_min: float = 1e-5       # cosine floor
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    epochs: int = 1500
    seed: int = 42             # canonical reference seed

PAPER = Config()
