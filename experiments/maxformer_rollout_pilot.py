"""
MaxFormer-backbone autoregressive-rollout PILOT on Burgers.

Swaps the Conv1d block of SpikingRollout for a real MaxFormer transformer block
(the lab's MaxFormerAttention = SDT spiking attention + high-frequency branch),
operating over SPATIAL tokens (each grid point x is a token).  FFN is either a
spiking dense MLP or a spiking soft-gated MoE (E2-soft-R4), mirroring the paper.

u^{t+1} = u^t + Unlift( FFN( Attn( Lift(u^t) + pos ) ) )   applied autoregressively.

Param budget is auto-matched to the current conv models: MoE~28.7k, dense~37.8k.

Usage:
    python maxformer_rollout_pilot.py --check     # just print param-matched configs
    python maxformer_rollout_pilot.py             # train both, report held-out relL2
"""
import os, sys, math, argparse, contextlib, io
import numpy as np, torch, torch.nn as nn
_HERE=os.path.dirname(os.path.abspath(__file__)); _REL=os.path.dirname(_HERE); _ROOT=os.path.dirname(os.path.dirname(_REL))
sys.path.insert(0,_ROOT); sys.path.insert(0,_REL)
from models_transformer.spiking_layers import QIFNeuronWithLearnableSurrogate, MaxFormerAttention
from spikemoe import generate, PAPER, SpikingRollout, count_params
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(cfg):
    """Load cached Burgers data or generate once (suppressing the verbose solver output)."""
    cache=f'{_HERE}/_cache_burgers_{cfg.n_traj}_{cfg.nx}_{cfg.nt_long}_{cfg.seed}.pt'
    if os.path.exists(cache):
        d=torch.load(cache); return d['U'],d['U_te'],d['m'],d['sd']
    with contextlib.redirect_stdout(io.StringIO()):
        U,U_te,m,sd=generate('burgers',cfg.n_traj,cfg.nx,cfg.nt_long,cfg.seed)
    torch.save({'U':U,'U_te':U_te,'m':m,'sd':sd},cache); return U,U_te,m,sd

TARGET_MOE=28713; TARGET_DENSE=37831   # current conv-model param counts

def sinusoidal(X, D):
    pe=torch.zeros(X,D); pos=torch.arange(X).unsqueeze(1).float()
    div=torch.exp(torch.arange(0,D,2).float()*(-math.log(10000.0)/D))
    pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
    return pe.unsqueeze(0)   # (1,X,D)

class MFBlock(nn.Module):
    """MaxFormer transformer block over spatial tokens, dense or soft-MoE FFN."""
    def __init__(self, D, mode='moe', ratio=4, experts=2, heads=2, X=128):
        super().__init__()
        self.mode, self.E = mode, experts; H=ratio*D
        self.lift=nn.Linear(1, D); self.unlift=nn.Linear(D, 1)
        self.register_buffer('pos', sinusoidal(X, D))
        self.attn=MaxFormerAttention(D, num_heads=heads, neuron_type='qif')
        self.norm=nn.LayerNorm(D)
        self.qif=QIFNeuronWithLearnableSurrogate()          # shared spiking nonlinearity
        if mode=='dense':
            self.up=nn.Linear(D,H); self.down=nn.Linear(H,D)
        else:
            self.experts=nn.ModuleList([nn.Linear(D,H) for _ in range(experts)])
            self.router=nn.Linear(D,experts); self.down=nn.Linear(H,D)
    def ffn(self, z, mem):
        if self.mode=='dense':
            h=self.up(z); sp,mem=self.qif(h,mem); return self.down(sp), mem, None
        g=torch.softmax(self.router(z),-1)                  # (B,X,E)
        mix=sum(g[...,e:e+1]*self.experts[e](z) for e in range(self.E))
        sp,mem=self.qif(mix,mem); return self.down(sp), mem, g
    def step(self, u, mem):                                 # u:(B,1,X)
        tok=self.lift(u.transpose(1,2))+self.pos            # (B,X,D)
        tok=tok+self.attn(tok)
        z=self.norm(tok)
        f,mem,g=self.ffn(z,mem)
        du=self.unlift(tok+f).transpose(1,2)                # (B,1,X)
        return u+du, mem, g

class MFRollout(nn.Module):
    def __init__(self, D, mode, ratio, experts, X=128):
        super().__init__(); self.blk=MFBlock(D,mode,ratio,experts,X=X)
    def step(self,u,mem): return self.blk.step(u,mem)
    @torch.no_grad()
    def rollout(self,u0,n):
        self.eval(); mem=None; u=u0.clone(); out=[]
        for _ in range(n):
            u,mem,_=self.step(u,mem); out.append(u)
        return torch.stack(out,1)                            # (B,n,1,X)... wait keep (n,1,X) per batch

def match_D(target, mode, ratio, experts, X=128, heads=2):
    best=None
    for D in range(8, 96, 2):
        if D%heads: continue
        m=MFRollout(D,mode,ratio,experts,X); p=count_params(m)
        if best is None or abs(p-target)<abs(best[1]-target): best=(D,p)
    return best

def relL2(model, U_te, nt):
    model.eval(); rel=[]
    with torch.no_grad():
        for b in range(U_te.shape[0]):
            mem=None; u=U_te[b,0:1].clone(); preds=[]
            for _ in range(nt-1):
                u,mem,_=model.step(u,mem); preds.append(u[0].cpu().numpy())
            preds=np.array(preds); true=U_te[b,1:].cpu().numpy()
            rel.append(np.linalg.norm(preds-true)/(np.linalg.norm(true)+1e-8))
    return float(np.mean(rel))

def train(mode, D, ratio, cfg=PAPER):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    U,U_te,m,sd=get_data(cfg); U=U.to(dev)
    model=MFRollout(D,mode,ratio,cfg.experts,X=cfg.nx).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr_min)
    crit=nn.MSELoss()
    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad(); mem=None; loss=0.0
        for t in range(cfg.nt_train-1):
            p,mem,_=model.step(U[:,t],mem); loss=loss+crit(p,U[:,t+1])
        (loss/(cfg.nt_train-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); sch.step()
        if ep%200==0: print(f'  [{mode}] ep{ep} loss={loss.item()/(cfg.nt_train-1):.4e}',flush=True)
    rel=relL2(model,U_te.to(dev),cfg.nt_long)
    torch.save({'state_dict':model.state_dict(),'D':D,'ratio':ratio,'mode':mode,
                'params':count_params(model),'heldout_relL2':rel,'U_heldout':U_te},
               f'{_HERE}/saved_maxformer_burgers_{mode}.pt')
    print(f'[maxformer/{mode}] D={D} params={count_params(model)} held-out relL2={rel:.4f}',flush=True)
    return rel

if __name__=='__main__':
    import copy
    ap=argparse.ArgumentParser(); ap.add_argument('--check',action='store_true')
    ap.add_argument('--epochs',type=int,default=None); ap.add_argument('--only',default=None,choices=['moe','dense'])
    a=ap.parse_args()
    cfg=copy.copy(PAPER)
    if a.epochs is not None: cfg.epochs=a.epochs
    Dm,pm=match_D(TARGET_MOE,'moe',PAPER.ratio,PAPER.experts,PAPER.nx)
    Dd,pd=match_D(TARGET_DENSE,'dense',PAPER.dense_ratio,PAPER.experts,PAPER.nx)
    print(f'MoE   : D={Dm} params={pm} (target {TARGET_MOE})',flush=True)
    print(f'dense : D={Dd} params={pd} (target {TARGET_DENSE})',flush=True)
    if a.check: sys.exit(0)
    res={}
    if a.only in (None,'moe'):
        print('=== training MoE ===',flush=True); res['moe']=train('moe',Dm,PAPER.ratio,cfg)
    if a.only in (None,'dense'):
        print('=== training dense ===',flush=True); res['dense']=train('dense',Dd,PAPER.dense_ratio,cfg)
    if 'moe' in res and 'dense' in res:
        rm,rd=res['moe'],res['dense']
        print(f'\nRESULT  MaxFormer-rollout Burgers held-out relL2:  MoE={rm:.4f}  dense={rd:.4f}  '
              f'({"MoE wins" if rm<rd else "dense wins"})',flush=True)
