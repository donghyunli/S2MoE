"""
FULLY-SPIKING conv rollout on Burgers: put a QIF after EVERY conv (except the final
linear readout c2 and the softmax router gate).  Compare to the current 1-QIF model.

Block (moe):
  u^t -> c1 -> QIF1 -> {router->softmax; experts; soft-mix} -> QIF2 -> down -> QIF3 -> c2 -> +u^t
Each QIF threads its own membrane across rollout timesteps.

Param count ~ unchanged (QIF adds only a few scalars) so C=48 stays param-matched.

Usage: python fully_spiking_pilot.py            # trains moe + dense, reports held-out relL2
"""
import os, sys, argparse, contextlib, io, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_HERE=os.path.dirname(os.path.abspath(__file__)); _REL=os.path.dirname(_HERE); _ROOT=os.path.dirname(os.path.dirname(_REL))
sys.path.insert(0,_ROOT); sys.path.insert(0,_REL)
from models_transformer.spiking_layers import QIFNeuronWithLearnableSurrogate as QIF
from spikemoe import generate, PAPER, count_params
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(cfg):
    cache=f'{_HERE}/_cache_burgers_{cfg.n_traj}_{cfg.nx}_{cfg.nt_long}_{cfg.seed}.pt'
    if os.path.exists(cache):
        d=torch.load(cache); return d['U'],d['U_te'],d['m'],d['sd']
    with contextlib.redirect_stdout(io.StringIO()):
        U,U_te,m,sd=generate('burgers',cfg.n_traj,cfg.nx,cfg.nt_long,cfg.seed)
    torch.save({'U':U,'U_te':U_te,'m':m,'sd':sd},cache); return U,U_te,m,sd

class FSRollout(nn.Module):
    """Fully-spiking conv rollout (QIF after every conv but the readout)."""
    def __init__(s, n_ch=1, C=48, mode='moe', ratio=4, experts=2, top_k=None, kernel=5):
        super().__init__(); s.mode, s.E, s.K = mode, experts, (top_k or experts); H=ratio*C; pad=kernel//2
        s.c1=nn.Conv1d(n_ch,C,kernel,padding=pad); s.q1=QIF()
        if mode=='dense':
            s.up=nn.Conv1d(C,H,1)
        else:
            s.experts=nn.ModuleList([nn.Conv1d(C,H,1) for _ in range(experts)]); s.router=nn.Conv1d(C,experts,1)
        s.qh=QIF(); s.down=nn.Conv1d(H,C,1); s.qd=QIF()
        s.c2=nn.Conv1d(C,n_ch,kernel,padding=pad)                 # linear readout
    def step(s, u, mems):
        m1,mh,md=mems if mems else (None,None,None)
        h,m1=s.q1(s.c1(u),m1)                                     # spike features
        if s.mode=='dense':
            z,mh=s.qh(s.up(h),mh); g=None
        else:
            g=F.softmax(s.router(h),dim=1)
            if s.K<s.E:
                idx=g.topk(s.K,dim=1).indices; mask=torch.zeros_like(g).scatter(1,idx,1.0); g=g*mask; g=g/(g.sum(1,keepdim=True)+1e-9)
            mix=sum(g[:,e:e+1]*s.experts[e](h) for e in range(s.E)); z,mh=s.qh(mix,mh)
        d,md=s.qd(s.down(z),md)                                   # spike
        return u+s.c2(d), (m1,mh,md), g
    @torch.no_grad()
    def rollout(s,u0,n):
        s.eval(); mems=None; u=u0.clone(); out=[]
        for _ in range(n): u,mems,_=s.step(u,mems); out.append(u)
        return torch.stack(out,1)

def relL2(model,U_te,nt):
    model.eval(); rel=[]
    with torch.no_grad():
        for b in range(U_te.shape[0]):
            mems=None; u=U_te[b,0:1].clone(); preds=[]
            for _ in range(nt-1): u,mems,_=model.step(u,mems); preds.append(u[0].cpu().numpy())
            preds=np.array(preds); true=U_te[b,1:].cpu().numpy()
            rel.append(np.linalg.norm(preds-true)/(np.linalg.norm(true)+1e-8))
    return float(np.mean(rel))

def train(mode, ratio, cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    U,U_te,m,sd=get_data(cfg); U=U.to(dev)
    model=FSRollout(U.shape[2],cfg.C,mode,ratio,cfg.experts,cfg.top_k).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr_min)
    crit=nn.MSELoss()
    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad(); mems=None; loss=0.0
        for t in range(cfg.nt_train-1):
            p,mems,_=model.step(U[:,t],mems); loss=loss+crit(p,U[:,t+1])
        (loss/(cfg.nt_train-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); sch.step()
        if ep%200==0: print(f'  [{mode}] ep{ep} loss={loss.item()/(cfg.nt_train-1):.4e}',flush=True)
    rel=relL2(model,U_te.to(dev),cfg.nt_long)
    torch.save({'state_dict':model.state_dict(),'mode':mode,'params':count_params(model),
                'heldout_relL2':rel,'U_heldout':U_te},f'{_HERE}/saved_fullspike_burgers_{mode}.pt')
    print(f'[fullspike/{mode}] params={count_params(model)} held-out relL2={rel:.4f}',flush=True)
    return rel

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=None); a=ap.parse_args()
    cfg=copy.copy(PAPER)
    if a.epochs is not None: cfg.epochs=a.epochs
    rm=train('moe',cfg.ratio,cfg); rd=train('dense',cfg.dense_ratio,cfg)
    print(f'\nRESULT  fully-spiking conv Burgers held-out relL2:  MoE={rm:.4f}  dense={rd:.4f}  '
          f'({"MoE wins" if rm<rd else "dense wins"})',flush=True)
    print('COMPARE  1-QIF conv (current):  MoE=0.0971  dense=0.1325',flush=True)
