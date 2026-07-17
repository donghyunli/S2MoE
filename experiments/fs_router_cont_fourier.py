"""
1-block router-continuous fully-spiking rollout + FOURIER-FEATURE LIFT of the field VALUE.

Lift:  u (B,1,X) -> [u, sin(2pi 2^k w_k u), cos(2pi 2^k w_k u)]_{k<K} -> Conv1d(1+2K, C, 5)
(Translation-equivariant: same value-wise Fourier transform at every spatial location,
 so it does NOT reintroduce absolute-position dependence.)
Rest identical to fs_router_cont (router reads continuous, experts/hidden/down spike).

Usage: python fs_router_cont_fourier.py --nfreq 4
"""
import os, sys, argparse, contextlib, io, copy, math
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

class FourierLift(nn.Module):
    """Per-location value-wise Fourier features, then Conv1d to C."""
    def __init__(s,n_ch,C,n_freq,kernel=5):
        super().__init__(); s.K=n_freq; s.nch=n_ch
        s.w=nn.Parameter(torch.randn(n_ch,n_freq)*0.1)
        s.c1=nn.Conv1d(n_ch*(1+2*n_freq),C,kernel,padding=kernel//2)
    def forward(s,u):                      # u:(B,n_ch,X)
        feats=[u]
        for k in range(s.K):
            freq=2.0*math.pi*(2**k)*(u*s.w[:,k].view(1,-1,1))
            feats+=[torch.sin(freq),torch.cos(freq)]
        return s.c1(torch.cat(feats,dim=1))

class Model(nn.Module):
    def __init__(s,n_ch,C,mode,ratio,experts,top_k,n_freq,kernel=5):
        super().__init__(); s.mode,s.E,s.K=mode,experts,(top_k or experts); H=ratio*C
        s.lift=FourierLift(n_ch,C,n_freq,kernel); s.q1=QIF()
        if mode=='dense': s.up=nn.Conv1d(C,H,1)
        else: s.experts=nn.ModuleList([nn.Conv1d(C,H,1) for _ in range(experts)]); s.router=nn.Conv1d(C,experts,1)
        s.qh=QIF(); s.down=nn.Conv1d(H,C,1); s.qd=QIF(); s.c2=nn.Conv1d(C,n_ch,kernel,padding=kernel//2)
    def step(s,u,mems):
        m1,mh,md=mems if mems else (None,None,None)
        c=s.lift(u); hs,m1=s.q1(c,m1)
        if s.mode=='dense':
            z,mh=s.qh(s.up(hs),mh); g=None
        else:
            g=F.softmax(s.router(c),dim=1)                     # gate from CONTINUOUS lifted features
            if s.K<s.E:
                idx=g.topk(s.K,dim=1).indices; mask=torch.zeros_like(g).scatter(1,idx,1.0); g=g*mask; g=g/(g.sum(1,keepdim=True)+1e-9)
            mix=sum(g[:,e:e+1]*s.experts[e](hs) for e in range(s.E)); z,mh=s.qh(mix,mh)
        d,md=s.qd(s.down(z),md)
        return u+s.c2(d),(m1,mh,md),g

def relL2(model,U_te,nt):
    model.eval(); rel=[]
    with torch.no_grad():
        for b in range(U_te.shape[0]):
            mems=None; u=U_te[b,0:1].clone(); preds=[]
            for _ in range(nt-1): u,mems,_=model.step(u,mems); preds.append(u[0].cpu().numpy())
            preds=np.array(preds); true=U_te[b,1:].cpu().numpy()
            rel.append(np.linalg.norm(preds-true)/(np.linalg.norm(true)+1e-8))
    return float(np.mean(rel))

def train(mode,ratio,n_freq,cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    U,U_te,m,sd=get_data(cfg); U=U.to(dev)
    model=Model(U.shape[2],cfg.C,mode,ratio,cfg.experts,cfg.top_k,n_freq).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr_min)
    crit=nn.MSELoss()
    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad(); mems=None; loss=0.0
        for t in range(cfg.nt_train-1):
            p,mems,_=model.step(U[:,t],mems); loss=loss+crit(p,U[:,t+1])
        (loss/(cfg.nt_train-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); sch.step()
        if ep%300==0: print(f'  [{mode}/fourier] ep{ep} loss={loss.item()/(cfg.nt_train-1):.4e}',flush=True)
    rel=relL2(model,U_te.to(dev),cfg.nt_long)
    print(f'[fs-rc-fourier/{mode}] nfreq={n_freq} params={count_params(model)} held-out relL2={rel:.4f}',flush=True)
    return rel,count_params(model)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--nfreq',type=int,default=4); ap.add_argument('--epochs',type=int,default=None); a=ap.parse_args()
    cfg=copy.copy(PAPER)
    if a.epochs is not None: cfg.epochs=a.epochs
    rm,pm=train('moe',cfg.ratio,a.nfreq,cfg); rd,pd=train('dense',cfg.dense_ratio,a.nfreq,cfg)
    print(f'\nRESULT  1-block fully-spiking(router-cont)+Fourier  Burgers held-out relL2:',flush=True)
    print(f'   MoE={rm:.4f} ({pm}p)   dense={rd:.4f} ({pd}p)   ({"MoE wins" if rm<rd else "dense wins"})',flush=True)
    print('COMPARE  no-Fourier router-cont: MoE=0.1441 dense=0.1649 | 1-QIF: MoE=0.0971 dense=0.1325',flush=True)
