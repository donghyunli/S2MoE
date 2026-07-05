"""Per-token expert-assignment (separation) figure from saved MoE weights (no retraining)."""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':18,'axes.titlesize':21,'axes.labelsize':19,
    'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16})
_HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,_HERE)  # spikemoe package
sys.path.insert(0,os.path.dirname(os.path.dirname(_HERE)))  # lab repo root (models_transformer, solvers)
from spikemoe import SpikingRollout

def rollout_gates(pde):
    d = torch.load(f'saved/{pde}_moe.pt', map_location='cpu'); c = d['config']
    model = SpikingRollout(d['n_ch'], c['C'], 'moe', c['ratio'], c['experts'], c['top_k'])
    model.load_state_dict(d['state_dict']); model.eval()
    U = d['U_heldout']
    b = int(np.argmax([np.abs(np.gradient(U[i,:,0].numpy(),axis=1)).max() for i in range(U.shape[0])]))
    preds, gates = model.rollout(U[b,0:1], U.shape[1]-1, return_gates=True)
    uf = preds.squeeze(1)[:,0].numpy(); g0 = gates[:,0,0].numpy()
    return uf, g0

pdes = [('burgers','Burgers (shock)'),('buckley','Buckley-Leverett (shock)'),('lwr','LWR (shock)'),
        ('ns','Navier-Stokes (forced)'),('heat','Heat (smooth control)')]
pdes = [(p,n) for p,n in pdes if os.path.exists(f'saved/{p}_moe.pt')]
fig, ax = plt.subplots(len(pdes), 2, figsize=(16, 5*len(pdes)))
if len(pdes)==1: ax = ax[None,:]
NX = 128
for i,(pde,nm) in enumerate(pdes):
    uf,g0 = rollout_gates(pde); shock=np.abs(np.gradient(uf,axis=1)); ts=uf.shape[0]//2; x=np.arange(NX); y=uf[ts]
    c=abs(np.corrcoef(g0.flatten(),shock.flatten())[0,1])
    im=ax[i,0].imshow(uf,aspect='auto',cmap='RdBu_r',extent=[0,NX,uf.shape[0],0]); plt.colorbar(im,ax=ax[i,0])
    dom=(g0>0.5).astype(float)
    if dom.std()>0: ax[i,0].contour(np.arange(NX),np.arange(uf.shape[0]),dom,levels=[0.5],colors='k',linewidths=3)
    ax[i,0].set_title(f'{nm}: u(x,t) [held-out]'); ax[i,0].set_ylabel('rollout t')
    ax[i,1].plot(x,y,'-',color='0.6',lw=1.5,zorder=1)
    sc=ax[i,1].scatter(x,y,c=g0[ts],cmap='coolwarm',vmin=0,vmax=1,s=90,edgecolors='k',linewidths=0.5,zorder=3)
    ax[i,1].plot(x,shock[ts]/(shock[ts].max()+1e-9)*(y.max()-y.min())*0.5+y.min(),color='orange',lw=2.2,ls='--',label='|du/dx|')
    ax[i,1].set_xlim(0,NX); ax[i,1].set_ylim(y.min()-0.3,y.max()+0.3)
    ax[i,1].set_title(f'{nm}: token$\\to$expert (corr={c:.2f})'); ax[i,1].legend(loc='best')
    cb=plt.colorbar(sc,ax=ax[i,1]); cb.set_label('gate: blue=E0, red=E1')
fig.suptitle('Per-token expert assignment (E2-soft-R4, seed 42, held-out)',fontsize=25,y=1.001)
plt.tight_layout(); os.makedirs('figures',exist_ok=True)
plt.savefig('figures/separation.png',dpi=135,bbox_inches='tight'); print('saved figures/separation.png')
