"""Analytic moment comparison; no simulated or fitted curves."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
p=ap.parse_args().output_dir
p.mkdir(parents=True,exist_ok=True)
n=np.arange(4,1001);theta=1/np.sqrt(n);lam=n*np.sin(theta)**2
native=(1+np.sin(2*theta)/4)**n
smoothed=np.exp(9*lam/16)
full=2*np.exp(8*np.pi*np.log(40)+9*lam/16)
fig,axes=plt.subplots(1,2,figsize=(8.4,3.5))
axes[0].semilogy(n,native,color='#245a81',label='Native: exact on target')
axes[0].semilogy(n,smoothed,'--',color='#a75c20',label='One smoothed score: bound')
axes[1].semilogy(n,native,color='#245a81',label='Native: exact on target')
axes[1].semilogy(n,full,'-.',color='#34765c',label='Combined score: bound V')
for ax in axes:
    ax.set_xlabel('Number of qubits n');ax.set_ylabel('Second moment');ax.grid(alpha=.22);ax.legend(fontsize=8,loc='best')
axes[0].set_title('Before coefficient mass');axes[1].set_title('Including coefficient mass')
fig.suptitle(r'Planar tilt $\theta=1/\sqrt{n}$; $T=1/4$, $\beta=1/40$',fontsize=11)
fig.tight_layout();fig.savefig(p/'moment_comparison.png',dpi=220);plt.close(fig)
(p/'moment_comparison.json').write_text(json.dumps({'n':n.tolist(),'native':native.tolist(),'single_smoothed_bound':smoothed.tolist(),'full_score_bound':full.tolist(),'note':'Moment comparison, not necessary high-probability copy counts. Full bound includes 2K^2 with K=40^(4pi).'},indent=2)+'\n')
