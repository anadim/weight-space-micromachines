import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import blogstyle  # noqa
G='#118659'; GR='#8a909a'
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
D='data/'
runs={'on_s0':'../../results/library/figdata/mix_m1_mixLIB_steps.jsonl','on_s1':D+'m1_mixLIB_s1_steps.jsonl','plain':D+'plain/m3_mixPlain64_steps.jsonl'}
def load(f):
    L=[json.loads(l) for l in open(f)]
    dev=[x['dev'] for x in L if isinstance(x.get('dev'),dict)]
    rw=[x for x in L if 'r_gsm' in x]
    return dev,rw
dat={k:load(v) for k,v in runs.items()}
def sm(y,w=9):
    y=np.array(y,float); k=np.ones(w)/w; p=np.pad(y,(w//2,w//2),mode='edge'); return np.convolve(p,k,'valid')
panels=[('dev_acc','All tasks'),('dev_gsm','GSM8K'),('dev_bbh','BBH'),('dev_ifeval','IFEval'),('dev_combo','GSM + IFEval combo')]
fig,ax=plt.subplots(1,5,figsize=(20,4.6),sharex=True)
for a,(k,t) in zip(ax,panels):
    for s,ls in [('on_s0','-'),('on_s1','--')]:
        d=dat[s][0]; a.plot([x['u'] for x in d],[x[k] for x in d],ls,color=G,lw=3,marker='o',ms=6,label='implants on, seed %s'%s[-1])
    d=dat['plain'][0]; a.plot([x['u'] for x in d],[x[k] for x in d],'-',color=GR,lw=3,marker='o',ms=6,label='plain RL')
    a.set_title(t,fontsize=20); a.set_xticks([0,50,100,150]); a.tick_params(labelsize=16); a.set_ylim(0,100); a.grid(axis='y',alpha=.3)
    for sp in ['top','right']: a.spines[sp].set_visible(False)
ax[0].set_ylabel('dev accuracy (%)',fontsize=18)
for a in ax: a.set_xlabel('RL update',fontsize=16)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,fontsize=17,frameon=False,bbox_to_anchor=(.5,-.08))
fig.tight_layout(rect=(0,.06,1,1)); fig.savefig('rl9_curves_dev.png',dpi=100,bbox_inches='tight'); fig.savefig('rl9_curves_dev.svg',bbox_inches='tight')
rp=[('reward','All tasks'),('r_gsm','GSM8K'),('r_bbh','BBH'),('r_ifeval','IFEval'),('r_combo','GSM + IFEval combo')]
fig,ax=plt.subplots(1,5,figsize=(20,4.6),sharex=True)
for a,(k,t) in zip(ax,rp):
    for s,ls in [('on_s0','-'),('on_s1','--')]:
        r=dat[s][1]; a.plot(range(1,len(r)+1),sm([x[k] for x in r]),ls,color=G,lw=2.5,label='implants on, seed %s'%s[-1])
    r=dat['plain'][1]; a.plot(range(1,len(r)+1),sm([x[k] for x in r]),'-',color=GR,lw=2.5,label='plain RL')
    a.set_title(t,fontsize=20); a.set_xticks([0,50,100,150]); a.tick_params(labelsize=16); a.set_ylim(0,1); a.grid(axis='y',alpha=.3)
    for sp in ['top','right']: a.spines[sp].set_visible(False)
    a.set_xlabel('RL update',fontsize=16)
ax[0].set_ylabel('training reward\n(9-update mean)',fontsize=18)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,fontsize=17,frameon=False,bbox_to_anchor=(.5,-.08))
fig.tight_layout(rect=(0,.06,1,1)); fig.savefig('rl9_curves_reward.png',dpi=100,bbox_inches='tight'); fig.savefig('rl9_curves_reward.svg',bbox_inches='tight')
