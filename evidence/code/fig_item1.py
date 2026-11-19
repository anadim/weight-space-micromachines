import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob,os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','item1'); OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures')
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','figures','fonts','*.ttf')): fm.fontManager.addfont(f)
FONT='Avenir Next' if any('Avenir' in f.name for f in fm.fontManager.ttflist) else 'DejaVu Sans'
INK,INK2,GRID,BG='#222222','#6b7280','#e5e7eb','#ffffff'; GRAY,GREEN='#b9bec6','#118659'
plt.rcParams.update({'font.family':FONT,'font.size':15,'text.color':INK,'axes.edgecolor':GRID,
 'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG})
mult=json.load(open(f'{D}/mult_heatmap.json'))
H=np.zeros((9,9))
for k,v in mult.items():
    da,db=map(int,k.split('x')); H[da-1,db-1]=v['base']*100
fig=plt.figure(figsize=(14,6.2))
# Panel A: heatmap
axA=fig.add_axes([0.06,0.12,0.40,0.70])
im=axA.imshow(H,origin='lower',cmap='Greens_r',vmin=0,vmax=100,extent=[0.5,9.5,0.5,9.5],aspect='auto')
axA.set_xticks(range(1,10)); axA.set_yticks(range(1,10))
axA.set_xlabel('digits in second operand'); axA.set_ylabel('digits in first operand')
for da in range(9):
    for db in range(9):
        val=H[da,db]; axA.text(db+1,da+1,f'{val:.0f}',ha='center',va='center',
            fontsize=9,color='#ffffff' if val<45 else INK)
cb=fig.colorbar(im,ax=axA,fraction=0.046,pad=0.04); cb.set_label('base accuracy (%)')
axA.set_title('Frozen model alone: multiplication',fontsize=16,loc='left',color=INK,pad=10)
# Panel B: lines vs digit length
axB=fig.add_axes([0.58,0.12,0.38,0.70])
ds=list(range(1,10))
multdiag=[mult[f'{d}x{d}']['base']*100 for d in ds]
add=json.load(open(f'{D}/add_bylen.json')); sub=json.load(open(f'{D}/sub_bylen.json'))
addb=[add[f'd{d}']['base']*100 for d in ds]; subb=[sub[f'd{d}']['base']*100 for d in ds]
axB.plot(ds,multdiag,'-o',color=GREEN,lw=2.5,ms=6,label='multiply (base)')
axB.plot(ds,addb,'-s',color='#2563eb',lw=2,ms=5,label='add (base)')
axB.plot(ds,subb,'-^',color='#b45309',lw=2,ms=5,label='subtract (base)')
axB.axhline(100,ls='--',color=INK,lw=2)
axB.text(5,101.5,'all three + calculator implant = 100%',color=INK,fontsize=12,ha='center')
axB.set_ylim(-4,108); axB.set_xticks(ds); axB.set_xlabel('digits per operand (d × d)')
axB.set_ylabel('accuracy (%)'); axB.legend(frameon=False,fontsize=12,loc='lower left')
axB.set_title('Base vs implant, by operand size',fontsize=16,loc='left',color=INK,pad=10)
fig.text(0.06,0.95,'The calculator implant: exact arithmetic on a frozen Qwen3-0.6B-Base',
    fontsize=19,fontweight='bold',va='top')
fig.text(0.06,0.895,'few-shot prompt, no training, n=100 per cell; implant = calc_s4',fontsize=13,color=INK2,va='top')
fig.savefig(f'{OUT}/item1_calculator.png',dpi=140); print('wrote item1_calculator.png')
