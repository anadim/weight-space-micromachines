import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import glob, os
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'fonts', '*.ttf')): fm.fontManager.addfont(f)
plt.rcParams['font.family']='Avenir Next'
G='#118659'; GD='#0b5e3f'; PALE='#e7f3ee'; GR='#b9bec6'; INK='#1f2328'; INK2='#5f6670'; RED='#c0392b'; PALER='#fbebeb'; Q='#eef0f3'
fig=plt.figure(figsize=(16,14.4)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,160); ax.set_ylim(-6,138); ax.axis('off')
def box(x,y,w,h,fc,ec=None,txt='',fs=15,c=INK,bold=False,lw=1.5,r=1.2,ha='center'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec or fc,lw=lw))
    if txt: ax.text(x+(w/2 if ha=='center' else 1.2),y+h/2,txt,ha=ha,va='center',fontsize=fs,color=c,fontweight='bold' if bold else 'normal')
def tag(x,y,n,txt,c=G):
    ax.text(x,y,f'{n}',ha='center',va='center',fontsize=12,color='white',fontweight='bold',bbox=dict(boxstyle='circle,pad=0.25',fc=c,ec=c))
    ax.text(x+1.8,y,txt,ha='left',va='center',fontsize=13.5,color=c,fontweight='bold')
def arrow(x1,y1,x2,y2,c=INK2,lw=2,st='-|>',ls='-',ms=16,rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=st,mutation_scale=ms,color=c,lw=lw,linestyle=ls,connectionstyle=f'arc3,rad={rad}'))
# ---------------- Panel A: one generation, many attempts
ax.text(3,134.5,'One generation call, many attempts',fontsize=22,fontweight='bold',color=INK)
ax.text(3,130.8,'Everything below is one stream of tokens from the frozen model; the green pieces are written by implants, not by the model.',fontsize=14,color=INK2)
y=116; h=8
box(3,y,17,h,Q,txt='Question',fs=15,bold=True)
box(21,y,25,h,'white',GR,txt='attempt 1 … #### 42',fs=14)
box(47,y,14,h,G,txt='Wait, let me\nsolve it again.',fs=11.5,c='white')
box(62,y,25,h,'white',GR,txt='attempt 2 … #### 45',fs=14)
box(88,y,14,h,G,txt='Wait, let me\nsolve it again.',fs=11.5,c='white')
ax.text(105.5,y+h/2,'· · ·',fontsize=20,color=INK2,ha='center',va='center')
box(109,y,25,h,'white',GR,txt='attempt k … #### 42',fs=14)
box(135,y,22,h,GD,txt='Final answer: #### 42',fs=14,c='white',bold=True)
tag(48.5,112.2,4,'retry writer');tag(136.5,112.2,7,'voter writes the answer',c=GD)
# register strip
ry=96
ax.text(21,ry-3.2,'answer register: side state, updated by fixed weights as the tokens go by',fontsize=13,color=INK2,style='italic')
cells=[('attempt 1','42','0.81'),('attempt 2','45','0.34'),('attempt 3','42','0.77'),('attempt 4','42','0.69')]
for i,(a,v,p) in enumerate(cells):
    x=21+i*22; box(x,ry,20,8.2,PALE,G,f'{a}:  {v}   p={p}',fs=13,lw=1.2)
for xs in (33.5,74.5): arrow(xs,y-0.3,xs,ry+8.6,c=G,lw=1.6,ms=12)
tag(112,ry+5.6,7,'register stores each answer,',c=G); ax.text(113.8,ry+1.8,'its flags and a confidence p\n(a fixed probe on layer 18)',fontsize=13,color=G,va='center',linespacing=1.15)
# ---------------- Panel B: attention inside a later attempt
by=40; ax.text(3,86,'Inside attention, for a token in attempt 3',fontsize=22,fontweight='bold',color=INK)
bh=8
box(3,by+16,17,bh,Q,txt='Question',fs=15,bold=True)
box(22,by+16,22,bh,PALER,'#e3b3b0',txt='attempt 1',fs=14,c='#8a3a33')
box(46,by+16,22,bh,PALER,'#e3b3b0',txt='attempt 2',fs=14,c='#8a3a33')
box(70,by+16,30,bh,'white',G,txt='attempt 3 (current token)',fs=14,lw=2)
# arrows from current token (above the boxes)
top=by+16+bh
arrow(90,top,11.5,top+0.3,c=G,lw=3,rad=0.42)
arrow(86,top,33,top+0.3,c=RED,lw=2,ls='--',rad=0.42); arrow(82,top,57,top+0.3,c=RED,lw=2,ls='--',rad=0.45)
ax.text(33,top+2.6,'×',fontsize=30,color=RED,ha='center',va='center',fontweight='bold'); ax.text(57,top+2.6,'×',fontsize=30,color=RED,ha='center',va='center',fontweight='bold')
ax.text(45,top+17.5,'reads the question as if it were right before it',fontsize=13,color=G,ha='center',va='center',fontweight='bold')
ax.text(45,by+11.5,'earlier attempts: hidden',fontsize=13,color=RED,ha='center',va='center',fontweight='bold')
tag(106,by+21.5,5,'forget-mask')
ax.text(106,by+17.6,'4 extra fixed q/k dims per head: keys of earlier attempts\nget −10,000 added to their score, so softmax gives them\n≈0 attention. The question and the current attempt get +0.',fontsize=12.5,color=INK,va='top',linespacing=1.25)
tag(106,by+5.5,6,'position reset')
ax.text(106,by+1.6,'An offset register = how far this attempt starts after the\nquestion. A fixed cos/sin table rotates q and k back by\nthat offset in every layer: the question looks adjacent.',fontsize=12.5,color=INK,va='top',linespacing=1.25)
# ---------------- Panel C: voter + adaptive tries
cy=-2; ax.text(3,26,'How the voter and adaptive tries decide (compiled ReLU comparators, no code)',fontsize=22,fontweight='bold',color=INK)
box(3,cy+4,40,20,'white',GR,lw=1.5)
ax.text(5,cy+20.5,'Group equal answers, add up confidence',fontsize=14,fontweight='bold',color=INK)
ax.text(5,cy+16.2,'42 :  0.81 + 0.77 + 0.69 = 2.27',fontsize=14,color=G,fontweight='bold')
ax.text(5,cy+12.4,'45 :  0.34',fontsize=14,color=INK2)
ax.text(5,cy+8.4,'impossible answers (negative, not\nan integer) are dropped first',fontsize=12,color=INK2,va='center',linespacing=1.2)
arrow(43.5,cy+14,52,cy+14,lw=2.5)
rules=[('all 4 agree and confident?','stop: write #### answer',GD),('winner has < 70% of the confidence?','retry writer fires: +4 attempts (up to 16)',G),('otherwise','write the winner',INK2)]
for i,(c1,c2,col) in enumerate(rules):
    yy=cy+18.5-i*7.3
    box(52,yy-2.9,44,5.8,PALE if col!=INK2 else Q,txt=c1,fs=13.5)
    arrow(96.5,yy,103,yy,lw=2,c=col)
    box(103,yy-2.9,54,5.8,'white',col,txt=c2,fs=13.5,c=col,bold=True,lw=1.6)
tag(54,cy-1.6,8,'adaptive tries (plus counters: at most 3 repairs, 400 tokens per attempt)')
fig.savefig('serial_implants.png',dpi=100); fig.savefig('serial_implants.svg')
print('ok')
