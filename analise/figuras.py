"""Produce Figures 5.1, 5.2 and 5.3 of the dissertation from resultados/.

Usage:
    python analise/figuras.py
"""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
RAIZ=Path(__file__).resolve().parents[1]; OUT=RAIZ/'analise'/'saida'; OUT.mkdir(parents=True,exist_ok=True)
M=['Mean/Mode','k-NN','MICE','missForest','SoftImpute','ISCA-k']
LAB={'Mean/Mode':'Média/moda','ISCA-k':'ISCA-k'}
COL={'Mean/Mode':'#888888','k-NN':'#6a51a3','MICE':'#2171b5','missForest':'#dd8500','SoftImpute':'#b12a65','ISCA-k':'#00685d'}
MK={'Mean/Mode':'x','k-NN':'s','MICE':'^','missForest':'D','SoftImpute':'v','ISCA-k':'o'}
lab=lambda m: LAB.get(m,m)
def fmt(v,nd=1):
    s=f'{abs(v):.{nd}f}'.replace('.',',')
    return ('−' if v<0 else '+')+s
b=pd.read_csv(RAIZ/'resultados'/'estudo_principal'/'benchmark_repeticoes.csv')
m=b.groupby(['dataset','condition','method'])[['nrmse','pfc']].mean().reset_index()
# Fig 5.1
mc=m[m.condition.str.startswith('mcar')].copy(); mc['rate']=mc.condition.str[-2:].astype(int)
fig,axes=plt.subplots(1,2,figsize=(7.1,3.45))
for ax,met,title in zip(axes,['nrmse','pfc'],['Variáveis numéricas','Variáveis categóricas']):
    for meth in M:
        s=mc[mc.method==meth].groupby('rate')[met].mean()
        ax.plot(s.index,s.values,marker=MK[meth],color=COL[meth],label=lab(meth),lw=2.4 if meth=='ISCA-k' else 1.3,ms=5)
    ax.set_title(title,fontsize=12); ax.set_xticks([5,10,20,30,40,50]); ax.set_xlabel('Taxa nominal MCAR (%)')
    ax.set_ylabel(met.upper()); ax.grid(alpha=.25); ax.spines[['top','right']].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p,met=met: f'{v:.1f}' if met=='nrmse' else f'{v:.3f}'))
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=(0,.14,1,1)); fig.savefig(OUT/'figura_5_1.png',dpi=220,bbox_inches='tight'); plt.close(fig)
# Fig 5.2
ORDER=['Iris','Wine','Diabetes','Breast Cancer','Heart Disease','Abalone','California Housing','Penguins','Titanic','Auto MPG','Ionosphere','Heart Failure']
C=['mcar_05','mcar_10','mcar_20','mcar_30','mcar_40','mcar_50','blockwise_20','mar_20']
p=m.pivot_table(index=['dataset','condition'],columns='method',values='nrmse')
o=p.drop(columns='ISCA-k'); diff=(100*(p['ISCA-k']/o.min(axis=1)-1)).unstack().loc[ORDER,C]
fig,ax=plt.subplots(figsize=(7.35,4.62))
im=ax.imshow(diff.values,cmap='PuOr_r',vmin=-45,vmax=45,aspect='auto')
for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        v=diff.values[i,j]; r=round(v,1)
        ax.text(j,i,fmt(r),ha='center',va='center',fontsize=9.5,fontweight='bold' if v<0 else 'normal',color='white' if abs(v)>30 else '#1a1a1a')
ax.axvline(5.5,color='white',lw=3)
ax.set_xticks(range(8),['5%','10%','20%','30%','40%','50%','Bloco\n20%','MAR\n20%']); ax.set_yticks(range(12),ORDER)
ax.set_xlabel('MCAR uniforme nas seis primeiras colunas'); ax.tick_params(length=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.02); cb.set_label('Diferença de NRMSE (%)')
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: ('−' if v<0 else '')+f'{abs(v):.0f}'))
fig.tight_layout(); fig.savefig(OUT/'figura_5_2.png',dpi=220,bbox_inches='tight'); plt.close(fig)
# Fig 5.3
t=pd.read_csv(RAIZ/'resultados'/'tempos'/'tempos_repeticoes.csv')
md=t[t.condition=='mcar_20'].groupby(['dataset','method']).seconds.median().unstack()
fig,ax=plt.subplots(figsize=(7.35,4.2))
OFF=dict(zip(M,[-.3,-.2,-.1,0,.1,.2]))
for meth in M:
    ys=[i+OFF[meth] for i in range(12)]
    ax.scatter(md.loc[ORDER,meth],ys,marker=MK[meth],color=COL[meth],label=lab(meth),s=38 if meth!='Mean/Mode' else 32,lw=1.8 if meth=='Mean/Mode' else 0)
ax.set_yticks(range(12),ORDER); ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlim(8e-5,2.2e3)
ax.set_xlabel('Tempo mediano por imputação (segundos)'); ax.grid(axis='x',alpha=.3); ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False)
fig.tight_layout(); fig.savefig(OUT/'figura_5_3.png',dpi=220,bbox_inches='tight'); plt.close(fig)
