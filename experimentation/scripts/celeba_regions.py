import numpy as np, pandas as pd
d=np.load("/home/claude/work/interpretable-transformers/data/celeba/celeba_test64.npz")
lm=d["lm"]; attr=d["attr"]; N=len(lm)
g=(np.arange(16)*4+2).astype(np.float32); PX,PY=np.meshgrid(g,g)
PX=PX.ravel()[None];PY=PY.ravel()[None]
def disk(c,r): return ((PX-c[:,0:1])**2+(PY-c[:,1:2])**2)<=r*r
def seg(a,b,r):
    ab=b-a; t=(((PX-a[:,0:1])*ab[:,0:1]+(PY-a[:,1:2])*ab[:,1:2])/(ab**2).sum(1,keepdims=True)).clip(0,1)
    return ((PX-(a[:,0:1]+t*ab[:,0:1]))**2+(PY-(a[:,1:2]+t*ab[:,1:2]))**2)<=r*r
le,re,no,lmo,rmo=[lm[:,i] for i in range(5)]
R={"ojos":disk(le,5)|disk(re,5),"nariz":disk(no,5),"boca":seg(lmo,rmo,5)}
x0=lm[...,0].min(1)-6;x1=lm[...,0].max(1)+6;y0=lm[...,1].min(1)-6;y1=lm[...,1].max(1)+6
core=(PX>=x0[:,None])&(PX<=x1[:,None])&(PY>=y0[:,None])&(PY<=y1[:,None])
R["cara (caja landmarks)"]=core; R["fuera de la cara"]=~core
eyes_y=(le[:,1]+re[:,1])/2; R["frente/pelo (arriba de ojos)"]=PY<(eyes_y[:,None]-7)
R["abajo de la boca"]=PY>(((lmo[:,1]+rmo[:,1])/2)[:,None]+5)
area={k:v.mean(1) for k,v in R.items()}
rows=[];prof={}
for v in ["unconstrained","constrained","constrained_rank"]:
  for s in range(3):
    a=np.load(f"/home/claude/work/att_{v}_s{s}.npy"); a=a/a.sum(-1,keepdims=True)
    for h in range(4):
      r={"modelo":v,"seed":s,"cabeza":h+1}
      for k,m in R.items():
        mass=(a[:,h]*m).sum(1); r[k]=float(np.mean(mass/np.clip(area[k],1e-6,None)))   # enriquecimiento = masa/área
      r["parches_efectivos"]=float(np.exp(-(a[:,h]*np.log(np.clip(a[:,h],1e-8,None))).sum(1)).mean())
      rows.append(r)
df=pd.DataFrame(rows); df.to_csv("/home/claude/work/celeba_region_enrichment.csv",index=False)
pd.set_option("display.width",220)
print(df.round(2).to_string(index=False))
