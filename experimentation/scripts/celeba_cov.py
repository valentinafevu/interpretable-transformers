exec(open("celeba_regions.py").read().split("rows=[]")[0])
import numpy as np, pandas as pd
names=["ojos","nariz","boca","fuera de la cara"]
an=np.array(["5_o_Clock_Shadow"])  # placeholder
rows=[]; gl={}
D=np.load("/home/claude/work/interpretable-transformers/data/celeba/celeba_test64.npz",allow_pickle=True)
an=list(D["attr_names"]); ia={k:an.index(k) for k in ["Eyeglasses","Wearing_Hat","Mouth_Slightly_Open","Smiling","Wearing_Lipstick"]}
for v in ["unconstrained","constrained","constrained_rank"]:
  for s in range(3):
    a=np.load(f"att_{v}_s{s}.npy"); a=a/a.sum(-1,keepdims=True)
    cov=set(); conc=0; best=[]
    for h in range(4):
      mass={k:(a[:,h]*R[k]).sum(1).mean() for k in names}
      eff=np.exp(-(a[:,h]*np.log(np.clip(a[:,h],1e-8,None))).sum(1)).mean()
      top=max(mass,key=mass.get)
      if mass[top]>0.5 and eff<12: cov.add(top); conc+=1
    # atributos: masa en ojos con/sin gafas (mejor cabeza), sombrero->frente, boca abierta->boca
    def delta(attrname,reg):
      m=(a*R[reg][:,None,:]).sum(-1)   # N,H
      y=D["attr"][:,ia[attrname]]==1
      return float(np.max(np.abs(m[y].mean(0)-m[~y].mean(0)))) , int(np.argmax(np.abs(m[y].mean(0)-m[~y].mean(0)))+1)
    rows.append({"modelo":v,"seed":s,"cabezas_concentradas_en_UNA_región":conc,"regiones distintas cubiertas":len(cov),"cubre":",".join(sorted(cov)),
                 "Δmasa ojos gafas(sí−no) máx":delta("Eyeglasses","ojos")[0],"Δmasa frente/pelo sombrero máx":delta("Wearing_Hat","frente/pelo (arriba de ojos)")[0],"Δmasa boca boca-abierta máx":delta("Mouth_Slightly_Open","boca")[0]})
df=pd.DataFrame(rows);df.to_csv("celeba_coverage.csv",index=False)
pd.set_option("display.width",250);print(df.round(3).to_string(index=False))
print(df.groupby("modelo")[["cabezas_concentradas_en_UNA_región","regiones distintas cubiertas","Δmasa ojos gafas(sí−no) máx","Δmasa frente/pelo sombrero máx","Δmasa boca boca-abierta máx"]].mean().round(3))
