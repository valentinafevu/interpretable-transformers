import json, numpy as np, torch, sys
from pathlib import Path
sys.path.append(".")
from modules.model import ViTForClassfication
d=np.load("data/celeba/celeba_test64.npz")
X8=torch.tensor(d["x"]).permute(0,3,1,2)
attr=torch.tensor(d["attr"]).float()
out={}
for v in ["unconstrained","constrained","constrained_rank"]:
  for s in range(3):
    name=f"vit-5-{v}-s{s}-CelebA"; p=Path("experimentation/experiments")/name
    m=ViTForClassfication(json.loads((p/"config.json").read_text())); m.load_state_dict(torch.load(p/"model_final.pt",map_location="cpu")); m.eval()
    atts=[];acc=[]
    with torch.no_grad():
      for i in range(0,len(X8),100):
        lo,att=m((X8[i:i+100].float()/255-0.5)/0.5,output_attentions=True)
        atts.append(att[3][:,:,0,1:].clone().numpy()); acc.append(((lo>0).float()==attr[i:i+100]).float().mean().item())
    a=np.concatenate(atts); np.save(f"/home/claude/work/att_{v}_s{s}.npy",a.astype(np.float32))
    print(name,"acc",round(float(np.mean(acc)),4),a.shape,flush=True)
