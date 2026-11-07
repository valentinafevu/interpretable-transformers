"""
Agrega las métricas de evaluate_checkpoints.py por variante (y semilla) y compara contra `constrained`.
Semilla 0 = los modelos originales (vit-15-AwA2, vit-15-constrained-AwA2, vit-15-constrained_rank-AwA2);
semillas 1, 2 = corridas de run_variant.py (vit-15-<variante>-s<semilla>-AwA2).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "experiments" / "eval_multi"
SEED0 = {"unconstrained": "vit-15-AwA2", "constrained": "vit-15-constrained-AwA2",
         "constrained_rank": "vit-15-constrained_rank-AwA2"}

rows = []
for f in sorted(OUT.glob("*.json")):
    r = json.loads(f.read_text())
    name = r["name"]
    m = re.match(r"vit-15-(.+)-s(\d+)-AwA2$", name)
    if m:
        variant, seed = m.group(1), int(m.group(2))
    else:
        inv = {v: k for k, v in SEED0.items()}
        if name not in inv:
            continue
        variant, seed = inv[name], 0
    rows.append({"variant": variant, "seed": seed, **{k: v for k, v in r.items() if not isinstance(v, list)}})
df = pd.DataFrame(rows).sort_values(["variant", "seed"])

KEY = ["accuracy", "b4_attn_cos", "b4_out_cka_cls", "b4_out_cka_tok", "b4_out_erank_cls_mean", "b4_out_erank_cls_concat",
       "b4_key_w_erank_mean", "b4_value_w_erank_mean", "probe_head_r2_mean", "probe_concat_r2",
       "probe_concat_r2_unseen", "probe_r2vec_pearson_mean", "resid_b1_r2_unseen", "resid_b2_r2_unseen",
       "resid_b3_r2_unseen", "resid_b4_r2_unseen", "abl_drop_b4_mean", "abl_drop_b4_max", "abl_b4_gini",
       "dmg_cosine_mean", "dmg_specialization_index", "dmg_split_same"]
KEY = [k for k in KEY if k in df.columns]

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print("n por variante:", df.groupby("variant").size().to_dict())
    g = df.groupby("variant")[KEY]
    mean, sd = g.mean(), g.std(ddof=1)
    pref = ["unconstrained", "constrained", "constrained_rank", "constrained_xcka", "constrained_rank_xcka",
            "constrained_l05", "constrained_rank_l05"]
    order = [v for v in pref if v in mean.index] + sorted(v for v in mean.index if v not in pref)
    out = pd.DataFrame({k: [f"{mean.loc[v, k]:.3f}±{(0 if np.isnan(sd.loc[v, k]) else sd.loc[v, k]):.3f}" for v in order]
                        for k in KEY}, index=order)
    print(out.T.to_string())
    out.T.to_csv(OUT / "variants_summary.csv")
    df.to_csv(OUT / "per_run.csv", index=False)
    print()
    print("Por semilla (valores individuales):")
    print(df[["variant", "seed"] + KEY].round(3).to_string(index=False))
