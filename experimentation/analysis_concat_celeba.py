"""
Dos preguntas sobre los checkpoints de CelebA, motivadas por "Neural Networks Are Already Clean" (Cammarata, Goodfire):

 (a) ¿Dónde aparece el solapamiento entre cabezas: ANTES de la concatenación + W_O (salidas z_h de cada cabeza) o
     DESPUÉS (lo que cada cabeza escribe en el flujo residual, c_h = W_O[:, h-ésimo bloque] @ z_h)?
       - CKA y rango efectivo entre cabezas, antes y después de W_O (token CLS, test completo).
       - Solapamiento entre los subespacios de escritura de W_O (0 = ortogonales; azar = head_size / hidden_size).
       - Razón de varianza: Var(sum_h c_h) / sum_h Var(c_h)  (1 = escrituras no correlacionadas; <1 se cancelan; >1 se refuerzan).
 (b) ¿Los atributos que codifica cada cabeza forman grupos con sentido (pelo, vello facial, accesorios...) o están
     repartidos al azar? Se usa el R² del probing lineal de cada atributo desde cada cabeza (bloque 4) y se compara
     contra una permutación de los grupos.

    python analysis_concat_celeba.py vit-5-unconstrained-s0-CelebA vit-5-constrained-s0-CelebA ...

Guarda experiments/eval_multi_celeba/concat_<nombre>.json (+ .npz con la matriz R² cabeza x atributo).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from modules.cka import linear_cka  # noqa: E402
from modules.datasets import CelebADataset  # noqa: E402
from modules.effective_rank import effective_rank  # noqa: E402
from modules.model import ViTForClassfication  # noqa: E402

EXP_DIR = HERE / "experiments"
OUT_DIR = EXP_DIR / "eval_multi_celeba"
OUT_DIR.mkdir(exist_ok=True)
IMAGE_SIZE = 64
BLOCK = 3          # bloque 4 (0-indexed) para el probing
N_PERM = 5000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Grupos semánticos (a priori, antes de mirar resultados). Atributos sin grupo claro quedan fuera.
GROUPS = {
    "color_pelo": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
    "forma_pelo": ["Bangs", "Straight_Hair", "Wavy_Hair", "Receding_Hairline", "Bald"],
    "vello_facial": ["5_o_Clock_Shadow", "Goatee", "Mustache", "Sideburns", "No_Beard"],
    "boca_expresion": ["Smiling", "Mouth_Slightly_Open", "High_Cheekbones", "Big_Lips"],
    "maquillaje": ["Heavy_Makeup", "Wearing_Lipstick", "Rosy_Cheeks"],
    "accesorios": ["Eyeglasses", "Wearing_Hat", "Wearing_Earrings", "Wearing_Necklace", "Wearing_Necktie"],
    "rasgos_cara": ["Oval_Face", "Chubby", "Double_Chin", "Pointy_Nose", "Big_Nose", "Bags_Under_Eyes",
                    "Narrow_Eyes", "Bushy_Eyebrows", "Arched_Eyebrows"],
}

_dataset = CelebADataset(image_size=IMAGE_SIZE, root=str(ROOT / "data" / "celeba"))
_dataset.get_loaders(batch_size=256, num_workers=0)
_test = _dataset.testset
attr_names = list(_dataset.classes)
Y = _test.attr.numpy().astype(np.float32)
# num_workers=0: con >0 el script se cuelga en macOS (los hijos reimportan este módulo).
_loader = torch.utils.data.DataLoader(_test, batch_size=256, shuffle=False, num_workers=0)
images = torch.cat([x for x, _ in _loader])
N = len(images)


def load(name):
    d = EXP_DIR / name
    config = json.loads((d / "config.json").read_text())
    m = ViTForClassfication(config)
    m.load_state_dict(torch.load(d / "model_final.pt", map_location="cpu"))
    return m.to(device).eval(), config


@torch.no_grad()
def collect_cls_head_outputs(model, n_blocks, n_heads):
    """z[b][h]: (N, head_size), salida de la cabeza h del bloque b en el token CLS (antes de concatenar)."""
    out = [[[] for _ in range(n_heads)] for _ in range(n_blocks)]
    hooks = []
    for b, block in enumerate(model.encoder.blocks):
        for h, head in enumerate(block.attention.heads):
            hooks.append(head.register_forward_hook(lambda _m, _i, o, b=b, h=h: out[b][h].append(o[0][:, 0].cpu())))
    for i0 in range(0, N, 512):
        model(images[i0:i0 + 512].to(device), output_attentions=False)
    for hk in hooks:
        hk.remove()
    return [[torch.cat(out[b][h]) for h in range(n_heads)] for b in range(n_blocks)]


def pair_mean(fn, n):
    return float(np.mean([fn(i, j) for i in range(n) for j in range(i + 1, n)]))


def probe_r2(X, Yt, folds):
    pred = np.zeros_like(Yt)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=10.0).fit(sc.transform(X[tr]), Yt[tr]).predict(sc.transform(X[te]))
    return 1 - ((Yt - pred) ** 2).sum(0) / np.maximum(((Yt - Yt.mean(0)) ** 2).sum(0), 1e-12)


def group_statistics(r2):
    """r2: (n_heads, 40). Mide si los atributos de un mismo grupo prefieren la misma cabeza.

    Perfil de un atributo = su R² por cabeza, normalizado a suma 1 y centrado (preferencia relativa por cabeza).
    - same_argmax: fracción de pares de atributos del mismo grupo cuya cabeza preferida coincide.
    - within_cos: coseno medio entre perfiles de atributos del mismo grupo.
    Se compara contra permutaciones de la asignación de grupos (misma estructura de tamaños).
    """
    n_heads = r2.shape[0]
    pos = np.clip(r2, 0, None) + 1e-9
    prof = pos / pos.sum(0, keepdims=True) - 1.0 / n_heads               # (H, 40)
    prof = (prof / np.maximum(np.linalg.norm(prof, axis=0, keepdims=True), 1e-12)).T   # (40, H)
    best = r2.argmax(0)
    idx_groups = [[attr_names.index(a) for a in g] for g in GROUPS.values()]
    flat = np.array([i for g in idx_groups for i in g])
    sizes = [len(g) for g in idx_groups]

    def stats(order):
        same, cos, k = [], [], 0
        for s in sizes:
            g = order[k:k + s]; k += s
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    same.append(best[g[a]] == best[g[b]]); cos.append(prof[g[a]] @ prof[g[b]])
        return float(np.mean(same)), float(np.mean(cos))

    obs_same, obs_cos = stats(flat)
    rng = np.random.default_rng(0)
    null = np.array([stats(rng.permutation(flat)) for _ in range(N_PERM)])
    per_group = {}
    for name, g in zip(GROUPS, idx_groups):
        per_group[name] = {"head_counts": np.bincount(best[g], minlength=n_heads).tolist(),
                           "r2_by_head": r2[:, g].mean(1).tolist()}
    return {"same_argmax": obs_same, "same_argmax_null_mean": float(null[:, 0].mean()),
            "same_argmax_p": float((null[:, 0] >= obs_same).mean()),
            "within_cos": obs_cos, "within_cos_null_mean": float(null[:, 1].mean()),
            "within_cos_p": float((null[:, 1] >= obs_cos).mean()),
            "head_of_best_per_attr": best.tolist(), "per_group": per_group}


def analyze(name):
    cache = OUT_DIR / f"concat_{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    model, config = load(name)
    n_blocks, n_heads = config["num_hidden_layers"], config["num_attention_heads"]
    hs = config["hidden_size"] // n_heads
    z = collect_cls_head_outputs(model, n_blocks, n_heads)
    res = {"name": name, "chance_subspace_overlap": hs / config["hidden_size"], "blocks": []}

    for b in range(n_blocks):
        W = model.encoder.blocks[b].attention.output_projection.weight.detach().cpu()       # (hidden, hidden)
        Wh = [W[:, h * hs:(h + 1) * hs] for h in range(n_heads)]                              # (hidden, hs)
        c = [z[b][h] @ Wh[h].T for h in range(n_heads)]                                       # (N, hidden): escritura de la cabeza
        U = [torch.linalg.svd(w, full_matrices=False)[0] for w in Wh]                         # base ortonormal del subespacio
        sv = torch.linalg.svdvals(W)
        total = torch.stack(c).sum(0)
        var = lambda t: (t - t.mean(0)).pow(2).sum().item()
        res["blocks"].append({
            "cka_z": pair_mean(lambda i, j: linear_cka(z[b][i], z[b][j]).item(), n_heads),
            "cka_c": pair_mean(lambda i, j: linear_cka(c[i], c[j]).item(), n_heads),
            "erank_z_mean": float(np.mean([effective_rank(z[b][h]).item() for h in range(n_heads)])),
            "erank_z_concat": float(effective_rank(torch.cat(z[b], 1)).item()),
            "erank_c_sum": float(effective_rank(total).item()),
            "subspace_overlap": pair_mean(lambda i, j: (U[i].T @ U[j]).pow(2).sum().item() / hs, n_heads),
            "Wo_erank": float(effective_rank(W).item()),
            "Wo_cond": float((sv.max() / sv.min()).item()),
            "var_ratio": var(total) / sum(var(t) for t in c),
        })

    # --- probing por cabeza y atributo (bloque 4, token CLS, salida de la cabeza antes de W_O)
    folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(N)))
    r2 = np.stack([probe_r2(z[BLOCK][h].numpy(), Y, folds) for h in range(n_heads)])          # (H, 40)
    res["probe"] = group_statistics(r2)
    res["probe_r2_mean_per_head"] = r2.mean(1).tolist()
    np.savez_compressed(OUT_DIR / f"concat_{name}.npz", r2=r2)
    cache.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        r = analyze(nm)
        b4 = r["blocks"][BLOCK]
        print(f"{nm}: cka_z={b4['cka_z']:.3f} cka_c={b4['cka_c']:.3f} overlap={b4['subspace_overlap']:.3f} "
              f"(azar {r['chance_subspace_overlap']:.2f}) var_ratio={b4['var_ratio']:.2f} | "
              f"grupos: same_argmax={r['probe']['same_argmax']:.2f} (nulo {r['probe']['same_argmax_null_mean']:.2f}, "
              f"p={r['probe']['same_argmax_p']:.3f})", flush=True)
