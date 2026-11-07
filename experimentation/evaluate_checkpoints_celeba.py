"""
Batería de métricas núcleo (similitud/rango, ablación, probing) sobre checkpoints de CelebA (multi-etiqueta, 40
atributos), para comparar variantes y semillas. Usa el test completo (19962 imágenes).

    python evaluate_checkpoints_celeba.py vit-5-unconstrained-s0-CelebA vit-5-constrained-s0-CelebA ...

Diferencias respecto a evaluate_checkpoints.py (AwA2): las "clases" son los 40 atributos binarios; accuracy = acierto
medio por atributo (umbral logit > 0); el probing usa los atributos como objetivo (sin partición por clases no vistas,
CelebA no trae identidades en los archivos disponibles); el daño por ablación es un vector de 40 caídas de acierto.
Guarda experiments/eval_multi_celeba/<nombre>.json (+ .npz) y reutiliza el caché si existe.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_TOK_SUBSET = 1000          # imágenes para métricas sobre todos los tokens
BLOCK = 3                    # bloque penalizado (0-indexed): el último
N_SPLITS = 50                # particiones para el índice de especialización por clase

_dataset = CelebADataset(image_size=IMAGE_SIZE, root=str(ROOT / "data" / "celeba"))
_dataset.get_loaders(batch_size=256, num_workers=0)
_test = _dataset.testset
attr_names = list(_dataset.classes)
n_attr = len(attr_names)
Y = _test.attr.numpy().astype(np.float32)                           # (N, 40) en {0, 1}
_loader = torch.utils.data.DataLoader(_test, batch_size=256, shuffle=False, num_workers=0)
images = torch.cat([x for x, _ in _loader])                         # (N, 3, H, W), ya transformadas
N = len(images)


def load(name):
    d = EXP_DIR / name
    config = json.loads((d / "config.json").read_text())
    m = ViTForClassfication(config)
    m.load_state_dict(torch.load(d / "model_final.pt", map_location="cpu"))
    return m.to(device).eval(), config, json.loads((d / "metrics.json").read_text())


@torch.no_grad()
def collect(model, n_blocks, n_heads, head_size):
    """Una pasada: salidas de cabeza (CLS y subset todos-los-tokens), atención CLS->parches, residual CLS."""
    cls_out = [[[] for _ in range(n_heads)] for _ in range(n_blocks)]
    tok_out = [[[] for _ in range(n_heads)] for _ in range(n_blocks)]
    cls_att = [[[] for _ in range(n_heads)] for _ in range(n_blocks)]
    resid = [[] for _ in range(n_blocks)]
    cur = {"i0": 0}
    hooks = []
    for b, block in enumerate(model.encoder.blocks):
        for h, head in enumerate(block.attention.heads):
            def hook(_m, _i, out, b=b, h=h):
                cls_out[b][h].append(out[0][:, 0].cpu())
                cls_att[b][h].append(out[1][:, 0, 1:].cpu())
                if cur["i0"] < N_TOK_SUBSET:
                    tok_out[b][h].append(out[0][: max(0, N_TOK_SUBSET - cur["i0"])].cpu())
            hooks.append(head.register_forward_hook(hook))
        hooks.append(block.register_forward_hook(lambda _m, _i, out, b=b: resid[b].append(out[0][:, 0].cpu())))
    logits = []
    for i0 in range(0, N, 512):
        cur["i0"] = i0
        logits.append(model(images[i0:i0 + 512].to(device), output_attentions=False)[0].cpu())
    for hk in hooks:
        hk.remove()
    cat = lambda xs: torch.cat(xs)
    return (torch.cat(logits),
            [[cat(cls_out[b][h]) for h in range(n_heads)] for b in range(n_blocks)],
            [[cat(tok_out[b][h]) for h in range(n_heads)] for b in range(n_blocks)],
            [[cat(cls_att[b][h]) for h in range(n_heads)] for b in range(n_blocks)],
            [cat(resid[b]) for b in range(n_blocks)])


def pair_mean(fn, n_heads):
    return float(np.mean([fn(i, j) for i in range(n_heads) for j in range(i + 1, n_heads)]))


def probe_r2(X, Y, folds):
    pred = np.zeros_like(Y)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        r = Ridge(alpha=10.0).fit(sc.transform(X[tr]), Y[tr])
        pred[te] = r.predict(sc.transform(X[te]))
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


@torch.no_grad()
def ablation_preds(model, n_blocks, n_heads):
    """preds[0] = sin ablar; preds[1 + b * n_heads + h] = cabeza (b, h) apagada. (n_blocks*n_heads + 1, N, 40) bool"""
    state = {"zero": None}
    hooks = []
    for b, block in enumerate(model.encoder.blocks):
        for h, head in enumerate(block.attention.heads):
            def hook(_m, _i, out, b=b, h=h):
                if state["zero"] == (b, h):
                    return (torch.zeros_like(out[0]), out[1])
            hooks.append(head.register_forward_hook(hook))
    out = []
    for target in [None] + [(b, h) for b in range(n_blocks) for h in range(n_heads)]:
        state["zero"] = target
        p = []
        for i0 in range(0, N, 1024):
            p.append(model(images[i0:i0 + 1024].to(device), output_attentions=False)[0].gt(0).cpu())
        out.append(torch.cat(p).numpy())
    for hk in hooks:
        hk.remove()
    return np.stack(out).astype(np.int8)


def damage_vectors(preds, block, n_heads, idx=None):
    idx = np.arange(N) if idx is None else idx
    tgt = (Y[idx] > 0.5)
    def per_class_acc(p):          # acierto por atributo (40,)
        return (p[idx].astype(bool) == tgt).mean(0)
    base = per_class_acc(preds[0])
    return np.stack([base - per_class_acc(preds[1 + block * n_heads + h]) for h in range(n_heads)])


def evaluate(name):
    cache = OUT_DIR / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    model, config, metrics = load(name)
    n_blocks, n_heads = config["num_hidden_layers"], config["num_attention_heads"]
    head_size = config["hidden_size"] // n_heads
    res = {"name": name, "accuracy": float(metrics["accuracy"])}

    logits, cls_out, tok_out, cls_att, resid = collect(model, n_blocks, n_heads, head_size)
    res["accuracy_recomputed"] = float(((logits.numpy() > 0) == (Y > 0.5)).mean())

    # --- similitud entre cabezas, por bloque
    for b in range(n_blocks):
        res[f"b{b+1}_attn_cos"] = pair_mean(lambda i, j: torch.nn.functional.cosine_similarity(
            cls_att[b][i], cls_att[b][j], dim=-1).mean().item(), n_heads)
        res[f"b{b+1}_out_cka_cls"] = pair_mean(lambda i, j: linear_cka(cls_out[b][i], cls_out[b][j]).item(), n_heads)
        res[f"b{b+1}_out_cka_tok"] = pair_mean(lambda i, j: linear_cka(
            tok_out[b][i].reshape(-1, head_size), tok_out[b][j].reshape(-1, head_size)).item(), n_heads)
        res[f"b{b+1}_out_erank_cls_mean"] = float(np.mean([effective_rank(cls_out[b][h]).item() for h in range(n_heads)]))
        res[f"b{b+1}_out_erank_cls_concat"] = float(effective_rank(torch.cat(cls_out[b], dim=1)).item())
    last = model.encoder.blocks[BLOCK].attention.heads
    for nm in ("query", "key", "value"):
        res[f"b4_{nm}_w_erank_mean"] = float(np.mean(
            [effective_rank(getattr(h, nm).weight.detach().cpu()).item() for h in last]))

    # --- probing de los 40 atributos (CLS, bloque 4): por cabeza, concat, y residual por bloque
    folds_img = list(KFold(5, shuffle=True, random_state=0).split(np.arange(N)))
    heads_r2 = np.stack([probe_r2(cls_out[BLOCK][h].numpy(), Y, folds_img) for h in range(n_heads)])
    concat = torch.cat(cls_out[BLOCK], dim=1).numpy()
    res["probe_head_r2_mean"] = float(heads_r2.mean())
    res["probe_head_r2_best_mean"] = float(heads_r2.max(0).mean())
    res["probe_head_r2_per_head"] = heads_r2.mean(1).tolist()
    res["probe_concat_r2"] = float(probe_r2(concat, Y, folds_img).mean())
    # solapamiento de la información entre cabezas: correlación entre vectores de R² por atributo
    cc = np.corrcoef(np.clip(heads_r2, 0, None))
    res["probe_r2vec_pearson_mean"] = float(cc[np.triu_indices(n_heads, 1)].mean())
    for b in range(n_blocks):
        r = resid[b].numpy()
        res[f"resid_b{b+1}_r2"] = float(probe_r2(r, Y, folds_img).mean())

    # --- ablación (test completo)
    preds = ablation_preds(model, n_blocks, n_heads)
    base_acc = float((preds[0].astype(bool) == (Y > 0.5)).mean())
    drops = np.array([[base_acc - float((preds[1 + b * n_heads + h].astype(bool) == (Y > 0.5)).mean())
                       for h in range(n_heads)] for b in range(n_blocks)])
    res["abl_acc"] = base_acc
    res["abl_drop_mean"] = float(drops.mean())
    res["abl_drop_max"] = float(drops.max())
    res["abl_drop_b4_mean"] = float(drops[BLOCK].mean())
    res["abl_drop_b4_max"] = float(drops[BLOCK].max())
    res["abl_drop_b4_per_head"] = drops[BLOCK].tolist()
    res["abl_drop_per_block_mean"] = drops.mean(1).tolist()
    # concentración de la importancia (Gini sobre caídas positivas del bloque 4)
    d = np.sort(np.clip(drops[BLOCK], 0, None)); k = len(d)
    res["abl_b4_gini"] = float((2 * np.arange(1, k + 1) - k - 1).dot(d) / (k * max(d.sum(), 1e-12)))

    # --- ablación por clase (bloque 4): similitud de vectores de daño y especialización con ruido de referencia
    dmg = damage_vectors(preds, BLOCK, n_heads)
    cs = np.corrcoef(np.nan_to_num(dmg))
    res["dmg_pearson_mean"] = float(cs[np.triu_indices(n_heads, 1)].mean())
    nrm = np.nan_to_num(dmg) / np.maximum(np.linalg.norm(np.nan_to_num(dmg), axis=1, keepdims=True), 1e-12)
    res["dmg_cosine_mean"] = float((nrm @ nrm.T)[np.triu_indices(n_heads, 1)].mean())
    rng = np.random.default_rng(0)
    same, cross = [], []
    for _ in range(N_SPLITS):
        perm = rng.permutation(N); A, B = perm[: N // 2], perm[N // 2:]
        dA, dB = np.nan_to_num(damage_vectors(preds, BLOCK, n_heads, A)), np.nan_to_num(damage_vectors(preds, BLOCK, n_heads, B))
        C = np.corrcoef(np.vstack([dA, dB]))[:n_heads, n_heads:]
        same.append(np.diag(C).mean()); cross.append(C[~np.eye(n_heads, dtype=bool)].mean())
    res["dmg_split_same"] = float(np.mean(same))          # fiabilidad: ¿el mismo vector aparece en ambas mitades?
    res["dmg_split_cross"] = float(np.mean(cross))
    res["dmg_specialization_index"] = float(np.mean(same) - np.mean(cross))

    np.savez_compressed(OUT_DIR / f"{name}.npz", abl_preds=preds, drops=drops, dmg=dmg)
    cache.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        r = evaluate(nm)
        print(f"{nm}: acc={r['accuracy']:.4f} probe_concat={r['probe_concat_r2']:.3f} "
              f"abl_b4_mean={r['abl_drop_b4_mean']:.4f} spec={r['dmg_specialization_index']:.3f}", flush=True)
