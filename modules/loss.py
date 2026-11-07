"""
Términos de penalización usados en el entrenamiento interpretable.

- `interpretability_loss`: producto punto (coseno) entre los mapas de atención del CLS de cada
  par de cabezas — penaliza redundancia entre cabezas.
- `head_output_collapse_loss`: anti-colapso por cabeza sobre la SALIDA de cada cabeza (piso de
  varianza + piso de rango efectivo). Evita que la ortogonalidad entre cabezas se consiga
  achicando o colapsando cabezas.
- `head_output_cka_loss`: similitud CKA lineal media entre las salidas de pares de cabezas
  (independencia estadística sobre el batch, invariante a escala).
- `attention_rank_floor_loss`: piso de rango efectivo (Roy & Vetterli, 2007) sobre la matriz
  (cabezas × parches) de atención del CLS de cada imagen — el mismo objeto sobre el que actúa
  `interpretability_loss`. Mide cuántas cabezas son linealmente independientes (máx. = nº de cabezas),
  algo que el producto punto por pares no garantiza.
- `effective_rank_floor_loss`: variante sobre los pesos Q/K de cada cabeza (alternativa; en las
  corridas penalizadas el rango de K de algunas cabezas cae a ~1.4).
- `attention_entropy_loss`: entropía (normalizada) del mapa de atención del CLS de cada cabeza —
  penaliza que una cabeza mire a muchos parches (localización), SIN exigir que las cabezas miren
  sitios distintos entre sí (a diferencia de `interpretability_loss`).
"""
import math

import torch
from torch.nn import functional as F

from .cka import linear_cka
from .effective_rank import effective_rank


def interpretability_loss(attention_maps):
    # attention_maps: (B, H, P) — atención del CLS hacia los parches por cada cabeza

    B, H, P = attention_maps.shape

    # Normalización L2 antes del producto punto entre cabezas.
    # Así la similitud queda en [-1, 1] (similitud coseno) y no depende de la magnitud
    # absoluta del vector de atención — evita que el modelo reduzca la loss
    # simplemente achicando los valores de atención.
    attention_maps = F.normalize(attention_maps, p=2, dim=-1)

    # Producto punto entre todos los pares de cabezas (i, j) posición a posición
    mm = torch.einsum('bhp,bkp->bhkp', attention_maps, attention_maps)

    # Máscara [H, H] → True en diagonal (no comparar cabeza consigo misma)
    diag_mask = torch.eye(H, dtype=torch.bool, device=attention_maps.device)

    # Anular la diagonal para que no contribuya a la loss
    mm_masked = mm.clone()
    mm_masked[:, diag_mask, :] = float(0)

    mm_sum = mm_masked.sum(dim=(1, 2))   # (B, P)
    mm_total = mm_sum.sum(dim=1)          # (B,)  una pérdida por imagen

    # Reducir a escalar para que .backward() funcione.
    # mean() mantiene la magnitud del gradiente independiente del batch size.
    return mm_total.mean()


def effective_rank_floor_loss(model, target_rank, params=("query", "key"), block_idx=-1):
    """
    Piso de rango efectivo por cabeza sobre los pesos de `params` (Q y/o K) del bloque `block_idx`
    (por defecto el último, el mismo que usa la penalización de atención y el que se loguea en MLflow).

    Por cabeza: relu(target_rank - erank) / target_rank, en [0, 1). Es 0 cuando la cabeza ya tiene
    rango efectivo >= target_rank, así que solo actúa sobre cabezas que colapsan. Con
    target_rank = head_size equivale a maximizar el rango efectivo sin tope.

    Devuelve el promedio sobre cabezas y sobre `params`. Es diferenciable respecto de los pesos.
    """
    heads = model.encoder.blocks[block_idx].attention.heads
    terms = []
    for name in params:
        weights = torch.stack([getattr(head, name).weight for head in heads])   # (H, head_size, hidden)
        max_rank = min(weights.shape[-2:])
        assert target_rank <= max_rank, f"target_rank={target_rank} > rango máximo posible {max_rank}"
        erank = effective_rank(weights)                                         # (H,)
        terms.append(F.relu(target_rank - erank) / target_rank)
    return torch.stack(terms).mean()


def attention_rank_floor_loss(attention_maps, target_rank):
    """
    Piso de rango efectivo sobre la matriz (H, P) de atención del CLS hacia los parches, por imagen.
    attention_maps: (B, H, P), el mismo tensor que recibe `interpretability_loss`.

    Por imagen: relu(target_rank - erank) / target_rank. Vale 0 si las cabezas ya son al menos
    `target_rank` "cabezas efectivamente distintas"; con target_rank = H equivale a maximizar el
    rango efectivo sin tope. Se promedia sobre el batch. Es diferenciable.
    """
    max_rank = min(attention_maps.shape[-2:])
    assert target_rank <= max_rank, f"target_rank={target_rank} > rango máximo posible {max_rank}"
    erank = effective_rank(attention_maps)                    # (B,)
    return (F.relu(target_rank - erank) / target_rank).mean()


def head_output_collapse_loss(head_outputs, target_rank, var_floor=0.5, rank_ceiling=False):
    """
    Anti-colapso por cabeza sobre su salida. head_outputs: (N, H, d), N = muestras (p.ej. el CLS de
    cada imagen del batch), H = cabezas, d = head_size.

    Se trabaja con la DIRECCIÓN de la salida (normalizada L2 por muestra) y centrada sobre el batch,
    así ambos términos son invariantes a la escala (que en un ViT pre-LN es arbitraria y se podría
    "arreglar" agrandando pesos):

    - Piso de varianza: std por dimensión >= gamma = var_floor / sqrt(d). Para direcciones
      isotrópicas en d dimensiones la std es 1/sqrt(d), así que var_floor es la fracción de esa
      variación mínima exigida. Ataca el colapso constante (la cabeza da lo mismo para toda
      imagen) y la cabeza muerta (norma ~0).
    - Piso de rango efectivo por cabeza: erank >= target_rank sobre la matriz centrada (N, d).
      Ataca el colapso dimensional (la cabeza usa pocas direcciones). Se ve la forma del
      espectro; la escala la cubre el término de varianza.

    rank_ceiling=True invierte SOLO el término de rango: en vez de un piso pasa a ser un techo,
    erank <= target_rank (penaliza que la cabeza use más de target_rank direcciones), normalizado por
    (d - target_rank). El piso de varianza se mantiene para que la cabeza no muera.

    Ambos términos están normalizados a [0, 1) (hinge / umbral) y se suman. Devuelve
    (pérdida, stats) con stats por cabeza (sin gradiente) para loguear: erank y std media.
    """
    N, H, d = head_outputs.shape
    if N < 2:
        zero = head_outputs.sum() * 0.0
        return zero, {"erank": torch.zeros(H), "std": torch.zeros(H)}

    z = F.normalize(head_outputs, p=2, dim=-1, eps=1e-6)       # (N, H, d)
    z = z - z.mean(dim=0, keepdim=True)

    std = torch.sqrt(z.pow(2).mean(dim=0) + 1e-8)               # (H, d)
    gamma = var_floor / math.sqrt(d)
    var_term = (F.relu(gamma - std) / gamma).mean()

    erank = effective_rank(z.transpose(0, 1))                   # (H, N, d) -> (H,)
    target = min(target_rank, N, d)                             # el último batch puede traer pocas muestras
    if rank_ceiling:
        rank_term = (F.relu(erank - target) / max(d - target, 1)).mean()
    else:
        rank_term = (F.relu(target - erank) / target).mean()

    return var_term + rank_term, {"erank": erank.detach().cpu(), "std": std.mean(dim=-1).detach().cpu()}


def head_output_cka_loss(head_outputs):
    """
    CKA lineal media entre las salidas de los pares de cabezas. head_outputs: (N, H, d).
    A diferencia del producto punto entre mapas de atención (por muestra), mide dependencia
    estadística sobre el batch: 0 = salidas independientes (linealmente), 1 = idénticas salvo
    rotación/escala. Es diferenciable y está en [0, 1].
    """
    N, H, d = head_outputs.shape
    if N < 3 or H < 2:
        return head_outputs.sum() * 0.0
    z = F.normalize(head_outputs, p=2, dim=-1, eps=1e-6).transpose(0, 1)     # (H, N, d)
    cka = linear_cka(z.unsqueeze(1), z.unsqueeze(0))                         # (H, H)
    off_diag = ~torch.eye(H, dtype=torch.bool, device=cka.device)
    return cka[off_diag].mean()


def attention_entropy_loss(attention_maps, eps=1e-8):
    """
    Entropía de Shannon normalizada del mapa de atención del CLS de cada cabeza.
    attention_maps: (B, H, P), el mismo tensor que recibe `interpretability_loss`.

    Cada mapa se renormaliza para que sume 1 sobre los P parches (la fila del CLS también reparte
    atención hacia sí mismo y hacia nadie más, que aquí se descarta), y luego:

        H_norm = -sum_i a_i log a_i / log(P)   en [0, 1]

    0 = la cabeza mira un único parche; 1 = atención uniforme sobre todos los parches. La
    normalización por log(P) hace que lambda signifique lo mismo con 64 parches (AwA2, 32 px) que con
    256 (CelebA, 64 px) y deja la pérdida en la misma escala [0, 1] que el coseno entre cabezas.

    Es una pérdida de MINIMIZACIÓN directa (sin objetivo K): el término de tarea hace de freno
    para que las cabezas no degeneren a atención dura sobre un parche; si lambda es grande eso sí
    puede pasar, por eso se loguea también `eff_patches = exp(H)` (nº efectivo de parches que mira
    la cabeza) como alarma de colapso.

    Devuelve (pérdida escalar = media sobre batch y cabezas, stats) con stats por cabeza sin gradiente:
      "entropy":     H_norm medio sobre el batch, (H,)
      "eff_patches": exp(H) medio sobre el batch, (H,)
    """
    B, H, P = attention_maps.shape
    a = attention_maps / attention_maps.sum(dim=-1, keepdim=True).clamp(min=eps)
    ent = -(a * torch.log(a.clamp(min=eps))).sum(dim=-1)            # (B, H), en nats
    ent_norm = ent / math.log(P)
    stats = {"entropy": ent_norm.detach().mean(dim=0).cpu(),
             "eff_patches": torch.exp(ent.detach()).mean(dim=0).cpu()}
    return ent_norm.mean(), stats
