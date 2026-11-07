import torch


def effective_rank(A: torch.Tensor) -> torch.Tensor:
    """
    Effective rank de Roy & Vetterli (2007): erank(A) = exp(H(p)),
    donde p_k = sigma_k / ||sigma||_1 son los valores singulares de A
    normalizados y H es su entropía de Shannon (con la convención 0*log(0) = 0).

    Soporta batches: A puede tener shape (..., M, N), y el rango efectivo
    se calcula sobre las últimas dos dimensiones, devolviendo shape (...,).
    Cumple 1 <= erank(A) <= rank(A) <= min(M, N).

    Es diferenciable y estable cuando hay valores singulares exactamente 0 (matrices de rango
    deficiente, p.ej. cabezas con atención idéntica): la convención 0*log(0) = 0 se implementa
    con un clamp en lugar de torch.where, porque el gradiente de where(p > 0, log(p), 0) en p = 0
    es inf * 0 = NaN y envenenaría los pesos si el rango efectivo entra en la pérdida.
    """
    # torch.linalg.svdvals no está implementado en el backend MPS; se calcula
    # en CPU (y en float64, más estable con valores singulares repetidos) y el resultado
    # se devuelve en el device y dtype originales de A.
    singular_values = torch.linalg.svdvals(A.cpu().double())  # (..., Q)
    p = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    entropy = -(p * torch.log(p.clamp_min(1e-30))).sum(dim=-1)

    # Primero el dtype (en CPU) y después el device: un único .to(device, dtype) haría que el backward
    # convierta el gradiente a float64 estando en MPS, que no lo soporta.
    return torch.exp(entropy).to(A.dtype).to(A.device)
