from pathlib import Path


def save_fig(fig, stem, output_dir, tag=None, project_root=None):
    """Guarda `fig` como PNG en `output_dir`, con un nombre único y descriptivo
    (`stem` + `tag`, p.ej. "damage_heatmap_vit-with-15-epochs-AwA2.png"), e imprime
    el nombre de archivo usado — así queda claro en la celda con qué modelo/corrida
    corresponde cada imagen guardada. Devuelve el `Path` guardado.

    `project_root`, si se pasa, hace que se imprima la ruta relativa a él en vez de
    la ruta absoluta completa (más legible en el notebook).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{tag}.png" if tag else f"{stem}.png"
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    shown = path.relative_to(project_root) if project_root else path
    print(f"guardado: {shown}")
    return path
