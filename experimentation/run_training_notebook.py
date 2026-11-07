"""
Ejecuta las celdas de ENTRENAMIENTO de training.ipynb (sin las de visualización) para CelebA o AwA2, con los parámetros
inyectados en memoria: training.ipynb no se modifica. Todo queda en el mlflow.db principal de experimentation/
(experimento "vit-<epochs>-epochs-<Dataset>", run "<variante>_s<semilla>") y en experiments/vit-<epochs>-<variante>-s<semilla>-<Dataset>/.

Hiperparámetros = los de las corridas AwA2 de MLflow (lr 0.01, batch 32, lambda_loss 0.1; constrained_rank: lambda_rank 0.1,
rank_target 6.0, var_floor 0.5, sin techo de rango). CelebA: 64 px, 5 épocas (semilla 0). AwA2: 32 px, 15 épocas.
La variante "entropy" (sin coseno entre cabezas) penaliza la entropía normalizada de la atención del CLS con --lambda-entropy;
el run se llama entropy_l<λ>_s<semilla> (λ=0.1 -> entropy_l01_s0).

    python run_training_notebook.py --name constrained_rank --seed 1                          # CelebA (como antes)
    python run_training_notebook.py --dataset AwA2 --name entropy --lambda-entropy 0.1 --seed 0
    python run_training_notebook.py --dataset AwA2 --name entropy --lambda-entropy 0.1 --seed 0 --smoke   # prueba rápida, fuera del mlflow.db principal

El notebook ejecutado (con salidas) se guarda en logs/executed_<variante>_s<semilla>.ipynb.
"""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("SMOKE_SCRATCH", "/private/tmp/claude-501/-Users-valefeve-Projects-interpretability-of-neural-networks/3ac1dabd-0e42-458f-984d-feb85d43d49e/scratchpad/smoke"))
VARIANTS = {   # nombre -> (is_constrained, penalize_rank)
    "unconstrained": (False, False),
    "constrained": (True, False),
    "constrained_rank": (True, True),
    "entropy": (False, False),   # solo penalización de entropía de la atención (usa --lambda-entropy)
}
USE_CELLS = [0, 1, 2, 4, 14, 15, 16, 17, 19]   # imports, device, dataset, índices, config+MLflow, Trainer, modelo, entrenamiento, mlflow.end_run
SAVE_MARKER = 'model_path = f"vit_final_'      # celda que guarda el .pt; se busca por contenido (su índice cambió al agregar celdas)


def sub(cell, old, new):
    src = cell.source
    assert old in src, f"no encontré {old!r} en la celda"
    cell.source = src.replace(old, new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, choices=list(VARIANTS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", choices=["CelebA", "AwA2"], default="CelebA")
    p.add_argument("--lambda-entropy", type=float, default=0.1, help="solo para --name entropy")
    p.add_argument("--epochs", type=int, default=None, help="por defecto 5 (CelebA) o 15 (AwA2)")
    p.add_argument("--image-size", type=int, default=None, help="por defecto 64 (CelebA) o 32 (AwA2)")
    p.add_argument("--mlflow-db", default=None, help="ruta de la base sqlite de MLflow (por defecto experimentation/mlflow.db); "
                   "útil cuando la carpeta está en un sistema de archivos que no soporta sqlite")
    p.add_argument("--smoke", action="store_true", help="1 época, 0.2%% del train, mlflow y experimentos en scratchpad")
    a = p.parse_args()
    is_c, pen = VARIANTS[a.name]
    if a.epochs is None:
        a.epochs = 5 if a.dataset == "CelebA" else 15
    if a.image_size is None:
        a.image_size = 64 if a.dataset == "CelebA" else 32
    epochs = 1 if a.smoke else a.epochs
    lam_ent = a.lambda_entropy if a.name == "entropy" else 0.0
    assert a.name != "entropy" or lam_ent > 0, "--name entropy necesita --lambda-entropy > 0"
    lam_tag = f"{lam_ent:g}".replace(".", "")
    run_name = f"entropy_l{lam_tag}" if a.name == "entropy" else a.name   # = variant del notebook

    nb = nbformat.read(HERE / "training.ipynb", 4)
    c_save = next(c for c in nb.cells if SAVE_MARKER in c.source)
    cells = [nb.cells[i] for i in USE_CELLS] + [c_save]
    c0, c2, c14, c16, c17 = (nb.cells[i] for i in (0, 2, 14, 16, 17))
    c19 = c_save

    if a.dataset == "CelebA":
        sub(c2, 'DATASET_NAME = "AwA2"', 'DATASET_NAME = "CelebA"')
        sub(c2, '"CelebA": lambda: CelebADataset(),',
            f'"CelebA": lambda: CelebADataset(image_size={a.image_size}, root=str(project_root / "data" / "celeba")),')
    else:
        assert a.image_size == 32, "el notebook usa AwA2 a 32 px (cache local)"
    sub(c14, "is_constrained = False #@param", f"is_constrained = {is_c} #@param")
    sub(c14, "penalize_rank = False #@param", f"penalize_rank = {pen} #@param")
    sub(c14, "epochs = 15 #@param", f"epochs = {epochs} #@param")
    sub(c14, "lambda_rank = 1.0 #@param", "lambda_rank = 0.1 #@param")
    sub(c14, "rank_target = 3.0 #@param", "rank_target = 6.0 #@param")
    sub(c14, "lambda_entropy = 0.1 #@param", f"lambda_entropy = {lam_ent} #@param")
    sub(c14, '"image_size": 32,', f'"image_size": {a.image_size},')
    # Poner la semilla en el nombre del run / experimento guardado (variant del notebook = run_name, p.ej. entropy_l01).
    sub(c14, """exp_name = f'vit-{epochs}{"-" + variant if variant != "unconstrained" else ""}-{dataset.name}'""",
        'variant_base = variant\nSEED = %d\nvariant = f"{variant_base}_s{SEED}"\n' % a.seed
        + "exp_name = f'vit-{epochs}-{variant_base}-s{SEED}-{dataset.name}'")
    assert "variant_base" in c14.source
    sub(c14, '"variant": variant,', '"variant": variant_base, "seed": str(SEED),')
    if a.smoke:
        sub(c14, "train_fraction = 1.0 #@param", "train_fraction = 0.002 #@param")
    sub(c16, "train_sample_size = int(", f"import random\nrandom.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\ntrain_sample_size = int(")
    if a.mlflow_db and not a.smoke:
        sub(c0, 'mlflow.set_tracking_uri("sqlite:///mlflow.db")', f'mlflow.set_tracking_uri("sqlite:///{a.mlflow_db}")')
    if a.smoke:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        sub(c0, 'mlflow.set_tracking_uri("sqlite:///mlflow.db")', f'mlflow.set_tracking_uri("sqlite:///{SCRATCH}/mlflow.db")')
        sub(c17, 'base_dir="experiments"', f'base_dir="{SCRATCH}/experiments"')
        sub(c19, 'model_path = f"vit_final_', f'model_path = f"{SCRATCH}/vit_final_')

    out = nbformat.v4.new_notebook(cells=cells, metadata=nb.metadata)
    (HERE / "logs").mkdir(exist_ok=True)
    ds_tag = "" if a.dataset == "CelebA" else f"{a.dataset}_"
    out_path = HERE / "logs" / f"executed_{ds_tag}{run_name}_s{a.seed}{'_smoke' if a.smoke else ''}.ipynb"
    client = NotebookClient(out, kernel_name="interpretable-transformers", timeout=None,
                            resources={"metadata": {"path": str(HERE)}}, allow_errors=False)
    try:
        client.execute()
        print("DONE", a.dataset, run_name, a.seed, flush=True)
    finally:
        nbformat.write(out, out_path)
        print("notebook ejecutado en", out_path, flush=True)


if __name__ == "__main__":
    main()
