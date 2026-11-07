"""
Entrena una variante (AwA2, 15 épocas, misma arquitectura que training.ipynb) desde la línea de comandos,
para poder lanzar varias semillas / variantes en paralelo sin abrir el notebook.

Guarda el experimento en experimentation/experiments/vit-15-<name>-s<seed>-AwA2/ (config, metrics, model_final.pt)
y el log de MLflow en su propia base sqlite (experimentation/mlflow_runs/<name>-s<seed>/), para que varios
procesos no compitan por el mismo mlflow.db.

Ejemplo:
    python run_variant.py --name constrained_rank --seed 1 --constrained 1 --lambda-rank 0.1
"""
import argparse
import os
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn, optim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from modules.datasets import AwA2Dataset  # noqa: E402
from modules.model import ViTForClassfication  # noqa: E402
from modules.training import Trainer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--constrained", type=int, default=1)
    p.add_argument("--lambda-loss", type=float, default=0.1)
    p.add_argument("--lambda-rank", type=float, default=0.0)
    p.add_argument("--rank-target", type=float, default=6.0)
    p.add_argument("--var-floor", type=float, default=0.5)
    p.add_argument("--lambda-xcka", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--train-sample", type=int, default=None, help="solo para pruebas rápidas")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    exp_name = f"vit-{args.epochs}-{args.name}-s{args.seed}-AwA2"
    run_dir = HERE / "mlflow_runs" / f"{args.name}-s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)                               # mlruns/ (artefactos) queda junto a su base de datos
    mlflow.set_tracking_uri(f"sqlite:///{run_dir / 'mlflow.db'}")
    mlflow.set_experiment(exp_name)

    dataset = AwA2Dataset(image_size=32, root=str(ROOT / "data" / "awa2"))
    trainloader, testloader, _ = dataset.get_loaders(batch_size=args.batch_size, num_workers=args.num_workers,
                                                     train_sample_size=args.train_sample)

    config = {
        "patch_size": 4, "hidden_size": 48, "num_hidden_layers": 4, "num_attention_heads": 4,
        "intermediate_size": 4 * 48, "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02, "image_size": 32, "num_classes": dataset.n_classes, "num_channels": 3,
        "qkv_bias": True, "use_faster_attention": False,
    }
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    note = (f"{args.name} seed={args.seed} | constrained={bool(args.constrained)} lambda_loss={args.lambda_loss} | "
            f"lambda_rank={args.lambda_rank} target={args.rank_target} var_floor={args.var_floor} | "
            f"lambda_xcka={args.lambda_xcka}")
    trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device, config=config,
                      is_constrained=bool(args.constrained), lambda_loss=args.lambda_loss,
                      batch_size=args.batch_size, lr=args.lr, dataset=dataset, log_indices=None,
                      base_dir=str(HERE / "experiments"),
                      lambda_rank=args.lambda_rank if args.constrained else 0.0,
                      rank_target=args.rank_target, rank_on="head_output", var_floor=args.var_floor,
                      rank_tokens="cls", lambda_xcka=args.lambda_xcka if args.constrained else 0.0,
                      mlflow_run_name=f"{args.name}_s{args.seed}",
                      mlflow_tags={"variant": args.name, "seed": str(args.seed)}, mlflow_note=note)
    metrics = trainer.train(trainloader, testloader, args.epochs)
    print("DONE", exp_name, metrics.get("accuracy"), flush=True)


if __name__ == "__main__":
    main()
