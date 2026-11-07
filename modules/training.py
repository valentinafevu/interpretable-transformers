"""
Pipeline de entrenamiento de `experimentation/training.ipynb`, empaquetado como módulo
para poder reutilizarlo desde otros notebooks (p.ej. AwA2) sin copiar celdas.

Diferencias respecto al notebook, todas de forma (no de comportamiento):
- Los globals del notebook (`is_constrained`, `lambda_loss`, `config`, `batch_size`, `lr`)
  pasan a ser argumentos del Trainer.
- Los batches pueden traer más de dos tensores (AwA2 devuelve (imagen, label, atributos));
  el Trainer usa solo los dos primeros.
- `base_dir` de `save_experiment` es configurable.
"""
import logging

import matplotlib.pyplot as plt
import mlflow
import torch
from tqdm.auto import tqdm

from .cka import linear_cka
from .effective_rank import effective_rank
from .experiments import save_checkpoint, save_experiment
from .loss import (attention_entropy_loss, attention_rank_floor_loss, effective_rank_floor_loss,
                   head_output_cka_loss, head_output_collapse_loss, interpretability_loss)
from .visualizations import visualize_attention, visualize_attention_on_images

logger = logging.getLogger("vit_training")


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device, config, is_constrained,
                 lambda_loss=0.1, batch_size=None, lr=None, dataset=None, log_indices=None,
                 base_dir="experiments", lambda_rank=0.0, rank_target=6.0, rank_on="head_output",
                 rank_params=("query", "key"), var_floor=0.5, rank_tokens="cls",
                 mlflow_run_name="run-experiment", mlflow_tags=None, mlflow_note=None, lambda_xcka=0.0,
                 rank_ceiling=False, lambda_entropy=0.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.config = config
        self.is_constrained = is_constrained
        self.lambda_loss = lambda_loss
        # Término anti-colapso (ver modules/loss.py); lambda_rank = 0 lo desactiva de la pérdida.
        #   rank_on="head_output":   piso de varianza + rango efectivo sobre la salida de cada cabeza del
        #                            último bloque (rank_tokens="cls" | "all"); rank_target <= head_size.
        #   rank_on="attention_map": sobre la matriz (cabezas × parches) de atención del CLS; máx. = nº de cabezas.
        #   rank_on="qk_weights":    sobre los pesos Q/K de la última capa (`rank_params`); máx. = head_size.
        # Con rank_on="head_output" las métricas por cabeza se calculan y loguean SIEMPRE (también con
        # lambda_rank = 0), para poder comparar contra el modelo sin el término.
        assert rank_on in ("head_output", "attention_map", "qk_weights")
        assert rank_tokens in ("cls", "all")
        self.var_floor = var_floor
        self.rank_ceiling = rank_ceiling   # True: el término de rango de head_output pasa de piso a techo (ver modules/loss.py)
        # Penalización de CKA entre salidas de cabezas (requiere rank_on="head_output", que captura las salidas).
        self.lambda_xcka = lambda_xcka
        # Penalización de entropía de la atención del CLS (último bloque), normalizada en [0, 1]. Se suma a la
        # pérdida total SIEMPRE que lambda_entropy > 0, también con is_constrained=False (variante "entropy":
        # solo localización, sin penalizar similitud entre cabezas). La entropía y el nº efectivo de parches
        # (exp(H)) por cabeza se calculan y loguean SIEMPRE (también con lambda_entropy = 0) para comparar
        # contra los modelos sin el término.
        self.lambda_entropy = lambda_entropy
        assert lambda_xcka == 0 or rank_on == "head_output"
        self.rank_tokens = rank_tokens
        self.mlflow_run_name = mlflow_run_name
        self.mlflow_tags = mlflow_tags or {}
        self.mlflow_note = mlflow_note
        self.lambda_rank = lambda_rank
        self.rank_target = rank_target
        self.rank_on = rank_on
        self.rank_params = tuple(rank_params)
        self.batch_size = batch_size
        self.lr = lr
        self.dataset = dataset
        self.log_indices = log_indices
        self.base_dir = base_dir
        self.multi_label = getattr(dataset, "multi_label", False) if dataset is not None else False

        # Salida de cada cabeza del último bloque, capturada con hooks (el modelo solo devuelve las
        # probabilidades de atención).
        self._head_outputs = {}
        self._stats = {}
        if self.rank_on == "head_output":
            assert not config.get("use_faster_attention", False), "se necesitan cabezas separadas (attention.heads)"
            heads = self.model.encoder.blocks[-1].attention.heads
            for k, head in enumerate(heads):
                head.register_forward_hook(lambda _m, _i, out, k=k: self._head_outputs.__setitem__(k, out[0]))

    def _flush_stats(self, prefix, epoch_i):
        for name, values in self._stats.items():
            mlflow.log_metric(f"{prefix}{name}", sum(values) / len(values), step=epoch_i)
        self._stats = {}

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        logger.info(f"Iniciando entrenamiento '{self.exp_name}': {epochs} épocas, "
                    f"batch_size={self.batch_size}, lr={self.lr}, device={self.device}")

        with mlflow.start_run(run_name=self.mlflow_run_name, description=self.mlflow_note):
            if self.mlflow_tags:
                mlflow.set_tags(self.mlflow_tags)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("learning_rate", self.lr)
            mlflow.log_param("lambda_loss", self.lambda_loss)
            mlflow.log_param("is_constrained", self.is_constrained)
            mlflow.log_param("lambda_rank", self.lambda_rank)
            mlflow.log_param("rank_target", self.rank_target)
            mlflow.log_param("rank_on", self.rank_on)
            mlflow.log_param("var_floor", self.var_floor)
            mlflow.log_param("rank_ceiling", self.rank_ceiling)
            mlflow.log_param("rank_tokens", self.rank_tokens)
            mlflow.log_param("lambda_xcka", self.lambda_xcka)
            mlflow.log_param("lambda_entropy", self.lambda_entropy)
            mlflow.log_param("rank_params", ",".join(self.rank_params))

            epoch_bar = tqdm(range(epochs), desc="Épocas", unit="epoch")
            for i in epoch_bar:
                train_losses = self.train_epoch(trainloader, epoch_i=i)
                test_losses = self.evaluate(testloader, epoch_i=i)

                epoch_bar.set_postfix(
                    train_loss=f"{train_losses['total_loss']:.4f}",
                    test_loss=f"{test_losses['total_loss']:.4f}",
                    test_acc=f"{test_losses['accuracy']:.4f}",
                )

                logger.info(f"Epoch {i+1}/{epochs} - Train Loss: {train_losses['total_loss']:.4f}, "
                            f"Train Diversity Loss: {train_losses['interpretability_loss']:.4f}, "
                            f"Test Loss: {test_losses['total_loss']:.4f}, "
                            f"Test Accuracy: {test_losses['accuracy']:.4f}")

                if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                    logger.info(f"Guardando checkpoint en época {i+1}")
                    save_checkpoint(self.exp_name, self.model, i+1, base_dir=self.base_dir)

        save_experiment(self.exp_name, self.config, self.model, test_losses, base_dir=self.base_dir)
        logger.info(f"Entrenamiento finalizado. Modelo y métricas guardados para '{self.exp_name}'.")
        return test_losses

    def _step_losses(self, images, labels):
        logits, attention_maps = self.model(images, output_attentions=True)

        cls_attention = attention_maps[-1][:, :, 0, 1:]
        interpretability_loss_value = interpretability_loss(cls_attention)
        # Rango efectivo de la MISMA matriz sobre la que actúa la penalización
        # (atención del CLS hacia los parches, por cabeza) — no de los pesos de
        # Q/K/V, sino del objeto que la pérdida realmente ve.
        attn_erank_value = effective_rank(cls_attention).mean()
        entropy_value, entropy_stats = attention_entropy_loss(cls_attention)
        self._stats.setdefault("attention_entropy/mean", []).append(entropy_value.item())
        self._stats.setdefault("attention_eff_patches/mean", []).append(entropy_stats["eff_patches"].mean().item())
        for h in range(cls_attention.shape[1]):
            self._stats.setdefault(f"attention_entropy/head_{h}", []).append(entropy_stats["entropy"][h].item())
            self._stats.setdefault(f"attention_eff_patches/head_{h}", []).append(entropy_stats["eff_patches"][h].item())
        # BCEWithLogitsLoss (multi-label) espera targets float; CrossEntropyLoss (single-label) espera índices long.
        cls_loss = self.loss_fn(logits, labels.float() if self.multi_label else labels)

        # Término anti-colapso: solo entra a la pérdida total en el modo constrained y con lambda_rank > 0.
        if self.rank_on == "head_output":
            n_heads = len(self._head_outputs)
            outs = torch.stack([self._head_outputs[k] for k in range(n_heads)], dim=1)   # (B, H, T, d)
            if self.rank_tokens == "cls":
                outs = outs[:, :, 0]                                                      # (B, H, d)
            else:
                outs = outs.permute(0, 2, 1, 3).reshape(-1, n_heads, outs.shape[-1])      # (B*T, H, d)
            rank_loss_value, rank_stats = head_output_collapse_loss(outs, self.rank_target, self.var_floor, self.rank_ceiling)
            xcka_value = head_output_cka_loss(outs)
            self._stats.setdefault("head_output/cka_mean", []).append(xcka_value.item())
            for h in range(n_heads):
                self._stats.setdefault(f"head_output/erank/head_{h}", []).append(rank_stats["erank"][h].item())
                self._stats.setdefault(f"head_output/std/head_{h}", []).append(rank_stats["std"][h].item())
        elif self.lambda_rank > 0 and self.rank_on == "attention_map":
            rank_loss_value = attention_rank_floor_loss(cls_attention, self.rank_target)
        elif self.lambda_rank > 0:
            rank_loss_value = effective_rank_floor_loss(self.model, self.rank_target, self.rank_params)
        else:
            rank_loss_value = torch.zeros((), device=logits.device)

        if self.is_constrained:
            total_loss = (1 - self.lambda_loss) * cls_loss + self.lambda_loss * interpretability_loss_value \
                + self.lambda_rank * rank_loss_value
            if self.lambda_xcka > 0:
                total_loss = total_loss + self.lambda_xcka * xcka_value
        else:
            total_loss = cls_loss
        if self.lambda_entropy > 0:
            total_loss = total_loss + self.lambda_entropy * entropy_value
        return logits, cls_loss, interpretability_loss_value, attn_erank_value, rank_loss_value, total_loss

    def train_epoch(self, trainloader, epoch_i=0):
        self.model.train()
        self._stats = {}
        losses = {
            "cls_loss": [],
            "interpretability_loss": [],
            "attn_erank": [],
            "rank_loss": [],
            "total_loss": [],
        }

        batch_bar = tqdm(trainloader, desc=f"Época {epoch_i+1} [train]", unit="batch", leave=False)
        for batch in batch_bar:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            _, cls_loss, interpretability_loss_value, attn_erank_value, rank_loss_value, total_loss = \
                self._step_losses(images, labels)

            total_loss.backward()
            self.optimizer.step()

            losses["cls_loss"].append(cls_loss.item())
            losses["interpretability_loss"].append(interpretability_loss_value.item())
            losses["attn_erank"].append(attn_erank_value.item())
            losses["rank_loss"].append(rank_loss_value.item())
            losses["total_loss"].append(total_loss.item())

            batch_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg = {k: sum(v) / len(v) for k, v in losses.items()}

        mlflow.log_metric("loss", avg["cls_loss"], step=epoch_i)
        mlflow.log_metric("interpretability_loss", avg["interpretability_loss"], step=epoch_i)
        mlflow.log_metric("effective_rank/attention_map/train_mean", avg["attn_erank"], step=epoch_i)
        mlflow.log_metric("rank_loss", avg["rank_loss"], step=epoch_i)
        mlflow.log_metric("total_loss", avg["total_loss"], step=epoch_i)
        self._flush_stats("train_", epoch_i)

        return avg

    @torch.no_grad()
    def evaluate(self, testloader, epoch_i=0):
        self.model.eval()
        self._stats = {}

        losses = {
            "cls_loss": [],
            "interpretability_loss": [],
            "attn_erank": [],
            "rank_loss": [],
            "total_loss": [],
        }

        num_classes = self.config["num_classes"]
        total_correct = 0
        total_samples = 0

        if self.multi_label:
            # Precision por atributo: cuenta de aciertos por columna, acumulada a lo largo de todo el testset.
            attr_correct = torch.zeros(num_classes, dtype=torch.long)
        else:
            # Matriz de confusión acumulada en CPU para no transferir tensores por batch.
            confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

        batch_bar = tqdm(testloader, desc=f"Época {epoch_i+1} [eval]", unit="batch", leave=False)
        for batch in batch_bar:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            logits, cls_loss, interpretability_loss_value, attn_erank_value, rank_loss_value, total_loss = \
                self._step_losses(images, labels)

            losses["cls_loss"].append(cls_loss.item())
            losses["interpretability_loss"].append(interpretability_loss_value.item())
            losses["attn_erank"].append(attn_erank_value.item())
            losses["rank_loss"].append(rank_loss_value.item())
            losses["total_loss"].append(total_loss.item())

            if self.multi_label:
                # Cada salida es un sigmoide binario independiente (atributo presente/ausente).
                preds = (torch.sigmoid(logits) > 0.5).long()
                matches = (preds == labels)
                total_correct += matches.sum().item()
                total_samples += labels.numel()  # accuracy promediada sobre todos los (sample, atributo)
                attr_correct += matches.sum(dim=0).cpu()
            else:
                preds = logits.argmax(dim=-1)

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                # Matriz de confusión acumulada vía bincount (O(N) por batch, sin loop Python).
                # Truco: index = true * num_classes + pred → bin único por celda (true, pred).
                idx = labels * num_classes + preds
                binc = torch.bincount(idx, minlength=num_classes * num_classes).cpu()
                confusion += binc.view(num_classes, num_classes)

            batch_bar.set_postfix(acc=f"{total_correct / total_samples:.4f}")

        avg = {k: sum(v) / len(v) for k, v in losses.items()}
        accuracy = total_correct / total_samples

        mlflow.log_metric("val_loss", avg["cls_loss"], step=epoch_i)
        mlflow.log_metric("val_interpretability_loss", avg["interpretability_loss"], step=epoch_i)
        mlflow.log_metric("effective_rank/attention_map/mean", avg["attn_erank"], step=epoch_i)
        mlflow.log_metric("val_rank_loss", avg["rank_loss"], step=epoch_i)
        mlflow.log_metric("val_total_loss", avg["total_loss"], step=epoch_i)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch_i)
        self._flush_stats("val_", epoch_i)

        # Rango efectivo (Roy & Vetterli, 2007) de las matrices Q/K/V de cada cabeza
        # de la última capa de atención — señal de qué tan redundante es cada cabeza.
        last_block = self.model.encoder.blocks[-1]
        heads = last_block.attention.heads

        query_weights = torch.stack([head.query.weight.detach() for head in heads])
        key_weights = torch.stack([head.key.weight.detach() for head in heads])
        value_weights = torch.stack([head.value.weight.detach() for head in heads])

        erank_query = effective_rank(query_weights)
        erank_key = effective_rank(key_weights)
        erank_value = effective_rank(value_weights)

        for h in range(len(heads)):
            mlflow.log_metric(f"effective_rank/Q/head_{h}", erank_query[h].item(), step=epoch_i)
            mlflow.log_metric(f"effective_rank/K/head_{h}", erank_key[h].item(), step=epoch_i)
            mlflow.log_metric(f"effective_rank/V/head_{h}", erank_value[h].item(), step=epoch_i)

        mlflow.log_metric("effective_rank/Q/mean", erank_query.mean().item(), step=epoch_i)
        mlflow.log_metric("effective_rank/K/mean", erank_key.mean().item(), step=epoch_i)
        mlflow.log_metric("effective_rank/V/mean", erank_value.mean().item(), step=epoch_i)

        # CKA (Kornblith et al. 2019) entre cada par de cabezas de Q/K/V del último bloque
        # (el mismo sobre el que actúa la penalización).
        num_heads = self.config["num_attention_heads"]
        head_pairs = [(i, j) for i in range(num_heads) for j in range(i + 1, num_heads)]
        last_block_idx = len(self.model.encoder.blocks) - 1

        block_weights = {"Q": query_weights, "K": key_weights, "V": value_weights}

        for name, weights in block_weights.items():
            cka = linear_cka(weights.unsqueeze(1), weights.unsqueeze(0))  # (num_heads, num_heads)

            for i, j in head_pairs:
                mlflow.log_metric(f"cka/{name}/block_{last_block_idx}/head_{i}_{j}", cka[i, j].item(), step=epoch_i)

            mean_cka = torch.stack([cka[i, j] for i, j in head_pairs]).mean().item()
            mlflow.log_metric(f"cka/{name}/block_{last_block_idx}/mean", mean_cka, step=epoch_i)

        class_names = getattr(self.dataset, "classes", None) if self.dataset is not None else None
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]

        if self.multi_label:
            attr_samples = total_samples / num_classes
            attr_acc = (attr_correct.float() / attr_samples).numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            order = attr_acc.argsort()
            ax.barh(range(num_classes), attr_acc[order], color="tab:blue")
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels([class_names[i] for i in order], fontsize=7)
            ax.set_xlabel("Accuracy")
            ax.set_xlim(0, 1)
            ax.set_title(f"Accuracy por atributo — epoch {epoch_i}")
            fig.tight_layout()

            mlflow.log_figure(fig, f"attribute_accuracy/epoch_{epoch_i:03d}.png")
            plt.close(fig)
        else:
            # Matriz de confusión normalizada por fila (true class): la diagonal es el recall por clase.
            cm = confusion.float()
            cm_norm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
            cm_np = cm_norm.numpy()

            # Con muchas clases (AwA2 = 50) anotar cada celda es ilegible y lento.
            annotate = num_classes <= 20
            size = 8 if annotate else 14
            fig, ax = plt.subplots(figsize=(size, size * 7 / 8))
            im = ax.imshow(cm_np, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"Confusion matrix (row-normalized) — epoch {epoch_i}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45 if annotate else 90, ha="right" if annotate else "center",
                               fontsize=None if annotate else 7)
            ax.set_yticklabels(class_names, fontsize=None if annotate else 7)

            if annotate:
                for i in range(num_classes):
                    for j in range(num_classes):
                        val = cm_np[i, j]
                        ax.text(j, i, f"{val:.2f}",
                                ha="center", va="center",
                                fontsize=8,
                                color="white" if val > 0.5 else "black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            mlflow.log_figure(fig, f"confusion_matrix/epoch_{epoch_i:03d}.png")
            plt.close(fig)

        # Log attention
        if self.dataset is not None and self.log_indices is not None:
            log_images, _ = self.dataset.get_samples_from_indices(self.log_indices, device=self.device, set="train")
            _, log_attention_maps = self.model(log_images, output_attentions=True)

            for layer_idx, amap in enumerate(log_attention_maps):
                attn_img = visualize_attention(amap)
                mlflow.log_image(attn_img, f"attention/layer_{layer_idx}/epoch_{epoch_i:03d}.png")

            overlay_imgs = visualize_attention_on_images(log_attention_maps, log_images)
            for layer_idx, overlay_img in enumerate(overlay_imgs):
                mlflow.log_image(overlay_img, f"attention_overlay/layer_{layer_idx}/epoch_{epoch_i:03d}.png")

        return {**avg, "accuracy": accuracy}
