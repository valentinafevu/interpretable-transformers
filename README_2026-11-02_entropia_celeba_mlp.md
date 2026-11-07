# Qué se hizo el 2026-11-02: entropía, grupos de atributos, regiones en CelebA y MLP por cabeza

Guía de lectura de todo el trabajo de esta sesión. Cada sección dice **qué pregunta responde, dónde está el código, dónde están los resultados, qué se encontró y qué NO se puede concluir**.

> Convención: "cabeza" = cabeza de atención; "bloque 4" = último bloque del ViT (4 bloques, 4 cabezas, hidden 48);
> AwA2 = 32 px, 64 parches; CelebA = 64 px, 256 parches.

---

## 0. Mapa rápido

| # | Qué | Dónde (código) | Resultados |
|---|---|---|---|
| 1 | Penalización de entropía de la atención | `modules/loss.py` (`attention_entropy_loss`), `modules/training.py`, `experimentation/training.ipynb`, `experimentation/run_training_notebook.py` | MLflow (`experimentation/mlflow.db`, experimento `vit-15-epochs-AwA2`); `experimentation/experiments/vit-15-entropy_l*-s0-AwA2/` |
| 2 | Ablación, probing y especialización con los modelos de entropía | `notebooks/22082026_ablation/ablation_entropy.ipynb`, `notebooks/25092026_awa2_probing/probing_awa2_entropy.ipynb`, `notebooks/30082026_head_specialization/head_specialization_entropy.ipynb` | `experimentation/experiments/*_entropy/` |
| 3 | Atributos de AwA2 agrupados en "animal" vs "fondo" | `notebooks/02112026_attribute_groups_entropy/attribute_groups_entropy.ipynb` | `experimentation/experiments/02112026_attribute_groups_entropy/` |
| 4 | ¿Dónde mira cada cabeza en CelebA? (landmarks) | `notebooks/02112026_celeba_regions/celeba_regions.ipynb` | `experimentation/experiments/02112026_celeba_regions/` |
| 5 | Probing del MLP por cabeza (AwA2) | `notebooks/02112026_mlp_per_head_probing/mlp_per_head_probing.ipynb` | `experimentation/experiments/02112026_mlp_per_head_probing/` |
| 6 | Atributo subjetivo `Attractive` en CelebA: dónde mira cada cabeza | `notebooks/02112026_celeba_attractive/celeba_attractive.ipynb` | `experimentation/experiments/02112026_celeba_attractive/` |
| 7 | Descripción del dataset CelebA (figuras listas para diapositivas) | `notebooks/02112026_celeba_dataset/celeba_dataset.ipynb` | `experimentation/experiments/02112026_celeba_dataset/` |
| 8 | CelebA con entropía λ=0.01 (1 corrida): duplicado del notebook de regiones, listo para correr cuando exista el checkpoint | `notebooks/02112026_celeba_regions_entropy/celeba_regions_entropy.ipynb` (entrenar con `experimentation/run_training_notebook.py --name entropy --lambda-entropy 0.01 --seed 0`) | `experimentation/experiments/02112026_celeba_regions_entropy/` (se crea al ejecutar) |

Los scripts sueltos `experimentation/scripts/celeba_*.py` son la primera versión de (4); el notebook los reemplaza. Se pueden borrar.

---

## 1. Penalización de entropía

**Idea.** Las cabezas penalizadas por coseno son geométricamente distintas pero no más interpretables. Hipótesis: pueden estar mirando regiones distintas sin que su atención esté *concentrada*. Se probó una penalización que castiga que una cabeza mire a muchos parches.

**Definición** (`attention_entropy_loss` en `modules/loss.py`). Para cada cabeza, la atención del [CLS] a los P parches, `a_i`, suma 1. Entropía normalizada `H/log P`, con `H = −Σ a_i log a_i`. La pérdida es el promedio sobre cabezas y batch, multiplicada por `lambda_entropy`. Sin objetivo K (no se fija a cuántos parches debe mirar); sin coseno entre cabezas. Solo bloque 4.

**Cómo usarlo.** En `experimentation/training.ipynb`: `lambda_entropy = 0.01` (la variante se llama `entropy_l001`, el tag sale de `f"{λ:g}"` sin el punto). El Trainer registra en MLflow `attention_entropy/*` y `attention_eff_patches/*` (parches efectivos = exp(H)) siempre, aunque λ = 0.

**Resultados (AwA2, 15 épocas, 1 semilla, CPU).**

| λ_entropy | accuracy | parches efectivos/cabeza | coseno medio entre cabezas |
|---|---|---|---|
| 0.001 | 32.7 % | 26 | 0.30 |
| 0.01 | 32.1 % | 16 | 0.22 |
| 0.1 | 29.0 % | 1.1 (colapso a un parche) | 0.36 |
| sin penalizar (3 semillas) | 29.7–32.0 % | 15–55 | ~0.24 |

**Lectura.** Con λ ≤ 0.01 el efecto queda dentro de la variación entre semillas del baseline. Con λ = 0.1 colapsa a atención de un solo parche en la primera época y las cabezas se vuelven redundantes. **No existe un régimen útil de entropía simple.** (Una afirmación mía anterior —que con λ bajo había poco riesgo de colapso— resultó errónea para λ = 0.1.) Los modelos con coseno ya estaban muy concentrados (1–10 parches efectivos): la dispersión nunca fue el problema.

## 2. Ablación, probing y especialización con entropía

Copias de los notebooks originales (los originales no se tocaron) apuntando a los modelos de entropía. Resumen (bloque 4, seed 0):

| | sin pen. | constrained | constr.+rango | ent 0.1 | ent 0.01 | ent 0.001 |
|---|---|---|---|---|---|---|
| R² probing, 4 cabezas concat. | 0.161 | 0.123 | 0.116 | 0.112 | 0.147 | 0.158 |
| Coseno entre vectores de daño por clase (menor = más especializadas) | 0.10 | 0.07 | 0.03 | 0.61 | 0.34 | 0.28 |
| Caída media al apagar 1 cabeza (pts, n=1000) | 1.6 | 0.7 | 1.2 | 1.4 | 0.7 | 1.9 |

Ninguna cabeza supera R² > 0.3 en ningún atributo. Diferencias de ablación ≤ 1.2 pts con n = 1000: indistinguibles del ruido. El R² concatenado baja al bajar los parches efectivos (observación correlacional, 6 modelos, 1 semilla).

## 3. Grupos de atributos de AwA2: animal vs fondo (`entropy_l001_s0`)

**Pregunta.** ¿Las cabezas codifican distinto los atributos del animal (color, textura, forma, partes) que los del fondo (hábitat)?

**Hallazgos.**
- Todos los grupos superan el nulo (p ≈ 0.01) pero con R² bajo (0.03–0.09). El hábitat se decodifica algo mejor que el animal (0.058–0.090 vs 0.045–0.065).
- Esa ventaja del hábitat aparece también sin penalizar (0.034) y en los demás modelos; en `entropy_l001` es menor (0.018). **No es un efecto de la penalización.**
- **Confusión a nivel de clase:** `ocean` y `water` correlacionan 0.97 y 0.88 con `flippers`; el 68 % y 61 % de su varianza se explica con atributos del animal. Al residualizar, `ocean` 0.237 → 0.107, `water` 0.217 → 0.082, `coastal` 0.103 → 0.046; `desert`, `forest`, `mountains` quedan en el nulo.
- Solo la cabeza 3 tiene atributos con R² > 0.1 (ocean, flippers, water, ground, strainteeth, blue, hairless, coastal): una cabeza "acuática" que mezcla animal y hábitat.

**No se puede concluir** que alguna cabeza lea el fondo: los atributos son constantes por clase y no hay máscaras.

## 4. CelebA: ¿dónde mira cada cabeza?

Notebook: `notebooks/02112026_celeba_regions/celeba_regions.ipynb` (muy comentado; léelo en orden).

**Por qué no se usó `list_bbox_celeba.csv`.** Está en las coordenadas de la foto original, no de las imágenes alineadas; el 87 % de las cajas se sale de la imagen de 178×218. Se usan los 5 landmarks alineados (ojos, nariz, comisuras de la boca), convertidos al marco de 64×64 con `x·64/178`, `(y−20)·64/178` (el `CenterCrop(178)` quita 20 px arriba y abajo).

**Método.** Regiones por imagen sobre la rejilla 16×16 (ojos, nariz, boca = parches a ≤ 5 px de los landmarks; cara = caja de landmarks +6 px; fuera de la cara = el resto). Para cada cabeza del bloque 4: *enriquecimiento* = masa de atención en la región ÷ fracción de área de la región (1 = mira parejo). *Parches efectivos* = exp(entropía). Test completo (19 962 imágenes), 3 variantes × 3 semillas (los 9 checkpoints `vit-5-*-CelebA`). **Chequeo:** la exactitud recalculada coincide con `metrics.json` (p. ej. 0.8785 en `constrained` s0).

**Hallazgos** (valores exactos en `celeba_region_enrichment.csv` y `celeba_coverage.csv`).
- Sin penalizar, las cabezas están repartidas (mediana ≈ 25 parches efectivos; la semilla 0 llega a 80–218) y casi no se dedican a una región.
- Con coseno, casi todas las cabezas bajan a 1–8 parches efectivos y 3 de 4 se concentran (>50 % de su masa) en **una** región. Cada modelo penalizado cubre 2 regiones distintas, y se repite en las 3 semillas: una cabeza en los ojos (enriquecimiento ×16–27) y otra en pelo/fondo; en `constrained_rank` aparece la boca en 2 de 3 semillas.
- Vínculo con atributos (exploratorio): con gafas, la cabeza más sensible cambia su masa en `ojos` en 0.155 (constrained) y 0.156 (constrained_rank) frente a 0.028 sin penalizar, pero depende de la semilla (0.002–0.288); el 0.55 del sombrero en `constrained` s2 es un caso aislado.

**Matiz:** varias cabezas dedicadas tienen **1 parche efectivo**: miran siempre el mismo parche. Como CelebA está alineada, un parche fijo equivale a una parte de la cara, pero es un *selector de posición*, no un detector de ojos.

**Límites.** Regiones definidas por mí (radio 5 px, umbrales 50 % y 12 parches; no barridos). 3 semillas. Es descriptivo: no se probó que esas cabezas sean más útiles para clasificar. **No hay modelos de entropía en CelebA** (habría que entrenarlos).

## 5. Probing del MLP por cabeza (AwA2)

Notebook: `notebooks/02112026_mlp_per_head_probing/mlp_per_head_probing.ipynb`.

**Problema de fondo.** El MLP del bloque recibe la *suma* de las 4 cabezas y es no lineal, así que no existe "el MLP de la cabeza h". Se usan dos definiciones contrafactuales sobre el [CLS]:
- `mlp_solo_h` = MLP alimentado solo con la cabeza h (otras apagadas);
- `mlp_loo_h` = MLP(todas) − MLP(todas menos h): cuánto depende la salida del MLP de la cabeza h.
Referencias: `cabeza` (salida cruda, lo ya sondeado) y `mlp_completo`. El notebook reconstruye el bloque a mano y verifica con `assert` que coincide con el modelo.

**Sonda y nulo:** Ridge, R² fuera de muestra (5 folds por clase), 47 atributos, 100 permutaciones de clase, selectividad = R² − nulo.

**Hallazgos** (valores en `mlp_probing_summary.csv`, `gain_by_group.csv`, `animal_vs_background.csv`, `per_attribute_gain.csv`).
- Selectividad media (R² − nulo), `entropy_l001`: cabeza cruda 0.029 / 0.031 / 0.047 / 0.049 (cabezas 1–4); `mlp_loo` 0.020 / 0.025 / 0.030 / 0.031; `mlp_solo` 0.047 / 0.044 / 0.043 / 0.046; `mlp_completo` 0.046. En `unconstrained`: cruda 0.047–0.064, `mlp_loo` 0.048–0.071.
- **El MLP no hace más legible lo de cada cabeza**: con `mlp_loo` baja o queda igual. Los atributos fuertes (`water`, `ocean`, `flippers`, cabeza 3) pierden ~0.08 de R².
- **`mlp_solo` no es limpio:** sus 4 cabezas salen casi iguales entre sí y a `mlp_completo`, porque la entrada contiene el estado residual `x`, que domina. Para atribuir algo a una cabeza usa `mlp_loo`.
- Animal vs fondo: en `entropy_l001` el exceso del hábitat no crece en el MLP (0.018 → 0.008 con `mlp_loo`). En `unconstrained` sube con `mlp_solo`/`mlp_completo`, pero viene del residual.
- **Límites:** contrafactuales no exactos (MLP no lineal), 1 semilla, 2 modelos, atributos por clase.


## 6. CelebA: el atributo subjetivo `Attractive`

Notebook: `notebooks/02112026_celeba_attractive/celeba_attractive.ipynb` (requiere la caché que crea el notebook 4). Mismos 9 modelos y mismas regiones que en la sección 4. Las figuras llevan leyenda en la propia imagen (qué significa rojo/azul, qué es cada barra).

**Pregunta.** `Attractive` es subjetivo y no tiene posición propia. Con cabezas que miran ojos, boca o pelo, ¿podemos explicar **cómo** se predice? Es decir: ¿qué cabezas pesan y qué miran?

**Método.** (1) Dónde mira cada cabeza con `Attractive`=1 vs 0 (mapas con leyenda). (2) Sonda lineal sobre el mapa de atención. (3) Control por correlatos (Male × Young). (4) **Probing de la salida de cada cabeza** (12 números por cabeza) y de las 4 juntas. (5) **Ablación**: apagar cada cabeza (reemplazada por su media) y medir el **peso** = AUC completo − AUC sin la cabeza, con intervalo bootstrap; también el **efecto por imagen** en la puntuación y apagar las 4 cabezas a la vez. (6) Imágenes de ejemplo con atención por cabeza y efecto de apagarla.

**Hallazgos** (valores en los CSV de `experiments/02112026_celeba_attractive/`).
- Las cabezas penalizadas tienen región reconocible (ojos, boca, pelo/fondo); sin penalizar son difusas.
- En imágenes concretas, la cabeza de ojos empuja la puntuación hacia la etiqueta correcta (p. ej. +0.7 a +1.1 en Attractive y −1.1 a −1.4 en no-Attractive en `constrained_rank` s0).
- **Pero el efecto promedio es pequeño:** alineación ~0.1 logits en cabezas de ojos/boca (la puntuación varía ~2 logits); apagar una cabeza baja el AUC ≤ 0.013 (sobre 0.87), y las 4 a la vez 0.001–0.030. La atención del último bloque apenas sostiene `Attractive`; la información viaja por el residual y el MLP.
- Las cabezas de ojos/boca tienden a empujar más que las de pelo/fondo (0.101 vs 0.049) pero p≈0.08 (con dependencia ignorada) y no se repite en todas las semillas: es una pista.
- La salida de cada cabeza sí contiene información (AUC 0.60–0.81; estratificado 0.58–0.74), de forma redundante entre cabezas.
- Sobre *dónde mira* solamente (sonda sobre el mapa de atención), en los modelos penalizados el AUC estratificado (0.64–0.67) queda en el techo de los correlatos (0.652).

**Límites.** Ablación por la media solo en el [CLS] del bloque 4; ejemplos muy seguros; control parcial (Male × Young); 3 semillas; etiqueta subjetiva y sesgada; solo bloque 4.

---

## 7. CelebA: descripción del dataset

Notebook `notebooks/02112026_celeba_dataset/celeba_dataset.ipynb` (10 figuras 16:9, `01_…` a `10_…`). Datos clave: 202 599 imágenes (162 770 / 19 867 / 19 962), 40 atributos binarios, imagen 178×218 → recorte 178×178 → **64×64**, parches 4×4 = **256**. Landmarks se mueven ≲ 3 px (< 1 parche); ojos a 13.8 ± 1.2 px. 58 % mujeres / 42 % hombres; Attractive 51 %, corr. con Male −0.39 y Young +0.39. Piso "clase mayoritaria" 0.80 frente a ~0.88 de los modelos.

## Cómo reproducir

1. Entorno del proyecto (`venv`), carpeta `interpretable-transformers/`. Los notebooks usan `Path.cwd().parents[1]` como raíz: **ejecútalos desde su propia carpeta** (Jupyter lo hace solo).
2. Notebook 4: necesita `data/celeba/` (Kaggle) y crea una caché `data/celeba/celeba_test64.npz` la primera vez.
3. Notebook 5: lento (los nulos); en MPS/CPU tarda varios minutos.
4. Entrenar entropía: en `experimentation/training.ipynb` fija `lambda_entropy`; o `python experimentation/run_training_notebook.py --dataset AwA2 --lambda-entropy 0.01 --name entropy`.

## Límites generales (léelos antes de citar algo en la tesis)

- Una sola semilla en todo lo de entropía y en AwA2; el entrenamiento de los modelos de entropía se hizo en CPU (los baselines, en MPS).
- Los atributos de AwA2 son **por clase**: no distinguen animal de fondo ni premian localización.
- En CelebA el análisis cubre solo las 3 variantes ya entrenadas; falta entropía y falta una prueba causal (ablación por región) del papel de cada cabeza.
- Todo es correlacional/descriptivo salvo la ablación, y esa tiene n = 1000 y diferencias dentro del ruido.

## Residuos en tu disco que puedes borrar

`data/awa2/_stage_tmp/` y `_stage_tmp/` (restos de transferencia), `notebooks/**/logs/executed_AwA2_entropy_l01_s0_smoke.ipynb`, y `experimentation/scripts/celeba_*.py` (reemplazados por el notebook).
