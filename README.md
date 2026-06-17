# DS-ResNets
The theory of Dynamical System for ResNets(DS-ResNets) — Explainability on modified ResNet architectures from dynamical systems perspectives.

## Install

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Train & Extract Features

Fine-tunes a DS-ResNet and saves per-block features for every test sample.

```bash
python main.py
```

Key options in `main.py` (edit directly):

| Variable | Description | Example |
|---|---|---|
| `MODEL_NAME` | Backbone to use | `"ds_resnet18"` / `"ds_resnet50"` |
| `DATA_NAME` | Dataset | `"MNIST"` / `"CIFAR10"` / `"IMAGENET10"` |
| `USE_BLOCK_FC` | Train per-block linear probes | `True` |
| `USE_AVGPOOL` | Use avgpool before main fc | `True` |

Outputs written to `prob/`, `prob_fc/`, `pix/`.

---

### 2. Distance Analysis

Computes inter/intra-class Minkowski distances, derives the expansive constant.
We build pseudo-orbit and compute the shadowing constant. By the main theorem we can get lower bound of topological stability constant from Shg(φ)/Lip(g) ≤ Tg(φ).

```bash
python dist_calc.py
```

Set `MODEL_NAME`, `DATA_NAME`, `USE_AVGPOOL` to match step 1.

Outputs written to `Result/task1/`, `task2/`, `Result/*_epsilon.npy`, `Result/*_theorem.npy`.

---

### 3. Inspect Examples

Selects and visualises concrete instability examples:

- **[Expansive]** Top-10 same-class pairs whose `block_fc` output distance is anomalously large (blocks ≥ 50 % depth).
- **[Shadowing]** Pseudo-orbit chains ranked by prediction-class flip count — most robust chain and most unstable chain.

```bash
python inspect_examples.py
```

Set `MODEL_NAME` and `DATA_NAME` to match steps 1–2. Requires `prob_fc/` (step 1 with `USE_BLOCK_FC=True`) and `task2/` (step 2).

Output PNGs saved to `Result/inspect/`.

---

## Utilities

```bash
# Architecture comparison (no GPU required)
python test_arch_compare.py

# End-to-end pipeline smoke test
python test_pipeline.py
```

## Output Structure

```
prob/          raw per-block features   (N, feat_dim) per block
prob_fc/       per-block fc logits      (N, n_class)  per block
pix/           labels saved alongside features
Result/task1/  class-wise distance matrices
task2/         SeqInfo, MaxList (pseudo-orbit chains)
Result/        ε, theorem constants, inspect PNGs
```
