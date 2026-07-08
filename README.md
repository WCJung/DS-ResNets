# DS-ResNets

The theory of Dynamical Systems for ResNets (DS-ResNets) — quantifying the
**topological stability** of trained ResNets (g-expansivity, g-shadowing,
topological g-stability) on dimension-preserving ResNet variants.

All stability quantities are computed in the pseudometric
**d_g(x, y) = d(g(x), g(y))** — the observation space of the block-wise
classifiers g — matching the definitions in the paper.

## Install

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Train & Extract — `main.py`

Trains a backbone, evaluates F1/Loss/Acc, and (DS-ResNet only) saves per-block
raw features and block-wise fc logits for every test sample.

```bash
python main.py --model ds_resnet18 --data MNIST
python main.py --model ds_resnet50 --data CIFAR10
python main.py --model resnet18   --data MNIST     # accuracy baseline only
```

| Option | Default | Description |
|---|---|---|
| `--model` | `ds_resnet18` | `ds_resnet18` (8 blocks) / `ds_resnet50` (16) / `resnet18` / `resnet50` |
| `--data` | `MNIST` | `MNIST` / `CIFAR10` / `IMAGENET10` (Imagenette) |
| `--use-block-fc / --no-use-block-fc` | on | train per-block linear probes (required for analysis) |
| `--use-avgpool / --no-use-avgpool` | on | avgpool before the main fc |
| `--epochs / --lr / --batch-size / --seed` | 100 / 5e-5 / 64 / 13 | training hyperparameters |

Outputs: `{model}_{data}.pt`, `{model}_{data}_multifc.pt`,
`Result/{data}_{model}_metrics.npy` (F1/Loss/Acc),
`prob/`, `prob_fc/`, `pix/.../{data}_label.pt`.

---

### 2. Stability Analysis — `dist_calc.py`

Computes, in the d_g observation space (default `--space prob` =
softmax probabilities of the block-wise fc):

1. **g-expansive constant** (Definition 1) — *min over cross-class pairs of the
   max over blocks* of d_g (min–max, as the definition requires).
2. **g-shadowing constant** (Definition 2) — builds nearest-neighbour
   pseudo-orbits, finds for each chain the best-tracing true orbit
   (eps_i = min over candidates of the **max over blocks** tracing error), and
   reports the ratio curve `delta*(eps)/eps` with its smallest-eps value as the
   Sh_g estimate (empirical analogue of the liminf in Morales–Nguyen).
3. **Lip(g)** — exact spectral norm of each block fc (single linear layer), with
   a softmax correction factor (≤ 1/2) when `--space prob`, and an avgpool
   correction `1/sqrt(k)` reported for the main fc.
4. **Table 1 row** printed to the terminal:
   F1 / Loss / g-expansive / g-shadowing / Lip(g).
   The topological g-stable constant is *not* printed — compute it via
   Theorem 1: `T_g(phi) >= Sh_g(phi) / Lip(g)`.

```bash
python dist_calc.py --model ds_resnet18 --data MNIST
python dist_calc.py --model ds_resnet50 --data CIFAR10 --device cuda
```

| Option | Default | Description |
|---|---|---|
| `--space` | `prob` | d_g space: `prob` (softmax), `logit`, `feat` (legacy raw features) |
| `--legacy-orbit` | off | old chain mode (always block 0→1 transition) instead of depth-consistent |
| `--allow-cross-class` | off | let pseudo-orbit neighbours cross class boundaries |
| `--cross-class-trace` | off | let tracing candidates cross class boundaries |
| `--n-samples` | all | subsample size (recommended for `--space feat`) |
| `--device` | cpu | `cuda` for GPU-accelerated distance computation |

Outputs: `Result/{data}_{model}_epsilon.npy`, `_shadowing.npy`, `_theorem.npy`,
`task2/{data}_{model}_{SeqInfo,MaxList,ClassInfo,TraceEps}.npy`.

---

### 3. Inspect Examples — `inspect_examples.py`

Selects and visualises concrete instability examples:

- **[Expansive]** Top-10 same-class pairs whose block_fc output distance is
  anomalously large (blocks ≥ 50 % depth).
- **[Shadowing]** Pseudo-orbit chains ranked by prediction-class flip count —
  most robust chain and most unstable chain.

```bash
python inspect_examples.py --model ds_resnet18 --data MNIST
```

Output PNGs saved to `Result/inspect/`.

---

### All combinations — `run_all.py`

Runs 4 models × 3 datasets sequentially (train → extract → analysis) with
logging to `logs/`. `print_results.py` prints the aggregated Table 1 and saves
example images.

```bash
python run_all.py
python print_results.py
```

## Utilities

```bash
# Architecture comparison + Clipper isometry check (no GPU required)
python test_arch_compare.py

# End-to-end smoke test of the analysis modules (no training required)
python test_pipeline.py
```

## Code Map

```
models/ResNets.py      dimension-preserving ResNet (Clipper = isometric reshape)
models/models.py       build_ds_resnet()
utils/trajectory.py    (N, T, D) trajectory loader in the d_g space
utils/expansive.py     g-expansive constant (min-max, vectorised torch.cdist)
utils/shadowing.py     pseudo-orbits, true-orbit tracing, Sh_g estimator
utils/lipschitz.py     Lip(g) from checkpoint state_dict (spectral norms)
utils/orbit_analysis.py  example selection for inspect/print scripts
utils/stubs.py         data loading, train/evaluate, block-output extraction
```

## Output Structure

```
prob/          raw per-block features   (N, 200704) per block
prob_fc/       per-block fc logits      (N, n_class) per block
pix/           test labels
task2/         SeqInfo, MaxList, ClassInfo, TraceEps (pseudo-orbit chains)
Result/        metrics, epsilon, shadowing, theorem, inspect PNGs
```
