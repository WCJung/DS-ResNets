# DS-ResNets

The theory of Dynamical Systems for ResNets (DS-ResNets) — quantifying the
**topological stability** (g-expansivity, g-shadowing, topological
g-stability) and **topological entropy** (FTTE) of trained ResNets on
dimension-preserving ResNet variants.

All quantities are computed in the pseudometric
**d_g(x, y) = d(g(x), g(y))** — the observation space of the block-wise
classifiers g — matching the definitions in the paper.

### Dimension-preserving models

Every residual block acts on a common phase space (flattened dim 200,704 —
the Clipper reshapes are coordinate permutations, hence l2-isometries).
The block *interior* is free; three families are provided:

| Name | Block | Idea |
|---|---|---|
| `ds_resnet18` / `ds_resnet50` | `Bottleneck` | grouped bottleneck (baseline) |
| `ds_wide18` / `ds_wide50` | `WideBottleneck` | Wide-ResNet: 2× bottleneck width + dropout, **pre-activation** (`φ(x) = x + F(x)` exactly) |
| `ds_resnext18` / `ds_resnext50` | `ResNeXtBottleneck` | ResNeXt: cardinality-32 aggregated transforms |

`*18` = 8 blocks, `*50` = 16 blocks. The registry lives in
`models/models.py` (`DS_MODELS`).

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
| `--model` | `ds_resnet18` | any name in `DS_MODELS` (table above) / `resnet18` / `resnet50` |
| `--data` | `MNIST` | `MNIST` / `CIFAR10` / `IMAGENET10` (Imagenette) |
| `--use-block-fc / --no-use-block-fc` | on | train per-block linear probes (required for analysis) |
| `--use-avgpool / --no-use-avgpool` | on | avgpool before the main fc |
| `--save-raw-feat / --no-save-raw-feat` | off | also dump raw per-block features (D=200,704) for legacy `--space feat`. Off by default — accumulating all blocks in RAM needs ~128 GB on `*50` models and was the OOM cause. |
| `--epochs / --lr / --batch-size / --seed` | 100 / 5e-5 / 64 / 13 | training hyperparameters |

Outputs: `{model}_{data}.pt`, `{model}_{data}_multifc.pt`,
`Result/{data}_{model}_metrics.npy` (F1/Loss/Acc),
`prob_fc/`, `pix/.../{data}_label.pt` (and `prob/` only with `--save-raw-feat`).
The block-wise fc logits in `prob_fc/` are all the `--space prob`/`logit`
analysis needs; the raw `prob/` features feed only the legacy `--space feat`.

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

To dump the actual image pair that attains the g-expansive constant
(the cross-class pair reported as `class a vs b, block k`):

```bash
python save_expansive_pair.py --model ds_resnet18 --data MNIST
```

Saves the two individual PNGs plus an annotated side-by-side figure to
`Result/expansive_pair/` (any model/data that dist_calc has been run on).

---

### 3. Entropy (FTTE) — `entropy_calc.py`

Computes the finite-time trajectory entropy in the same d_g space, using the
trajectory pseudometric `d_g^T(x,y) = max_t d(g_t(φ^t x), g_t(φ^t y))`:

1. **Proposition 1 diagnostics** — `intra_max` (largest within-class d_g^T),
   `cross_min` (smallest cross-class d_g^T, which equals the g-expansive
   constant), and the ε-window `[intra_max, cross_min)` on which `s_T = m`
   (hence `Δh_T = 0`) is guaranteed.
2. **FTTE grid** — for each ε (auto quantile grid, or `--eps 0.05,0.1,...`):
   `s_T(ε)` via greedy maximal (T, ε)-separated packing (a lower bound of the
   NP-hard maximum), `h_T(ε) = log(s_T)/T`, and the FTTE gap
   `Δh_T(ε) = h_T(ε) − log(m)/T` with its Proposition-2 sign (`s` vs `m`).

```bash
python entropy_calc.py --model ds_resnet18 --data MNIST
python entropy_calc.py --model ds_wide50 --data CIFAR10 --device cuda
```

Outputs: `Result/{data}_{model}_entropy.npy`.

---

### 4. Inspect Examples — `inspect_examples.py`

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

Runs all models × 3 datasets sequentially (train → extract → stability →
entropy) with logging to `logs/`. Trim `MODELS` in the config block if you
don't need all 8. `print_results.py` prints the aggregated Table 1
(including `h_T` / `Δh_T` columns) and saves example images.

```bash
python run_all.py
python print_results.py
```

## IsoLift-ResNeXt — multi-dataset common-state model (experimental)

A second architecture family (`models/isolift.py`): instead of sharing the
post-stem space, each dataset's **raw input** is lifted isometrically into
one common state space ℝ^{48×56×56} (= 3·224·224 = 150,528):

| Dataset | E_d (exact isometry, frozen) |
|---|---|
| MNIST 1×28×28 | center zero-embed → 56×56, unit-norm 1→48 channel lift |
| CIFAR10 3×32×32 | center zero-pad → 56×56, semi-orthogonal 1×1 (WᵀW=I₃) |
| IMAGENET10 3×224×224 | `PixelUnshuffle(4)` (coordinate permutation) |

Pipeline: `x_d → E_d → A_d (domain adapter) → shared backbone
(stage transitions PixelUnshuffle(2): 48×56² = 192×28² = 768×14²) → GAP →
domain head`.  The shared backbone comes in the same three families as the
DS models (`--family`): `resnet` (width C/4 baseline), `wide` (width C/2 +
dropout, pre-activation), `resnext` (width C/3, grouped 3×3).  Two modes: `provable` (hard spectral constraints, i-ResNet
style, Lip(αF) ≤ ρ < 1 → invertible blocks, no norm layers / adapters) and
`performance` (domain-specific BatchNorm + adapters + soft spectral
penalty).  Loss: Σ CE + λ_geo·(distance-ratio hinge on u₀) + λ_lip·(spectral
penalty above ρ).

```bash
python train_isolift.py --family resnet --mode performance
python train_isolift.py --mode provable --lambda-lip 0
python test_isolift.py    # isometry / dimension / Lip<ρ smoke test
```

Outputs: `isolift_{family}_{mode}.pt`, `Result/isolift_{family}_{mode}_metrics.npy`.

**Stability / entropy analysis on IsoLift** — train block-wise probes on the
frozen backbone, then run the existing analysis with the IsoLift tag:

```bash
python extract_isolift.py --family resnet --mode performance
python dist_calc.py    --model isolift_resnet_performance --data MNIST --space logit --device cuda
python entropy_calc.py --model isolift_resnet_performance --data MNIST --space logit --device cuda
```

`extract_isolift.py` writes `prob_fc/{data}/isolift_{family}_{mode}/`,
labels, and a `*_multifc.pt` probe checkpoint (block count is inferred from
the saved files, so `dist_calc`/`entropy_calc` accept the tag directly).

## Utilities

```bash
# Architecture comparison + Clipper isometry check (no GPU required)
python test_arch_compare.py

# End-to-end smoke test of the analysis modules (no training required)
python test_pipeline.py
```

## Code Map

```
models/ResNets.py      dimension-preserving ResNet skeleton (Clipper = isometric reshape)
models/blocks.py       WideBottleneck / ResNeXtBottleneck (φ: R^n → R^n variants)
models/models.py       DS_MODELS registry + build_ds_model()
utils/trajectory.py    (N, T, D) trajectory loader in the d_g space
utils/expansive.py     g-expansive constant (min-max, vectorised torch.cdist)
utils/shadowing.py     pseudo-orbits, true-orbit tracing, Sh_g estimator
utils/entropy.py       FTTE: separated sets, h_T, Δh_T, Prop.1 diagnostics
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
Result/        metrics, epsilon, shadowing, theorem, entropy, inspect PNGs
```
