# IsoLift ResNets: Finite-Time Topological Dynamics

This repository implements the models and empirical analyses used in
*Stability of ResNets: A Topological Dynamics Perspective*.

The main experiments use **IsoLift-ResNet**, **IsoLift-Wide ResNet**, and
**IsoLift-ResNeXt** on **MNIST**, **CIFAR-10**, and **Imagenette**. Fixed
isometric input lifts and dimension-preserving stage transitions place all
representations in a common ambient Euclidean state space.

## What the repository estimates

For a trained network, let $x_t$ be the representation after residual block
$t$, and let $g_t$ be the linear observation map attached to that block.
The empirical analysis uses the block-wise observation family

```math
G=\{g_t\}_{t=0}^{T-1}
```
and the finite-depth trajectory distance

```math
d_G^T(x,y) = \max_{0\le t<T} \|g_t(x_t)-g_t(y_t)\|_2.
```
The repository estimates:

- the empirical $g$-expansive constant $\varepsilon_g$, which identifies
  the weakest separation between samples with different labels;
- the empirical $g$-shadowing constant $Sh_g$;
- the observation sensitivity
```math
\mathrm{Lip}(G)=\max_t\mathrm{Lip}(g_t);
```
- the certified stability lower bound
```math
B_g=\frac{Sh_g}{\mathrm{Lip}(G)}\le T_g;
```
- finite-time trajectory entropy (FTTE), estimated by greedy separated-set
  packing.

The exact topological $g$-stability constant $T_g$ is **not** computed.
The code reports the theorem-based lower bound $B_g$.

The paper reports the main IsoLift results in **logit space**. Use
`--space logit` when reproducing the reported tables and figures.

## Terminology

The code retains two legacy mode tags:

| Code tag | Terminology used in the paper |
|---|---|
| `performance` | Lipschitz-unconstrained configuration |
| `provable` | Lipschitz-constrained configuration |

The term *Lipschitz-unconstrained* means that no hard branch-wise Lipschitz
bound is enforced; a soft spectral penalty may still be used.

## IsoLift architecture

Each dataset is mapped by a fixed, non-trainable lift
$E_d:\mathcal X_d\to\mathbb R^{48\times56\times56}$.

| Dataset | Fixed lift $E_d$ |
|---|---|
| MNIST | center zero-embedding into $56\times56$, followed by a unit-norm $1\to48$ channel lift |
| CIFAR-10 | center zero-padding into $56\times56$, followed by a semi-orthogonal $1\times1$ channel lift |
| Imagenette | `PixelUnshuffle(4)`, a coordinate permutation |

The fixed lifts preserve within-dataset Euclidean distances exactly at the
lift. Stage transitions use `PixelUnshuffle(2)` and preserve the flattened
state dimension:

```text
48×56² = 192×28² = 768×14² = 3072×7² = 150,528
```

The fixed lifts and `PixelUnshuffle` transitions are isometries at the points
where they are applied. Learned residual blocks may expand or contract
distances; measuring that evolution is the purpose of the trajectory
analysis.

In the Lipschitz-unconstrained configuration, a dataset-specific adapter is
applied after the fixed lift. The complete entry map is therefore not
necessarily isometric; its distortion is controlled empirically by the
geometry loss.

Raw observation-space quantities such as $\varepsilon_g$ remain dependent
on the scale of the learned probes. They should be interpreted together with
$\mathrm{Lip}(G)$ or a scale-normalized quantity.

### Model families

| Family | Residual-branch design |
|---|---|
| `resnet` | standard bottleneck |
| `wide` | wider bottleneck |
| `resnext` | grouped transform with increased cardinality |

The paper evaluates each family at:

- $T=16$: ResNet-50-type stage configuration `[3,4,6,3]`;
- $T=33$: ResNet-101-type stage configuration `[3,4,23,3]`.

Within each training configuration, the architecture families share the same
fixed lifts, state dimensions, stage transitions, block counts, and
optimization protocol. The Lipschitz-constrained and
Lipschitz-unconstrained configurations differ in several architectural and
training components; their comparison should therefore be interpreted as a
comparison of the complete configurations rather than as an isolated causal
effect of the hard spectral constraint.

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the paper experiments

### 1. Train IsoLift models

The checkpoint tag is

```text
isolift_{family}_{mode}{tag_suffix}
```

For example, `isolift_resnet_performance_50` denotes the
Lipschitz-unconstrained IsoLift-ResNet-50 configuration.

#### $T=16$: ResNet-50-type models

```bash
# Lipschitz-unconstrained
python train_isolift.py --family resnet  --layers 3,4,6,3 --tag-suffix _50 --mode performance
python train_isolift.py --family wide    --layers 3,4,6,3 --tag-suffix _50 --mode performance
python train_isolift.py --family resnext --layers 3,4,6,3 --tag-suffix _50 --mode performance \
    --width-ratio 2 --cardinality 32

# Lipschitz-constrained
python train_isolift.py --family resnet  --layers 3,4,6,3 --tag-suffix _50 --mode provable --lambda-lip 0
python train_isolift.py --family wide    --layers 3,4,6,3 --tag-suffix _50 --mode provable --lambda-lip 0
python train_isolift.py --family resnext --layers 3,4,6,3 --tag-suffix _50 --mode provable \
    --width-ratio 2 --cardinality 32 --lambda-lip 0
```

#### $T=33$: ResNet-101-type models

```bash
# Lipschitz-unconstrained
python train_isolift.py --family resnet  --layers 3,4,23,3 --tag-suffix _101 --mode performance
python train_isolift.py --family wide    --layers 3,4,23,3 --tag-suffix _101 --mode performance
python train_isolift.py --family resnext --layers 3,4,23,3 --tag-suffix _101 --mode performance \
    --width-ratio 2 --cardinality 32

# Lipschitz-constrained
python train_isolift.py --family resnet  --layers 3,4,23,3 --tag-suffix _101 --mode provable --lambda-lip 0
python train_isolift.py --family wide    --layers 3,4,23,3 --tag-suffix _101 --mode provable --lambda-lip 0
python train_isolift.py --family resnext --layers 3,4,23,3 --tag-suffix _101 --mode provable \
    --width-ratio 2 --cardinality 32 --lambda-lip 0
```

### 2. Extract block-wise observations

The backbone is frozen and a dataset-specific linear probe is trained after
each residual block.

```bash
python extract_isolift.py --family resnet --mode performance --tag-suffix _50
```

The extraction step writes block-wise observations under `prob_fc/`, test
labels, and a `*_multifc.pt` probe checkpoint. Downstream scripts infer the
block count from the saved files, so the same checkpoint tag must be used
throughout training, extraction, and analysis.

### 3. Compute $\varepsilon_g$, $Sh_g$, $\mathrm{Lip}(G)$, and $B_g$

```bash
python dist_calc.py \
    --model isolift_resnet_performance_50 \
    --data MNIST \
    --space logit \
    --device cuda
```

The empirical $g$-expansive constant is computed as

```math
\varepsilon_g
=
\min_{y_i\ne y_j}
\max_t
\|g_t(x_t^{(i)})-g_t(x_t^{(j)})\|_2.
```
The shadowing estimator constructs depth-consistent pseudo-orbits and
compares them with true trajectories in the same observation space.

For logit-space linear probes, the spectral norm of the linear weight gives
the exact Lipschitz constant with respect to the pooled representation. If
softmax or global average pooling is included explicitly in a different
analysis mode, the reported composition value is an upper bound rather than
an exact global constant.

The exact $T_g$ is not computed. The reported certificate is

```math
B_g=\frac{Sh_g}{\mathrm{Lip}(G)}.
```
Some output filenames retain legacy terminology, including
`*_theorem.npy`; these files store the theorem-based certificate rather than
the unknown exact stability constant.

### 4. Compute FTTE

```bash
python entropy_calc.py \
    --model isolift_resnet_performance_50 \
    --data MNIST \
    --space logit \
    --device cuda
```

For an observation scale $\varepsilon$, the script constructs a greedy
maximal $(T,\varepsilon)$-separated set. Its size

```math
\widehat{s}_T(\varepsilon)
```
is a lower bound on the exact packing number $s_T(\varepsilon)$. The
reported empirical FTTE is therefore based on $\widehat{s}_T$:

```math
\widehat{h}_T(\varepsilon)
=
\frac{1}{T}\log \widehat{s}_T(\varepsilon).
```
The script also reports:

- `intra_max`: largest within-class trajectory distance;
- `cross_min`: smallest different-label trajectory distance, equal to the
  empirical $g$-expansive constant;
- the candidate class-scale interval `[intra_max, cross_min)` when it is
  nonempty;
- the empirical entropy gap relative to $m$ class labels.

A value $\widehat{s}_T(\varepsilon)=m$ indicates that the greedy packing
contains $m$ visible trajectory patterns. It does not by itself prove a
one-to-one match between those patterns and the $m$ classes.

### 5. Plot block-wise observation sensitivity

```bash
python lip_curve.py \
    --families resnet,wide,resnext \
    --depths 50,101 \
    --modes performance,provable \
    --space logit
```

The script plots

```math
t\mapsto \mathrm{Lip}(g_t)
```
and summarizes the maximum, mean, standard deviation, peak ratio, and peak
block. In the paper terminology, solid `performance` curves correspond to
Lipschitz-unconstrained configurations and dashed `provable` curves
correspond to Lipschitz-constrained configurations.

### 6. Run the full IsoLift analysis

```bash
python run_isolift_analysis.py \
    --families resnet,wide,resnext \
    --modes performance,provable \
    --tag-suffix _50 \
    --space logit
```

Run the command again with `--tag-suffix _101` for $T=33$.

## Example visualization

To save the pair that attains the empirical $g$-expansive constant:

```bash
python save_expansive_pair.py \
    --model isolift_resnet_performance_50 \
    --data MNIST
```

This pair consists of samples with different labels and realizes the weakest
finite-depth trajectory separation.

The script `inspect_examples.py` also contains a legacy same-label analysis.
Those examples should be described as **within-class fragmentation** or
**large within-class trajectory variation**, not as $g$-expansive pairs.

## Main outputs

```text
prob_fc/       block-wise probe outputs
pix/           test labels
Result/        metrics, expansive constants, shadowing estimates,
               stability certificates, FTTE results, curves, and figures
```

Common outputs include:

```text
Result/{data}_{model}_metrics.npy
Result/{data}_{model}_epsilon.npy
Result/{data}_{model}_shadowing.npy
Result/{data}_{model}_theorem.npy     # legacy filename for the certificate
Result/{data}_{model}_entropy.npy
Result/lip_curves/
```

## Scope and interpretation

- The theoretical appendix uses an autonomous self-map and a theoretical
  observation map $g$. The code evaluates finite-depth empirical
  quantities along the actual sequence of trained residual blocks with
  block-wise maps $G=\{g_t\}$.
- $B_g$ is a lower bound, not the exact topological stability constant.
- Raw $\varepsilon_g$ and $\mathrm{Lip}(G)$ depend on probe scale.
- The greedy FTTE packing is a lower bound on the exact packing number.
- The constrained and unconstrained model configurations differ in more than
  the hard spectral constraint.

- [Legacy DS-ResNet pipeline](LEGACY.md)
- [Developer notes and experimental features](DEVELOPER.md)
