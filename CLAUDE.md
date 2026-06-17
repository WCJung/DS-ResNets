# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DS-ResNets (Dynamic System for ResNets) is a research project for feature extraction and distance-based analysis using modified ResNet architectures. The pipeline has two stages:

1. **Train & extract** (`main.py`): Fine-tune a ResNet, extract intermediate block features per sample, and save them as `.pt` files.
2. **Distance analysis** (`dist_calc.py`): Load saved features, compute inter/intra-class Minkowski distances, and build nearest-neighbor sequences.

## Running the Pipelines

```bash
# Training + feature extraction
python main.py --model resnet --bc [3,4,6,3] --weight False

# Distance analysis on saved features
python dist_calc.py
```

No build step, test suite, or linter is configured.

## Dependencies

PyTorch, torchvision, NumPy, scikit-learn, matplotlib, numba. No `requirements.txt` exists — install manually as needed.

## Architecture

### `models/ResNets.py`
Modified ResNet with a `Clipper` class inserted between residual layers. `Clipper(h_split, w_split)` spatially tiles a feature map (e.g., 2×2) and re-packs the tiles into the channel dimension, increasing channel count while reducing spatial resolution — this is the "dynamic system" mechanism. Clipper instances (`clip1`, `clip2`, `clip3`) are applied in `_forward_impl` between `layer1`–`layer4`. The `_make_layer` method uses per-layer group counts (4, 8, 16, 32) rather than standard ResNet grouping. The `Bottleneck` here reads from `planes * expansion` in `conv1` rather than from `inplanes`.

### `models/models.py`
Thin factory: `model_(params, device)` constructs a `ResNet(Bottleneck, layers)` with optional pretrained weights; `transfer(model, seed, n_class, feat)` replaces `model.fc` for fine-tuning.

### `utils/task.py` — `DistanceMeasure`
Core analysis class. Initialized with feature array `(N, n_blocks, feat_dim)` and labels; builds a class-indexed dict. `task_1()` computes per-class Minkowski distances across all pairs and saves `.npy` results under `Result/task1/`. `task_2()` (module-level function) finds nearest-sequence matches.

### `utils/builders.py`
- `seq_builder`: for each sample, chains the `n_blocks` nearest neighbors by block-0 Minkowski distance; saves Targets, Series, SeqInfo, MaxList `.npy` files under `task2/`.
- `create_dict`: groups features by label into a `{label: (n, n_blocks, feat_dim)}` dict, optionally applying softmax/sigmoid.
- `load_files`: loads chunked `.npy` feature files (100 chunks per block) from a structured directory; optionally generates t-SNE plots per block.

### `utils/distance.py`
Four metrics: `minkovski` (PyTorch, generalizes to any p), `euclidean`, `manhattan`, `chebychev` (NumPy). Note: `utils/task.py` also imports `hausdorff_distance` from here, but it is not yet implemented.

### `utils/norms.py`
`softmax`, `sigmoid`, `norm` (min-max), `harm_mean`. The `init_random` helper (referenced in `main.py` as `init_rand`) should seed NumPy/random/torch for reproducibility.

## Known Incomplete Areas

`main.py` references `init_rand`, `load_data`, `Exprob`, and `tt_set` which are not imported or defined — the training/extraction entry point is a scaffold that needs these stubs filled in. `dist_calc.py` has a typo (`our_dir` vs `out_dir`) on line 18.
