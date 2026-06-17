#!/usr/bin/env python3
"""
print_results.py — Results summary table + example image extraction.

Reads all outputs produced by run_all.py (or main.py + dist_calc.py) and:
  1. Prints a formatted results table (Acc, ε, Shg, Lip, Tg lower bound)
  2. Saves example images for:
       [Expansive]  Top-K same-class pairs with the largest block_fc distance
       [Shadowing]  Most robust and most unstable pseudo-orbit chains

Output images saved to:  Result/examples/{DATA}_{model}/

Prerequisites:
  - run_all.py (or main.py + dist_calc.py) completed for each combination
  - prob_fc/, task2/, Result/*_accuracy.npy, Result/*_epsilon.npy,
    Result/*_theorem.npy must exist for the combinations you want to inspect

Usage:
  python print_results.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')           # headless server — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.stubs import load_data
from utils.orbit_analysis import find_expansive_outliers, analyze_pseudo_orbit_stability


# ── Configuration ─────────────────────────────────────────────────────────────
MODELS   = ['resnet18', 'resnet50', 'ds_resnet18', 'ds_resnet50']
DATASETS = ['MNIST', 'CIFAR10', 'IMAGENET10']

DS_LAYERS_MAP = {
    'ds_resnet18': [2, 2, 2, 2],
    'ds_resnet50': [3, 4, 6, 3],
}

NORM_STATS = {
    'MNIST':      ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'CIFAR10':    ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    'IMAGENET10': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

TOP_K_EXPANSIVE = 3    # number of same-class outlier pairs to visualise
TOP_K_SHADOWING = 1    # number of robust / unstable chains to visualise each
MIN_DEPTH_RATIO = 0.5  # only inspect blocks >= 50% depth for expansive

OUT_DIR = "Result/examples"
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_npy(path, key=None):
    """Return float value from a .npy result file, or None if missing."""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True).item()
        return float(d[key]) if key else d
    except Exception:
        return None


def unnormalize(img_tensor, mean, std):
    img = img_tensor.clone().float()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


# ── 1. Results table ───────────────────────────────────────────────────────────

def _collect(model_name, data_name):
    tag = f"{data_name}_{model_name}"
    acc  = _load_npy(f"Result/{tag}_accuracy.npy",  "accuracy")
    eps  = _load_npy(f"Result/{tag}_epsilon.npy",   "epsilon")
    thm  = _load_npy(f"Result/{tag}_theorem.npy")
    shg  = float(thm["Shg_phi"])           if thm else None
    lip  = float(thm["Lip_g"])             if thm else None
    tg   = float(thm["Tg_phi_lower_bound"]) if thm else None
    return {"acc": acc, "eps": eps, "shg": shg, "lip": lip, "tg": tg}


def _fmt(val, fmt=".4f", na="—"):
    return f"{val:{fmt}}" if val is not None else na


def print_table():
    print()
    print("=" * 72)
    print("  DS-ResNets — Experiment Results")
    print("=" * 72)

    col_headers = f"  {'Model':<18} {'Acc(%)':>7}  {'ε':>8}  {'Shg(φ)':>8}  {'Lip(g)':>8}  {'Tg(φ)≥':>8}"
    separator   = "  " + "-" * 68

    for data_name in DATASETS:
        print(f"\n  ── {data_name} {'─' * (64 - len(data_name))}")
        print(col_headers)
        print(separator)
        for model_name in MODELS:
            r   = _collect(model_name, data_name)
            acc = _fmt(r["acc"], ".2f") if r["acc"] is not None else "—"
            if model_name in DS_LAYERS_MAP:
                row = (f"  {model_name:<18} {acc:>7}  "
                       f"{_fmt(r['eps'],'.5f'):>8}  "
                       f"{_fmt(r['shg'],'.4f'):>8}  "
                       f"{_fmt(r['lip'],'.4f'):>8}  "
                       f"{_fmt(r['tg'], '.4f'):>8}")
            else:
                row = (f"  {model_name:<18} {acc:>7}  "
                       f"{'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}")
            print(row)

    print()
    print("  Columns (DS-ResNet only):")
    print("    ε      : expansive constant (min inter-class block_fc distance)")
    print("    Shg(φ) : max pseudo-orbit step error across all chains")
    print("    Lip(g) : spectral norm of block_fc weights (max over blocks)")
    print("    Tg(φ)≥ : theorem lower bound  Tg ≥ Shg / Lip")
    print("=" * 72)


# ── 2. Image helpers ───────────────────────────────────────────────────────────

def _load_prob_fc(data_name, model_name, n_blocks):
    """Load all block_fc outputs. Returns dict {b: Tensor(N, n_class)} or None."""
    fc_dir = f"prob_fc/{data_name}/{model_name}"
    if not os.path.isdir(fc_dir):
        return None
    blocks = {}
    for b in range(n_blocks):
        p = f"{fc_dir}/{data_name}_block{b}.pt"
        if not os.path.exists(p):
            return None
        blocks[b] = torch.load(p)
    return blocks


def _load_labels(data_name, model_name):
    p = f"pix/resnet/{data_name}/{model_name}/test/{data_name}_label.pt"
    if not os.path.exists(p):
        return None
    return torch.load(p)


def _load_seq(data_name, model_name):
    si = f"task2/{data_name}_{model_name}_SeqInfo.npy"
    ml = f"task2/{data_name}_{model_name}_MaxList.npy"
    if not (os.path.exists(si) and os.path.exists(ml)):
        return None, None
    return (np.load(si, allow_pickle=True),
            np.load(ml, allow_pickle=True))


# ── 3. Expansive example images ───────────────────────────────────────────────

def plot_expansive(test_dataset, mean, std, outliers, data_name, model_name, save_path):
    """Grid: each row = one same-class pair (sample_i | sample_j)."""
    n = len(outliers)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.6 * n),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.5})
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, r in enumerate(outliers):
        for col, key in enumerate(["sample_i", "sample_j"]):
            idx = r[key]
            img, _ = test_dataset[idx]
            axes[row, col].imshow(unnormalize(img, mean, std))
            axes[row, col].axis('off')
            axes[row, col].set_title(
                f"idx={idx}\nlabel={r['label_i']}",
                fontsize=7, pad=2)
        # left annotation: block and distance
        axes[row, 0].set_ylabel(
            f"block {r['block']}\nd={r['distance']:.4f}",
            fontsize=7, rotation=0, labelpad=48, va='center')

    fig.suptitle(
        f"[Expansive] same-class outlier pairs  Top-{n}\n"
        f"{data_name} / {model_name}  (block_fc distance, blocks >= 50% depth)",
        fontsize=8, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved: {save_path}")


# ── 4. Shadowing example images ───────────────────────────────────────────────

def plot_chain(test_dataset, mean, std, info, title, save_path):
    """Horizontal strip of chain images, green=correct pred, red=wrong."""
    chain      = info["chain"]
    preds      = info["chain_pred_seq"]
    true_label = info["true_label"]
    n          = len(chain)

    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0),
                             gridspec_kw={'wspace': 0.08})
    if n == 1:
        axes = [axes]

    for t, (sidx, pred) in enumerate(zip(chain, preds)):
        img, _ = test_dataset[sidx]
        axes[t].imshow(unnormalize(img, mean, std))
        axes[t].axis('off')
        color = '#2ecc71' if pred == true_label else '#e74c3c'
        axes[t].set_title(
            f"t={t}\nidx={sidx}\npred={pred}",
            fontsize=6.5, color=color, pad=2)

    # legend patches
    correct_patch = mpatches.Patch(color='#2ecc71', label='pred = true')
    wrong_patch   = mpatches.Patch(color='#e74c3c', label='pred != true')
    fig.legend(handles=[correct_patch, wrong_patch],
               loc='lower center', ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    fig.suptitle(
        f"{title}  (true={true_label}, flips={info['flip_count']}, "
        f"deviations={info['deviate_count']}/{n})",
        fontsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved: {save_path}")


# ── 5. Per-combination image extraction ───────────────────────────────────────

def extract_images(model_name, data_name):
    if model_name not in DS_LAYERS_MAP:
        return   # images only for DS-ResNet (block analysis available)

    n_blocks = sum(DS_LAYERS_MAP[model_name])
    tag      = f"{data_name}_{model_name}"
    out_dir  = os.path.join(OUT_DIR, tag)

    # ── check prerequisites ────────────────────────────────────────────────
    prob_fc = _load_prob_fc(data_name, model_name, n_blocks)
    labels  = _load_labels(data_name, model_name)
    if prob_fc is None or labels is None:
        print(f"  [skip] {tag}: prob_fc/ or labels missing — run run_all.py first.")
        return

    # ── load dataset for actual images ────────────────────────────────────
    print(f"  Loading dataset {data_name} for image extraction ...")
    try:
        _, test_dataset = load_data(data_name)
    except Exception as e:
        print(f"  [skip] {tag}: dataset load failed — {e}")
        return

    mean, std = NORM_STATS[data_name]
    os.makedirs(out_dir, exist_ok=True)

    # ── [Expansive] ────────────────────────────────────────────────────────
    print(f"  [{tag}] [Expansive] searching top-{TOP_K_EXPANSIVE} same-class outlier pairs ...")
    outliers = find_expansive_outliers(
        prob_fc, labels, n_blocks,
        top_k=TOP_K_EXPANSIVE, min_depth_ratio=MIN_DEPTH_RATIO)

    if outliers:
        for r in outliers:
            print(f"    label={r['label_i']}  block={r['block']:>2d}  "
                  f"dist={r['distance']:.4f}  "
                  f"idx ({r['sample_i']}, {r['sample_j']})")
        plot_expansive(
            test_dataset, mean, std, outliers, data_name, model_name,
            save_path=os.path.join(out_dir, "expansive.png"))
    else:
        print(f"    no same-class pairs found.")

    # ── [Shadowing] ────────────────────────────────────────────────────────
    seq_info, maxlist = _load_seq(data_name, model_name)
    if seq_info is None:
        print(f"  [{tag}] [Shadowing] SeqInfo/MaxList missing — run dist_calc / run_all first.")
        return

    print(f"  [{tag}] [Shadowing] ranking chains by flip count ...")
    stable_top, unstable_top = analyze_pseudo_orbit_stability(
        seq_info, maxlist, prob_fc[0], labels, top_k=TOP_K_SHADOWING)

    if stable_top:
        r = stable_top[0]
        print(f"    robust   — sample {r['sample']}  true={r['true_label']}  "
              f"flips={r['flip_count']}  deviations={r['deviate_count']}/{len(r['chain'])}")
        plot_chain(
            test_dataset, mean, std, r,
            title=f"[Shadowing] robust chain — {data_name}/{model_name}",
            save_path=os.path.join(out_dir, "shadowing_stable.png"))

    if unstable_top:
        r = unstable_top[0]
        print(f"    unstable — sample {r['sample']}  true={r['true_label']}  "
              f"flips={r['flip_count']}  deviations={r['deviate_count']}/{len(r['chain'])}")
        plot_chain(
            test_dataset, mean, std, r,
            title=f"[Shadowing] unstable chain — {data_name}/{model_name}",
            save_path=os.path.join(out_dir, "shadowing_unstable.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── 1. Results table ──────────────────────────────────────────────────
    print_table()

    # ── 2. Example images (DS-ResNet only) ────────────────────────────────
    ds_models = [m for m in MODELS if m in DS_LAYERS_MAP]
    total     = len(ds_models) * len(DATASETS)
    done      = 0

    print(f"\n[Image extraction]  {total} combinations  ->  {OUT_DIR}/\n")
    for data_name in DATASETS:
        for model_name in ds_models:
            done += 1
            print(f"[{done}/{total}]  {model_name} x {data_name}")
            extract_images(model_name, data_name)
            print()

    print(f"Done.  Images saved under  {OUT_DIR}/")
    print(f"Directory structure:")
    print(f"  {OUT_DIR}/")
    print(f"  ├── {{DATA}}_{{model}}/")
    print(f"  │     ├── expansive.png          # top-{TOP_K_EXPANSIVE} same-class outlier pairs")
    print(f"  │     ├── shadowing_stable.png   # most robust pseudo-orbit chain")
    print(f"  │     └── shadowing_unstable.png # most unstable pseudo-orbit chain")
