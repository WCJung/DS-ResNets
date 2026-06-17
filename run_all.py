#!/usr/bin/env python3
"""
run_all.py — Automated full experiment runner.
4 models x 3 datasets = 12 combinations, executed sequentially.

Per combination:
  1. Train backbone (up to EPOCHS epochs, early stop ES)
  2. Extract per-block raw features          -> prob/{DATA}/{model}/
  3. Train block-wise linear probes (DS-only) -> prob_fc/{DATA}/{model}/
  4. Distance analysis: task_1, seq_builder, epsilon, theorem

Usage:
  python run_all.py

All output is mirrored to logs/run_all_<timestamp>.log
"""

import os
import sys
import re
import time
import traceback
import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.ResNets import ResNet, Bottleneck
from utils.norms import init_random
from utils.stubs import load_data, train, train_block_fc, Exprob
from utils.task import DistanceMeasure, task_2
from utils.builders import seq_builder
from utils.lipschitz import analyze_theorem


# ── Experiment configuration ───────────────────────────────────────────────────
MODELS   = ['resnet18', 'resnet50', 'ds_resnet18', 'ds_resnet50']
DATASETS = ['MNIST', 'CIFAR10', 'IMAGENET10']

SEED              = 13
N_CLASS           = 10
EPOCHS            = 100
EARLY_STOP        = 20
LR                = 5e-5
BATCH_SIZE        = 64        # 16 -> 64: better A100 utilization, negligible accuracy delta

USE_BLOCK_FC      = True      # DS-ResNet: train per-block linear probes
USE_AVGPOOL       = True      # DS-ResNet: avgpool before main fc (feat=2048, tight Lip)
ALLOW_CROSS_CLASS = False     # seq_builder: same-class neighbors only

DS_LAYERS_MAP = {
    'ds_resnet18': [2, 2, 2, 2],   #  8 blocks
    'ds_resnet50': [3, 4, 6, 3],   # 16 blocks
}
FEAT_DIM = 2048 * 7 * 14           # 200,704 — DS-ResNet block feature dim
LOG_DIR  = "logs"
# ──────────────────────────────────────────────────────────────────────────────


# ── Terminal helpers ───────────────────────────────────────────────────────────

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
C = "\033[96m"; B = "\033[94m"; RST = "\033[0m"; BLD = "\033[1m"

def _strip(s):
    return re.sub(r'\033\[[0-9;]*m', '', s)

def fmt_time(sec):
    h, m, s = int(sec) // 3600, (int(sec) % 3600) // 60, int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def eta(elapsed, done, total):
    if done == 0:
        return "--:--:--"
    return fmt_time(elapsed / done * (total - done))

def _box(msg, width=72, char='═'):
    inner = _strip(msg)
    left  = (width - len(inner) - 2) // 2
    right = width - len(inner) - 2 - left
    return f"{char * left} {msg} {char * right}"

def _section(msg):
    print(f"\n  {B}>>>{RST} {msg}")


class _Tee:
    """Write to both terminal and a log file (ANSI codes stripped for the file)."""
    def __init__(self, path):
        self._real = sys.stdout
        self._f    = open(path, 'w', encoding='utf-8')
    def write(self, msg):
        self._real.write(msg)
        self._f.write(_strip(msg))
    def flush(self):
        self._real.flush()
        self._f.flush()
    def fileno(self):           # needed by some libs that check sys.stdout.fileno()
        return self._real.fileno()
    def close(self):
        self._f.close()
        sys.stdout = self._real


# ── Step 1: train + feature extraction ────────────────────────────────────────

def _eval_accuracy(model, testloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred  = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def _make_model(model_name, device):
    is_ds = model_name in DS_LAYERS_MAP
    if is_ds:
        layers = DS_LAYERS_MAP[model_name]
        model  = ResNet(block=Bottleneck, layers=layers,
                        num_classes=N_CLASS, use_avgpool=USE_AVGPOOL)
    elif model_name == 'resnet18':
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, N_CLASS)
    elif model_name == 'resnet50':
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, N_CLASS)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def step_train_extract(model_name, data_name, device,
                       train_dataset, test_dataset):
    is_ds     = model_name in DS_LAYERS_MAP
    model_tag = model_name
    ckpt_name = f"{model_tag}_{data_name}"

    num_workers = 4 if sys.platform != 'win32' else 0
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # ── backbone training ──────────────────────────────────────────────────
    _section("Training backbone ...")
    t0    = time.time()
    model = _make_model(model_name, device)
    train(model, trainloader, testloader, device,
          epochs=EPOCHS, es=EARLY_STOP, lpth=ckpt_name, lr=LR)
    # reload best checkpoint (train() keeps saving best-loss, last epoch may not be best)
    model.load_state_dict(torch.load(f"{ckpt_name}.pt", map_location=device))
    acc = _eval_accuracy(model, testloader, device)
    os.makedirs("Result", exist_ok=True)
    np.save(f"Result/{data_name}_{model_tag}_accuracy.npy",
            {"accuracy": acc, "model": model_tag, "data": data_name})
    print(f"  [{G}done{RST}] training  {fmt_time(time.time() - t0)}"
          f"  |  test acc = {G}{acc:.2f}%{RST}")

    if not is_ds:
        print(f"  {Y}[skip]{RST} {model_name}: standard ResNet — accuracy baseline only, "
              "no block analysis.")
        return

    # ── DS-ResNet: raw block feature extraction ────────────────────────────
    ds_layers = DS_LAYERS_MAP[model_name]
    extractor = Exprob(FEAT_DIM, N_CLASS, layers=ds_layers,
                       multi_fc=USE_BLOCK_FC, use_avgpool=USE_AVGPOOL)
    state = torch.load(f"{ckpt_name}.pt", map_location=device)
    extractor.load_state_dict(state, strict=False)
    extractor.to(device).eval()

    _section("Extracting raw block features ...")
    t0 = time.time()
    block_acc = {}
    with torch.no_grad():
        for xx, _ in testloader:
            for b, feat in extractor(xx.to(device)).items():
                fc = feat.detach().cpu()
                block_acc[b] = fc if b not in block_acc else torch.cat((block_acc[b], fc))

    feat_dir = f"prob/{data_name}/{model_tag}"
    os.makedirs(feat_dir, exist_ok=True)
    for b, feat in block_acc.items():
        torch.save(feat, f"{feat_dir}/{data_name}_block{b}.pt")
    print(f"  [{G}done{RST}] raw features -> {feat_dir}/  {fmt_time(time.time() - t0)}")

    # ── block-wise linear probes (block_fc) ───────────────────────────────
    if USE_BLOCK_FC:
        _section("Training block-wise linear probes ...")
        t0 = time.time()
        train_block_fc(extractor, trainloader, device, epochs=5)
        torch.save(extractor.state_dict(), f"{ckpt_name}_multifc.pt")

        extractor.eval()
        fc_acc = {}
        with torch.no_grad():
            for xx, _ in testloader:
                for b, logit in extractor(xx.to(device), use_block_fc=True).items():
                    lc = logit.detach().cpu()
                    fc_acc[b] = lc if b not in fc_acc else torch.cat((fc_acc[b], lc))

        fc_dir = f"prob_fc/{data_name}/{model_tag}"
        os.makedirs(fc_dir, exist_ok=True)
        for b, logit in fc_acc.items():
            torch.save(logit, f"{fc_dir}/{data_name}_block{b}.pt")
        print(f"  [{G}done{RST}] block_fc outputs -> {fc_dir}/  {fmt_time(time.time() - t0)}")

    # ── label + chunked feature save (for inspect_examples.py) ───────────
    _section("Saving labels and chunked features ...")
    t0 = time.time()
    extractor.load_state_dict(
        torch.load(f"{ckpt_name}.pt", map_location=device), strict=False)
    extractor.to(device).eval()

    yout = None
    with torch.no_grad():
        for i, (x, y) in enumerate(testloader):
            out  = extractor(x.to(device))
            yout = y if yout is None else torch.cat((yout, y))
            pix_dir = f"pix/resnet/{data_name}/{model_tag}/test"
            os.makedirs(pix_dir, exist_ok=True)
            for key, feat in out.items():
                torch.save(feat.cpu(), f"{pix_dir}/{data_name}_block{key}_{i}.pt")
    torch.save(yout, f"pix/resnet/{data_name}/{model_tag}/test/{data_name}_label.pt")
    print(f"  [{G}done{RST}] labels -> pix/resnet/{data_name}/{model_tag}/test/  "
          f"{fmt_time(time.time() - t0)}")


# ── Step 2: distance analysis ──────────────────────────────────────────────────

def step_analysis(model_name, data_name, device):
    if model_name not in DS_LAYERS_MAP:
        return   # standard ResNets skipped

    ds_layers = DS_LAYERS_MAP[model_name]
    n_blocks  = sum(ds_layers)
    model_tag = model_name
    ckpt_name = f"{model_tag}_{data_name}"

    feat_dir = f"prob/{data_name}/{model_tag}"
    l_path   = f"pix/resnet/{data_name}/{model_tag}/test"

    # ── load features (N, n_blocks, feat_dim) ────────────────────────────
    _section("Loading features for analysis ...")
    hold = None
    for b in range(n_blocks):
        x     = torch.load(f"{feat_dir}/{data_name}_block{b}.pt")
        chunk = x.detach().numpy().reshape(x.shape[0], 1, x.shape[1])
        hold  = chunk if hold is None else np.concatenate((hold, chunk), axis=1)
    y    = torch.load(f"{l_path}/{data_name}_label.pt")
    y_np = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    labels_unique = np.unique(y_np)
    print(f"  N={hold.shape[0]}  blocks={n_blocks}  feat_dim={hold.shape[2]}")

    # ── task_1: class-wise Minkowski distance matrices ─────────────────────
    _section("task_1: class-wise Minkowski distances ...")
    t0 = time.time()
    feat_dict = DistanceMeasure(hold / 1000.0, y, norm="softmax")
    feat_dict.task_1(data_name, model_tag)
    print(f"  [{G}done{RST}] task_1  {fmt_time(time.time() - t0)}")

    # ── seq_builder: pseudo-orbit chains ──────────────────────────────────
    _section("seq_builder: building pseudo-orbit chains ...")
    t0 = time.time()
    seqs = seq_builder(hold, data_name, model_tag, n_blocks,
                       labels=y, allow_cross_class=ALLOW_CROSS_CLASS)
    best_stack, best_stack_mean = task_2(seqs, data_name,
                                         labels=y, allow_cross_class=ALLOW_CROSS_CLASS)
    print(f"  [{G}done{RST}] seq_builder + task_2  {fmt_time(time.time() - t0)}")

    # ── expansive constant ε ───────────────────────────────────────────────
    _section("Computing expansive constant ε ...")
    t0        = time.time()
    task1_dir = "Result/task1"
    epsilon   = float("inf")
    eps_cls_a = eps_cls_b = eps_idx_a = eps_idx_b = eps_block = None

    for cls_a in labels_unique:
        fname = os.path.join(task1_dir,
                             f"{data_name}_{model_tag}_Class_{cls_a}_prob.npy")
        if not os.path.exists(fname):
            continue
        arr = np.load(fname, allow_pickle=True)
        for cls_b in labels_unique:
            if cls_b == cls_a:
                continue
            b_idx = sorted(labels_unique.tolist()).index(int(cls_b))
            try:
                sub = arr[b_idx]
            except IndexError:
                continue
            sub_arr  = np.array(sub, dtype=float)
            flat_min = np.nanmin(sub_arr)
            if flat_min < epsilon:
                epsilon   = flat_min
                eps_cls_a = int(cls_a)
                eps_cls_b = int(cls_b)
                flat_loc  = np.unravel_index(np.nanargmin(sub_arr), sub_arr.shape)
                eps_block = int(flat_loc[0]) if sub_arr.ndim > 1 else 0
                cls_a_g   = np.where(y_np == cls_a)[0]
                cls_b_g   = np.where(y_np == cls_b)[0]
                if sub_arr.ndim >= 2:
                    eps_idx_a = int(cls_a_g[flat_loc[-2]]) if flat_loc[-2] < len(cls_a_g) else None
                    eps_idx_b = int(cls_b_g[flat_loc[-1]]) if flat_loc[-1] < len(cls_b_g) else None

    os.makedirs("Result", exist_ok=True)
    np.save(f"Result/{data_name}_{model_tag}_epsilon.npy",
            {"epsilon": epsilon, "class_a": eps_cls_a, "class_b": eps_cls_b,
             "block": eps_block, "sample_a": eps_idx_a, "sample_b": eps_idx_b})
    print(f"  ε = {epsilon:.6f}  (class {eps_cls_a} <-> class {eps_cls_b}"
          f", block {eps_block})  {fmt_time(time.time() - t0)}")

    # ── theorem: Shg / Lip → Tg lower bound ───────────────────────────────
    multifc_ckpt = f"{ckpt_name}_multifc.pt"
    if USE_BLOCK_FC and os.path.exists(multifc_ckpt):
        _section("Theorem: Shg(phi) / Lip(g) -> Tg(phi) lower bound ...")
        t0 = time.time()
        analyze_theorem(
            d_name=data_name, model=model_tag,
            ckpt_path=multifc_ckpt,
            feat_dim=FEAT_DIM, n_class=N_CLASS,
            layers=ds_layers, use_avgpool=USE_AVGPOOL,
            save_path="Result",
        )
        print(f"  [{G}done{RST}] theorem  {fmt_time(time.time() - t0)}")
    else:
        print(f"  {Y}[skip]{RST} theorem: {multifc_ckpt} not found.")


# ── Single combination runner ──────────────────────────────────────────────────

def run_combo(model_name, data_name, device, idx, total, global_start):
    elapsed = time.time() - global_start
    tag     = f"{BLD}{model_name}{RST} x {BLD}{data_name}{RST}"
    print()
    print(_box(
        f"{C}{BLD}[{idx}/{total}]{RST}  {tag}"
        f"  |  elapsed {fmt_time(elapsed)}"
        f"  |  ETA {eta(elapsed, idx - 1, total)}"
    ))
    t0 = time.time()

    try:
        init_random(SEED)
        _section(f"Loading dataset: {data_name} ...")
        train_ds, test_ds = load_data(data_name)
        print(f"  train={len(train_ds):,}  test={len(test_ds):,}")

        step_train_extract(model_name, data_name, device, train_ds, test_ds)
        step_analysis(model_name, data_name, device)

        dur = fmt_time(time.time() - t0)
        print(f"\n  {G}{BLD}[{idx}/{total}] OK{RST}  {model_name} x {data_name}  ({dur})")
        return ("OK", dur, "")

    except Exception:
        tb  = traceback.format_exc()
        dur = fmt_time(time.time() - t0)
        print(f"\n  {R}{BLD}[{idx}/{total}] FAILED{RST}  {model_name} x {data_name}  ({dur})")
        print(f"  {R}{tb.splitlines()[-1]}{RST}")
        err_path = os.path.join(LOG_DIR, f"err_{model_name}_{data_name}.txt")
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write(tb)
        print(f"  Traceback -> {err_path}")
        return ("FAILED", dur, tb.splitlines()[-1])


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 x 모델 순서: 같은 데이터셋을 연속으로 실행해 DataLoader 캐시 효율 최대화
    combos = [(m, d) for d in DATASETS for m in MODELS]

    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"run_all_{ts}.log")

    tee = _Tee(log_path)
    sys.stdout = tee

    # ── header ───────────────────────────────────────────────────────────
    print(_box(f"DS-ResNets Full Experiment — {ts}"))
    print(f"  Device    : {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Models    : {MODELS}")
    print(f"  Datasets  : {DATASETS}")
    print(f"  Batch     : {BATCH_SIZE}  |  Epochs: {EPOCHS}  |  ES: {EARLY_STOP}  |  LR: {LR}")
    print(f"  block_fc  : {USE_BLOCK_FC}  |  avgpool: {USE_AVGPOOL}")
    print(f"  Log       : {log_path}")
    print(f"  Total     : {len(combos)} combinations")

    global_start = time.time()
    results      = {}

    for i, (model_name, data_name) in enumerate(combos, 1):
        results[(model_name, data_name)] = run_combo(
            model_name, data_name, device, i, len(combos), global_start)

    # ── summary table ─────────────────────────────────────────────────────
    total_elapsed = time.time() - global_start
    ok_count      = sum(1 for s, _, _ in results.values() if s == "OK")

    print()
    print(_box("SUMMARY"))
    print(f"\n  {'Model':<18} {'Dataset':<12} {'Status':<8}  Time      Note")
    print(f"  {'-'*18} {'-'*12} {'-'*8}  {'-'*9}  {'-'*30}")

    for (mn, dn), (status, dur, err) in results.items():
        is_ds = mn in DS_LAYERS_MAP
        sym   = f"{G}OK      {RST}" if status == "OK" else f"{R}FAILED  {RST}"
        note  = "" if status == "OK" else err[:35]
        if status == "OK" and not is_ds:
            note = "(accuracy baseline only)"
        print(f"  {mn:<18} {dn:<12} {sym}  {dur}  {note}")

    print(f"\n  {'-'*65}")
    print(f"  {G if ok_count == len(combos) else Y}{ok_count}/{len(combos)} combinations succeeded{RST}"
          f"  |  Total wall time: {fmt_time(total_elapsed)}")
    print(f"  Full log: {log_path}")

    tee.close()
