#!/usr/bin/env python3
"""
run_all.py — Automated full experiment runner.
4 models x 3 datasets = 12 combinations, executed sequentially.

Per combination:
  1. Train backbone (up to EPOCHS epochs, early stop ES)
  2. Evaluate F1/Loss/Acc                     -> Result/{DATA}_{model}_metrics.npy
  3. Extract per-block raw features           -> prob/{DATA}/{model}/
  4. Train block-wise linear probes (DS-only) -> prob_fc/{DATA}/{model}/
  5. Stability analysis (dist_calc.run_analysis):
     g-expansive, pseudo-orbits, Sh_g, Lip(g), Table 1 print

Usage:
  python run_all.py

All output is mirrored to logs/run_all_<timestamp>.log
"""

import datetime
import os
import re
import sys
import time
import traceback

import torch
import torch.nn as nn
import torchvision.models as tv_models

from dist_calc import run_analysis
from entropy_calc import run_entropy
from models.models import DS_MODELS, build_ds_model, ds_block, ds_layers
from utils.norms import init_random
from utils.stubs import (Exprob, evaluate, extract_block_outputs, load_data,
                         save_block_outputs, save_labels, save_metrics,
                         train, train_block_fc)


# ── Experiment configuration ───────────────────────────────────────────────────
# 필요 시 조합을 줄여서 실행하세요 (8 models x 3 datasets = 24 combos는 무겁습니다).
MODELS   = ['resnet18', 'resnet50'] + list(DS_MODELS)
DATASETS = ['MNIST', 'CIFAR10', 'IMAGENET10']

SEED              = 13
N_CLASS           = 10
EPOCHS            = 100
EARLY_STOP        = 20
LR                = 5e-5
BATCH_SIZE        = 64

USE_BLOCK_FC      = True      # DS models: train per-block linear probes
USE_AVGPOOL       = True      # DS models: avgpool before main fc
SPACE             = 'prob'    # d_g 관측 공간: softmax 확률 (dist_calc 참조)
ALLOW_CROSS_CLASS = False     # pseudo-orbit: same-class neighbors only
RUN_ENTROPY       = True      # 안정성 분석 후 FTTE(entropy_calc)도 실행
ANALYSIS_DEVICE   = 'cuda' if torch.cuda.is_available() else None

LOG_DIR = "logs"
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


# ── Step 1: train + extraction ────────────────────────────────────────────────

def _make_model(model_name, device):
    if model_name in DS_MODELS:
        model = build_ds_model(model_name, N_CLASS, use_avgpool=USE_AVGPOOL)
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
    is_ds     = model_name in DS_MODELS
    ckpt_name = f"{model_name}_{data_name}"

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
    # reload best checkpoint (train() saves best test-loss state)
    model.load_state_dict(torch.load(f"{ckpt_name}.pt", map_location=device))
    metrics = evaluate(model, testloader, device)
    save_metrics(metrics, data_name, model_name)
    print(f"  [{G}done{RST}] training  {fmt_time(time.time() - t0)}"
          f"  |  F1 = {G}{metrics['f1']:.4f}{RST}"
          f"  |  loss = {metrics['loss']:.4f}"
          f"  |  acc = {metrics['acc']*100:.2f}%")

    if not is_ds:
        print(f"  {Y}[skip]{RST} {model_name}: standard ResNet — accuracy baseline "
              "only, no block analysis.")
        return

    # ── DS models: block outputs ───────────────────────────────────────────
    extractor = Exprob(N_CLASS, layers=ds_layers(model_name),
                       multi_fc=USE_BLOCK_FC, use_avgpool=USE_AVGPOOL,
                       block=ds_block(model_name))
    extractor.load_state_dict(
        torch.load(f"{ckpt_name}.pt", map_location=device), strict=False)
    extractor.to(device)

    _section("Extracting raw block features + labels ...")
    t0 = time.time()
    feats, labels = extract_block_outputs(extractor, testloader, device)
    feat_dir = save_block_outputs(feats, "prob", data_name, model_name)
    del feats
    pix_dir = save_labels(labels, data_name, model_name)
    print(f"  [{G}done{RST}] raw features -> {feat_dir}/  |  labels -> {pix_dir}/  "
          f"{fmt_time(time.time() - t0)}")

    if USE_BLOCK_FC:
        _section("Training block-wise linear probes ...")
        t0 = time.time()
        train_block_fc(extractor, trainloader, device, epochs=5)
        torch.save(extractor.state_dict(), f"{ckpt_name}_multifc.pt")

        logits, _ = extract_block_outputs(extractor, testloader, device,
                                          use_block_fc=True)
        fc_dir = save_block_outputs(logits, "prob_fc", data_name, model_name)
        print(f"  [{G}done{RST}] block_fc outputs -> {fc_dir}/  "
              f"{fmt_time(time.time() - t0)}")


# ── Step 2: stability analysis ─────────────────────────────────────────────────

def step_analysis(model_name, data_name):
    if model_name not in DS_MODELS:
        return   # standard ResNets skipped

    _section("Stability analysis: eps / Sh_g / Lip(g) / Table 1 ...")
    t0 = time.time()
    run_analysis(
        data_name=data_name,
        model_tag=model_name,
        layers=ds_layers(model_name),
        space=SPACE,
        allow_cross_class=ALLOW_CROSS_CLASS,
        device=ANALYSIS_DEVICE,
        seed=SEED,
    )
    print(f"  [{G}done{RST}] analysis  {fmt_time(time.time() - t0)}")

    if RUN_ENTROPY:
        _section("FTTE: separated sets / h_T / Δh_T ...")
        t0 = time.time()
        run_entropy(
            data_name=data_name,
            model_tag=model_name,
            layers=ds_layers(model_name),
            space=SPACE,
            device=ANALYSIS_DEVICE,
            seed=SEED,
        )
        print(f"  [{G}done{RST}] entropy  {fmt_time(time.time() - t0)}")


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
        step_analysis(model_name, data_name)

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
    print(f"  block_fc  : {USE_BLOCK_FC}  |  avgpool: {USE_AVGPOOL}  |  space: {SPACE}")
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
        is_ds = mn in DS_MODELS
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
