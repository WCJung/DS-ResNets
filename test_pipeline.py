"""
더미 데이터로 파이프라인 전체 + 정리 계산을 테스트합니다.

구조:
  - N=15 샘플 (클래스 3개 × 5샘플)
  - n_blocks=4 블록
  - feat_dim=16 특징 차원
  - n_class=3 클래스

실행: python test_pipeline.py
"""

import numpy as np
import torch
import torch.nn as nn
import os

from utils.norms import init_random
from utils.task import DistanceMeasure, task_2
from utils.builders import seq_builder
from utils.lipschitz import shg_phi, lip_g, theorem_lower_bound

# ── 재현성 ──────────────────────────────────────────────
init_random(42)

# ── 더미 데이터 생성 ────────────────────────────────────
N, n_blocks, feat_dim, n_class = 15, 4, 16, 3
hold = np.random.randn(N, n_blocks, feat_dim).astype(np.float32)
y = torch.tensor([i % n_class for i in range(N)])   # 0,1,2,0,1,2,...

print(f"hold shape : {hold.shape}")   # (15, 4, 16)
print(f"labels     : {y}")

# ── Phase 1: DistanceMeasure.task_1 ────────────────────
print("\n[Phase 1] task_1 시작...")
os.makedirs("Result/task1", exist_ok=True)

dm = DistanceMeasure(hold / 1000, y, norm="softmax")
dm.task_1("dummy", "TestModel")

saved = [f for f in os.listdir("Result/task1") if f.startswith("dummy")]
print(f"  저장된 파일: {saved}")
print("[Phase 1] 완료\n")

# ── Phase 2: seq_builder + task_2 (class-aware) ────────
print("[Phase 2] seq_builder 시작...")
os.makedirs("task2", exist_ok=True)

seqs = seq_builder(hold, "dummy", "TestModel", n_blocks,
                   labels=y, allow_cross_class=False)
print(f"  seqs[0] (Targets)   shape : {seqs[0].shape}")
print(f"  seqs[1] (Series)    shape : {seqs[1].shape}")
print(f"  seqs[2] (SeqInfo)   shape : {seqs[2].shape}")
print(f"  seqs[3] (MaxList)   shape : {seqs[3].shape}")
print(f"  seqs[4] (ClassInfo) shape : {seqs[4].shape}")

best_stack, best_stack_mean = task_2(seqs, "dummy",
                                      labels=y, allow_cross_class=False)
print(f"  best_stack shape      : {best_stack.shape}")
print(f"  best_stack_mean shape : {best_stack_mean.shape}")
print("[Phase 2] 완료\n")

# ── Phase 3: 정리 계산 — Shg(φ) / Lip(g) → Tg(φ) 하한 ──
print("[Phase 3] 정리 계산...")

# Shg(φ): MaxList로부터 pseudo-orbit 최대 오차
maxlist = seqs[3]   # (N, n_blocks)
shg, per_chain = shg_phi(maxlist)
print(f"  MaxList shape          : {maxlist.shape}")
print(f"  샘플별 최대 오차       : {per_chain.round(4)}")
print(f"  Shg(φ)                 = {shg:.6f}")

# Lip(g): 더미 fc 레이어의 spectral norm 계산
# (실제 실행에서는 train_block_fc()로 학습된 extractor.block_fc를 사용)
class DummyExtractor:
    """테스트용 더미 — block_fc는 실제 nn.Linear와 동일 구조"""
    def __init__(self):
        self.block_fc = nn.ModuleList(
            [nn.Linear(feat_dim, n_class) for _ in range(n_blocks)]
        )

dummy_ext = DummyExtractor()
lip_max, lip_list = lip_g(dummy_ext)
print(f"  블록별 Lip(g_b)        : {[round(l,4) for l in lip_list]}")
print(f"  Lip(g) (전체 최댓값)   = {lip_max:.6f}")

# Tg(φ) 하한
tg_lb = theorem_lower_bound(shg, lip_max)
print(f"\n  정리:  Shg(φ) ≤ Lip(g) · Tg(φ)")
print(f"  ∴ Tg(φ) ≥ Shg(φ) / Lip(g) = {shg:.4f} / {lip_max:.4f} = {tg_lb:.6f}")
print("[Phase 3] 완료\n")

print("=== 전체 파이프라인 테스트 성공 ===")
