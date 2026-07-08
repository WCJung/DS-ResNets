"""
더미 데이터로 안정성 분석 파이프라인 전체를 스모크 테스트합니다.
(학습/데이터 다운로드 없이 utils 모듈만 검사)

구조:
  - N=30 샘플 (클래스 3개 × 10샘플)
  - T=5 블록
  - D=3 (관측 공간 차원 = 클래스 수)

실행: python test_pipeline.py
"""
import math

import numpy as np
import torch
import torch.nn as nn

from utils.entropy import class_scale_diagnostics, ftte_report, separated_set_size
from utils.expansive import expansive_constant
from utils.lipschitz import spectral_norm_fc
from utils.norms import init_random
from utils.shadowing import (build_pseudo_orbits, shadowing_constant,
                             trace_orbits)

init_random(42)

N, T, n_class = 30, 5, 3
labels = torch.tensor([i % n_class for i in range(N)])

# 클래스별로 분리된 궤적 생성: 클래스 중심 + 소음 → softmax로 확률 공간에 배치
centers = torch.eye(n_class) * 4.0
raw = centers[labels].unsqueeze(1).repeat(1, T, 1) + 0.5 * torch.randn(N, T, n_class)
traj = torch.softmax(raw, dim=-1)

print(f"traj  : {tuple(traj.shape)}  (N, T, D)")
print(f"labels: {labels.tolist()}")

# ── Phase 1: g-expansive 상수 (min over 쌍, max over 블록) ─────────────
print("\n[Phase 1] expansive_constant ...")
eps_res = expansive_constant(traj, labels, chunk=8, verbose=False)
assert eps_res["epsilon"] > 0
assert eps_res["class_a"] != eps_res["class_b"]
print(f"  eps = {eps_res['epsilon']:.4f}  "
      f"(class {eps_res['class_a']} vs {eps_res['class_b']}, "
      f"block {eps_res['block']})")
print("[Phase 1] OK\n")

# ── Phase 2: pseudo-orbit + 추적 + Sh_g ────────────────────────────────
print("[Phase 2] build_pseudo_orbits / trace_orbits / shadowing_constant ...")
seq, step_err = build_pseudo_orbits(traj, labels, allow_cross_class=False,
                                    depth_consistent=True, chunk=8)
assert seq.shape == (N, T) and step_err.shape == (N, T - 1)
# 클래스 제한: 체인 멤버 전원이 시작 샘플과 같은 클래스여야 함
assert (labels[seq] == labels[seq[:, :1]]).all()
# 방문 중복 없음
for i in range(N):
    assert len(set(seq[i].tolist())) == T

trace_eps, tracer = trace_orbits(traj, seq, labels, same_class_only=True, chunk=8)
assert trace_eps.shape == (N,)

sh = shadowing_constant(step_err.numpy(), trace_eps.numpy())
print(f"  Sh_g = {sh['Sh_g']:.4f}  (eps0={sh['eps0']:.4f}, "
      f"delta*={sh['delta_star']:.4f})")
print(f"  곡선 점 개수: {len(sh['curve'])}")

# legacy 모드도 동작 확인
seq_l, step_l = build_pseudo_orbits(traj, labels, depth_consistent=False, chunk=8)
assert seq_l.shape == (N, T)
print("[Phase 2] OK\n")

# ── Phase 3: Lip(g) — 선형층 스펙트럴 노름 ─────────────────────────────
print("[Phase 3] spectral_norm_fc ...")
fc = nn.Linear(16, n_class)
sigma = spectral_norm_fc(fc)
ref = float(np.linalg.svd(fc.weight.detach().numpy(), compute_uv=False)[0])
assert abs(sigma - ref) < 1e-5
print(f"  sigma_max = {sigma:.4f} (numpy 참조값과 일치)")
print("[Phase 3] OK\n")

# ── Phase 4: FTTE — separated set / h_T / Δh_T ─────────────────────────
print("[Phase 4] entropy (FTTE) ...")
diag = class_scale_diagnostics(traj, labels, chunk=8)
assert diag["cross_min"] > 0
# 클래스가 잘 분리된 궤적이므로 Prop.1 창이 존재해야 함
assert diag["prop1_window"] is not None, "잘 분리된 더미인데 Prop.1 창이 없음"
a, b = diag["prop1_window"]

# 창 내부 eps → s = m (Proposition 1)
eps_mid = (a + b) / 2
s_mid = separated_set_size(traj, eps_mid, chunk=8)
assert s_mid == n_class, f"Prop.1 창 내부인데 s={s_mid} != m={n_class}"

# 아주 작은 eps → 모든 점이 분리 → s = N
s_tiny = separated_set_size(traj, 1e-8, chunk=8)
assert s_tiny == N, f"eps→0인데 s={s_tiny} != N={N}"

rep = ftte_report(traj, labels, eps_list=[1e-8, eps_mid], chunk=8,
                  verbose=False)
row_mid = [r for r in rep["rows"] if r["eps"] == eps_mid][0]
assert row_mid["cmp"] == '=' and abs(row_mid["gap"]) < 1e-12
assert abs(rep["h_class"] - math.log(n_class) / T) < 1e-12
print(f"  intra_max={diag['intra_max']:.4f}  cross_min={diag['cross_min']:.4f}"
      f"  창=[{a:.4f}, {b:.4f})")
print(f"  s(eps_mid)={s_mid}=m,  s(eps→0)={s_tiny}=N,  "
      f"Δh(eps_mid)={row_mid['gap']:+.4f}")
print("[Phase 4] OK\n")

print("=== 전체 파이프라인 스모크 테스트 성공 ===")
