"""
entropy_calc.py — FTTE(finite-time trajectory entropy) 계산 + 요약 출력.

dist_calc.py와 같은 d_g 관측 공간에서 궤적 pseudometric
d_g^T(x,y) = max_t d(g_t(phi^t x), g_t(phi^t y)) 를 사용해:

  1) Proposition 1 진단 — intra_max(클래스 내 최대 d_g^T),
     cross_min(클래스 간 최소 d_g^T = g-expansive 상수),
     그리고 s = m이 이론적으로 보장되는 eps 창.
  2) eps 그리드별  s_T(eps) / h_T(eps) / Δh_T(eps)  (greedy packing 하한)
     — Proposition 2의 부호 해석(s vs m) 포함.

실행 예:
  python entropy_calc.py --model ds_resnet18 --data MNIST
  python entropy_calc.py --model ds_wide50 --data CIFAR10 --device cuda
  python entropy_calc.py --model ds_resnet18 --data MNIST --eps 0.05,0.1,0.2

산출물:
  Result/{data}_{model}_entropy.npy — 진단 + 그리드 결과 + 대표값
"""
import argparse
import os

import numpy as np

from models.models import DS_MODELS, ds_layers
from utils.entropy import ftte_report
from utils.norms import init_random
from utils.trajectory import load_trajectory


def parse_args():
    p = argparse.ArgumentParser(description="DS-ResNets FTTE 계산")
    p.add_argument('--model', default='ds_resnet18', choices=list(DS_MODELS))
    p.add_argument('--data', default='MNIST',
                   choices=['MNIST', 'CIFAR10', 'IMAGENET10'])
    p.add_argument('--space', default='prob', choices=['prob', 'logit', 'feat'],
                   help="d_g 관측 공간 (dist_calc.py와 동일하게 맞출 것)")
    p.add_argument('--eps', default=None,
                   help="쉼표로 구분한 eps 그리드 (예: 0.05,0.1,0.2). "
                        "생략 시 d_g^T 분위수 + Prop.1 창 중점으로 자동 구성")
    p.add_argument('--n-samples', type=int, default=None)
    p.add_argument('--chunk', type=int, default=1024)
    p.add_argument('--device', default=None, help="'cuda' 지정 시 GPU 계산")
    p.add_argument('--seed', type=int, default=13)
    return p.parse_args()


def run_entropy(data_name, model_tag, layers, space='prob', eps_list=None,
                n_samples=None, chunk=1024, device=None, seed=13):
    """FTTE 파이프라인. 결과 dict 반환 및 Result/에 저장."""
    n_blocks = sum(layers)
    tag = f"{data_name}_{model_tag}"

    traj, labels, _ = load_trajectory(
        data_name, model_tag, n_blocks, space=space,
        n_samples=n_samples, seed=seed)
    N, T, D = traj.shape
    print(f"[궤적] space={space}  N={N}  T={T}  D={D}")
    print("\n[FTTE] 계산 시작...")

    rep = ftte_report(traj, labels, eps_list=eps_list, chunk=chunk,
                      device=device)

    # ── 요약 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  FTTE  —  {data_name} / {model_tag}   (space={space}, T={T}, N={N})")
    print("-" * 70)
    print(f"  h_class = log(m)/T = log({rep['m']})/{rep['T']} "
          f"= {rep['h_class']:.4f}")
    d = rep["diagnostics"]
    print(f"  intra_max = {d['intra_max']:.6f}   "
          f"cross_min = {d['cross_min']:.6e} (= g-expansive)")
    if d["prop1_window"]:
        a, b = d["prop1_window"]
        print(f"  Prop.1 창 : eps ∈ [{a:.6f}, {b:.6f})  →  s_T = m, Δh_T = 0 보장")
    else:
        print("  Prop.1 창 : 없음 (intra_max >= cross_min)")
    print("-" * 70)
    print(f"  {'eps':>12}  {'s_T(eps)':>9}  {'h_T(eps)':>9}  "
          f"{'Δh_T(eps)':>10}  s vs m")
    for r in rep["rows"]:
        print(f"  {r['eps']:>12.6f}  {r['s']:>9d}  {r['h']:>9.4f}  "
              f"{r['gap']:>+10.4f}   s {r['cmp']} m")
    if rep["recommended"]:
        r = rep["recommended"]
        print("-" * 70)
        print(f"  대표값 (Prop.1 창 우선): eps={r['eps']:.6f}  s_T={r['s']}  "
              f"h_T={r['h']:.4f}  Δh_T={r['gap']:+.4f}")
    print("=" * 70)
    print("  (s_T는 greedy maximal packing 하한 — 실제 s_g^T(eps) >= 표기값)")

    os.makedirs("Result", exist_ok=True)
    np.save(f"Result/{tag}_entropy.npy", {
        "T": rep["T"], "m": rep["m"], "h_class": rep["h_class"],
        "intra_max": d["intra_max"], "cross_min": d["cross_min"],
        "prop1_window": d["prop1_window"],
        "rows": rep["rows"], "recommended": rep["recommended"],
        "space": space,
    })
    print(f"  저장: Result/{tag}_entropy.npy")
    return rep


if __name__ == '__main__':
    args = parse_args()
    init_random(args.seed)
    eps_list = None
    if args.eps:
        eps_list = sorted(float(v) for v in args.eps.split(','))
    run_entropy(
        data_name=args.data,
        model_tag=args.model,
        layers=ds_layers(args.model),
        space=args.space,
        eps_list=eps_list,
        n_samples=args.n_samples,
        chunk=args.chunk,
        device=args.device,
        seed=args.seed,
    )
