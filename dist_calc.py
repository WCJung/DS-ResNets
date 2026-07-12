"""
dist_calc.py — 안정성 상수 계산 (g-expansive, g-shadowing, Lip(g)) + Table 1 출력.

모든 거리는 논문의 pseudometric  d_g(x, y) = d(g(x), g(y))  위에서 계산된다
(기본 space='prob': 블록별 fc logit의 softmax 확률 공간, l2 거리).

실행 예:
  python dist_calc.py --model ds_resnet18 --data MNIST
  python dist_calc.py --model ds_resnet50 --data CIFAR10 --device cuda

계산 내용:
  1) g-expansive 상수 eps  = min_{다른 클래스 쌍} max_{블록} d_g   (Definition 1)
  2) pseudo-orbit 생성 → 진짜 궤도 추적 → Sh_g 추정 곡선          (Definition 2)
  3) Lip(g) = max_b sigma_max(W_b) × (softmax 보정)               (정리 1 우변)
  4) Table 1 행 출력 — F1 / Loss / g-expansive / g-shadowing / Lip(g)
     (topological g-stable 상수는 정리 1의 Sh_g(phi) <= Lip(g)·T_g(phi)에
      따라 사용자가 직접 계산: T_g(phi) >= Sh_g(phi) / Lip(g))

산출물:
  Result/{data}_{model}_epsilon.npy    — expansive 상수와 해당 쌍 정보
  Result/{data}_{model}_shadowing.npy  — Sh_g 곡선 및 체인별 (delta, eps)
  Result/{data}_{model}_theorem.npy    — Shg_phi, Lip_g, 블록별 sigma_max
  task2/{data}_{model}_SeqInfo.npy 등  — 체인 정보 (inspect_examples.py 호환)
"""
import argparse
import os

import numpy as np
import torch

from models.models import ds_layers, resolve_model_name
from utils.expansive import expansive_constant
from utils.lipschitz import lip_report_from_checkpoint
from utils.norms import init_random
from utils.shadowing import (build_pseudo_orbits, save_orbit_files,
                             shadowing_constant, trace_orbits)
from utils.trajectory import infer_n_blocks, load_trajectory


def resolve_model_and_layers(model, data, space):
    """DS 모델명(유연 표기) 또는 커스텀 태그 → (정식 태그, layers).

    DS 레지스트리에 없는 태그(예: isolift_resnet_performance)는
    저장된 블록 파일 수로 T 를 추론한다.
    """
    try:
        m = resolve_model_name(model)
        return m, ds_layers(m)
    except ValueError:
        n = infer_n_blocks(data, model, space)
        if n == 0:
            raise SystemExit(
                f"[에러] '{model}' 은 DS 레지스트리에 없고 "
                f"prob_fc/{data}/{model}/ 블록 파일도 없습니다. "
                "DS 모델은 main.py, IsoLift 태그는 extract_isolift.py 를 "
                "먼저 실행하세요.")
        return model, [n]


def parse_args():
    p = argparse.ArgumentParser(description="DS-ResNets 안정성 상수 계산")
    p.add_argument('--model', default='ds_resnet18',
                   help="DS 모델명(유연 표기: RESNET18, wrn50 등) 또는 "
                        "커스텀 태그 (예: isolift_resnet_performance)")
    p.add_argument('--data', default='MNIST',
                   choices=['MNIST', 'CIFAR10', 'IMAGENET10', 'IMAGENET1K'])
    p.add_argument('--space', default='prob', choices=['prob', 'logit', 'feat'],
                   help="d_g 관측 공간 (기본 prob = softmax 확률)")
    p.add_argument('--n-samples', type=int, default=None,
                   help="서브샘플 수 (space=feat 사용 시 권장)")
    p.add_argument('--allow-cross-class', action='store_true',
                   help="pseudo-orbit 이웃 탐색 시 클래스 제한 해제")
    p.add_argument('--legacy-orbit', action='store_true',
                   help="체인 생성을 구버전 방식(블록 0→1 전이 고정)으로")
    p.add_argument('--cross-class-trace', action='store_true',
                   help="추적 후보를 다른 클래스까지 허용")
    p.add_argument('--chunk', type=int, default=1024)
    p.add_argument('--device', default=None, help="'cuda' 지정 시 GPU 계산")
    p.add_argument('--seed', type=int, default=13)
    return p.parse_args()


def run_analysis(data_name, model_tag, layers, space='prob', n_samples=None,
                 allow_cross_class=False, depth_consistent=True,
                 same_class_trace=True, chunk=1024, device=None, seed=13):
    """안정성 분석 전체 파이프라인. Table 1에 필요한 상수를 dict로 반환."""
    n_blocks = sum(layers)
    tag = f"{data_name}_{model_tag}"
    os.makedirs("Result", exist_ok=True)

    # ── 궤적 로드 (d_g 공간) ─────────────────────────────────────────────
    traj, labels, orig_idx = load_trajectory(
        data_name, model_tag, n_blocks, space=space,
        n_samples=n_samples, seed=seed)
    N, T, D = traj.shape
    print(f"[궤적] space={space}  N={N}  T={T}  D={D}")

    # ── 1) g-expansive 상수 (min-max) ────────────────────────────────────
    print("\n[eps] g-expansive 상수 계산 (min over 쌍, max over 블록)...")
    eps_res = expansive_constant(traj, labels, chunk=chunk, device=device)
    eps_res["sample_a"] = int(orig_idx[eps_res["sample_a"]])
    eps_res["sample_b"] = int(orig_idx[eps_res["sample_b"]])
    eps_res["space"] = space
    np.save(f"Result/{tag}_epsilon.npy", eps_res)
    print(f"[eps] eps = {eps_res['epsilon']:.6e}")
    print(f"      클래스 {eps_res['class_a']} vs {eps_res['class_b']}  |  "
          f"샘플 {eps_res['sample_a']} <-> {eps_res['sample_b']}  |  "
          f"블록 {eps_res['block']}")
    if eps_res['epsilon'] == 0.0:
        print("[경고] g-expansive 상수가 정확히 0 — 서로 다른 클래스의 두 샘플이 "
              "모든 블록에서 동일한 g 출력을 가집니다 (관측 공간 붕괴).")
        print("       원인: softmax 포화(space='prob') 또는 block_fc probe 미학습. "
              "--space logit으로 재시도하거나 main.py --use-block-fc로 "
              "probe를 재학습하세요.")

    # ── 2) g-shadowing: pseudo-orbit → 추적 → Sh_g ───────────────────────
    mode = "depth-consistent" if depth_consistent else "legacy(블록0→1)"
    print(f"\n[Sh] pseudo-orbit 생성 ({mode}, cross_class={allow_cross_class})...")
    seq, step_err = build_pseudo_orbits(
        traj, labels, allow_cross_class=allow_cross_class,
        depth_consistent=depth_consistent, chunk=max(chunk, 2048), device=device)

    print("[Sh] 진짜 궤도 추적 오차 계산 (max over 블록, min over 후보)...")
    trace_eps, tracer = trace_orbits(
        traj, seq, labels, same_class_only=same_class_trace,
        chunk=min(chunk, 512), device=device)

    sh_res = shadowing_constant(step_err.numpy(), trace_eps.numpy())
    save_orbit_files(data_name, model_tag, seq.numpy(), step_err.numpy(),
                     labels, trace_eps.numpy(), orig_idx)
    np.save(f"Result/{tag}_shadowing.npy", {
        "Sh_g": sh_res["Sh_g"], "eps0": sh_res["eps0"],
        "delta_star": sh_res["delta_star"], "curve": sh_res["curve"],
        "space": space, "depth_consistent": depth_consistent,
        "degenerate": sh_res["degenerate"],
    })
    if sh_res["degenerate"]:
        print(f"[Sh] 경고: {sh_res['note']}")
        print("[Sh] Sh_g = nan  (계산 불가 — Table 1에는 '—'로 표기)")
    else:
        if sh_res["note"]:
            print(f"[Sh] 참고: {sh_res['note']}")
        print(f"[Sh] Sh_g 추정 곡선  (delta*(eps) = eps-추적 실패 체인이 생기기 "
              "직전까지의 delta):")
        print(f"      {'eps':>12}  {'delta*':>12}  {'delta*/eps':>12}")
        for e, d, r in sh_res["curve"]:
            print(f"      {e:>12.6f}  {d:>12.6f}  {r:>12.6f}")
        print(f"[Sh] Sh_g = {sh_res['Sh_g']:.4f}  "
              f"(eps0={sh_res['eps0']:.4f}, delta*={sh_res['delta_star']:.4f})")

    # ── 3) Lip(g) ────────────────────────────────────────────────────────
    multifc_ckpt = f"{model_tag}_{data_name}_multifc.pt"
    lip_res = None
    if os.path.exists(multifc_ckpt):
        print(f"\n[Lip] {multifc_ckpt} 에서 블록별 sigma_max 계산...")
        lip_res = lip_report_from_checkpoint(multifc_ckpt, space=space)
        for b in sorted(lip_res["sigma_per_block"]):
            print(f"      block {b:02d}: sigma_max = "
                  f"{lip_res['sigma_per_block'][b]:.4f}")
        print(f"[Lip] Lip(g) = {lip_res['Lip_g']:.4f}  "
              f"(= sigma_max {lip_res['sigma_max']:.4f} × softmax 보정 "
              f"{lip_res['softmax_factor']})")
        if lip_res["main_fc"] is not None:
            mf = lip_res["main_fc"]
            print(f"      (참고: main fc  sigma_max={mf['sigma_max']:.4f}, "
                  f"avgpool k={mf['avgpool_k']} → 보정 {mf['avgpool_factor']:.4f}, "
                  f"Lip={mf['Lip']:.4f})")
        np.save(f"Result/{tag}_theorem.npy", {
            "Shg_phi": sh_res["Sh_g"],
            "Lip_g": lip_res["Lip_g"],
            "sigma_max": lip_res["sigma_max"],
            "softmax_factor": lip_res["softmax_factor"],
            "Lip_g_per_block": np.array(
                [lip_res["sigma_per_block"][b]
                 for b in sorted(lip_res["sigma_per_block"])]),
            "space": space,
        })
    else:
        print(f"\n[Lip] 건너뜀 — {multifc_ckpt} 없음. "
              "main.py를 --use-block-fc로 먼저 실행하세요.")

    # ── 4) Table 1 출력 ──────────────────────────────────────────────────
    metrics = None
    metrics_path = f"Result/{tag}_metrics.npy"
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True).item()

    def _f(v, fmt):
        return format(v, fmt) if v is not None else "—"

    print()
    print("=" * 70)
    print(f"  Table 1  —  {data_name} / {model_tag}   "
          f"(space={space}, T={T}, N={N})")
    print("-" * 70)
    print(f"  F1 Score     : {_f(metrics['f1'] if metrics else None, '.4f')}")
    print(f"  Loss         : {_f(metrics['loss'] if metrics else None, '.4f')}")
    print(f"  g-expansive  : {eps_res['epsilon']:.6e}   "
          f"(class {eps_res['class_a']} vs {eps_res['class_b']}, "
          f"block {eps_res['block']})")
    if sh_res["degenerate"]:
        print("  g-shadowing  : —   (관측 공간 붕괴 — space 변경 또는 probe 재학습 필요)")
    else:
        print(f"  g-shadowing  : {sh_res['Sh_g']:.4f}   "
              f"(eps0={sh_res['eps0']:.4f}, delta*={sh_res['delta_star']:.4f})")
    print(f"  Lip(g)       : "
          f"{_f(lip_res['Lip_g'] if lip_res else None, '.4f')}")
    print("-" * 70)
    print("  topological g-stable: 정리 1  Sh_g(phi) <= Lip(g)·T_g(phi) 에 따라")
    print("                        T_g(phi) >= Sh_g / Lip(g) 로 직접 계산하세요.")
    print("=" * 70)

    if metrics is None:
        print(f"  (F1/Loss는 {metrics_path} 없음 — main.py 실행 시 자동 저장)")

    return {"epsilon": eps_res, "shadowing": sh_res, "lipschitz": lip_res,
            "metrics": metrics}


if __name__ == '__main__':
    args = parse_args()
    init_random(args.seed)
    if args.space == 'feat' and args.n_samples is None:
        print("[안내] space='feat'는 D=200,704라 메모리 소모가 큽니다. "
              "--n-samples 1000 등을 권장합니다.")
    args.model, layers = resolve_model_and_layers(
        args.model, args.data, args.space)
    run_analysis(
        data_name=args.data,
        model_tag=args.model,
        layers=layers,
        space=args.space,
        n_samples=args.n_samples,
        allow_cross_class=args.allow_cross_class,
        depth_consistent=not args.legacy_orbit,
        same_class_trace=not args.cross_class_trace,
        chunk=args.chunk,
        device=args.device,
        seed=args.seed,
    )
