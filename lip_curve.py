"""
lip_curve.py — 블록별 Lipschitz 곡선  t -> Lip(g_t)  저장 + 플롯.

기존 block-wise probe checkpoint({model}_{dataset}_multifc.pt)만 읽어서
dist_calc.py 와 "동일한 방식"(utils.lipschitz.lip_report_from_checkpoint 재사용,
space='prob' 이면 softmax 보정 0.5 포함)으로 블록별 Lip(g_t) = sigma_max(W_t)
x factor 를 계산한다. 재학습/재추출 없음. 곡선의 최댓값 L_max 는 dist_calc.py
Table 1 의 Lip(g) 와 정확히 일치한다 (스크립트 내부에서 assert 로 보장).

지표 정의:
  L_max        = max_t Lip(g_t)            (= Table 1 의 Lip(g))
  L_mean/L_std = 블록 평균/표준편차
  R_peak       = L_max / L_mean            (정점 대 평균 비 — 프로파일 뾰족함)
  argmax_block = L_max 를 달성한 블록 t

산출물 (--out-dir, 기본 Result/lip_curves):
  {dataset}_{model}_lip_curve.npy   — dict(dataset, model, space, block_lip,
                                      L_max, L_mean, L_std, R_peak,
                                      argmax_block, T)
  lip_curve_summary.csv             — dataset,model,space,T,L_max,L_mean,
                                      L_std,R_peak,argmax_block
  {dataset}_lip_curve.png / .pdf    — 데이터셋별 곡선 (performance 실선 /
                                      provable 점선)

실행 예시:
  # 기본: families=resnet, depths=50, modes=performance,provable
  #   -> isolift_resnet_performance_50, isolift_resnet_provable_50
  python lip_curve.py

  # ResNet-50 provable/performance
  python lip_curve.py --families resnet --depths 50 --modes performance,provable

  # Wide-101 provable 만
  python lip_curve.py --families wide --depths 101 --modes provable

  # 전체 12개 조합
  python lip_curve.py --families resnet,wide,resnext --depths 50,101 \
      --modes performance,provable

  # 직접 model tag 지정 (--families/--depths/--modes 무시)
  python lip_curve.py --models isolift_resnet_provable_50,isolift_wide_performance_101

sanity check (사용자 실측값과 대조):
  CIFAR10    provable    L_max ~= 7.8964
  CIFAR10    performance L_max ~= 31.7566
  IMAGENET10 provable    L_max ~= 1.2980
  MNIST      performance L_max ~= 44.5439
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.lipschitz import lip_report_from_checkpoint

FAMILIES = ("resnet", "wide", "resnext")
DEPTHS = ("50", "101")
MODES = ("performance", "provable")


def parse_args():
    p = argparse.ArgumentParser(
        description="블록별 Lip(g_t) 곡선 저장/플롯 (probe ckpt 만 읽음)")
    p.add_argument("--models", default=None,
                   help="쉼표 구분 model tag 직접 지정 — 지정 시 "
                        "--families/--depths/--modes 는 무시됨 "
                        "(예: isolift_resnet_provable_50,...)")
    p.add_argument("--datasets", default="CIFAR10,IMAGENET10,MNIST",
                   help="쉼표 구분 (기본: CIFAR10,IMAGENET10,MNIST)")
    p.add_argument("--space", default="logit", choices=["logit", "prob"],
                   help="dist_calc 와 동일한 관측 공간 (prob 이면 x0.5 보정)")
    p.add_argument("--out-dir", default="Result/lip_curves")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--families", default="resnet",
                   help=f"쉼표 구분, 선택지 {FAMILIES} (기본 resnet)")
    p.add_argument("--depths", default="50",
                   help=f"쉼표 구분, 선택지 {DEPTHS} (기본 50)")
    p.add_argument("--modes", default="performance,provable",
                   help=f"쉼표 구분, 선택지 {MODES} (기본 둘 다)")
    return p.parse_args()


def build_model_tags(args):
    """--models 명시 시 그대로, 아니면 families x modes x depths 조합 생성."""
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]

    def _split(value, allowed, name):
        items = [v.strip() for v in value.split(",") if v.strip()]
        bad = [v for v in items if v not in allowed]
        if bad:
            raise SystemExit(f"[에러] --{name} 값 {bad} — 선택지: {allowed}")
        return items

    fams = _split(args.families, FAMILIES, "families")
    depths = _split(args.depths, DEPTHS, "depths")
    modes = _split(args.modes, MODES, "modes")
    return [f"isolift_{f}_{m}_{d}"
            for f in fams for d in depths for m in modes]


def compute_curve(model, dataset, space):
    """probe ckpt 에서 t -> Lip(g_t) 곡선과 요약 지표 계산. 없으면 None."""
    ckpt = f"{model}_{dataset}_multifc.pt"
    if not os.path.exists(ckpt):
        print(f"missing checkpoint: {ckpt}")
        print("block-wise probe extraction or dist_calc.py pipeline "
              "must be run first.")
        return None

    rep = lip_report_from_checkpoint(ckpt, space=space)
    factor = rep["softmax_factor"]
    blocks = sorted(rep["sigma_per_block"])
    block_lip = np.array([rep["sigma_per_block"][b] * factor
                          for b in blocks], dtype=float)

    l_max = float(block_lip.max())
    # dist_calc Table 1 의 Lip(g) 와 동일함을 보장
    assert abs(l_max - rep["Lip_g"]) < 1e-9, (l_max, rep["Lip_g"])

    l_mean = float(block_lip.mean())
    return {
        "dataset": dataset,
        "model": model,
        "space": space,
        "block_lip": block_lip,
        "L_max": l_max,
        "L_mean": l_mean,
        "L_std": float(block_lip.std()),
        "R_peak": l_max / l_mean,
        "argmax_block": int(block_lip.argmax()),
        "T": int(len(block_lip)),
    }


def plot_dataset(dataset, curves, space, out_dir):
    """한 데이터셋의 모든 모델 곡선 — performance 실선 / provable 점선."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for c in curves:
        style = "--" if "_provable" in c["model"] else "-"
        line, = ax.plot(range(c["T"]), c["block_lip"], style,
                        marker="o", markersize=4, label=c["model"])
        ax.plot(c["argmax_block"], c["L_max"], "*", markersize=11,
                color=line.get_color())
    ax.set_xlabel("block index t")
    ax.set_ylabel("Lip(g_t)")
    ax.set_title(f"{dataset} — block-wise Lip(g_t)  (space={space})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{dataset}_lip_curve.{ext}"),
                    dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    models = build_model_tags(args)
    datasets = [d.strip().upper() for d in args.datasets.split(",")
                if d.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[lip_curve] models={models}")
    print(f"[lip_curve] datasets={datasets}  space={args.space}  "
          f"out={args.out_dir}")

    rows = []
    for dataset in datasets:
        curves = []
        for model in models:
            c = compute_curve(model, dataset, args.space)
            if c is None:
                continue
            np.save(os.path.join(
                args.out_dir, f"{dataset}_{model}_lip_curve.npy"), c)
            curves.append(c)
            rows.append(c)
            print(f"  {dataset:<11} {model:<34} T={c['T']:>3}  "
                  f"L_max={c['L_max']:8.4f} (block {c['argmax_block']})  "
                  f"L_mean={c['L_mean']:8.4f}  R_peak={c['R_peak']:.3f}")
        if args.plot and curves:
            plot_dataset(dataset, curves, args.space, args.out_dir)

    if not rows:
        raise SystemExit("[에러] 처리된 조합이 없습니다 — checkpoint 경로를 "
                         "확인하세요.")

    csv_path = os.path.join(args.out_dir, "lip_curve_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "space", "T", "L_max", "L_mean",
                    "L_std", "R_peak", "argmax_block"])
        for c in rows:
            w.writerow([c["dataset"], c["model"], c["space"], c["T"],
                        f"{c['L_max']:.6f}", f"{c['L_mean']:.6f}",
                        f"{c['L_std']:.6f}", f"{c['R_peak']:.6f}",
                        c["argmax_block"]])
    print(f"[lip_curve] 요약 CSV → {csv_path}"
          + (f"  |  플롯 → {args.out_dir}/{{dataset}}_lip_curve.png/.pdf"
             if args.plot else ""))


if __name__ == "__main__":
    main()
