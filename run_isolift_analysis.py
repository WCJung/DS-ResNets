"""
run_isolift_analysis.py — IsoLift 안정성/엔트로피 분석 일괄 실행.

학습된 isolift_{family}_{mode}.pt 마다:
  1. prob_fc 관측값이 없으면 extract_isolift.run_extract 로 자동 추출
     (블록별 probe 학습 포함)
  2. 도메인마다 dist_calc.run_analysis  (g-expansive / Sh_g / Lip(g) / Table 1)
  3. 도메인마다 entropy_calc.run_entropy (FTTE / s_T=m 대역)
을 순차 실행하고 요약 표를 출력한다. 조합 하나가 실패해도 다음으로 진행.

실행 예:
  python run_isolift_analysis.py                              # 3계열 x performance
  python run_isolift_analysis.py --families resnet --modes performance,provable
  python run_isolift_analysis.py --datasets MNIST,CIFAR10     # 도메인 제한
"""
import argparse
import os
import time
import traceback

import torch

from dist_calc import run_analysis
from entropy_calc import run_entropy
from extract_isolift import run_extract, save_domain_metrics
from models.isolift import ISOLIFT_FAMILIES
from utils.trajectory import infer_n_blocks


def parse_args():
    p = argparse.ArgumentParser(description="IsoLift 분석 일괄 실행")
    p.add_argument("--families", default="resnet,wide,resnext",
                   help="쉼표 구분 (기본: 3계열 전부)")
    p.add_argument("--modes", default="performance",
                   help="쉼표 구분: performance / provable")
    p.add_argument("--datasets", default=None,
                   help="분석할 도메인 제한 (기본: 체크포인트의 전체 도메인)")
    p.add_argument("--space", default="logit", choices=["prob", "logit"],
                   help="d_g 관측 공간 (기본 logit — softmax 포화 회피)")
    p.add_argument("--probe-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=None,
                   help="분석 서브샘플 수 — IMAGENET1K(val 5만 장) 등 큰 "
                        "테스트셋에 권장 (예: 10000)")
    p.add_argument("--imagenet-root", default=None,
                   help="ImageNet-1k ImageFolder 루트 (IMAGENET1K 도메인용)")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--device", default=None,
                   help="기본: cuda 가능하면 cuda")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main():
    args = parse_args()
    if args.imagenet_root:
        os.environ["IMAGENET_ROOT"] = args.imagenet_root
    device = args.device or ("cuda" if torch.cuda.is_available() else None)
    families = [f.strip() for f in args.families.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    only = ([d.strip().upper() for d in args.datasets.split(",")]
            if args.datasets else None)
    bad = [f for f in families if f not in ISOLIFT_FAMILIES]
    if bad:
        raise SystemExit(f"알 수 없는 계열 {bad} — 사용 가능: "
                         f"{list(ISOLIFT_FAMILIES)}")

    results = {}
    t_start = time.time()

    for family in families:
        for mode in modes:
            tag = f"isolift_{family}_{mode}"
            ckpt = f"{tag}.pt"
            if not os.path.exists(ckpt):
                print(f"\n[skip] {ckpt} 없음 — train_isolift.py 미실행 조합")
                results[(tag, "-")] = ("SKIP", "체크포인트 없음")
                continue

            # ── 도메인 목록: 체크포인트에서 복원 ─────────────────────
            from extract_isolift import _infer_structure
            domains, _ = _infer_structure(
                torch.load(ckpt, map_location="cpu"))
            if only:
                domains = [d for d in domains if d in only]

            # ── 1) 관측값 없으면 자동 추출 ───────────────────────────
            missing = [d for d in domains
                       if infer_n_blocks(d, tag, args.space) == 0]
            if missing:
                print(f"\n{'='*70}\n[{tag}] 관측값 없음({missing}) — "
                      f"probe 추출 실행\n{'='*70}")
                try:
                    run_extract(family, mode, probe_epochs=args.probe_epochs,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                device=device, seed=args.seed)
                except Exception:
                    tb = traceback.format_exc()
                    print(tb.splitlines()[-1])
                    for d in domains:
                        results[(tag, d)] = ("FAILED", "추출 실패")
                    continue

            # ── 1.5) Table 1 성능 열 (F1/Loss/Acc) 누락분 채우기 ─────
            # 이전 버전 extract 로 추출한 조합은 metrics 파일이 없다 —
            # probe 재학습 없이 평가만 수행해 채운다.
            if any(not os.path.exists(f"Result/{d}_{tag}_metrics.npy")
                   for d in domains):
                print(f"[{tag}] 도메인별 F1/Loss/Acc 평가 (metrics 누락분)...")
                try:
                    save_domain_metrics(family, mode, device=device,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers)
                except Exception:
                    print(traceback.format_exc().splitlines()[-1])
                    print("[경고] metrics 평가 실패 — Table 1 의 F1/Loss 는 "
                          "'—' 로 표시됩니다 (분석은 계속 진행).")

            # ── 2)+3) 도메인별 분석 ──────────────────────────────────
            for d in domains:
                print(f"\n{'='*70}\n[{tag} / {d}] 안정성 분석 + FTTE  "
                      f"(space={args.space})\n{'='*70}")
                t0 = time.time()
                try:
                    n_blocks = infer_n_blocks(d, tag, args.space)
                    run_analysis(data_name=d, model_tag=tag,
                                 layers=[n_blocks], space=args.space,
                                 n_samples=args.n_samples,
                                 chunk=args.chunk, device=device,
                                 seed=args.seed)
                    run_entropy(data_name=d, model_tag=tag,
                                layers=[n_blocks], space=args.space,
                                n_samples=args.n_samples,
                                chunk=args.chunk, device=device,
                                seed=args.seed)
                    dur = time.time() - t0
                    results[(tag, d)] = ("OK", f"{dur:.0f}s")
                except Exception:
                    tb = traceback.format_exc()
                    print(tb)
                    results[(tag, d)] = ("FAILED", tb.splitlines()[-1][:40])

    # ── 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SUMMARY  (총 {time.time()-t_start:.0f}s)\n{'-'*70}")
    print(f"  {'Tag':<32} {'Dataset':<12} {'Status':<8} Note")
    for (tag, d), (status, note) in results.items():
        print(f"  {tag:<32} {d:<12} {status:<8} {note}")
    ok = sum(1 for s, _ in results.values() if s == "OK")
    print(f"{'-'*70}\n  {ok}/{len(results)} 성공")


if __name__ == "__main__":
    main()
