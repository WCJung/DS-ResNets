"""
extract_isolift.py — IsoLift 백본의 블록별 probe 학습 + d_g 관측값 추출.

기존 DS 파이프라인(main.py 의 block_fc 단계)에 대응하는 IsoLift 버전:
학습된 IsoLift 체크포인트를 얼리고, 블록마다 선형 probe
g_b = Linear(C_b, n_class) ∘ GAP 를 도메인별로 학습한 뒤 테스트셋의
블록별 logit 을 기존 분석 파이프라인이 읽는 포맷으로 저장한다.

산출물 (도메인 d 마다, tag = isolift_{family}_{mode}):
  prob_fc/{d}/{tag}/{d}_block{b}.pt          — 블록별 probe logit (N, n_class)
  pix/resnet/{d}/{tag}/test/{d}_label.pt     — 테스트 라벨
  {tag}_{d}_multifc.pt                       — probe 가중치 (block_fc.{b}.* 키,
                                               utils.lipschitz 가 그대로 읽음)

이후 기존 스크립트를 IsoLift 태그로 실행하면 Table 1 분석이 그대로 됨:
  python extract_isolift.py --family resnet --mode performance
  python dist_calc.py    --model isolift_resnet_performance --data MNIST \
      --space logit --device cuda
  python entropy_calc.py --model isolift_resnet_performance --data MNIST \
      --space logit --device cuda
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.isolift import ISOLIFT_FAMILIES, IsoLiftNet
from train_isolift import native_datasets
from utils.norms import init_random
from utils.stubs import save_block_outputs, save_labels


def parse_args():
    p = argparse.ArgumentParser(
        description="IsoLift 블록별 probe 학습 + 관측값 추출")
    p.add_argument("--family", default="resnext",
                   choices=list(ISOLIFT_FAMILIES))
    p.add_argument("--mode", default="performance",
                   choices=["performance", "provable"])
    p.add_argument("--ckpt", default=None,
                   help="체크포인트 경로 (기본 isolift_{family}_{mode}.pt)")
    p.add_argument("--cardinality", type=int, default=8,
                   help="학습 때와 동일해야 함 (shape 불일치 시 로드 에러)")
    p.add_argument("--probe-epochs", type=int, default=5)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def _infer_structure(state):
    """체크포인트 키에서 (domains, layers) 를 복원."""
    domains = sorted({k.split(".")[1] for k in state if k.startswith("heads.")})
    n_stage = 1 + max(int(k.split(".")[1]) for k in state
                      if k.startswith("stages."))
    layers = []
    for s in range(n_stage):
        layers.append(1 + max(int(k.split(".")[2]) for k in state
                              if k.startswith(f"stages.{s}.")))
    return domains, tuple(layers)


def main():
    args = parse_args()
    init_random(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = f"isolift_{args.family}_{args.mode}"
    ckpt = args.ckpt or f"{tag}.pt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"[에러] {ckpt} 없음 — train_isolift.py 를 먼저 "
                         f"실행하세요 (--family {args.family} --mode {args.mode}).")

    state = torch.load(ckpt, map_location="cpu")
    domains, layers = _infer_structure(state)
    model = IsoLiftNet(domains=domains, layers=layers, mode=args.mode,
                       family=args.family, cardinality=args.cardinality)
    model.load_state_dict(state)          # strict — 구조 불일치 시 즉시 에러
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dims = model.block_channels()
    n_class = model.heads[domains[0]].out_features
    print(f"[extract] {ckpt}  domains={domains}  blocks={len(dims)}  "
          f"dims={dims}  device={device}")

    for d in domains:
        tr, te = native_datasets(d)
        trainloader = torch.utils.data.DataLoader(
            tr, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            te, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        probes = nn.ModuleList(
            [nn.Linear(c, n_class) for c in dims]).to(device)
        opt = torch.optim.Adam(probes.parameters(), lr=args.probe_lr)

        # ── probe 학습 (백본 고정) ─────────────────────────────────────
        print(f"[{d}] 블록별 probe 학습 ({args.probe_epochs} epochs)...")
        for epoch in range(args.probe_epochs):
            total = 0.0
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    feats = model.block_features(x, d)
                opt.zero_grad()
                loss = sum(
                    F.cross_entropy(
                        probes[b](F.adaptive_avg_pool2d(f, 1).flatten(1)), y)
                    for b, f in enumerate(feats))
                loss.backward()
                opt.step()
                total += float(loss.detach())
            print(f"    epoch {epoch+1}/{args.probe_epochs}  "
                  f"loss={total/len(trainloader):.4f}")

        # ── 테스트셋 블록별 logit 추출 ─────────────────────────────────
        print(f"[{d}] 테스트셋 블록별 logit 추출...")
        probes.eval()
        outs, labels = {b: [] for b in range(len(dims))}, []
        with torch.no_grad():
            for x, y in testloader:
                feats = model.block_features(x.to(device), d)
                for b, f in enumerate(feats):
                    logit = probes[b](
                        F.adaptive_avg_pool2d(f, 1).flatten(1))
                    outs[b].append(logit.cpu())
                labels.append(y)
        outs = {b: torch.cat(v) for b, v in outs.items()}
        labels = torch.cat(labels)

        fc_dir = save_block_outputs(outs, "prob_fc", d, tag)
        pix_dir = save_labels(labels, d, tag)

        # Lip(g) 용 probe 체크포인트 — utils.lipschitz 의 block_fc.{b}.* 포맷
        probe_state = {}
        for b, lin in enumerate(probes):
            probe_state[f"block_fc.{b}.weight"] = lin.weight.detach().cpu()
            probe_state[f"block_fc.{b}.bias"] = lin.bias.detach().cpu()
        torch.save(probe_state, f"{tag}_{d}_multifc.pt")

        acc = float((outs[len(dims)-1].argmax(1) == labels).float().mean())
        print(f"[{d}] 완료 — logit → {fc_dir}/  |  라벨 → {pix_dir}/  |  "
              f"probe ckpt → {tag}_{d}_multifc.pt  "
              f"(마지막 블록 probe acc {acc*100:.2f}%)")

    print("\n다음 단계 (도메인별 안정성/엔트로피 분석):")
    for d in domains:
        print(f"  python dist_calc.py    --model {tag} --data {d} "
              f"--space logit --device cuda")
        print(f"  python entropy_calc.py --model {tag} --data {d} "
              f"--space logit --device cuda")


if __name__ == "__main__":
    main()
