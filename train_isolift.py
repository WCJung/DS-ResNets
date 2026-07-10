"""
train_isolift.py — IsoLift-ResNeXt 멀티 데이터셋 공동 학습.

구조:  x_d -E_d-> z_0(48x56x56) -A_d-> u_0 -T(공유)-> u_L -H_d-> y_d
손실:  L = sum_d CE^(d) + λ_geo·L_geo + λ_lip·L_lip     (자세한 정의는
       utils/isolift_losses.py, 아키텍처는 models/isolift.py 참조)

각 데이터셋은 "원본 해상도"로 로드된다 (기존 utils.stubs.load_data 의
224 리사이즈와 다름 — E_d 가 원본 차원에서 등거리 lifting 을 수행):
    MNIST 1x28x28 / CIFAR10 3x32x32 / IMAGENET10(Imagenette) 3x224x224

실행 예:
  python train_isolift.py --mode performance --datasets MNIST,CIFAR10
  python train_isolift.py --mode provable --datasets MNIST,CIFAR10,IMAGENET10 \
      --epochs 60 --lambda-lip 0
산출물:
  isolift_{mode}.pt                    — best 평균 정확도 체크포인트
  Result/isolift_{mode}_metrics.npy   — 도메인별 acc 이력
"""
import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from models.isolift import LIFTS, IsoLiftNet
from utils.isolift_losses import geometry_loss, lipschitz_penalty
from utils.norms import init_random


# ── 원본 해상도 로더 (E_d 의 IN_SHAPE 와 일치해야 함) ─────────────────────────

def native_datasets(name):
    if name == "MNIST":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tr = torchvision.datasets.MNIST("data", train=True, download=True,
                                        transform=tf)
        te = torchvision.datasets.MNIST("data", train=False, download=True,
                                        transform=tf)
    elif name == "CIFAR10":
        norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        tr = torchvision.datasets.CIFAR10(
            "data", train=True, download=True,
            transform=T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(), T.ToTensor(), norm]))
        te = torchvision.datasets.CIFAR10(
            "data", train=False, download=True,
            transform=T.Compose([T.ToTensor(), norm]))
    elif name == "IMAGENET10":
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        tr = torchvision.datasets.Imagenette(
            "data", split="train", size="320px", download=True,
            transform=T.Compose([T.RandomResizedCrop(224),
                                 T.RandomHorizontalFlip(), T.ToTensor(), norm]))
        te = torchvision.datasets.Imagenette(
            "data", split="val", size="320px", download=True,
            transform=T.Compose([T.Resize(256), T.CenterCrop(224),
                                 T.ToTensor(), norm]))
    else:
        raise ValueError(f"알 수 없는 데이터셋: {name}")
    return tr, te


def parse_args():
    p = argparse.ArgumentParser(description="IsoLift-ResNeXt 공동 학습")
    p.add_argument("--datasets", default="MNIST,CIFAR10,IMAGENET10",
                   help="쉼표 구분 (기본: 3개 전부)")
    p.add_argument("--mode", default="performance",
                   choices=["performance", "provable"])
    p.add_argument("--layers", default="3,3,3",
                   help="stage 별 블록 수 (쉼표 구분)")
    p.add_argument("--cardinality", type=int, default=8)
    p.add_argument("--rho", type=float, default=0.9,
                   help="branch Lipschitz 목표 (<1)")
    p.add_argument("--lambda-geo", type=float, default=0.1)
    p.add_argument("--lambda-lip", type=float, default=0.01,
                   help="performance 모드 soft penalty 계수 (provable 은 0 권장)")
    p.add_argument("--geo-m", type=float, default=0.5)
    p.add_argument("--geo-M", type=float, default=2.0)
    p.add_argument("--geo-eps", type=float, default=0.05,
                   help="geometry 쌍 x' = x + eps·N(0,1) 의 eps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loaders, device):
    model.eval()
    accs = {}
    for d, loader in loaders.items():
        correct = total = 0
        for x, y in loader:
            pred = model(x.to(device), d).argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += y.numel()
        accs[d] = correct / total
    model.train()
    return accs


def main():
    args = parse_args()
    init_random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domains = [s.strip().upper() for s in args.datasets.split(",")]
    unknown = [d for d in domains if d not in LIFTS]
    if unknown:
        raise SystemExit(f"알 수 없는 데이터셋 {unknown} — 사용 가능: {list(LIFTS)}")
    layers = tuple(int(v) for v in args.layers.split(","))

    model = IsoLiftNet(domains=domains, layers=layers,
                       cardinality=args.cardinality, mode=args.mode,
                       rho=args.rho).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[IsoLift] mode={args.mode}  domains={domains}  layers={layers}  "
          f"params={n_param/1e6:.2f}M  device={device}")

    train_loaders, test_loaders = {}, {}
    for d in domains:
        tr, te = native_datasets(d)
        train_loaders[d] = torch.utils.data.DataLoader(
            tr, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loaders[d] = torch.utils.data.DataLoader(
            te, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        print(f"  {d:<11} train={len(tr):,}  test={len(te):,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    ce = nn.CrossEntropyLoss()
    steps = max(len(l) for l in train_loaders.values())

    os.makedirs("Result", exist_ok=True)
    history, best = [], 0.0

    for epoch in range(args.epochs):
        iters = {d: itertools.cycle(l) for d, l in train_loaders.items()}
        sums = {k: 0.0 for k in ("cls", "geo", "lip")}

        for _ in range(steps):
            optimizer.zero_grad()
            loss_cls = loss_geo = 0.0
            for d in domains:
                x, y = next(iters[d])
                x, y = x.to(device), y.to(device)
                logits, u0 = model(x, d, return_u0=True)
                loss_cls = loss_cls + ce(logits, y)
                if args.lambda_geo > 0:
                    x2 = x + args.geo_eps * torch.randn_like(x)
                    u02 = model.lift_and_adapt(x2, d)
                    loss_geo = loss_geo + geometry_loss(
                        x, x2, u0, u02, m=args.geo_m, M=args.geo_M)
            loss_lip = (lipschitz_penalty(model, rho=args.rho)
                        if args.lambda_lip > 0 else torch.tensor(0.0))
            loss = (loss_cls + args.lambda_geo * loss_geo
                    + args.lambda_lip * loss_lip)
            loss.backward()
            optimizer.step()
            sums["cls"] += float(loss_cls.detach())
            sums["geo"] += float(loss_geo.detach()) \
                if torch.is_tensor(loss_geo) else float(loss_geo)
            sums["lip"] += float(loss_lip.detach())

        accs = evaluate(model, test_loaders, device)
        avg = sum(accs.values()) / len(accs)
        history.append({"epoch": epoch + 1, "accs": accs,
                        **{k: v / steps for k, v in sums.items()}})
        acc_str = "  ".join(f"{d}={a*100:.2f}%" for d, a in accs.items())
        print(f"[{epoch+1:03d}/{args.epochs}] cls={sums['cls']/steps:.4f}  "
              f"geo={sums['geo']/steps:.4f}  lip={sums['lip']/steps:.5f}  |  "
              f"{acc_str}  (avg {avg*100:.2f}%)")

        if avg > best:
            best = avg
            torch.save(model.state_dict(), f"isolift_{args.mode}.pt")

    np.save(f"Result/isolift_{args.mode}_metrics.npy",
            {"history": history, "best_avg_acc": best,
             "domains": domains, "mode": args.mode, "layers": layers,
             "rho": args.rho, "lambda_geo": args.lambda_geo,
             "lambda_lip": args.lambda_lip})
    print(f"완료.  best 평균 acc = {best*100:.2f}%  "
          f"→ isolift_{args.mode}.pt / Result/isolift_{args.mode}_metrics.npy")


if __name__ == "__main__":
    main()
