"""
train_isolift.py — IsoLift-ResNeXt 멀티 데이터셋 공동 학습.

구조:  x_d -E_d-> z_0(48x56x56) -A_d-> u_0 -T(공유)-> u_L -H_d-> y_d
손실:  L = sum_d CE^(d) + λ_geo·L_geo + λ_lip·L_lip     (자세한 정의는
       utils/isolift_losses.py, 아키텍처는 models/isolift.py 참조)

각 데이터셋은 "원본 해상도"로 로드된다 (기존 utils.stubs.load_data 의
224 리사이즈와 다름 — E_d 가 원본 차원에서 등거리 lifting 을 수행):
    MNIST 1x28x28 / CIFAR10 3x32x32 / IMAGENET10(Imagenette) 3x224x224

실행 예 (기본값 = 실험 C: AdamW 3e-4 + cosine/warmup5 + ES50 + ls0.1):
  python train_isolift.py --mode performance
  python train_isolift.py --mode provable --lr 1e-4 --lambda-lip 0
  # 실험 A (기존 DS-ResNets Table 1과 동일 조건 — 비교용):
  python train_isolift.py --optimizer adam --lr 5e-5 --weight-decay 0 \
      --scheduler none --warmup-epochs 0 --epochs 100 --early-stop 20 \
      --label-smoothing 0
산출물:
  isolift_{family}_{mode}.pt                  — best 평균 정확도 체크포인트
  Result/isolift_{family}_{mode}_metrics.npy — 도메인별 acc 이력

backbone 계열은 --family {resnet,wide,resnext} 로 선택 (DS 3계열과 평행).
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
    p.add_argument("--family", default="resnext",
                   choices=["resnet", "wide", "resnext"],
                   help="공유 backbone 블록 계열 (DS 3계열과 평행): "
                        "resnet=폭 C/4 baseline / wide=폭 C/2+dropout, "
                        "pre-act / resnext=폭 C/3 grouped")
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
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW decoupled weight decay "
                        "(BN·bias·alpha·beta 등 1차원 파라미터 제외)")
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="선형 warmup 에폭 수 (cosine 사용 시 권장)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=50,
                   help="평균 정확도가 이 에폭 수 동안 개선 없으면 중단 (0=끔)")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def build_optimizer(model, args):
    """weight decay 를 2차원 이상 가중치에만 적용 (BN·bias·alpha·beta 제외)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr)
    return torch.optim.Adam(groups, lr=args.lr)


def build_scheduler(optimizer, args):
    """선형 warmup 후 cosine decay (에폭 단위 step). 'none' 이면 None."""
    if args.scheduler == "none":
        return None
    warm = max(args.warmup_epochs, 0)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warm, 1))
    if warm == 0:
        return sched_cos
    sched_warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=warm)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [sched_warm, sched_cos], milestones=[warm])


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
                       rho=args.rho, family=args.family).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[IsoLift] family={args.family}  mode={args.mode}  "
          f"domains={domains}  layers={layers}  "
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

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    steps = max(len(l) for l in train_loaders.values())
    print(f"  optim={args.optimizer}  lr={args.lr}  wd={args.weight_decay}  "
          f"sched={args.scheduler}(warmup {args.warmup_epochs})  "
          f"epochs={args.epochs}  early_stop={args.early_stop}  "
          f"label_smoothing={args.label_smoothing}")

    os.makedirs("Result", exist_ok=True)
    history, best, best_epoch = [], 0.0, 0

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

        if scheduler is not None:
            scheduler.step()

        accs = evaluate(model, test_loaders, device)
        avg = sum(accs.values()) / len(accs)
        history.append({"epoch": epoch + 1, "accs": accs,
                        "lr": optimizer.param_groups[0]["lr"],
                        **{k: v / steps for k, v in sums.items()}})
        acc_str = "  ".join(f"{d}={a*100:.2f}%" for d, a in accs.items())
        print(f"[{epoch+1:03d}/{args.epochs}] cls={sums['cls']/steps:.4f}  "
              f"geo={sums['geo']/steps:.4f}  lip={sums['lip']/steps:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  |  "
              f"{acc_str}  (avg {avg*100:.2f}%)")

        if avg > best:
            best, best_epoch = avg, epoch
            torch.save(model.state_dict(), f"isolift_{args.family}_{args.mode}.pt")
        elif args.early_stop > 0 and epoch - best_epoch >= args.early_stop:
            print(f"[early stop] {args.early_stop} 에폭 동안 개선 없음 "
                  f"(best = epoch {best_epoch+1}, avg {best*100:.2f}%)")
            break

    np.save(f"Result/isolift_{args.family}_{args.mode}_metrics.npy",
            {"history": history, "best_avg_acc": best,
             "domains": domains, "mode": args.mode, "layers": layers,
             "rho": args.rho, "lambda_geo": args.lambda_geo,
             "lambda_lip": args.lambda_lip,
             "optimizer": args.optimizer, "lr": args.lr,
             "weight_decay": args.weight_decay, "scheduler": args.scheduler,
             "warmup_epochs": args.warmup_epochs,
             "early_stop": args.early_stop,
             "label_smoothing": args.label_smoothing})
    print(f"완료.  best 평균 acc = {best*100:.2f}%  "
          f"→ isolift_{args.family}_{args.mode}.pt / Result/isolift_{args.family}_{args.mode}_metrics.npy")


if __name__ == "__main__":
    main()
