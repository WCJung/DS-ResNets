"""
main.py — 백본 학습 + 블록별 출력 추출.

실행 예:
  python main.py --model ds_resnet18 --data MNIST
  python main.py --model ds_resnet50 --data CIFAR10 --batch-size 64
  python main.py --model resnet18 --data MNIST        # 정확도 baseline 전용

산출물:
  {model}_{data}.pt              — 백본 best 체크포인트 (테스트 손실 기준)
  {model}_{data}_multifc.pt      — 블록별 fc 포함 체크포인트 (--use-block-fc)
  Result/{data}_{model}_metrics.npy — F1/Loss/Acc (Table 1의 성능 열)
  prob_fc/{data}/{model}/        — 블록별 fc logit (N, n_class) — 분석 입력
  pix/resnet/{data}/{model}/test/{data}_label.pt — 테스트 라벨
  prob/{data}/{model}/           — 블록별 raw 특징 (N, 200704) — --save-raw-feat
                                   지정 시에만. legacy space='feat' 전용, OOM 주의.
"""
import argparse
import sys

import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.models import DS_MODELS, build_ds_model, ds_block, ds_layers
from utils.norms import init_random
from utils.stubs import (Exprob, evaluate, extract_block_outputs, load_data,
                         save_block_outputs, save_labels, save_metrics,
                         train, train_block_fc)


def parse_args():
    p = argparse.ArgumentParser(description="DS-ResNets 학습 + 블록 출력 추출")
    p.add_argument('--model', default='ds_resnet18',
                   choices=list(DS_MODELS) + ['resnet18', 'resnet50'])
    p.add_argument('--data', default='MNIST',
                   choices=['MNIST', 'CIFAR10', 'IMAGENET10'])
    p.add_argument('--n-class', type=int, default=10)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--early-stop', type=int, default=20)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--use-block-fc', action=argparse.BooleanOptionalAction,
                   default=True, help='블록별 fc(linear probe) 학습/추출')
    p.add_argument('--use-avgpool', action=argparse.BooleanOptionalAction,
                   default=True, help='main fc 앞 avgpool (fc 입력 2048)')
    p.add_argument('--save-raw-feat', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='블록별 raw 특징(D=200,704) 추출·저장 — legacy '
                        "space='feat' 전용, 메모리 대량 사용 (기본 꺼짐)")
    return p.parse_args()


def build_model(args):
    if args.model in DS_MODELS:
        return build_ds_model(args.model, args.n_class,
                              use_avgpool=args.use_avgpool)
    if args.model == 'resnet18':
        m = tv_models.resnet18(weights=None)
    else:
        m = tv_models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, args.n_class)
    return m


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ds = args.model in DS_MODELS
    ckpt_name = f"{args.model}_{args.data}"

    init_random(args.seed)
    train_dataset, test_dataset = load_data(args.data)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    if is_ds:
        n_blocks = sum(ds_layers(args.model))
        print(f"[설정] {args.model}  |  data={args.data}  |  blocks={n_blocks}"
              f"  |  avgpool={args.use_avgpool}  |  block_fc={args.use_block_fc}")
    else:
        print(f"[설정] {args.model}  |  data={args.data}  (정확도 baseline 모델 — "
              f"레이어별 특징 차원이 달라 블록 분석은 생략)")

    # ── 백본 학습 ─────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    train(model, trainloader, testloader, device,
          epochs=args.epochs, es=args.early_stop, lpth=ckpt_name, lr=args.lr)

    # ── Table 1 성능 지표 (best 체크포인트 기준) ──────────────────────────
    model.load_state_dict(torch.load(f"{ckpt_name}.pt", map_location=device))
    metrics = evaluate(model, testloader, device)
    path = save_metrics(metrics, args.data, args.model)
    print(f"[성능] F1={metrics['f1']:.4f}  Loss={metrics['loss']:.4f}  "
          f"Acc={metrics['acc']*100:.2f}%  → {path}")

    if not is_ds:
        sys.exit(0)

    # ── DS 모델 전용: 블록 단위 출력 추출 ─────────────────────────────────
    extractor = Exprob(args.n_class, layers=ds_layers(args.model),
                       multi_fc=args.use_block_fc, use_avgpool=args.use_avgpool,
                       block=ds_block(args.model))
    extractor.load_state_dict(
        torch.load(f"{ckpt_name}.pt", map_location=device), strict=False)
    extractor.to(device)

    # 분석 파이프라인(space=prob/logit)은 블록 fc logit(D=n_class)만 사용한다.
    # raw 특징(D=200,704)은 legacy space='feat' 전용이며 모든 블록을 CPU RAM에
    # 누적해 ds_resnet50(16블록)에서 ~128GB를 요구, OOM의 원인이었다. 따라서
    # 값싼 block_fc 추출을 먼저 하고(라벨도 여기서 확보), raw 추출은 opt-in으로.
    labels = None

    if args.use_block_fc:
        print("블록별 fc 학습 중...")
        train_block_fc(extractor, trainloader, device, epochs=5)
        torch.save(extractor.state_dict(), f"{ckpt_name}_multifc.pt")

        print("블록별 fc logit 추출 중...")
        logits, labels = extract_block_outputs(extractor, testloader, device,
                                               use_block_fc=True)
        fc_dir = save_block_outputs(logits, "prob_fc", args.data, args.model)
        print(f"fc logit → {fc_dir}/")

    if args.save_raw_feat:
        print("블록별 raw 특징 추출 중 (D=200,704 — 메모리 대량 사용)...")
        feats, labels_raw = extract_block_outputs(extractor, testloader, device)
        feat_dir = save_block_outputs(feats, "prob", args.data, args.model)
        del feats
        labels = labels if labels is not None else labels_raw
        print(f"raw 특징 → {feat_dir}/  (space='feat' 전용)")

    if labels is not None:
        pix_dir = save_labels(labels, args.data, args.model)
        print(f"라벨 → {pix_dir}/")
    else:
        print("[안내] --no-use-block-fc 且 --no-save-raw-feat: 분석용 출력·라벨 "
              "없음 (정확도 baseline만). 분석하려면 --use-block-fc로 재실행하세요.")

    print(f"완료. 다음 단계:")
    print(f"  python dist_calc.py   --model {args.model} --data {args.data}"
          f"   # 안정성 (Table 1)")
    print(f"  python entropy_calc.py --model {args.model} --data {args.data}"
          f"   # FTTE")
