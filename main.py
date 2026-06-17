import sys
import os
import torch
import torch.nn as nn
import argparse
import torchvision.models as tv_models
import models.models as ml
from utils.norms import init_random
from utils.stubs import load_data, train, train_block_fc, Exprob

if __name__ == '__main__':
    param = argparse.ArgumentParser()
    param.add_argument(
        '--model',
        default='resnet',
        type=str
    )
    param.add_argument(
        '--bc',
        help='block configuration',
        default=[3, 4, 6, 3],
        type=list
    )
    param.add_argument(
        '--weight',
        default=False,
        type=bool
    )
    params = param.parse_args()
    seed = 13
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ══════════════════════════════════════════════════════════════════════════
    # 실험 설정  ← 여기만 바꾸면 됩니다
    # ══════════════════════════════════════════════════════════════════════════
    #
    # 최종 실험 스코프: 4개 모델 x 3개 데이터셋
    #   모델   : 'resnet18' | 'resnet50' | 'ds_resnet18' | 'ds_resnet50'
    #   데이터 : 'MNIST' | 'CIFAR10' | 'IMAGENET10'  (IMAGENET10 = Imagenette,
    #            fast.ai의 라이선스-free 10클래스 ImageNet 서브셋, 자동 다운로드)
    #
    # DS-ResNet 명명 규칙: 블록 수가 아니라 "비교 대상 ResNet"을 직접 가리킴
    #   ds_resnet18 -> layers=[2,2,2,2] (8 블록,  ResNet-18과 블록 수 동일)
    #   ds_resnet50 -> layers=[3,4,6,3] (16 블록, ResNet-50과 블록 수 동일)

    MODEL_NAME = 'ds_resnet18'   # 'resnet18' | 'resnet50' | 'ds_resnet18' | 'ds_resnet50'
    DATA_NAME  = 'MNIST'         # 'MNIST' | 'CIFAR10' | 'IMAGENET10'
    n_class    = 10

    USE_BLOCK_FC = True    # DS-ResNet 전용: 블록별 fc 학습/추출 여부
    USE_AVGPOOL  = True    # DS-ResNet 전용: avgpool 사용 -> fc 입력 2,048, Lip(g) tight

    DS_LAYERS_MAP = {
        'ds_resnet18': [2, 2, 2, 2],
        'ds_resnet50': [3, 4, 6, 3],
    }

    # ══════════════════════════════════════════════════════════════════════════

    is_ds_resnet = MODEL_NAME in DS_LAYERS_MAP
    model_tag    = MODEL_NAME
    ckpt_name    = f"{model_tag}_{DATA_NAME}"

    init_random(seed)
    train_dataset, test_dataset = load_data(DATA_NAME)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)
    testloader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=16, shuffle=False)

    if is_ds_resnet:
        ds_layers = DS_LAYERS_MAP[MODEL_NAME]
        n_blocks  = sum(ds_layers)
        model = ml.model_(params, device, use_avgpool=USE_AVGPOOL, layers=ds_layers)
        ml.transfer(model, seed, n_class)
        print(f"[설정] {model_tag}  |  data={DATA_NAME}  |  blocks={n_blocks}"
              f"  |  avgpool={USE_AVGPOOL}")
    else:
        if MODEL_NAME == 'resnet18':
            model = tv_models.resnet18(weights=None)
        elif MODEL_NAME == 'resnet50':
            model = tv_models.resnet50(weights=None)
        else:
            raise ValueError(f"알 수 없는 MODEL_NAME: {MODEL_NAME}")
        model.fc = nn.Linear(model.fc.in_features, n_class)
        print(f"[설정] {model_tag}  |  data={DATA_NAME}  (정확도 기준 모델 — "
              f"레이어별 특징 차원이 달라 블록 분석 생략)")

    model.to(device)
    train(model, trainloader, testloader, device,
          epochs=100, es=20, lpth=ckpt_name, lr=5e-5)

    if not is_ds_resnet:
        # 표준 ResNet-18/50은 레이어 진행에 따라 채널 수가 달라지므로
        # (64->128->256->512, Clipper의 차원-보존 특성이 없음) 기존 Exprob 기반
        # 블록간 Minkowski 거리/shadowing 분석이 직접 성립하지 않는다.
        # 따라서 두 모델은 정확도 baseline 으로만 사용하고 블록별 분석은 생략한다.
        sys.exit(0)

    # ── (이하 DS-ResNet 전용) 블록 단위 특징 추출 / 거리 분석 파이프라인 ──────
    extractor = Exprob(2048 * 7 * 14, n_class, layers=ds_layers,
                       multi_fc=USE_BLOCK_FC, use_avgpool=USE_AVGPOOL)
    extractor.load_state_dict(torch.load(f"{ckpt_name}.pt"), strict=False)
    extractor.to(device)
    extractor.eval()

    block_acc = {}
    with torch.no_grad():
        for xx, _ in testloader:
            out = extractor(xx.to(device))        # {block_idx: tensor}
            for b_idx, feat in out.items():
                feat_cpu = feat.detach().cpu()
                block_acc[b_idx] = (
                    feat_cpu if b_idx not in block_acc
                    else torch.cat((block_acc[b_idx], feat_cpu), 0)
                )

    feat_dir = f"prob/{DATA_NAME}/{model_tag}"
    os.makedirs(feat_dir, exist_ok=True)
    for b_idx, feat in block_acc.items():
        torch.save(feat, f"{feat_dir}/{DATA_NAME}_block{b_idx}.pt")

    # ── 블록별 fc 출력 저장 (USE_BLOCK_FC=True) ───────────────────────────
    if USE_BLOCK_FC:
        print("블록별 fc 학습 중...")
        train_block_fc(extractor, trainloader, device, epochs=5)
        torch.save(extractor.state_dict(), f"{ckpt_name}_multifc.pt")

        extractor.eval()
        fc_acc = {}
        with torch.no_grad():
            for xx, _ in testloader:
                out = extractor(xx.to(device), use_block_fc=True)
                for b_idx, logit in out.items():
                    logit_cpu = logit.detach().cpu()
                    fc_acc[b_idx] = (
                        logit_cpu if b_idx not in fc_acc
                        else torch.cat((fc_acc[b_idx], logit_cpu), 0)
                    )

        fc_dir = f"prob_fc/{DATA_NAME}/{model_tag}"
        os.makedirs(fc_dir, exist_ok=True)
        for b_idx, logit in fc_acc.items():
            torch.save(logit, f"{fc_dir}/{DATA_NAME}_block{b_idx}.pt")
        print(f"fc 출력 저장 완료: {fc_dir}/")

    # ── 라벨 및 청크 저장 ─────────────────────────────────────────────────
    extractor.load_state_dict(torch.load(f"{ckpt_name}.pt"), strict=False)
    extractor.to(device)
    extractor.eval()
    yout = None
    for i, (x, y) in enumerate(testloader):
        out  = extractor(x.to(device))
        yout = y if i == 0 else torch.cat((yout, y), 0)
        pix_dir = f"pix/resnet/{DATA_NAME}/{model_tag}/test"
        os.makedirs(pix_dir, exist_ok=True)
        for key, feat in out.items():
            torch.save(feat.cpu(), f"{pix_dir}/{DATA_NAME}_block{key}_{i}.pt")
    torch.save(yout, f"pix/resnet/{DATA_NAME}/{model_tag}/test/{DATA_NAME}_label.pt")
