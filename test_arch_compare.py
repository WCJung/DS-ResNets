"""
ResNet-18, ResNet-50, DS-ResNet 구조 및 파라미터 비교.

GPU 없이도 실행 가능 (forward pass만 확인).

실행: python test_arch_compare.py
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.ResNets import ResNet, Bottleneck


# ── 모델 생성 ────────────────────────────────────────────────────────────────

def make_resnet18(n_class=10):
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(512, n_class)
    return m


def make_resnet50(n_class=10):
    m = tv_models.resnet50(weights=None)
    m.fc = nn.Linear(2048, n_class)
    return m


def make_ds_resnet(n_class=10, layers=None, use_avgpool=False, use_50176=False):
    if layers is None:
        layers = [3, 4, 6, 3]
    m = ResNet(block=Bottleneck, layers=layers, num_classes=n_class,
               use_avgpool=use_avgpool, use_50176=use_50176)
    return m


# ── 분석 함수 ─────────────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def analyze_model(name, model, input_tensor):
    total, trainable = count_params(model)
    model.eval()
    with torch.no_grad():
        try:
            out = model(input_tensor)
            out_shape = tuple(out.shape)
            status = "OK"
        except Exception as e:
            out_shape = None
            status = f"ERROR: {e}"

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  총 파라미터       : {total:>12,} 개")
    print(f"  학습 파라미터     : {trainable:>12,} 개")
    print(f"  fc 파라미터 비중  : "
          f"{sum(p.numel() for p in model.fc.parameters()):>12,} 개")
    print(f"  forward 출력 shape: {out_shape}")
    print(f"  상태              : {status}")


def layer_channel_trace(name, model, x):
    """각 layer 이후 채널/공간 크기 추적 (DS-ResNet 전용)."""
    print(f"\n  [{name}] 내부 텐서 흐름:")
    model.eval()
    with torch.no_grad():
        x = model.conv1(x);   print(f"    conv1+bn1+relu : {tuple(x.shape)}")
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x); print(f"    maxpool        : {tuple(x.shape)}")
        x = model.clip1(x);   print(f"    clip1(2,2)     : {tuple(x.shape)}")
        x = model.layer1(x);  print(f"    layer1         : {tuple(x.shape)}")
        x = model.clip2(x);   print(f"    clip2(2,1)     : {tuple(x.shape)}")
        x = model.layer2(x);  print(f"    layer2         : {tuple(x.shape)}")
        x = model.clip3(x);   print(f"    clip3(1,2)     : {tuple(x.shape)}")
        x = model.layer3(x);  print(f"    layer3         : {tuple(x.shape)}")
        x = model.clip2(x);   print(f"    clip2(2,1)     : {tuple(x.shape)}")
        x = model.layer4(x);  print(f"    layer4         : {tuple(x.shape)}")
        flat = torch.flatten(x, 1)
        print(f"    flatten        : {tuple(flat.shape)}")
        print(f"    ※ 총 특징 원소: {flat.shape[1]:,}  "
              f"(= {x.shape[1]}ch × {x.shape[2]} × {x.shape[3]})")


def clipper_isometry_check(model, x):
    """Clipper가 좌표 재배열(l2-등거리 변환)임을 수치로 확인.

    블록 특징이 전부 같은 R^n에 산다는 프레임워크의 전제 근거."""
    a = torch.randn_like(x)
    b = torch.randn_like(x)
    d_before = torch.norm((a - b).flatten(1), dim=1)
    d_after  = torch.norm((model.clip1(a) - model.clip1(b)).flatten(1), dim=1)
    gap = (d_before - d_after).abs().max().item()
    print(f"\n  [Clipper 등거리성] max |d_before - d_after| = {gap:.2e}"
          f"  ({'OK' if gap < 1e-4 else 'FAIL'})")


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dummy3 = torch.zeros(2, 3, 224, 224)

    L8  = [2, 2, 2, 2]   # 8 blocks  — ResNet-18 블록 수와 동일
    L16 = [3, 4, 6, 3]   # 16 blocks — ResNet-50 블록 수와 동일

    models_info = [
        ("ResNet-18",              make_resnet18(),                              dummy3),
        ("ResNet-50",              make_resnet50(),                              dummy3),
        ("DS-ResNet-18(no pool)",  make_ds_resnet(layers=L8,  use_avgpool=False), dummy3),
        ("DS-ResNet-18(avgpool)",  make_ds_resnet(layers=L8,  use_avgpool=True),  dummy3),
        ("DS-ResNet-50(no pool)",  make_ds_resnet(layers=L16, use_avgpool=False), dummy3),
        ("DS-ResNet-50(avgpool)",  make_ds_resnet(layers=L16, use_avgpool=True),  dummy3),
    ]

    print("\n[구조 비교] 입력: 3ch × 224 × 224")

    for name, model, dummy in models_info:
        analyze_model(name, model, dummy)

    # DS-ResNet-50 내부 흐름 상세 출력 (Clipper 구조 확인)
    _, ds16_model, _ = models_info[4]
    layer_channel_trace("DS-ResNet", ds16_model, dummy3)
    clipper_isometry_check(ds16_model, torch.zeros(2, 64, 56, 56))

    # ── 요약 테이블 ───────────────────────────────────────────────────────────
    print("\n\n[요약 비교]")
    specs = [
        ("ResNet-18",             8,  make_resnet18(),                              512,    "O", "기준",   "R^{512}"),
        ("ResNet-50",             16, make_resnet50(),                              2048,   "O", "기준",   "R^{2048}"),
        ("DS-ResNet-18(no pool)", 8,  make_ds_resnet(layers=L8,  use_avgpool=False), 200704, "X", "vs R18", "R^{200704}"),
        ("DS-ResNet-18(avgpool)", 8,  make_ds_resnet(layers=L8,  use_avgpool=True),  2048,   "O", "vs R18", "R^{2048}"),
        ("DS-ResNet-50(no pool)", 16, make_ds_resnet(layers=L16, use_avgpool=False), 200704, "X", "vs R50", "R^{200704}"),
        ("DS-ResNet-50(avgpool)", 16, make_ds_resnet(layers=L16, use_avgpool=True),  2048,   "O", "vs R50", "R^{2048}"),
    ]
    print(f"{'모델':<26} {'블록':>5} {'파라미터':>12} {'fc 입력':>10} {'pool':>5} {'비교':>8}  X 공간")
    print("-" * 85)
    for name, n_blk, m, fc_dim, pool, target, x_space in specs:
        total, _ = count_params(m)
        print(f"{name:<26} {n_blk:>5} {total:>12,} {fc_dim:>10,} {pool:>5} {target:>8}  {x_space}")

    print("\n[Lipschitz 상수]")
    print("  블록 fc(분석용 g): Lip = sigma_max(W_b)  — 블록 특징 200,704 직결")
    print("  main fc(avgpool) : Lip = sigma_max(W_fc) / sqrt(98)  (98 = 7 x 14 평균)")
    print("  main fc(no pool) : Lip = sigma_max(W_fc)")

    print("\n[결론]")
    print("  DS-ResNet-18 vs ResNet-18: 블록 8개 동일,  Clipper 유무 차이")
    print("  DS-ResNet-50 vs ResNet-50: 블록 16개 동일, Clipper 유무 차이")
    print("  ※ 실험 스코프(run_all.py): 4개 모델 × MNIST/CIFAR-10/IMAGENET10")
