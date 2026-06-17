"""
ResNet-18, ResNet-50, DS-ResNet 구조 및 파라미터 비교.

GPU 없이도 실행 가능 (forward pass만 확인).
실제 학습 비교는 A100에서 진행.

실행: python test_arch_compare.py
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from models.ResNets import ResNet, Bottleneck
from models.ResNets_X import DSResNetX


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


def make_dsresnetx(in_channels=1, n_class=10, n_blocks=16, n_clipper=3):
    """선택 1: X = R^{in_channels x 224 x 224} (차원 보존 동력계)."""
    return DSResNetX(in_channels, 224, 224, n_class, n_blocks, n_clipper)


# ── fc 입력을 50,176으로 통일한 버전 (X 공간 비교 공정성 확보) ──────────────
#
# 목표 차원: 50,176 = 1ch x 224 x 224 (MNIST 입력 차원)
#   ResNet-18 : layer4 (512, 7, 7)   -avgpool 제거-> Conv2d(512->1024,1) -> 1024x7x7  = 50,176
#   ResNet-50 : layer4 (2048, 7, 7)  -avgpool 제거-> Conv2d(2048->1024,1)-> 1024x7x7  = 50,176
#   DS-ResNet : layer4 (2048, 7, 14) -dim_reducer-> Conv2d(2048->512,1) -> 512x7x14  = 50,176
#   DSResNetX : 설계상 이미 50,176 (변경 없음)
#
# 주의: g(분류 헤드)가 Linear 한 층이 아니라 [Conv2d 채널압축 + Linear] 합성이 됨.
#       Lip(g) = sigma_max(W_fc) * Lip(conv1x1).  DSResNetX만 g가 순수 Linear.

def make_resnet18_50176(n_class=10):
    m = tv_models.resnet18(weights=None)
    m.avgpool = nn.Identity()            # (B, 512, 7, 7) 유지 (공간정보 보존)
    m.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),    # (B, 25088) -> (B, 512, 7, 7)
        nn.Conv2d(512, 1024, 1),         # -> (B, 1024, 7, 7)
        nn.Flatten(),                    # -> (B, 50176)
        nn.Linear(50176, n_class),
    )
    return m


def make_resnet50_50176(n_class=10):
    m = tv_models.resnet50(weights=None)
    m.avgpool = nn.Identity()            # (B, 2048, 7, 7) 유지
    m.fc = nn.Sequential(
        nn.Unflatten(1, (2048, 7, 7)),   # (B, 100352) -> (B, 2048, 7, 7)
        nn.Conv2d(2048, 1024, 1),        # -> (B, 1024, 7, 7)
        nn.Flatten(),                    # -> (B, 50176)
        nn.Linear(50176, n_class),
    )
    return m


def make_ds_resnet_50176(n_class=10, layers=None):
    """ResNets.py의 use_50176=True 내장 옵션 사용 (dim_reducer: 2048->512)."""
    return make_ds_resnet(n_class=n_class, layers=layers, use_50176=True)


# ── 분석 함수 ─────────────────────────────────────────────────────────────────

def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
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
    if name != "DS-ResNet":
        return
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


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 3ch 입력 (기존 DS-ResNet / ResNet)
    dummy3 = torch.zeros(2, 3, 224, 224)
    # 1ch 입력 (DSResNetX — MNIST 원본 채널)
    dummy1 = torch.zeros(2, 1, 224, 224)

    L8  = [2, 2, 2, 2]   # 8 blocks  — ResNet-18 블록 수와 동일
    L16 = [3, 4, 6, 3]   # 16 blocks — ResNet-50 블록 수와 동일

    models_info = [
        ("ResNet-18",              make_resnet18(),                           dummy3),
        ("ResNet-50",              make_resnet50(),                           dummy3),
        ("DS-ResNet-18(no pool)",  make_ds_resnet(layers=L8, use_avgpool=False), dummy3),
        ("DS-ResNet-18(avgpool)",  make_ds_resnet(layers=L8, use_avgpool=True),  dummy3),
        ("DS-ResNet-50(no pool)",  make_ds_resnet(layers=L16,use_avgpool=False), dummy3),
        ("DS-ResNet-50(avgpool)",  make_ds_resnet(layers=L16,use_avgpool=True),  dummy3),
        # 선택 1: X = R^{input_dim} (1ch MNIST, 224x224)
        ("DSResNetX-8  (1ch)",     make_dsresnetx(in_channels=1, n_blocks=8),  dummy1),
        ("DSResNetX-16 (1ch)",     make_dsresnetx(in_channels=1, n_blocks=16), dummy1),
        # fc 입력 50,176 통일 버전
        ("ResNet-18 (fc50176)",    make_resnet18_50176(),                       dummy3),
        ("ResNet-50 (fc50176)",    make_resnet50_50176(),                       dummy3),
        ("DS-ResNet-16(fc50176)",  make_ds_resnet_50176(layers=L16),            dummy3),
    ]

    print("\n[구조 비교] 입력: 3ch=224x224 (DS-ResNet), 1ch=224x224 (DSResNetX)")

    for name, model, dummy in models_info:
        analyze_model(name, model, dummy)

    # DS-ResNet-16 내부 흐름 상세 출력 (Clipper 구조 확인)
    _, ds16_model, _ = models_info[4]
    layer_channel_trace("DS-ResNet", ds16_model, dummy3)

    # ── 요약 테이블 ───────────────────────────────────────────────────────────
    print("\n\n[요약 비교]")
    x8_1ch  = make_dsresnetx(in_channels=1, n_blocks=8)
    x16_1ch = make_dsresnetx(in_channels=1, n_blocks=16)
    specs = [
        ("ResNet-18",             8,  make_resnet18(),                             512,    "O", "기준",    "R^{512}"),
        ("ResNet-50",             16, make_resnet50(),                             2048,   "O", "기준",    "R^{2048}"),
        ("DS-ResNet-18(no pool)", 8,  make_ds_resnet(layers=L8, use_avgpool=False),200704, "X", "vs R18",  "R^{200704}"),
        ("DS-ResNet-18(avgpool)", 8,  make_ds_resnet(layers=L8, use_avgpool=True), 2048,   "O", "vs R18",  "R^{2048}"),
        ("DS-ResNet-50(no pool)", 16, make_ds_resnet(layers=L16,use_avgpool=False),200704, "X", "vs R50",  "R^{200704}"),
        ("DS-ResNet-50(avgpool)", 16, make_ds_resnet(layers=L16,use_avgpool=True), 2048,   "O", "vs R50",  "R^{2048}"),
        ("DSResNetX-8  (1ch)",    8,  x8_1ch,  x8_1ch.input_dim,                  "-", "[선택1]", "R^{50176}"),
        ("DSResNetX-16 (1ch)",    16, x16_1ch, x16_1ch.input_dim,                 "-", "[선택1]", "R^{50176}"),
    ]
    print(f"{'모델':<26} {'블록':>5} {'파라미터':>12} {'fc 입력':>10} {'pool':>5} {'비교':>8}  X 공간")
    print("-" * 85)
    for name, n_blk, m, fc_dim, pool, target, x_space in specs:
        total, _ = count_params(m)
        print(f"{name:<26} {n_blk:>5} {total:>12,} {fc_dim:>10,} {pool:>5} {target:>8}  {x_space}")

    print("\n[Lipschitz 상수 (원본)]")
    print("  avgpool: Lip(g) = sigma_max(W_fc) / sqrt(98)  (fc 2048, 하한 tight)")
    print("  no pool: Lip(g) = sigma_max(W_fc)             (fc 200704)")
    print("  DSResNetX: Lip(g) = sigma_max(W_fc)           (fc = input_dim)")

    # ── fc 입력 50,176 통일 비교 ────────────────────────────────────────────
    print("\n\n[fc 입력 50,176 통일 비교] - 모든 모델이 동일한 g: R^50176 -> R^10")
    specs_u = [
        ("ResNet-18 (fc50176)",   8,  make_resnet18_50176(),          "Conv2d(512->1024,1)+Linear"),
        ("ResNet-50 (fc50176)",   16, make_resnet50_50176(),          "Conv2d(2048->1024,1)+Linear"),
        ("DS-ResNet-50(fc50176)", 16, make_ds_resnet_50176(layers=L16), "Conv2d(2048->512,1)+Linear"),
        ("DSResNetX-16 (1ch)",    16, x16_1ch,                        "Linear (단일층, 압축 없음)"),
    ]
    print(f"{'모델':<26} {'블록':>5} {'파라미터':>12} {'fc 입력':>10}  g 구조")
    print("-" * 90)
    for name, n_blk, m, g_struct in specs_u:
        total, _ = count_params(m)
        print(f"{name:<26} {n_blk:>5} {total:>12,} {50176:>10,}  {g_struct}")

    print("\n[Lipschitz 상수 (50,176 통일 버전)]")
    print("  ResNet-18/50, DS-ResNet : Lip(g) = sigma_max(W_fc) * sigma_max(W_conv1x1)")
    print("                            (Conv2d 채널압축이 g에 포함 -> 2단 합성)")
    print("  DSResNetX               : Lip(g) = sigma_max(W_fc)")
    print("                            (g가 순수 Linear 한 층 -> 가장 단순하고 tight)")

    print("\n[동력계 X 공간 비교]")
    for in_ch, tag in [(1, "MNIST 1ch")]:
        m = make_dsresnetx(in_channels=in_ch, n_blocks=16)
        print(f"  DSResNetX ({tag}): "
              f"input_dim={m.input_dim:,}, block_C={m.block_C}, "
              f"block spatial=28x28, phi: R^{{{m.input_dim}}} -> R^{{{m.input_dim}}}")

    print("\n[결론]")
    print("  DS-ResNet-18 vs ResNet-18: 블록 8개 동일,  Clipper 유무 차이")
    print("  DS-ResNet-50 vs ResNet-50: 블록 16개 동일, Clipper 유무 차이")
    print("  DSResNetX-{8,16}: 선택 1, X = R^{input_dim}, phi: X->X 보장")
    print("\n  ※ 최종 실험 스코프(main.py): ResNet-18/50, DS-ResNet-18/50 (4개 모델)")
    print("    x MNIST/CIFAR-10/IMAGENET10(Imagenette) (3개 데이터셋)")
