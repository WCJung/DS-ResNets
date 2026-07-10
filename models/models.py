"""
DS 모델 레지스트리 — 모델 이름 → (블록 클래스, layer 구성)의 단일 소스.

모든 스크립트(main / dist_calc / entropy_calc / run_all / inspect / print)가
이 레지스트리를 import해서 사용한다.

layer 구성:
  [2, 2, 2, 2] →  8 블록 (ResNet-18과 블록 수 동일)
  [3, 4, 6, 3] → 16 블록 (ResNet-50과 블록 수 동일)
"""
from models import ResNets as resnet
from models.blocks import ResNeXtBottleneck, WideBottleneck

DS_MODELS = {
    # name           : (block class,        layers)
    'ds_resnet18':     (resnet.Bottleneck,  [2, 2, 2, 2]),
    'ds_resnet50':     (resnet.Bottleneck,  [3, 4, 6, 3]),
    'ds_wide18':       (WideBottleneck,     [2, 2, 2, 2]),
    'ds_wide50':       (WideBottleneck,     [3, 4, 6, 3]),
    'ds_resnext18':    (ResNeXtBottleneck,  [2, 2, 2, 2]),
    'ds_resnext50':    (ResNeXtBottleneck,  [3, 4, 6, 3]),
}


def resolve_model_name(name):
    """유연한 모델명 → 레지스트리 정식 이름.

    대소문자·하이픈·공백, 'ds_' 접두사 생략, WRN/WideResNet 표기를 허용:
      'RESNET18' / 'ds-resnet18'  → 'ds_resnet18'
      'WRN50' / 'WideResNet-50'   → 'ds_wide50'
      'ResNeXt18'                 → 'ds_resnext18'

    레지스트리에 없는 구성(예: wrn16, wrn28 — 깊이는 *18/*50만 존재)이면
    사용 가능한 이름 목록을 담아 ValueError를 던진다.
    """
    compact = {k.replace('_', ''): k for k in DS_MODELS}   # 'dsresnet18' → 정식명
    key = name.strip().lower()
    for sep in ('-', '_', ' '):
        key = key.replace(sep, '')
    if key in compact:
        return compact[key]
    base = key[2:] if key.startswith('ds') else key
    base = base.replace('wideresnet', 'wide').replace('wrn', 'wide')
    if f"ds{base}" in compact:
        return compact[f"ds{base}"]
    raise ValueError(
        f"알 수 없는 모델 '{name}'. 사용 가능: {', '.join(DS_MODELS)} "
        f"(깊이는 *18=8블록 / *50=16블록만 존재 — wrn16/28 같은 다른 깊이 없음. "
        f"표준 resnet18/50 baseline은 블록 분석 대상이 아님)")


def ds_block(name):
    return DS_MODELS[name][0]


def ds_layers(name):
    return DS_MODELS[name][1]


def build_ds_model(name, n_class, use_avgpool=False, use_50176=False):
    """차원 보존(DS) 백본 생성 — 이름으로 블록/레이어 구성을 선택."""
    block, layers = DS_MODELS[name]
    return resnet.ResNet(block=block, layers=layers, num_classes=n_class,
                         use_avgpool=use_avgpool, use_50176=use_50176)
