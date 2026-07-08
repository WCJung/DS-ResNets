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


def ds_block(name):
    return DS_MODELS[name][0]


def ds_layers(name):
    return DS_MODELS[name][1]


def build_ds_model(name, n_class, use_avgpool=False, use_50176=False):
    """차원 보존(DS) 백본 생성 — 이름으로 블록/레이어 구성을 선택."""
    block, layers = DS_MODELS[name]
    return resnet.ResNet(block=block, layers=layers, num_classes=n_class,
                         use_avgpool=use_avgpool, use_50176=use_50176)
