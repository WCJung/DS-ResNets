"""
DS 블록 변형 — 모두 phi_t: R^{C×H×W} → R^{C×H×W} (차원 보존).

프레임워크가 요구하는 것은 블록의 입력/출력 차원 동일성뿐이고 블록 내부
구조는 자유이므로, 대표적인 두 가지 변형을 제공한다:

  WideBottleneck    — Wide ResNet (Zagoruyko & Komodakis 2016) 아이디어:
                      내부 병목 폭을 widen_factor배로 넓히고 dropout을 넣는다.
                      pre-activation 구조라 forward가 문자 그대로
                      x + F(x) — 논문의 phi(x) = x + F(x) 표기와 정확히 일치
                      (마지막 ReLU가 identity 경로를 건드리지 않는다).

  ResNeXtBottleneck — ResNeXt (Xie et al. 2017) 아이디어: grouped 3x3 conv로
                      cardinality개의 병렬 변환을 집계(aggregated transforms).
                      병목 폭은 ResNeXt 규약대로 기존 Bottleneck의 2배.

두 블록 모두 기존 Bottleneck과 동일한 생성자 시그니처와 expansion=4를
유지하므로 models.ResNets.ResNet 골격(_make_layer, Clipper 채널 흐름:
256 → 512 → 1024 → 2048)에 그대로 꽂힌다. 기존 Bottleneck과 마찬가지로
입력 채널은 planes * expansion으로 고정되어 있다 (차원 보존 설계).

하이퍼파라미터는 클래스 속성으로 조정한다:
  WideBottleneck.widen_factor (기본 2), WideBottleneck.dropout_p (기본 0.3)
  ResNeXtBottleneck.cardinality (기본 32)
"""
from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor

from models.ResNets import conv1x1, conv3x3


class WideBottleneck(nn.Module):
    """Pre-activation wide bottleneck: R^{C×H×W} → R^{C×H×W}.

    BN-ReLU-conv1x1(C→w) → BN-ReLU-conv3x3(w→w) → dropout
    → BN-ReLU-conv1x1(w→C),  w = planes × widen_factor
    (기존 Bottleneck의 w = planes 대비 widen_factor배 넓음)
    """

    expansion: int = 4
    widen_factor: int = 2
    dropout_p: float = 0.3

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        channels = planes * self.expansion          # 블록이 사는 공간의 채널 수
        width = planes * self.widen_factor

        self.bn1 = norm_layer(channels)
        self.conv1 = conv1x1(channels, width)
        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=1, groups=groups,
                             dilation=dilation)
        self.drop = nn.Dropout(self.dropout_p)
        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.drop(out)
        out = self.conv3(self.relu(self.bn3(out)))
        return x + out                              # phi(x) = x + F(x)


class ResNeXtBottleneck(nn.Module):
    """ResNeXt-style aggregated bottleneck: R^{C×H×W} → R^{C×H×W}.

    conv1x1(C→w) → grouped conv3x3(w→w, groups=cardinality) → conv1x1(w→C),
    w = planes × 2 (ResNeXt 규약: 병목 폭 2배).
    w ∈ {128, 256, 512, 1024}는 모두 cardinality=32로 나누어떨어진다.
    """

    expansion: int = 4
    cardinality: int = 32

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        channels = planes * self.expansion
        width = planes * 2
        if width % self.cardinality != 0:
            raise ValueError(
                f"bottleneck width {width}가 cardinality "
                f"{self.cardinality}로 나누어떨어지지 않습니다.")

        self.conv1 = conv1x1(channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=1, groups=self.cardinality,
                             dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, channels)
        self.bn3 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)
