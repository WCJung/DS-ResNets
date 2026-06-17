from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import sys
import torch
import torch.nn as nn
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet"
]


class Clipper:
    def __init__(self, h_split, w_split):
        super().__init__()
        self.h_split = h_split
        self.w_split = w_split

    def __call__(self, input_data):
        channel_amp = self.h_split * self.w_split
        (b, c, h, w) = input_data.shape
        h_size = int(h / self.h_split)
        w_size = int(w / self.w_split)
        # device=input_data.device: GPU 텐서로 직접 생성 (CPU ↔ GPU 전송 제거)
        new_shape = torch.zeros((b, c * channel_amp, h_size, w_size),
                                dtype=input_data.dtype, device=input_data.device)
        for h_cur in range(self.h_split):
            for w_cur in range(self.w_split):
                hc = h_size * h_cur
                wc = w_size * w_cur
                cc = int((((self.h_split * h_cur) + w_cur) * c)/max((self.h_split/self.w_split), 1))
                new_shape[:, cc: cc+c] = input_data[:, :, hc: hc+h_size, wc: wc+w_size]
        return new_shape


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion: int = 1

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
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

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
        width = int(planes * (base_width / 64.0)) * 1

        self.conv1 = conv1x1(planes * self.expansion, width, groups=groups)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, groups=groups)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_avgpool: bool = False,
        use_50176:   bool = False,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # use_50176=True  : layer4(2048,7,14) → dim_reducer(Conv2d 2048→512) → flatten 50,176
        #                   X = R^50176 = MNIST 입력공간 (1ch×224×224), 동력계 원형
        #                   Lip(g) = σ_max(W_fc) · Lip(dim_reducer)
        # use_avgpool=True : layer4 → AdaptiveAvgPool(1,1) → flatten 2,048
        #                   Lip(g) = σ_max(W_fc) · (1/√98), g가 수축적 → 하한 tight
        # (둘 다 False)    : spatial 그대로 flatten 200,704
        #                   Lip(g) = σ_max(W_fc),  X = R^200704
        self.use_avgpool = use_avgpool
        self.use_50176   = use_50176

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], groups=4)
        self.layer2 = self._make_layer(block, 128, layers[1], groups=8, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], groups=16, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], groups=32, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if use_50176:
            # 채널 압축 2048→512 (학습됨); 7×14 spatial은 보존 → 512×7×14 = 50,176
            self.dim_reducer = nn.Conv2d(512 * block.expansion, 512, 1, bias=False)
            fc_in_features   = 512 * 7 * 14                  # 50,176
        elif use_avgpool:
            fc_in_features = 512 * block.expansion          # 2,048
        else:
            fc_in_features = 512 * block.expansion * 7 * 14  # 200,704
        self.fc = nn.Linear(fc_in_features, num_classes)

        self.clip1 = Clipper(2, 2)
        self.clip2 = Clipper(2, 1)
        self.clip3 = Clipper(1, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        groups: int = None,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
          
        if groups is None:
            groups = self.groups

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.clip1(x).to(DEVICE)
        x = self.layer1(x)
        x = self.clip2(x).to(DEVICE)
        x = self.layer2(x)
        x = self.clip3(x).to(DEVICE)
        x = self.layer3(x)
        x = self.clip2(x).to(DEVICE)
        x = self.layer4(x)
        if self.use_50176:
            x = self.dim_reducer(x)  # (B, 2048, 7, 14) → (B, 512, 7, 14)
        elif self.use_avgpool:
            x = self.avgpool(x)      # (B, 2048, 7, 14) → (B, 2048, 1, 1)
        x = torch.flatten(x, 1)      # (B, 50176) or (B, 2048) or (B, 200704)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}

