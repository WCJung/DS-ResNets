"""
DSResNetX — Dimension-preserving DS-ResNet for X = R^{input_dim} framework.

Dynamical systems structure:
    X = R^{in_channels × in_H × in_W}   (input space, preserved throughout)
    phi: X -> X                           (DimPreservingBlock)
    g:   X -> R^{n_class}                 (Linear)
    Full system: g o phi^{n_blocks}

Clipper (no learned parameters): lossless channel-space reshape.
    Clipper(2,2) x n_clipper: channels x 4^k, H // 2^k, W // 2^k
    Total elements invariant => phi maps within the same X.
"""
import torch
import torch.nn as nn
from torch import Tensor
from models.ResNets import Clipper


class DimPreservingBlock(nn.Module):
    """Bottleneck residual block: R^{C x H x W} -> R^{C x H x W}."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = max(channels // 4, 1)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class DSResNetX(nn.Module):
    """
    Dimension-preserving DS-ResNet.

    Parameters
    ----------
    in_channels : input channel count  (1 = MNIST grayscale, 3 = CIFAR-10)
    in_H, in_W  : input spatial size   (e.g. 224 x 224)
    n_class     : number of output classes
    n_blocks    : residual block count (phi repetition: g o phi^n_blocks)
    n_clipper   : Clipper(2,2) pre-reshapes before blocks (default 3)
    multi_fc    : attach one Linear per block for block-level distance analysis
    """

    def __init__(
        self,
        in_channels: int,
        in_H: int,
        in_W: int,
        n_class: int,
        n_blocks: int,
        n_clipper: int = 3,
        multi_fc: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = in_channels * in_H * in_W

        # After n_clipper Clipper(2,2): channels x 4^k, H // 2^k, W // 2^k
        block_C = in_channels * (4 ** n_clipper)
        block_H = in_H // (2 ** n_clipper)
        block_W = in_W // (2 ** n_clipper)
        assert block_C * block_H * block_W == self.input_dim, (
            f"Clipper reshape mismatch: {block_C}x{block_H}x{block_W} != {self.input_dim}"
        )

        self.n_clipper = n_clipper
        self.block_C   = block_C
        # Clipper has no nn.Parameters — plain list, not ModuleList
        self._clippers = [Clipper(2, 2) for _ in range(n_clipper)]

        self.blocks   = nn.ModuleList([DimPreservingBlock(block_C) for _ in range(n_blocks)])
        self.multi_fc = multi_fc
        if multi_fc:
            self.block_fc = nn.ModuleList(
                [nn.Linear(self.input_dim, n_class) for _ in range(n_blocks)]
            )
        self.fc = nn.Linear(self.input_dim, n_class)

        self._block_feats: dict = {}
        self._hooks: list       = []
        self._register_hooks()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _register_hooks(self) -> None:
        for idx, block in enumerate(self.blocks):
            h = block.register_forward_hook(
                lambda m, inp, out, i=idx: self._block_feats.__setitem__(i, out.flatten(1))
            )
            self._hooks.append(h)

    def forward(self, x: Tensor, use_block_fc: bool = False):
        """
        Returns
        -------
        default             : (B, n_class) logits           [training / eval]
        use_block_fc=True   : {block_idx: (B, n_class)}     [block-level fc analysis]

        Raw block features are always captured via hooks.
        Access them with get_block_features() after any forward() call.
        """
        self._block_feats = {}
        for clipper in self._clippers:
            x = clipper(x)
        for block in self.blocks:
            x = block(x)
        if use_block_fc and self.multi_fc:
            return {i: self.block_fc[i](feat) for i, feat in self._block_feats.items()}
        return self.fc(torch.flatten(x, 1))

    def get_block_features(self) -> dict:
        """Return {block_idx: (B, input_dim)} from the last forward() call."""
        return dict(self._block_feats)
