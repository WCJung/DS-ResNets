"""
IsoLift-ResNeXt — Common-State Geometry-Controlled Residual Network.

서로 다른 입력 차원의 데이터셋(MNIST / CIFAR10 / IMAGENET10)을 하나의 공통
상태 공간 R^{48x56x56} (= 150,528 = 3*224*224) 으로 "정확한 등거리 lifting"
한 뒤, 공유 residual backbone 으로 처리하는 멀티 도메인 아키텍처:

    x_d --E_d--> z_0 in R^{48x56x56} --A_d--> u_0 --T_theta--> u_L --H_d--> y_d

  E_d : 데이터셋별 고정(비학습) 등거리 lifting — ||E_d(x)-E_d(x')|| = ||x-x'||
          MNIST      1x28x28  --(중앙 zero-embed)--> 1x56x56 --(정규직교 열 1x1)--> 48x56x56
          CIFAR10    3x32x32  --(중앙 zero-pad)----> 3x56x56 --(semi-orthogonal 1x1)--> 48x56x56
          IMAGENET10 3x224x224 --PixelUnshuffle(4)--> 48x56x56   (좌표 순열)
  A_d : 데이터셋별 소규모 residual adapter  A_d(z) = z + beta_d * G_d(z)
        (performance 모드 전용 — provable 모드에서는 항등)
  T   : 공유 ResNeXt-병목 residual 동역학  u <- u + alpha * F(u),
        F: c -> c/3 -> (3x3 grouped) -> c.  stage 전이는 PixelUnshuffle(2)
        (정확한 좌표 순열 = 등거리)이며 전체 유클리드 차원 150,528 유지:
            48*56^2 = 192*28^2 = 768*14^2 = 150,528
  H_d : 데이터셋별 classification head (GAP -> Linear)

mode:
  'provable'    — residual branch 의 모든 conv 에 하드 스펙트럴 제약
                  (i-ResNet 방식, Behrmann et al. 2019: W <- W*min(1, c/sigma))
                  을 걸어 Lip(alpha*F) <= rho < 1 을 보장 -> 각 블록이
                  가역(bi-Lipschitz). 정규화층·adapter 없음.
  'performance' — 제약 없는 conv + 도메인별 BatchNorm(Chang et al. 2019)
                  + adapter. Lipschitz 는 utils/isolift_losses 의
                  soft spectral penalty 로만 유도.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COMMON_DIM = 3 * 224 * 224                       # 150,528
STAGE_CHANNELS = (48, 192, 768)                  # 48*56^2 = 192*28^2 = 768*14^2


def _orthonormal_cols(rows, cols, seed):
    """열이 정규직교(Q^T Q = I_cols)인 (rows, cols) 행렬 — 고정 seed 로 재현.

    float64 로 생성·보관해 등거리성이 기계 정밀도로 성립하게 한다
    (forward 에서 입력 dtype 으로 캐스팅).
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)                    # reduced QR: (rows, cols)
    return q.contiguous()


# ── E_d : 데이터셋별 정확한 등거리 lifting (비학습) ───────────────────────────

class MNISTLift(nn.Module):
    """1x28x28 -> 48x56x56.  중앙 zero-embed 후 단위 노름 열로 채널 lift."""

    IN_SHAPE = (1, 28, 28)

    def __init__(self):
        super().__init__()
        # 1 -> 48 채널: 단위 노름 열 벡터 (W^T W = I_1) -> 픽셀별 등거리
        w = torch.full((48, 1, 1, 1), 1.0, dtype=torch.float64)
        self.register_buffer("weight", w / w.norm())

    def forward(self, x):
        x = F.pad(x, (14, 14, 14, 14))           # 28 -> 56 (중앙, 나머지 0)
        return F.conv2d(x, self.weight.to(x.dtype))


class CIFARLift(nn.Module):
    """3x32x32 -> 48x56x56.  중앙 zero-pad 후 semi-orthogonal 1x1 conv."""

    IN_SHAPE = (3, 32, 32)

    def __init__(self, seed=13):
        super().__init__()
        w = _orthonormal_cols(48, 3, seed).view(48, 3, 1, 1)   # W^T W = I_3
        self.register_buffer("weight", w)

    def forward(self, x):
        x = F.pad(x, (12, 12, 12, 12))           # 32 -> 56
        return F.conv2d(x, self.weight.to(x.dtype))


class ImageNetLift(nn.Module):
    """3x224x224 -> 48x56x56.  PixelUnshuffle(4) — 전단사 좌표 순열."""

    IN_SHAPE = (3, 224, 224)

    def __init__(self):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x):
        return self.unshuffle(x)


LIFTS = {"MNIST": MNISTLift, "CIFAR10": CIFARLift,
         "IMAGENET10": ImageNetLift, "IMAGENET1K": ImageNetLift}

# 도메인별 클래스 수 — head H_d 의 출력 차원 (IsoLiftNet 기본값)
DOMAIN_CLASSES = {"MNIST": 10, "CIFAR10": 10,
                  "IMAGENET10": 10, "IMAGENET1K": 1000}


# ── 스펙트럴 제약 conv (provable 모드) ────────────────────────────────────────

class SNConv2d(nn.Conv2d):
    """하드 스펙트럴 제약 conv — i-ResNet 스타일 W <- W * min(1, coef/sigma).

    sigma 는 reshape 행렬 (out, in/groups*kh*kw) 의 최대 특이값을 power
    iteration 으로 추정. conv "연산자" 노름은 이 행렬 노름의 sqrt(kh*kw) 배
    까지 클 수 있으므로 (Tsuzuku et al. 2018) coef 를 sqrt(kh*kw) 로 나눠
    보수적으로 보장한다. grouped conv 는 블록대각 연산자라 stacked 행렬의
    sigma 가 참값 이상 -> 과대추정이므로 역시 안전한 방향이다.
    """

    def __init__(self, *args, coef=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        kh, kw = self.kernel_size
        self.coef = coef / math.sqrt(kh * kw)
        self.register_buffer("_u", torch.randn(self.out_channels))

    def _sigma(self):
        w = self.weight.flatten(1)                       # (out, in/g*kh*kw)
        with torch.no_grad():
            v = F.normalize(w.t() @ self._u, dim=0)
            u = F.normalize(w @ v, dim=0)
            self._u.copy_(u)
        # 버퍼가 아니라 로컬 u/v 를 참조해야 같은 conv 를 한 그래프에서
        # 여러 번(도메인별) 통과시켜도 in-place 버전 충돌이 없다.
        return torch.dot(u, w @ v)                       # W 에 미분 가능

    def forward(self, x):
        sigma = self._sigma()
        scale = torch.clamp(self.coef / (sigma + 1e-12), max=1.0)
        return self._conv_forward(x, self.weight * scale, self.bias)


# ── 도메인별 정규화 (performance 모드) ────────────────────────────────────────

class DomainNorm2d(nn.Module):
    """도메인별 BatchNorm2d — 통계·affine 을 도메인마다 분리.

    backbone 파라미터는 공유하되 정규화만 분리하는 domain-specific BN
    (Chang et al., CVPR 2019) 구성.
    """

    def __init__(self, channels, domains):
        super().__init__()
        self.norms = nn.ModuleDict({d: nn.BatchNorm2d(channels) for d in domains})

    def forward(self, x, domain):
        return self.norms[domain](x)


# ── A_d : 데이터셋별 residual adapter ────────────────────────────────────────

class DomainAdapter(nn.Module):
    """A_d(z) = z + beta_d * G_d(z),  G_d = 3x3 conv -> ReLU -> 3x3 conv."""

    def __init__(self, channels=48, beta=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, z):
        return z + self.beta * self.conv2(F.relu(self.conv1(z)))


# ── 공유 residual block (ResNeXt 병목) ───────────────────────────────────────

# 계열별 병목 구성: (폭 비율, 3x3 grouped 여부, pre-activation, dropout p)
#   resnet  — DS-ResNets baseline 대응: 폭 C/4, 일반 3x3, post-act
#   wide    — DS-Wide 대응: 폭 C/2 (2x), dropout 0.3, pre-activation
#   resnext — 코멘트 명세 그대로: 폭 C/3, 3x3 grouped (cardinality)
ISOLIFT_FAMILIES = {
    "resnet":  {"ratio": 4, "grouped": False, "preact": False, "dropout": 0.0},
    "wide":    {"ratio": 2, "grouped": False, "preact": True,  "dropout": 0.3},
    "resnext": {"ratio": 3, "grouped": True,  "preact": False, "dropout": 0.0},
}


class IsoLiftBlock(nn.Module):
    """u + alpha * F(u),  F: c -> c/ratio -> (3x3 [grouped]) -> c.

    family 로 branch 내부를 선택 (ISOLIFT_FAMILIES — DS 3계열과 평행):
      resnet / wide / resnext.
    provable    : conv = SNConv2d(coef=rho^{1/3}) 3개 -> Lip(F) <= rho < 1
                  (ReLU 는 1-Lipschitz), alpha = 1 고정 -> 블록 가역.
                  dropout 은 학습 중 1/(1-p) 스케일로 Lip 을 깨므로 제외.
    performance : 일반 conv + DomainNorm2d, alpha 학습 (soft penalty 대상).
                  wide 계열은 pre-activation(norm -> relu -> conv) 순서.
    """

    def __init__(self, channels, domains, cardinality=8,
                 mode="performance", rho=0.9, family="resnext"):
        super().__init__()
        cfg = ISOLIFT_FAMILIES[family]
        assert channels % cfg["ratio"] == 0
        width = channels // cfg["ratio"]
        groups = cardinality if cfg["grouped"] else 1
        assert width % groups == 0, \
            f"width {width} 가 cardinality {groups} 로 나누어떨어지지 않음"
        self.mode = mode
        self.family = family
        self.preact = cfg["preact"]

        if mode == "provable":
            coef = rho ** (1.0 / 3.0)
            self.conv1 = SNConv2d(channels, width, 1, coef=coef)
            self.conv2 = SNConv2d(width, width, 3, padding=1,
                                  groups=groups, coef=coef)
            self.conv3 = SNConv2d(width, channels, 1, coef=coef)
            self.norms = None
            self.drop = None
            self.register_buffer("alpha", torch.tensor(1.0))
        else:
            self.conv1 = nn.Conv2d(channels, width, 1)
            self.conv2 = nn.Conv2d(width, width, 3, padding=1, groups=groups)
            self.conv3 = nn.Conv2d(width, channels, 1)
            if self.preact:      # norm 은 conv "앞" 채널 크기
                dims = (channels, width, width)
            else:                # norm 은 conv "뒤" 채널 크기
                dims = (width, width, channels)
            self.norms = nn.ModuleList(
                [DomainNorm2d(c, domains) for c in dims])
            self.drop = (nn.Dropout(cfg["dropout"])
                         if cfg["dropout"] > 0 else None)
            self.alpha = nn.Parameter(torch.tensor(1.0))

    def branch_convs(self):
        return (self.conv1, self.conv2, self.conv3)

    def _post(self, u, domain):
        """conv -> norm -> relu 순서 (resnet / resnext)."""
        out = self.conv1(u)
        if self.norms is not None:
            out = self.norms[0](out, domain)
        out = F.relu(out)
        out = self.conv2(out)
        if self.norms is not None:
            out = self.norms[1](out, domain)
        out = F.relu(out)
        out = self.conv3(out)
        if self.norms is not None:
            out = self.norms[2](out, domain)
        return out

    def _pre(self, u, domain):
        """norm -> relu -> conv 순서 + dropout (wide, DS-Wide 와 동일 구조)."""
        out = u
        if self.norms is not None:
            out = self.norms[0](out, domain)
        out = self.conv1(F.relu(out))
        if self.norms is not None:
            out = self.norms[1](out, domain)
        out = self.conv2(F.relu(out))
        if self.drop is not None:
            out = self.drop(out)
        if self.norms is not None:
            out = self.norms[2](out, domain)
        out = self.conv3(F.relu(out))
        return out

    def branch(self, u, domain):
        return self._pre(u, domain) if self.preact else self._post(u, domain)

    def forward(self, u, domain):
        return u + self.alpha * self.branch(u, domain)


# ── 전체 네트워크 ────────────────────────────────────────────────────────────

class IsoLiftNet(nn.Module):
    """E_d -> A_d -> 공유 T (3 stage, PixelUnshuffle 전이) -> GAP -> H_d."""

    def __init__(self, domains=("MNIST", "CIFAR10", "IMAGENET10"),
                 n_classes=None, layers=(3, 3, 3), cardinality=8,
                 mode="performance", rho=0.9, adapter_beta=0.1,
                 family="resnext"):
        """n_classes: None 이면 DOMAIN_CLASSES 레지스트리 사용
        (IMAGENET1K=1000, 나머지=10). int 를 주면 전 도메인 공통,
        dict {domain: n} 으로 도메인별 지정도 가능."""
        super().__init__()
        assert mode in ("provable", "performance")
        assert family in ISOLIFT_FAMILIES, \
            f"family 는 {list(ISOLIFT_FAMILIES)} 중 하나: {family}"
        unknown = [d for d in domains if d not in LIFTS]
        assert not unknown, f"지원하지 않는 도메인: {unknown}"
        self.domains = tuple(domains)
        self.mode = mode
        self.rho = rho
        self.family = family

        self.lifts = nn.ModuleDict({d: LIFTS[d]() for d in self.domains})
        for p in self.lifts.parameters():
            p.requires_grad_(False)              # E_d 는 고정 (버퍼가 대부분)

        if mode == "performance":
            self.adapters = nn.ModuleDict(
                {d: DomainAdapter(STAGE_CHANNELS[0], adapter_beta)
                 for d in self.domains})
        else:
            self.adapters = None                 # provable: A_d = 항등

        self.stages = nn.ModuleList([
            nn.ModuleList([
                IsoLiftBlock(c, self.domains, cardinality, mode, rho,
                             family=family)
                for _ in range(n)])
            for c, n in zip(STAGE_CHANNELS, layers)])
        self.shuffle = nn.PixelUnshuffle(2)      # 등거리 stage 전이

        if n_classes is None:
            ncls = {d: DOMAIN_CLASSES[d] for d in self.domains}
        elif isinstance(n_classes, int):
            ncls = {d: n_classes for d in self.domains}
        else:
            ncls = dict(n_classes)
        self.heads = nn.ModuleDict(
            {d: nn.Linear(STAGE_CHANNELS[-1], ncls[d]) for d in self.domains})

    # E_d / A_d ---------------------------------------------------------------
    def lift(self, x, domain):
        """z_0 = E_d(x) — 정확한 등거리 lifting."""
        return self.lifts[domain](x)

    def lift_and_adapt(self, x, domain):
        """u_0 = A_d(E_d(x)) — geometry loss 의 관측점."""
        z = self.lift(x, domain)
        if self.adapters is not None:
            z = self.adapters[domain](z)
        return z

    # T / H_d -----------------------------------------------------------------
    def features(self, u, domain):
        for s, stage in enumerate(self.stages):
            for blk in stage:
                u = blk(u, domain)
            if s < len(self.stages) - 1:
                u = self.shuffle(u)
        return u

    def forward(self, x, domain, return_u0=False):
        u0 = self.lift_and_adapt(x, domain)
        u = self.features(u0, domain)
        u = F.adaptive_avg_pool2d(u, 1).flatten(1)
        logits = self.heads[domain](u)
        return (logits, u0) if return_u0 else logits

    def residual_blocks(self):
        for stage in self.stages:
            yield from stage

    def block_features(self, x, domain):
        """각 residual block 통과 직후의 특징 리스트 — probe 학습/추출용.

        기존 DS 파이프라인의 Exprob(블록별 출력 수집)에 대응.
        stage 전이(PixelUnshuffle) 는 등거리이므로 어느 쪽에서 관측해도
        d_g 의 정의와 충돌하지 않는다 (블록 통과 직후, 전이 전에 기록).
        """
        u = self.lift_and_adapt(x, domain)
        feats = []
        for s, stage in enumerate(self.stages):
            for blk in stage:
                u = blk(u, domain)
                feats.append(u)
            if s < len(self.stages) - 1:
                u = self.shuffle(u)
        return feats

    def block_channels(self):
        """블록 순서대로 채널 수 리스트 (probe 입력 차원)."""
        return [c for c, stage in zip(STAGE_CHANNELS, self.stages)
                for _ in range(len(stage))]
