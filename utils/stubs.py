import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.datasets import FakeData
from torch.utils.data import random_split
from models.ResNets import ResNet, Bottleneck


def _load_mnist(img_size=224):
    transform = T.Compose([
        T.Resize(img_size),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.MNIST(root='data', train=True,
                                           download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root='data', train=False,
                                           download=True, transform=transform)
    return train_set, test_set


def _load_mnist_1ch(img_size=224):
    """MNIST 1채널 그대로 유지 — X = R^{img_size x img_size} (선택 1 전용)."""
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.1307], std=[0.3081]),   # MNIST 통계 (1ch)
    ])
    train_set = torchvision.datasets.MNIST(root='data', train=True,
                                           download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root='data', train=False,
                                           download=True, transform=transform)
    return train_set, test_set


def _load_cifar10(img_size=224):
    # CIFAR-10 표준 augmentation + ImageNet 정규화
    train_transform = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=img_size // 8),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]),
    ])
    test_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = torchvision.datasets.CIFAR10(root='data', train=True,
                                             download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root='data', train=False,
                                             download=True, transform=test_transform)
    return train_set, test_set


def _load_imagenette(img_size=224, size='320px'):
    """Imagenette (fast.ai) — ImageNet 10클래스 subset, 라이선스 없이 자동 다운로드.

    클래스(10개): tench, English springer, cassette player, chainsaw, church,
                  French horn, garbage truck, gas pump, golf ball, parachute
    size='320px'는 다운로드 용량/속도와 화질의 균형점 (어차피 224로 리사이즈됨).
    """
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),   # 원본이 정사각형이 아니므로 크롭 필요
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.Imagenette(
        root='data', split='train', size=size, download=True, transform=transform)
    test_set = torchvision.datasets.Imagenette(
        root='data', split='val', size=size, download=True, transform=transform)
    return train_set, test_set


def load_data(data_name, img_size=224, n_class=2, n_samples=40):
    name = data_name.upper()
    if name == 'MNIST':
        return _load_mnist(img_size)
    if name in ('MNIST_1CH', 'MNIST1CH'):
        return _load_mnist_1ch(img_size)
    if name == 'CIFAR10' or name == 'CIFAR-10':
        return _load_cifar10(img_size)
    if name in ('IMAGENET10', 'IMAGENET-10', 'IMAGENETTE'):
        return _load_imagenette(img_size)
    transform = T.Compose([T.ToTensor()])
    dataset = FakeData(size=n_samples, image_size=(3, img_size, img_size),
                       num_classes=n_class, transform=transform)
    n_train = int(len(dataset) * 0.8)
    return random_split(dataset, [n_train, len(dataset) - n_train])


def train(model, trainloader, testloader, device, epochs=2, es=5, lpth="model", lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    patience = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}")
        if total_loss < best_loss:
            best_loss = total_loss
            patience = 0
            try:
                torch.save(model.state_dict(), f"{lpth}.pt")
            except RuntimeError:
                pass
        else:
            patience += 1
            if patience >= es:
                print("Early stopping")
                break


def train_block_fc(extractor, trainloader, device, epochs=5):
    """블록별 fc 레이어만 학습 (backbone 고정, linear probing).

    각 블록 출력에 달린 16개 fc를 동시에 학습.
    손실 = 16개 블록 fc 출력의 CrossEntropyLoss 합산.
    """
    # backbone 고정, block_fc만 학습
    for param in extractor.parameters():
        param.requires_grad = False
    for param in extractor.block_fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(extractor.block_fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    extractor.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = extractor(x, use_block_fc=True)   # {0~15: (B, n_class)}
            loss = sum(criterion(logits, y) for logits in out.values())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[block_fc] Epoch {epoch+1}/{epochs}  loss={total_loss/len(trainloader):.4f}")

    # 학습 후 모든 파라미터 gradient 복원
    for param in extractor.parameters():
        param.requires_grad = True


class Exprob(ResNet):
    """ResNet 기반 특징 추출기.

    두 가지 모드:
      forward(x)                    → {0~15: (B, feat_dim)}  블록 raw 특징
      forward(x, use_block_fc=True) → {0~15: (B, n_class)}  블록별 fc 출력 (logit)

    Parameters
    ----------
    feat_dim    : 블록 출력의 flatten 차원 (200,704 or 2,048)
    n_class     : 분류 클래스 수
    use_avgpool : True  → 최종 fc 앞에 AdaptiveAvgPool(1,1) 적용 (feat_dim=2048)
                  False → spatial 정보 그대로 flatten           (feat_dim=200704)
    multi_fc    : True  → 블록별 block_fc 생성 (train_block_fc()로 학습)

    block_fc의 입력 차원은 항상 hook이 캡처한 block output dim (200,704)입니다.
    avgpool은 분류 헤드(main fc)에만 영향을 주며, 동력계 분석용 block feature는
    avgpool 이전의 200,704-dim 값을 그대로 사용합니다.
    """

    def __init__(self, feat_dim, n_class, layers=None, multi_fc=False,
                 use_avgpool=False):
        if layers is None:
            layers = [3, 4, 6, 3]
        super().__init__(block=Bottleneck, layers=layers, num_classes=n_class,
                         use_avgpool=use_avgpool)
        # main fc는 ResNet.__init__ 에서 use_avgpool 기반으로 이미 생성됨
        # transfer()가 호출되면 교체되므로 여기서 추가 생성하지 않음
        self.multi_fc = multi_fc
        if multi_fc:
            n_blocks = sum(layers)   # 3+4+6+3 = 16
            # block_fc는 항상 200,704-dim hook 출력을 입력으로 받음
            block_feat_dim = 2048 * 7 * 14   # 200,704
            self.block_fc = nn.ModuleList(
                [nn.Linear(block_feat_dim, n_class) for _ in range(n_blocks)]
            )
        self._block_feats = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        idx = 0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                h = block.register_forward_hook(
                    lambda m, inp, out, i=idx: self._block_feats.__setitem__(i, out.flatten(1))
                )
                self._hooks.append(h)
                idx += 1

    def forward(self, x, use_block_fc=False):
        self._block_feats = {}
        super().forward(x)
        if use_block_fc and self.multi_fc:
            # 각 블록 출력을 해당 블록의 fc에 통과 → logit (B, n_class)
            return {i: self.block_fc[i](feat) for i, feat in self._block_feats.items()}
        return dict(self._block_feats)
