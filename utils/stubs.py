import os

import numpy as np
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


def _load_cifar10(img_size=224):
    # CIFAR-10 표준 augmentation + CIFAR 정규화
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
    """Imagenette (fast.ai) — ImageNet 10클래스 subset, 자동 다운로드."""
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
    if name == 'CIFAR10' or name == 'CIFAR-10':
        return _load_cifar10(img_size)
    if name in ('IMAGENET10', 'IMAGENET-10', 'IMAGENETTE'):
        return _load_imagenette(img_size)
    transform = T.Compose([T.ToTensor()])
    dataset = FakeData(size=n_samples, image_size=(3, img_size, img_size),
                       num_classes=n_class, transform=transform)
    n_train = int(len(dataset) * 0.8)
    return random_split(dataset, [n_train, len(dataset) - n_train])


@torch.no_grad()
def evaluate(model, loader, device):
    """테스트 손실 / 정확도 / macro F1을 계산 (Table 1의 F1 Score, Loss)."""
    from sklearn.metrics import f1_score

    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    loss_sum, n = 0.0, 0
    preds, ys = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item()
        n += y.size(0)
        preds.append(out.argmax(dim=1).cpu())
        ys.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(ys).numpy()
    return {
        "loss": loss_sum / n,
        "acc": float((y_pred == y_true).mean()),
        "f1": float(f1_score(y_true, y_pred, average='macro')),
    }


def train(model, trainloader, testloader, device, epochs=2, es=5,
          lpth="model", lr=1e-4):
    """백본 학습. 매 에폭 테스트 손실을 평가해 best 체크포인트 저장 및
    early stopping 판단 (구버전은 train loss 기준이었음)."""
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
        train_avg = total_loss / len(trainloader)
        metrics = evaluate(model, testloader, device)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_avg:.4f}  "
              f"test_loss={metrics['loss']:.4f}  test_acc={metrics['acc']*100:.2f}%")
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            patience = 0
            torch.save(model.state_dict(), f"{lpth}.pt")
        else:
            patience += 1
            if patience >= es:
                print("Early stopping")
                break


def train_block_fc(extractor, trainloader, device, epochs=5):
    """블록별 fc 레이어만 학습 (backbone 고정, linear probing).

    손실 = 모든 블록 fc 출력의 CrossEntropyLoss 합산.
    """
    for param in extractor.parameters():
        param.requires_grad = False
    for param in extractor.block_fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(extractor.block_fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # eval 모드 유지: train()으로 두면 frozen backbone의 BatchNorm running
    # stats가 오염된다. block_fc는 Linear라 train/eval 동작 차이가 없다.
    extractor.eval()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = extractor(x, use_block_fc=True)   # {block_idx: (B, n_class)}
            loss = sum(criterion(logits, y) for logits in out.values())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[block_fc] Epoch {epoch+1}/{epochs}  loss={total_loss/len(trainloader):.4f}")

    for param in extractor.parameters():
        param.requires_grad = True


@torch.no_grad()
def extract_block_outputs(extractor, testloader, device, use_block_fc=False):
    """테스트셋 전체에 대해 블록별 출력과 라벨을 한 번의 pass로 수집.

    Returns
    -------
    outs   : {block_idx: Tensor (N, D)} — raw 특징(D=200704) 또는 logit(D=n_class)
    labels : Tensor (N,)
    """
    extractor.eval()
    outs, labels = {}, None
    for x, y in testloader:
        batch = extractor(x.to(device), use_block_fc=use_block_fc)
        for b, feat in batch.items():
            fc = feat.detach().cpu()
            outs[b] = fc if b not in outs else torch.cat((outs[b], fc))
        labels = y if labels is None else torch.cat((labels, y))
    return outs, labels


def save_block_outputs(outs, root, data_name, model_tag):
    """{root}/{data_name}/{model_tag}/{data_name}_block{b}.pt 형식으로 저장."""
    out_dir = f"{root}/{data_name}/{model_tag}"
    os.makedirs(out_dir, exist_ok=True)
    for b, feat in outs.items():
        torch.save(feat, f"{out_dir}/{data_name}_block{b}.pt")
    return out_dir


def save_labels(labels, data_name, model_tag):
    """라벨 저장 — 구버전 경로 유지 (분석/시각화 스크립트 호환).

    구버전이 이 디렉토리에 함께 저장하던 배치 단위 특징 청크는
    어디에서도 읽지 않으므로 더 이상 저장하지 않는다.
    """
    pix_dir = f"pix/resnet/{data_name}/{model_tag}/test"
    os.makedirs(pix_dir, exist_ok=True)
    torch.save(labels, f"{pix_dir}/{data_name}_label.pt")
    return pix_dir


def save_metrics(metrics, data_name, model_tag):
    """Table 1의 F1/Loss/Acc를 Result/에 저장."""
    os.makedirs("Result", exist_ok=True)
    path = f"Result/{data_name}_{model_tag}_metrics.npy"
    np.save(path, metrics)
    return path


class Exprob(ResNet):
    """ResNet 기반 특징 추출기.

    두 가지 모드:
      forward(x)                    → {block_idx: (B, feat_dim)}  블록 raw 특징
      forward(x, use_block_fc=True) → {block_idx: (B, n_class)}   블록별 fc logit

    block_fc의 입력 차원은 항상 hook이 캡처한 블록 출력 차원(200,704)입니다.
    avgpool은 분류 헤드(main fc)에만 영향을 주며, 동력계 분석용 블록 특징은
    avgpool 이전의 200,704-dim 값을 그대로 사용합니다.
    """

    BLOCK_FEAT_DIM = 2048 * 7 * 14   # 200,704 — 224 입력 기준 블록 flatten 차원

    def __init__(self, n_class, layers=None, multi_fc=False, use_avgpool=False):
        if layers is None:
            layers = [3, 4, 6, 3]
        super().__init__(block=Bottleneck, layers=layers, num_classes=n_class,
                         use_avgpool=use_avgpool)
        self.multi_fc = multi_fc
        if multi_fc:
            n_blocks = sum(layers)
            self.block_fc = nn.ModuleList(
                [nn.Linear(self.BLOCK_FEAT_DIM, n_class) for _ in range(n_blocks)]
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
            return {i: self.block_fc[i](feat) for i, feat in self._block_feats.items()}
        return dict(self._block_feats)
