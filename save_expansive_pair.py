"""
save_expansive_pair.py — g-expansive 상수를 달성한 이미지 쌍을 PNG로 저장.

dist_calc.py가 저장한 Result/{DATA}_{model}_epsilon.npy 에는 min-max
(min over 클래스 간 쌍, max over 블록)를 달성한 두 샘플의 원본 테스트셋
인덱스(sample_a / sample_b)와 클래스, 블록 번호가 들어 있다. 이 스크립트는
그 인덱스로 테스트셋에서 이미지를 꺼내:

  1) 개별 PNG 2장  — {DATA}_{model}_expansive_A_class{a}_idx{i}.png / ..._B_...
  2) 주석(eps, 클래스, 블록, space)이 달린 나란히 비교 그림 1장 — ..._pair.png

를 저장한다. 모델/데이터셋에 무관하게 동작한다 (dist_calc를 먼저 실행할 것).

실행:
  python save_expansive_pair.py --model ds_resnet18 --data MNIST
  python save_expansive_pair.py --model ds_resnet50 --data CIFAR10 --out my_dir
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")               # 디스플레이 없는 서버에서도 저장 가능
import matplotlib.pyplot as plt

from models.models import DS_MODELS
from utils.stubs import load_data

# 화면 표시용 역정규화 통계 (utils/stubs.py의 load_data 정규화 값과 동일해야 함)
NORM_STATS = {
    "MNIST":      ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "CIFAR10":    ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "IMAGENET10": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="g-expansive 상수를 달성한 이미지 쌍 저장")
    p.add_argument('--model', default='ds_resnet18', choices=list(DS_MODELS))
    p.add_argument('--data', default='MNIST',
                   choices=['MNIST', 'CIFAR10', 'IMAGENET10'])
    p.add_argument('--out', default='Result/expansive_pair',
                   help="PNG 저장 디렉토리 (기본 Result/expansive_pair)")
    return p.parse_args()


def unnormalize(img_tensor, mean, std):
    """(C,H,W) 정규화 텐서 → [0,1] (H,W,C) numpy."""
    img = img_tensor.clone().float()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def save_pair(eps_res, test_set, data_name, model_tag, out_dir):
    """epsilon.npy 결과 dict + 테스트셋으로 이미지 쌍 PNG 3장을 저장.

    Returns: 저장된 파일 경로 리스트
    """
    mean, std = NORM_STATS.get(data_name, ([0.0] * 3, [1.0] * 3))
    os.makedirs(out_dir, exist_ok=True)

    picks = [("A", int(eps_res["sample_a"]), int(eps_res["class_a"])),
             ("B", int(eps_res["sample_b"]), int(eps_res["class_b"]))]
    saved, images = [], []

    for tag_ab, idx, cls in picks:
        img, label = test_set[idx]
        label = int(label)
        if label != cls:
            # 인덱스가 다른 데이터셋/순서를 가리키면 잘못된 이미지를 뽑은 것
            print(f"[경고] 테스트셋[{idx}] 라벨({label}) != epsilon.npy의 "
                  f"클래스({cls}) — dist_calc와 동일한 data/seed인지 확인하세요.")
        arr = unnormalize(img, mean, std)
        path = os.path.join(
            out_dir,
            f"{data_name}_{model_tag}_expansive_{tag_ab}_class{cls}_idx{idx}.png")
        plt.imsave(path, arr)
        saved.append(path)
        images.append((arr, cls, idx, label))

    # 나란히 비교 그림 (수치 주석 포함)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2))
    for ax, (arr, cls, idx, label) in zip(axes, images):
        ax.imshow(arr)
        ax.set_title(f"class {cls}  (test idx {idx})", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"g-expansive pair — {data_name} / {model_tag}\n"
        f"eps = {eps_res['epsilon']:.6e}   |   block {eps_res['block']}"
        f"   |   space = {eps_res.get('space', '?')}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    pair_path = os.path.join(
        out_dir, f"{data_name}_{model_tag}_expansive_pair.png")
    fig.savefig(pair_path, dpi=150)
    plt.close(fig)
    saved.append(pair_path)
    return saved


if __name__ == '__main__':
    args = parse_args()
    eps_path = f"Result/{args.data}_{args.model}_epsilon.npy"
    if not os.path.exists(eps_path):
        raise FileNotFoundError(
            f"{eps_path} 없음 — dist_calc.py를 동일한 --model/--data로 먼저 "
            "실행하세요.")
    eps_res = np.load(eps_path, allow_pickle=True).item()

    print(f"[expansive] eps = {eps_res['epsilon']:.6e}  |  "
          f"class {eps_res['class_a']} vs {eps_res['class_b']}  |  "
          f"샘플 {eps_res['sample_a']} <-> {eps_res['sample_b']}  |  "
          f"블록 {eps_res['block']}")

    _, test_set = load_data(args.data)
    saved = save_pair(eps_res, test_set, args.data, args.model, args.out)
    print("저장 완료:")
    for pth in saved:
        print(f"  {pth}")
