"""
[Expansive]/[Shadowing] 개별 이미지·이미지 쌍 선별.

학습된 DS-ResNet의 block_fc 출력(prob_fc/)과, dist_calc.py가 만든 pseudo-orbit
체인(task2/*_SeqInfo.npy, *_MaxList.npy)을 이용해:

  1) [Expansive] 같은 클래스 쌍인데도 block_fc 거리가 비정상적으로 "큰"
     이미지 쌍 Top-10 (블록 전체의 50% 이상을 통과한 블록만 검사)
  2) [Shadowing] pseudo-orbit 체인을 따라 block_fc[0] 예측 클래스가 가장
     안정적으로(바뀌지 않고) 유지된 체인(robust) / 가장 자주 바뀌어 클래스에서
     계속 벗어난 체인(unstable)

을 선별하고, 거리/flip 수치와 함께 실제 이미지를 PNG로 저장한다.

사전 조건:
  - main.py에서 동일한 MODEL_NAME/DATA_NAME으로 USE_BLOCK_FC=True 학습을 마쳐
    prob_fc/{DATA_NAME}/{model_tag}/, pix/resnet/{DATA_NAME}/{model_tag}/test/ 가 존재해야 함.
  - dist_calc.py를 동일한 MODEL_NAME/DATA_NAME으로 실행해 task2/{DATA_NAME}_{model_tag}_
    SeqInfo.npy, MaxList.npy 가 생성되어 있어야 함 (pseudo-orbit 체인 분석용).

실행: python inspect_examples.py --model ds_resnet18 --data MNIST
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.stubs import load_data
from utils.orbit_analysis import find_expansive_outliers, analyze_pseudo_orbit_stability


from models.models import DS_MODELS, ds_layers

_p = argparse.ArgumentParser(description="Expansive/Shadowing 사례 시각화")
_p.add_argument('--model', default='ds_resnet18', choices=list(DS_MODELS))
_p.add_argument('--data', default='MNIST',
                choices=['MNIST', 'CIFAR10', 'IMAGENET10'])
_args = _p.parse_args()

MODEL_NAME = _args.model
DATA_NAME  = _args.data

TOP_K_EXPANSIVE = 10
TOP_K_SHADOWING = 5
MIN_DEPTH_RATIO = 0.5        # [Expansive]: 이 비율 이상 통과한 블록만 검사

OUT_DIR = "Result/inspect"

n_blocks = sum(ds_layers(MODEL_NAME))
model_tag = MODEL_NAME

# 화면 표시용 역정규화 통계 (utils/stubs.py의 load_data 정규화 값과 동일해야 함)
NORM_STATS = {
    "MNIST":      ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "CIFAR10":    ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "IMAGENET10": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def unnormalize(img, mean, std):
    img = img.clone()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_chain(test_dataset, mean, std, info, title, save_path):
    chain = info["chain"]
    preds = info["chain_pred_seq"]
    true_label = info["true_label"]
    n = len(chain)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.8))
    if n == 1:
        axes = [axes]
    for t, (sample_idx, pred) in enumerate(zip(chain, preds)):
        img, _ = test_dataset[sample_idx]
        axes[t].imshow(unnormalize(img, mean, std))
        color = 'green' if pred == true_label else 'red'
        axes[t].set_title(f"t={t}  idx={sample_idx}\npred={pred}", fontsize=7, color=color)
        axes[t].axis('off')
    fig.suptitle(f"{title}  (true_label={true_label}, "
                 f"flips={info['flip_count']}, deviates={info['deviate_count']})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    fc_dir = f"prob_fc/{DATA_NAME}/{model_tag}"
    l_path = f"pix/resnet/{DATA_NAME}/{model_tag}/test"
    task2_dir = "task2"

    prob_fc_blocks = {b: torch.load(f"{fc_dir}/{DATA_NAME}_block{b}.pt") for b in range(n_blocks)}
    labels = torch.load(f"{l_path}/{DATA_NAME}_label.pt")

    print(f"[설정] {model_tag}  |  data={DATA_NAME}  |  blocks={n_blocks}  |  N={labels.shape[0]}")

    # ── [Expansive] 같은 클래스 쌍 중 block_fc 거리 이상치 ──────────────
    start_block = max(int(n_blocks * MIN_DEPTH_RATIO) - 1, 0)
    while (start_block + 1) / n_blocks < MIN_DEPTH_RATIO:
        start_block += 1
    print(f"\n[Expansive] block_fc 출력 기준, 블록 {start_block}~{n_blocks - 1}"
          f"(50% 이상 통과) 구간에서 같은 클래스 쌍 중 거리가 가장 큰 "
          f"Top-{TOP_K_EXPANSIVE} 탐색...")
    expansive_top = find_expansive_outliers(
        prob_fc_blocks, labels, n_blocks, top_k=TOP_K_EXPANSIVE, min_depth_ratio=MIN_DEPTH_RATIO
    )
    for rank, r in enumerate(expansive_top, 1):
        print(f"  #{rank:2d}  sample {r['sample_i']:>6d}  vs  sample {r['sample_j']:>6d}  "
              f"(label={r['label_i']})  block={r['block']:>2d}  distance={r['distance']:.4f}")

    # ── [Shadowing] pseudo-orbit 체인의 예측 클래스 안정성 ──────────────
    seqinfo_path = f"{task2_dir}/{DATA_NAME}_{model_tag}_SeqInfo.npy"
    maxlist_path = f"{task2_dir}/{DATA_NAME}_{model_tag}_MaxList.npy"
    if not (os.path.exists(seqinfo_path) and os.path.exists(maxlist_path)):
        raise FileNotFoundError(
            f"{seqinfo_path} 또는 {maxlist_path} 없음. "
            f"dist_calc.py --model {model_tag} --data {DATA_NAME} 를 먼저 실행하세요."
        )
    seq_info = np.load(seqinfo_path, allow_pickle=True)
    maxlist = np.load(maxlist_path, allow_pickle=True)

    print(f"\n[Shadowing] pseudo-orbit 체인(길이 {seq_info.shape[1]})을 따라 "
          f"block_fc[0] 예측 클래스가 가장 안정적/불안정적으로 유지된 체인 "
          f"Top-{TOP_K_SHADOWING}개씩 선별...")
    stable_top, unstable_top = analyze_pseudo_orbit_stability(
        seq_info, maxlist, prob_fc_blocks[0], labels, top_k=TOP_K_SHADOWING
    )
    print("  [안정적 (robust - 예측 클래스가 거의 바뀌지 않음)]")
    for rank, r in enumerate(stable_top, 1):
        print(f"  #{rank}  sample {r['sample']:>6d}(true={r['true_label']})  "
              f"flips={r['flip_count']}  deviates={r['deviate_count']}/{len(r['chain'])}  "
              f"chain={r['chain']}")
        print(f"       chain_pred_seq={r['chain_pred_seq']}")
        print(f"       step_errors={r['step_errors']}")
    print("  [불안정적 (클래스에서 계속 벗어난 체인)]")
    for rank, r in enumerate(unstable_top, 1):
        print(f"  #{rank}  sample {r['sample']:>6d}(true={r['true_label']})  "
              f"flips={r['flip_count']}  deviates={r['deviate_count']}/{len(r['chain'])}  "
              f"chain={r['chain']}")
        print(f"       chain_pred_seq={r['chain_pred_seq']}")
        print(f"       step_errors={r['step_errors']}")

    # ── 시각화 (실제 이미지 저장) ────────────────────────────────────
    _, test_dataset = load_data(DATA_NAME)
    mean, std = NORM_STATS[DATA_NAME]

    # [Expansive] 쌍 그리드
    n_pairs = len(expansive_top)
    if n_pairs > 0:
        fig, axes = plt.subplots(n_pairs, 2, figsize=(4, 2 * n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, 2)
        for row, r in enumerate(expansive_top):
            for col, idx_key in enumerate(["sample_i", "sample_j"]):
                img, _ = test_dataset[r[idx_key]]
                axes[row, col].imshow(unnormalize(img, mean, std))
                axes[row, col].set_title(f"idx={r[idx_key]} y={r['label_i']}", fontsize=8)
                axes[row, col].axis('off')
            axes[row, 0].set_ylabel(f"block{r['block']}\nd={r['distance']:.2f}", fontsize=7)
        fig.suptitle(f"[Expansive] same-class distance outliers Top-{n_pairs} ({DATA_NAME}/{model_tag})")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/{DATA_NAME}_{model_tag}_expansive_top{n_pairs}.png", dpi=150)
        plt.close(fig)

    # [Shadowing] 1위 robust/unstable 체인 전체를 이미지 시퀀스로 시각화
    if stable_top:
        plot_chain(test_dataset, mean, std, stable_top[0], "[Shadowing] robust pseudo-orbit",
                   f"{OUT_DIR}/{DATA_NAME}_{model_tag}_shadowing_stable_chain.png")
    if unstable_top:
        plot_chain(test_dataset, mean, std, unstable_top[0], "[Shadowing] unstable pseudo-orbit",
                   f"{OUT_DIR}/{DATA_NAME}_{model_tag}_shadowing_unstable_chain.png")

    print(f"\n[저장 완료] 시각화 PNG -> {OUT_DIR}/")
