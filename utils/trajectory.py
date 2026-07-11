"""
관측 공간(d_g) 궤적 로더.

논문의 모든 안정성 개념은 pseudometric  d_g(x, y) = d(g(x), g(y))  위에서
정의되므로, 거리 계산은 g의 "출력 공간"에서 이루어져야 한다.
(구버전은 raw 블록 특징에 softmax를 씌운 거리를 사용했는데, 이는 d_g가 아니다.)

space 옵션:
  'prob'  : softmax(block_fc logit) — 논문의 "difference of probability" (기본값).
            D = n_class 차원이므로 전체 파이프라인이 가볍다.
  'logit' : block_fc logit 그대로 — Lip(g) = sigma_max(W)가 정확히 성립.
  'feat'  : raw 블록 특징 (prob/ 디렉토리) — 구버전 호환/참고용. d_g가 아님.
            D = 200,704라서 n_samples로 반드시 서브샘플링할 것.
"""
import os
import numpy as np
import torch


def infer_n_blocks(data_name, model_tag, space='prob'):
    """저장된 블록 파일 수로 T(블록 수)를 추론.

    DS 레지스트리 밖 커스텀 태그(예: extract_isolift.py 가 만든
    isolift_{family}_{mode})의 분석을 위해 사용. 파일이 없으면 0.
    """
    root = "prob_fc" if space in ('prob', 'logit') else "prob"
    n = 0
    while os.path.exists(
            f"{root}/{data_name}/{model_tag}/{data_name}_block{n}.pt"):
        n += 1
    return n


def load_trajectory(data_name, model_tag, n_blocks, space='prob',
                    n_samples=None, seed=13):
    """블록별 저장 파일에서 (N, T, D) 궤적 텐서를 구성해 반환.

    Parameters
    ----------
    space     : 'prob' | 'logit' | 'feat'
    n_samples : int 또는 None — None이면 전체 사용. 클래스 비율 무관 균등 추출.

    Returns
    -------
    traj   : torch.FloatTensor (N, T, D)
    labels : torch.LongTensor  (N,)
    idx    : np.ndarray (N,) — 원본 테스트셋 기준 샘플 인덱스
             (task2/SeqInfo 등 저장 시 이 인덱스로 되돌려 저장해야
              inspect_examples.py가 원본 데이터셋과 맞춰볼 수 있다)
    """
    if space not in ('prob', 'logit', 'feat'):
        raise ValueError(f"space는 'prob'|'logit'|'feat' 중 하나여야 합니다: {space}")

    label_path = f"pix/resnet/{data_name}/{model_tag}/test/{data_name}_label.pt"
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"라벨 파일 없음: {label_path}\nmain.py를 먼저 실행하세요.")
    labels = torch.load(label_path).long()

    root = "prob_fc" if space in ('prob', 'logit') else "prob"
    feat_dir = f"{root}/{data_name}/{model_tag}"

    blocks = []
    for b in range(n_blocks):
        pth = f"{feat_dir}/{data_name}_block{b}.pt"
        if not os.path.exists(pth):
            hint = ("main.py를 --use-block-fc로 실행해 prob_fc/를 생성하세요."
                    if root == "prob_fc" else "main.py를 먼저 실행하세요.")
            raise FileNotFoundError(f"블록 파일 없음: {pth}\n{hint}")
        blocks.append(torch.load(pth).float())
    traj = torch.stack(blocks, dim=1)          # (N, T, D)

    if space == 'prob':
        traj = torch.softmax(traj, dim=-1)

    idx = np.arange(traj.shape[0])
    if n_samples is not None and n_samples < traj.shape[0]:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(traj.shape[0], n_samples, replace=False))
        traj = traj[idx]
        labels = labels[idx]

    if space == 'feat' and n_samples is None and traj.shape[2] > 4096:
        print(f"[경고] space='feat' (D={traj.shape[2]:,})를 전체 샘플로 사용 중 — "
              "메모리가 부족하면 n_samples로 서브샘플링하세요.")

    return traj, labels, idx
