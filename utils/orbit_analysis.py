"""
[Expansive]/[Shadowing] 개별 샘플·쌍 단위 분석.

- find_expansive_outliers        : 같은 클래스 쌍 중 block_fc 출력 거리가
  비정상적으로 "큰" Top-K 쌍을 탐색 (block_fc 단계에서의 클래스 내부 불일치 사례).
- analyze_pseudo_orbit_stability : seq_builder가 만든 pseudo-orbit 체인을 따라
  block_fc[0] 예측 클래스가 얼마나 안정적으로(바뀌지 않고) 유지되는지를 평가.
"""

import torch


def find_expansive_outliers(prob_fc_blocks, labels, n_blocks, top_k=10, min_depth_ratio=0.5):
    """같은 클래스 쌍 중 block_fc 출력 거리가 비정상적으로 큰 Top-K 쌍.

    같은 클래스라면 g(phi^n(x))는 서로 가까워야 한다는 직관과 달리, block_fc
    출력 공간에서 거리가 유독 크게 벌어지는 쌍 -- 즉 같은 클래스 내부에서
    분류기가 불일치하게(비일관적으로) 취급한 사례를 찾는다.

    Parameters
    ----------
    prob_fc_blocks : dict[int, Tensor]   {block_idx: (N, n_class)}
    labels         : Tensor (N,)
    n_blocks       : int    — 전체 블록 수
    top_k          : int
    min_depth_ratio: float  — 이 비율 이상 통과한 블록만 검사 (기본 0.5 = 50%)

    Returns
    -------
    list[dict] — distance 내림차순, 길이 top_k
        {sample_i, sample_j, label_i, label_j, block, distance}
        (label_i == label_j 항상 성립)
    """
    y = labels
    N = y.shape[0]
    # (b+1)/n_blocks >= min_depth_ratio 를 만족하는 최소 블록 인덱스 b
    start_block = max(int(n_blocks * min_depth_ratio) - 1, 0)
    while (start_block + 1) / n_blocks < min_depth_ratio:
        start_block += 1

    candidates = []
    with torch.no_grad():
        for b in range(start_block, n_blocks):
            P = prob_fc_blocks[b].float()
            D = torch.cdist(P, P, p=2)
            same_mask = y.unsqueeze(0) == y.unsqueeze(1)
            tri_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            mask = same_mask & tri_mask
            if not mask.any():
                continue
            D_masked = torch.where(mask, D, torch.full_like(D, float('-inf')))
            k = min(top_k, int(mask.sum().item()))
            vals, idxs = torch.topk(D_masked.flatten(), k)
            for v, idx in zip(vals.tolist(), idxs.tolist()):
                i, j = divmod(idx, N)
                candidates.append((v, i, j, b))

    candidates.sort(key=lambda t: -t[0])
    top = candidates[:top_k]

    return [
        {
            "sample_i": i, "sample_j": j,
            "label_i": int(y[i]), "label_j": int(y[j]),
            "block": b, "distance": v,
        }
        for v, i, j, b in top
    ]


def analyze_pseudo_orbit_stability(seq_info, maxlist, prob_fc_block0, labels, top_k=5):
    """pseudo-orbit 체인(seq_builder 출력)을 따라 예측 클래스 안정성을 평가.

    seq_builder가 만든 체인의 모든 단계(Series)는 항상 "어떤 실제 샘플의
    block-0 특징"이다 (block을 진행할 때마다 가장 가까운 새로운 같은-클래스
    샘플의 block-0 특징으로 점프하는 구조). 따라서 체인 각 멤버의 예측 클래스는
    block_fc[0]을 적용해서 구한다.

    Parameters
    ----------
    seq_info       : np.ndarray (N, n_blocks+1) int — seq_builder의 SeqInfo.
                      각 행 = 샘플 idx의 pseudo-orbit 체인을 구성하는 실제 샘플
                      인덱스 (allow_cross_class=False라면 전부 idx와 같은 클래스).
    maxlist        : np.ndarray (N, n_blocks) — seq_builder의 MaxList. 체인
                      각 단계의 raw-feature 거리(pseudo-orbit step error).
    prob_fc_block0 : Tensor (N, n_class) — block_fc[0] 출력 (전체 샘플).
    labels         : Tensor (N,) — 전체 샘플의 ground-truth 라벨.
    top_k          : int

    Returns
    -------
    stable_top, unstable_top : list[dict] (길이 top_k)
        {sample, true_label, flip_count, deviate_count, chain,
         chain_pred_seq, step_errors}
        - flip_count    : 체인을 따라 예측 클래스가 바뀐 횟수 (작을수록 robust)
        - deviate_count : 체인 멤버 중 true_label과 다르게 예측된 개수
        - chain         : 체인을 구성하는 실제 샘플 인덱스 리스트
        - chain_pred_seq: 체인 각 멤버의 block_fc[0] 예측 클래스
        - step_errors   : 체인 각 단계의 raw-feature pseudo-orbit 거리(MaxList)
    """
    preds0 = prob_fc_block0.argmax(dim=1)
    N = seq_info.shape[0]

    flip_counts = []
    deviate_counts = []
    chain_preds_all = []
    for idx in range(N):
        chain = seq_info[idx]
        chain_preds = preds0[chain]
        flips = int((chain_preds[1:] != chain_preds[:-1]).sum().item())
        deviate = int((chain_preds != int(labels[idx])).sum().item())
        flip_counts.append(flips)
        deviate_counts.append(deviate)
        chain_preds_all.append(chain_preds.tolist())

    flip_counts_t = torch.tensor(flip_counts)
    order_stable = torch.argsort(flip_counts_t, stable=True)
    order_unstable = torch.argsort(flip_counts_t, descending=True, stable=True)

    def build(order):
        out = []
        for idx in order[:top_k].tolist():
            out.append({
                "sample": idx,
                "true_label": int(labels[idx]),
                "flip_count": flip_counts[idx],
                "deviate_count": deviate_counts[idx],
                "chain": seq_info[idx].tolist(),
                "chain_pred_seq": chain_preds_all[idx],
                "step_errors": [round(float(v), 4) for v in maxlist[idx].tolist()],
            })
        return out

    return build(order_stable), build(order_unstable)
