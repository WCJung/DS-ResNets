import numpy as np
import os
from utils.task import DistanceMeasure, task_2
from utils.builders import seq_builder, load_files
from utils.read import task_old1, task2_read
from utils.lipschitz import analyze_theorem
import sys
from utils.norms import softmax, sigmoid, norm, init_random
from utils.distance import minkovski
import torch


if __name__ == '__main__':

    # ── 옵션 설정 (main.py에서 학습한 값과 반드시 일치시킬 것) ─────────
    MODEL_NAME        = "ds_resnet18"  # 'ds_resnet18' | 'ds_resnet50' (DS-ResNet 전용 분석)
    DATA_NAME         = "MNIST"        # main.py의 DATA_NAME과 동일하게
    USE_AVGPOOL       = True           # main.py의 USE_AVGPOOL과 동일하게
    USE_FC_OUTPUT     = False  # True: 블록별 fc 출력(logit) 기반 거리 / False: raw 블록 특징 기반 거리
    ALLOW_CROSS_CLASS = False  # False: 같은 클래스 내에서만 이웃 탐색 / True: 클래스 무관 (클래스 혼합 관찰용)

    DS_LAYERS_MAP = {
        'ds_resnet18': [2, 2, 2, 2],
        'ds_resnet50': [3, 4, 6, 3],
    }
    if MODEL_NAME not in DS_LAYERS_MAP:
        raise ValueError(
            f"dist_calc.py는 DS-ResNet 전용 분석입니다 ('{MODEL_NAME}' 미지원). "
            "ResNet-18/50은 레이어마다 채널 수가 달라 블록간 거리 비교가 성립하지 않습니다."
        )
    ds_layers = DS_LAYERS_MAP[MODEL_NAME]
    n_blocks  = sum(ds_layers)
    model_tag = MODEL_NAME

    if USE_FC_OUTPUT:
        f_path = f"prob_fc"      # main.py에서 저장한 fc 출력 경로
        scale  = 1.0             # logit은 /1000 불필요
    else:
        f_path = f"prob"         # raw 블록 특징 경로
        scale  = 1000.0          # 큰 특징값 스케일 조정

    feat_dir = f"{f_path}/{DATA_NAME}/{model_tag}"
    l_path   = f"pix/resnet/{DATA_NAME}/{model_tag}/test"

    for b in range(n_blocks):
        pth = f"{feat_dir}/{DATA_NAME}_block{b}.pt"
        x = torch.load(pth)
        if b == 0:
            hold = x.detach().numpy().reshape(x.shape[0], 1, x.shape[1])
            y = torch.load(f"{l_path}/{DATA_NAME}_label.pt")
        else:
            hold = np.concatenate((hold, x.detach().numpy().reshape(x.shape[0], 1, x.shape[1])), 1)

    feat_dict = DistanceMeasure(hold / scale, y, norm="softmax")
    feat_dict.task_1(DATA_NAME, model_tag)

    seqs = seq_builder(hold, DATA_NAME, model_tag, n_blocks,
                       labels=y, allow_cross_class=ALLOW_CROSS_CLASS)
    best_stack, best_stack_mean = task_2(seqs, DATA_NAME,
                                         labels=y, allow_cross_class=ALLOW_CROSS_CLASS)

    # ── Expansive constant ε 계산 ─────────────────────────────────────────
    # task_1 결과 파일에서 서로 다른 클래스 쌍의 최솟값(ε)과 해당 이미지 쌍을 탐색.
    # task_1은 클래스 key별로 Result/task1/{DATA_NAME}_{model_tag}_Class_{key}_prob.npy 저장.
    # 각 파일의 shape: (n_class, n_class, n_blocks, n_samples_i, n_samples_j)
    # 여기서는 블록별 fc 출력(10-dim)을 사용하는 경우에만 의미 있으므로
    # USE_FC_OUTPUT=True 일 때 실행을 권장하지만, raw 특징에도 동작함.
    print("\n[ε] Expansive constant 계산 시작...")
    task1_path = "Result/task1"
    y_np = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    labels_unique = np.unique(y_np)

    epsilon     = float("inf")
    eps_cls_a   = None   # 클래스 A
    eps_cls_b   = None   # 클래스 B
    eps_idx_a   = None   # 클래스 A 내 샘플 인덱스
    eps_idx_b   = None   # 클래스 B 내 샘플 인덱스
    eps_block   = None   # 최솟값이 나타난 블록

    for cls_a in labels_unique:
        fname = os.path.join(task1_path, f"{DATA_NAME}_{model_tag}_Class_{cls_a}_prob.npy")
        if not os.path.exists(fname):
            continue
        arr = np.load(fname, allow_pickle=True)
        # arr shape 확인: task_1 저장 형태에 따라 달라질 수 있음
        # tmp_ shape: (n_class, n_class, n_blocks, n_samp_a, n_samp_b) 또는 유사
        # task_1 코드에서 key별로 key2를 쌓으므로 dim-0=key2, dim-1 이후 블록/샘플 정보
        # 실제 배열 구조를 그대로 탐색:  모든 원소 중 클래스-간 최솟값 탐색
        for cls_b in labels_unique:
            if cls_b == cls_a:
                continue
            # arr의 key2 인덱스 순서가 sorted(key_list) 이므로 cls_b의 위치 = sorted index
            b_idx_sorted = sorted(labels_unique.tolist()).index(int(cls_b))
            try:
                sub = arr[b_idx_sorted]   # (n_class, n_blocks, n_samp_a) or similar
            except IndexError:
                continue
            sub_arr = np.array(sub, dtype=float)
            flat_min = np.nanmin(sub_arr)
            if flat_min < epsilon:
                epsilon   = flat_min
                eps_cls_a = int(cls_a)
                eps_cls_b = int(cls_b)
                flat_loc  = np.unravel_index(np.nanargmin(sub_arr), sub_arr.shape)
                eps_block = int(flat_loc[0]) if sub_arr.ndim > 1 else 0
                # 클래스별 전역 샘플 인덱스 복원
                cls_a_global = np.where(y_np == cls_a)[0]
                cls_b_global = np.where(y_np == cls_b)[0]
                if sub_arr.ndim >= 2:
                    eps_idx_a = int(cls_a_global[flat_loc[-2]]) if flat_loc[-2] < len(cls_a_global) else None
                    eps_idx_b = int(cls_b_global[flat_loc[-1]]) if flat_loc[-1] < len(cls_b_global) else None
                else:
                    eps_idx_a = eps_idx_b = None

    print(f"[ε] Expansive constant ε = {epsilon:.6f}")
    print(f"    클래스 쌍 : {eps_cls_a} vs {eps_cls_b}")
    print(f"    블록      : {eps_block}")
    print(f"    샘플 인덱스: {eps_idx_a} (class {eps_cls_a})  ↔  {eps_idx_b} (class {eps_cls_b})")

    eps_result = {
        "epsilon":   epsilon,
        "class_a":   eps_cls_a,
        "class_b":   eps_cls_b,
        "block":     eps_block,
        "sample_a":  eps_idx_a,
        "sample_b":  eps_idx_b,
    }
    os.makedirs("Result", exist_ok=True)
    np.save(f"Result/{DATA_NAME}_{model_tag}_epsilon.npy", eps_result)
    print(f"[ε] 저장 완료: Result/{DATA_NAME}_{model_tag}_epsilon.npy")

    # ── 정리 계산: Shg(φ) / Lip(g) → Tg(φ) 하한 ─────────────────────────
    # main.py에서 USE_BLOCK_FC=True로 학습하면 "{model_tag}_{DATA_NAME}_multifc.pt"가 저장됨.
    multifc_ckpt = f"{model_tag}_{DATA_NAME}_multifc.pt"
    if USE_FC_OUTPUT and os.path.exists(multifc_ckpt):
        theorem_result = analyze_theorem(
            d_name      = DATA_NAME,
            model       = model_tag,
            ckpt_path   = multifc_ckpt,
            feat_dim    = 2048 * 7 * 14,   # 블록 특징 차원 (DS-ResNet 공통)
            n_class     = 10,
            layers      = ds_layers,
            use_avgpool = USE_AVGPOOL,
            save_path   = "Result",
        )
    else:
        print("\n[정리] block_fc 체크포인트가 없거나 USE_FC_OUTPUT=False.")
        print(f"       main.py에서 MODEL_NAME='{MODEL_NAME}', USE_BLOCK_FC=True로 학습 후"
              " dist_calc.py를 다시 실행하세요.")

    sys.exit()
    dim.class_wise(data_name, result_)
    old_task1 = task_old1(n_class, data_name, result_)

    f_name = f"Task_1_{data_name}"
    if not os.path.exists(f_name+'_9.npy'):
        dim.task_1(f_name)
    for key in np.arange(10):
        dist_vals = np.load(f'{f_name}_{key}.npy', allow_pickle=True)
        print(dist_vals.shape)
        print(np.max(dist_vals, 3).shape)
    
    if os.path.exists(f"Targets_{data_name}.npy"):
        seqs = task2_read(data_name)
    else:
        seqs = seq_builder(feats, labels)
    # distance = minkovski(seqs[0], seqs[1], 2)

    seq_max = np.max(seqs[3], axis=1)
    seq_mean = np.mean(seqs[3], axis=1)
    print(seq_max)
    print(seq_mean)
    # delta
    d1 = np.min(seq_max)
    d2 = np.min(seq_mean)

    if os.path.exists(f"Task2_{data_name}_best_max.npy"):
        best_stack = np.load(f"Task2_{data_name}_best_max.npy", allow_pickle=True)
        best_stack_mean = np.load(f"Task2_{data_name}_best_mean.npy", allow_pickle=True)
    else:
        best_stack, best_stack_mean = task_2(seqs, f"Task2_{data_name}")
    best = np.argsort(best_stack[:, 1])[0]
    best_mean = np.argsort(best_stack_mean[:, 1])[0]
    # epsilon
    e1 = np.min(best_stack[:, 1])
    e1s = np.argsort(best_stack[:, 1])[:10]
    e2 = np.min(best_stack_mean[:, 1])
    e2s = np.argsort(best_stack_mean[:, 1])[:10]

    tos = np.ndarray((2, 3), dtype=object)
    tos[0] = np.array([1, e1, d1])
    tos[1] = np.array([2, e2, d2])
    np.save("epdel12", tos)







