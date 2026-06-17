from utils.norms import softmax, sigmoid, init_random
from utils.distance import minkovski
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def seq_builder(raw, d_name, model, n_blocks, labels=None, allow_cross_class=False):
    """nearest-neighbor 체인(pseudo-orbit) 생성.

    Parameters
    ----------
    labels : array-like, optional
        각 샘플의 클래스 레이블. None이면 클래스 무시.
    allow_cross_class : bool
        False (기본) : 같은 클래스 내에서만 이웃 탐색
        True         : 클래스 무관하게 탐색 (어디서 클래스가 섞이는지 관찰용)

    Output (out 리스트)
    -------------------
    out[0] Targets   (N, n_blocks, feat_dim)  – 각 샘플의 원본 블록 특징
    out[1] Series    (N, n_blocks+1, feat_dim) – 체인의 block-0 특징 시퀀스
    out[2] SeqInfo   (N, n_blocks+1)           – 체인에 포함된 샘플 인덱스
    out[3] MaxList   (N, n_blocks)             – 체인 각 단계의 거리
    out[4] ClassInfo (N, n_blocks+1)           – 체인 각 단계의 클래스 (labels 제공 시)
    """
    out = [None, None, None, None, None]

    raw = softmax(raw)

    if labels is not None:
        labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    else:
        labels_np = None

    for idx, rows in enumerate(raw):
        pick = rows
        current_class = int(labels_np[idx]) if labels_np is not None else None

        # valid_mask: True = 이웃 후보로 사용 가능
        valid_mask = np.ones(len(raw), dtype=bool)
        valid_mask[idx] = False                           # 자기 자신 제외
        if labels_np is not None and not allow_cross_class:
            valid_mask[labels_np != current_class] = False  # 다른 클래스 제외

        if out[0] is None:
            out[0] = pick.reshape(1, pick.shape[0], pick.shape[1])
        else:
            out[0] = np.concatenate((out[0], pick.reshape(1, pick.shape[0], pick.shape[1])), 0)

        temp        = pick[0].reshape(1, 1, -1)
        temp_seqs   = [idx]
        temp_classes = [current_class] if labels_np is not None else []
        maxd        = []

        for _ in range(n_blocks):
            valid_indices = np.where(valid_mask)[0]
            raw_valid     = raw[valid_indices, 0, :]          # (n_valid, feat_dim)

            dist      = minkovski(pick[1].reshape(1, 1, -1), raw_valid, 2)
            dist_np   = dist.detach().numpy() if hasattr(dist, 'detach') else np.array(dist)
            local_loc = int(np.argmin(dist_np))
            loc       = int(valid_indices[local_loc])

            temp = np.concatenate((temp, raw[loc, 0].copy().reshape(1, 1, -1)), 1)
            maxd.append(float(dist_np.flatten()[local_loc]))
            temp_seqs.append(loc)
            if labels_np is not None:
                temp_classes.append(int(labels_np[loc]))
            valid_mask[loc] = False   # 방문한 샘플 제외
            pick = raw[loc]

        if out[1] is None:
            out[1] = temp
            out[2] = np.array(temp_seqs).reshape(1, -1)
            out[3] = np.array(maxd).reshape(1, -1)
            if labels_np is not None:
                out[4] = np.array(temp_classes).reshape(1, -1)
        else:
            out[1] = np.concatenate((out[1], temp), 0)
            out[2] = np.concatenate((out[2], np.array(temp_seqs).reshape(1, -1)), 0)
            out[3] = np.concatenate((out[3], np.array(maxd).reshape(1, -1)), 0)
            if labels_np is not None:
                out[4] = np.concatenate((out[4], np.array(temp_classes).reshape(1, -1)), 0)

    np.save(f"task2/{d_name}_{model}_Targets.npy",  out[0])
    np.save(f"task2/{d_name}_{model}_Series.npy",   out[1])
    np.save(f"task2/{d_name}_{model}_SeqInfo.npy",  out[2])
    np.save(f"task2/{d_name}_{model}_MaxList.npy",  out[3])
    if labels_np is not None and out[4] is not None:
        np.save(f"task2/{d_name}_{model}_ClassInfo.npy", out[4])

    return out


def create_dict(data, labels, norm=None):
    out = dict()
    feat_dim = data.shape[2]
    n_blocks = data.shape[1]
    if norm == 'softmax':
        for x, y in zip(data, labels.detach().numpy()):
            if y not in out.keys():
                out[y] = softmax(x).reshape(1, n_blocks, feat_dim)
            else:
                out[y] = np.concatenate((out[y], softmax(x).reshape(1, n_blocks, feat_dim)), 0)
    elif norm == 'sigmoid':
        for x, y in zip(data, labels):
            if y not in out.keys():
                out[y] = sigmoid(x).reshape(1, n_blocks, feat_dim)
            else:
                out[y] = np.concatenate((out[y], sigmoid(x).reshape(1, n_blocks, feat_dim)), 0)
    else:
        for x, y in zip(data, labels):
            if y not in out.keys():
                out[y] = x.reshape(1, n_blocks, feat_dim)
            else:
                out[y] = np.concatenate((out[y], x.reshape(1, n_blocks, feat_dim)), 0)

    return out


def load_files(path, n_blocks, n_feats=None, hist=False, set_name="/", md_name="", stat=""):
    if hist:
        hist_path = 'Result/hist'
        if not os.path.exists(hist_path):
            os.makedirs(hist_path)
    dir_name = path + stat + f"/{set_name}_{md_name}/"
    label = np.load(f"D:/aw_ext/{set_name}_{md_name}/{set_name}_{md_name}_test_label_raw.npy", allow_pickle=True)
    for i in range(n_blocks + 1):
        if stat == "probs":
            for cc in range(100):
                test_f = f"{set_name}_{md_name}_b{i}_feat_prob_{cc}.npy"
                if cc == 0:
                    tmp_data = np.load(dir_name + test_f, allow_pickle=True)
                else:
                    tmp_data = np.concatenate((tmp_data, np.load(dir_name + test_f, allow_pickle=True)), 0)
        else:
            if (set_name == "MNIST" and md_name == "resnet101") or (set_name == "MNIST" and md_name == "resnet50"):
                test_f = f"{set_name}_{md_name}_test_{i}_feat_raw.npy"
                tmp_data = np.load(dir_name + test_f, allow_pickle=True)[-10000:]
                tmp_data = tmp_data.reshape(10000, -1)
            else:
                for cc in range(100):
                    test_f = f"{set_name}_{md_name}_test_{i}_feat_raw_{cc}.npy"
                    if cc == 0:
                        tmp_data = np.load(dir_name + test_f, allow_pickle=True)
                    else:
                        tmp_data = np.concatenate((tmp_data, np.load(dir_name + test_f, allow_pickle=True)), 0)

        if i == 0:
            if n_feats is None:
                n_feats = tmp_data.shape[-1]
            data_cat = tmp_data.reshape((-1, 1, n_feats))
        else:
            data_cat = np.concatenate((data_cat, tmp_data.reshape((-1, 1, n_feats))), axis=1)

        if hist:
            clist = ["plum", "darkslateblue", "rosybrown", "darkkhaki", "darkseagreen",
                     "darkcyan", "cadetblue", "deeppink", "greenyellow", "crimson"]
            init_random(13)
            perp = 40
            tsne = TSNE(perplexity=perp)
            tt = tsne.fit_transform(tmp_data, label)
            for y in np.sort(np.unique(label))[::-1]:
                lt = np.asarray(label == y).nonzero()
                lt = lt[0]
                plt.scatter(tt[lt, 0], tt[lt, 1], c=clist[int(y)], s=3)
            plt.savefig(f"{hist_path}_{set_name}_Block_{stat}_{i}")
            plt.cla()
            plt.clf()

    return data_cat, label
