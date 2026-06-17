"""
Computes Shg(φ) and Lip(g) for the theorem:

    Shg(φ) ≤ Lip(g) · Tg(φ)

which gives the lower bound:

    Tg(φ) ≥ Shg(φ) / Lip(g)

- φ  : ResNet (dynamic system, implemented as Exprob)
- g  : block-wise fc layers (block_fc) mapping features → probabilities
- Shg(φ): maximum local error across all pseudo-orbits (from seq_builder MaxList)
- Lip(g): Lipschitz constant of g = spectral norm of fc weight matrix
- Tg(φ): lower bound derived from the theorem
"""

import numpy as np
import torch
import os


# ── Lip(g) ────────────────────────────────────────────────────────────────────

def spectral_norm_fc(fc_layer):
    """단일 fc 레이어의 Lipschitz 상수 = 가중치 행렬의 최대 특이값(spectral norm).

    y = Wx + b 에서 ||g(x) - g(y)|| ≤ sigma_max(W) · ||x - y||
    따라서 Lip(g) = sigma_max(W).
    """
    W = fc_layer.weight.detach().float()          # (n_class, feat_dim)
    # SVD: W = U Σ V^T, 특이값 Σ는 내림차순
    sv = torch.linalg.svd(W, full_matrices=False).S
    return sv[0].item()                            # 최대 특이값


def lip_g(extractor):
    """모든 block_fc 레이어의 Lipschitz 상수 목록과 전체 최댓값을 반환.

    Parameters
    ----------
    extractor : Exprob
        multi_fc=True 로 생성된 추출기 (16개 block_fc 포함).

    Returns
    -------
    lip_max  : float  — 전체 블록 중 최대 Lipschitz 상수 (= Lip(g))
    lip_list : list   — 블록별 Lipschitz 상수 [block0, block1, ..., block15]
    """
    if not hasattr(extractor, 'block_fc'):
        raise ValueError(
            "extractor에 block_fc가 없습니다. Exprob(multi_fc=True)로 생성하세요."
        )
    lip_list = [spectral_norm_fc(fc) for fc in extractor.block_fc]
    return max(lip_list), lip_list


def lip_g_from_checkpoint(ckpt_path, feat_dim, n_class, layers=None,
                          use_avgpool=False, device='cpu'):
    """저장된 체크포인트에서 Lip(g)를 계산.

    A100에서 학습한 뒤 여기에서 불러와 계산할 때 사용.

    Parameters
    ----------
    ckpt_path   : str   — .pt 파일 경로 (예: "ds_resnet18_MNIST_multifc.pt")
    feat_dim    : int   — 블록 특징 차원 (예: 200704)
    n_class     : int   — 클래스 수 (MNIST = 10)
    use_avgpool : bool  — main.py 학습 시 사용한 USE_AVGPOOL 값과 반드시 일치해야 함
                  (main fc의 in_features가 달라져 load_state_dict 시 shape mismatch 발생)
    """
    from utils.stubs import Exprob
    if layers is None:
        layers = [3, 4, 6, 3]
    ext = Exprob(feat_dim, n_class, layers=layers, multi_fc=True,
                use_avgpool=use_avgpool)
    state = torch.load(ckpt_path, map_location=device)
    ext.load_state_dict(state, strict=False)
    ext.eval()
    return lip_g(ext)


# ── Shg(φ) ────────────────────────────────────────────────────────────────────

def shg_phi(maxlist):
    """pseudo-orbit들의 최대 오차 Shg(φ) 계산.

    seq_builder가 저장한 MaxList (N, n_blocks) 배열로부터:
      - 각 샘플 i의 체인 최대 오차 = max over blocks of MaxList[i, :]
      - Shg(φ) = 전체 샘플 중 최댓값

    Parameters
    ----------
    maxlist : np.ndarray, shape (N, n_blocks)
        seq_builder의 out[3] 또는 task2/{d_name}_{model}_MaxList.npy

    Returns
    -------
    shg       : float        — 전체 Shg(φ)
    per_chain : np.ndarray   — 샘플별 체인 최대 오차 (N,)
    """
    maxlist = np.array(maxlist, dtype=float)
    per_chain = np.max(maxlist, axis=1)      # 각 체인의 최악 단계 오차
    shg = float(np.max(per_chain))           # 모든 체인 중 최댓값
    return shg, per_chain


def shg_phi_from_file(d_name, model, path='task2'):
    """저장된 MaxList .npy 파일에서 Shg(φ)를 계산."""
    fpath = os.path.join(path, f"{d_name}_{model}_MaxList.npy")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"MaxList 파일 없음: {fpath}\n"
                                "dist_calc.py를 먼저 실행해 seq_builder를 완료하세요.")
    maxlist = np.load(fpath, allow_pickle=True)
    return shg_phi(maxlist)


# ── 정리 하한 계산 ────────────────────────────────────────────────────────────

def theorem_lower_bound(shg, lip):
    """Tg(φ)의 하한 계산.

    정리:  Shg(φ) ≤ Lip(g) · Tg(φ)
    따라서: Tg(φ) ≥ Shg(φ) / Lip(g)

    Parameters
    ----------
    shg : float — Shg(φ)
    lip : float — Lip(g)

    Returns
    -------
    lower_bound : float — Tg(φ)의 하한선
    """
    if lip == 0:
        raise ValueError("Lip(g) = 0: g가 상수 함수입니다.")
    return shg / lip


# ── 통합 분석 함수 ─────────────────────────────────────────────────────────────

def analyze_theorem(d_name, model, extractor=None, ckpt_path=None,
                    feat_dim=200704, n_class=10, layers=None, use_avgpool=False,
                    maxlist_path='task2', save_path='Result', verbose=True):
    """Shg(φ), Lip(g), Tg(φ) 하한을 한 번에 계산하고 출력.

    두 가지 사용 방식:
      (A) 이미 로드된 extractor 전달
      (B) ckpt_path 전달 → 내부에서 로드

    Parameters
    ----------
    extractor   : Exprob (multi_fc=True) 또는 None
    ckpt_path   : str — "ds_resnet18_MNIST_multifc.pt" 등
    layers      : list — ckpt_path 사용 시 main.py와 동일한 DS_LAYERS 값 필요
    use_avgpool : bool — ckpt_path 사용 시 main.py의 USE_AVGPOOL 값과 일치해야 함
    maxlist_path: str — MaxList .npy가 저장된 디렉토리
    save_path   : str — 결과 .npy 저장 위치
    """
    # ── Shg(φ) ──────────────────────────────────────────────────────────────
    shg, per_chain = shg_phi_from_file(d_name, model, path=maxlist_path)

    # ── Lip(g) ──────────────────────────────────────────────────────────────
    if extractor is not None:
        lip_max, lip_list = lip_g(extractor)
    elif ckpt_path is not None:
        lip_max, lip_list = lip_g_from_checkpoint(
            ckpt_path, feat_dim, n_class, layers=layers, use_avgpool=use_avgpool)
    else:
        raise ValueError("extractor 또는 ckpt_path 중 하나를 제공하세요.")

    # ── Tg(φ) 하한 ──────────────────────────────────────────────────────────
    tg_lb = theorem_lower_bound(shg, lip_max)

    if verbose:
        print("=" * 55)
        print("  정리:  Shg(φ) ≤ Lip(g) · Tg(φ)")
        print("=" * 55)
        print(f"  Shg(φ)        = {shg:.6f}  (pseudo-orbit 최대 오차)")
        print(f"  Lip(g)        = {lip_max:.6f}  (fc 레이어 spectral norm 최댓값)")
        print(f"  Tg(φ) 하한선  ≥ {tg_lb:.6f}  (= Shg / Lip)")
        print("-" * 55)
        print("  블록별 Lip(g_b):")
        for i, l in enumerate(lip_list):
            print(f"    block {i:02d}: {l:.4f}")
        print("=" * 55)

    result = {
        "Shg_phi":        shg,
        "per_chain_max":  per_chain,
        "Lip_g":          lip_max,
        "Lip_g_per_block": np.array(lip_list),
        "Tg_phi_lower_bound": tg_lb,
    }

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"{d_name}_{model}_theorem.npy"), result)
    if verbose:
        print(f"  저장: {save_path}/{d_name}_{model}_theorem.npy")

    return result
