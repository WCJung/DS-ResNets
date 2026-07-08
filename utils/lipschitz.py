"""
Lip(g) 계산 — 정리 1  Sh_g(phi) <= Lip(g) * T_g(phi)  의 우변 상수.

g가 단일 선형층(block_fc)이면 Lipschitz 상수는 가중치 행렬의 최대 특이값
sigma_max(W)와 정확히 일치한다 (LipSDP 같은 상한 추정이 필요 없다).

관측 공간(space)에 따른 보정:
  space='logit' : d_g가 logit 공간 거리 → Lip(g) = sigma_max(W)  (정확)
  space='prob'  : d_g가 softmax 확률 공간 거리 → g = softmax ∘ fc.
                  softmax의 l2-Lipschitz 상수는 1/2 이하이므로
                  Lip(g) <= 0.5 * sigma_max(W)  를 사용.

main fc(분류 헤드)에 대한 참고 정보도 함께 계산한다:
  use_avgpool=True로 학습된 경우 g_main = fc ∘ avgpool 이고, k개 원소 평균의
  l2-Lipschitz 상수는 1/sqrt(k)이므로  Lip(g_main) = sigma_max(W_fc)/sqrt(k).
  (k = 블록 특징 차원 / fc 입력 차원, 예: 200704/2048 = 98)

체크포인트 state_dict에서 직접 읽으므로 모델 인스턴스화가 필요 없고,
feat_dim/use_avgpool 불일치로 인한 shape mismatch 문제도 발생하지 않는다.
"""
import math
import re

import torch


def spectral_norm(weight):
    """가중치 행렬의 최대 특이값 = 선형층의 정확한 l2-Lipschitz 상수."""
    sv = torch.linalg.svd(weight.detach().float(), full_matrices=False).S
    return float(sv[0])


def spectral_norm_fc(fc_layer):
    """nn.Linear 레이어의 Lipschitz 상수."""
    return spectral_norm(fc_layer.weight)


def lip_report_from_checkpoint(ckpt_path, space='prob', device='cpu'):
    """multifc 체크포인트에서 블록별 sigma_max와 유효 Lip(g)를 계산.

    Parameters
    ----------
    ckpt_path : str — "{model_tag}_{data_name}_multifc.pt"
    space     : 'prob' | 'logit' | 'feat' — 거리를 계산한 관측 공간과 일치시킬 것

    Returns
    -------
    dict:
      sigma_per_block : {block_idx: sigma_max}
      sigma_max       : float — 블록 fc 중 최댓값
      softmax_factor  : float — space='prob'이면 0.5, 아니면 1.0
      Lip_g           : float — sigma_max * softmax_factor  (Table 1의 Lip(g))
      main_fc         : dict 또는 None — 분류 헤드 참고 정보
                        {sigma_max, avgpool_k, avgpool_factor, Lip}
    """
    state = torch.load(ckpt_path, map_location=device)

    sigma_per_block = {}
    for key, w in state.items():
        m = re.fullmatch(r"block_fc\.(\d+)\.weight", key)
        if m:
            sigma_per_block[int(m.group(1))] = spectral_norm(w)
    if not sigma_per_block:
        raise ValueError(
            f"{ckpt_path}에 block_fc 가중치가 없습니다. "
            "main.py를 --use-block-fc로 학습한 multifc 체크포인트인지 확인하세요."
        )

    sigma_max = max(sigma_per_block.values())
    softmax_factor = 0.5 if space == 'prob' else 1.0
    lip_g = sigma_max * softmax_factor

    # ── main fc 참고 정보 (avgpool 보정 포함) ───────────────────────────
    main_fc = None
    if 'fc.weight' in state:
        fc_sigma = spectral_norm(state['fc.weight'])
        fc_in = state['fc.weight'].shape[1]
        block_in = state[f"block_fc.{min(sigma_per_block)}.weight"].shape[1]
        k = block_in // fc_in if fc_in and block_in % fc_in == 0 else 1
        avgpool_factor = 1.0 / math.sqrt(k) if k > 1 else 1.0
        main_fc = {
            "sigma_max": fc_sigma,
            "avgpool_k": k,
            "avgpool_factor": avgpool_factor,
            "Lip": fc_sigma * avgpool_factor * softmax_factor,
        }

    return {
        "sigma_per_block": sigma_per_block,
        "sigma_max": sigma_max,
        "softmax_factor": softmax_factor,
        "Lip_g": lip_g,
        "main_fc": main_fc,
    }
