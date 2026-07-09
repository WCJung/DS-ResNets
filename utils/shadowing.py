"""
g-shadowing (논문 Definition 2) — pseudo-orbit 생성, 추적(tracing), Sh_g 추정.

이론 요약
---------
(delta, g)-pseudo-orbit : d_g(g phi(x_n), g(x_{n+1})) <= delta 를 만족하는 샘플열.
eps-추적됨               : 어떤 진짜 궤도 x가 존재해 모든 t에서
                           d_g(g phi^t(x), g(x_t)) <= eps.
Sh_g(phi) = liminf_{eps->0}  sup{ delta : 모든 delta-pseudo-orbit이 eps-추적됨 } / eps
            (Morales & Nguyen 2023의 양적 shadowing 상수)

경험적 추정 절차
----------------
1) build_pseudo_orbits : 각 샘플에서 시작하는 최근접-이웃 체인을 만들고,
   체인 i의 스텝 오차의 최댓값을 delta_i 로 둔다.
2) trace_orbits        : 체인 i를 가장 잘 추적하는 진짜 궤도를 찾아
   eps_i = min_x max_t d(traj[x, t], traj[s_{i,t}, t]) 를 계산한다.
   (max over t — 추적은 "모든 블록에서" 가까워야 하므로 sup 오차.
    구버전은 블록에 대해 min을 취해 정의와 달랐다.)
3) shadowing_constant  : eps 그리드에 대해
   delta*(eps) = min{ delta_i : eps_i > eps }   (eps-추적 안 되는 체인이 생기기
                                                 직전까지 허용되는 delta)
   Sh_g(eps) = delta*(eps) / eps 곡선을 만들고, 가장 작은 eps에서의 값을
   Sh_g 추정치로 보고한다 (liminf의 경험적 대응).

체인 생성 모드
--------------
depth_consistent=True (기본): 스텝 t에서 현재 샘플의 "블록 t 표현"과 후보들의
  블록 t 표현을 비교 — 논문의 '각 residual block을 통과하며' 서사에 대응.
depth_consistent=False      : 구버전 방식. 항상 블록 0 -> 블록 1 전이만 사용
  (one-parameter 단순화 Phi = {phi}에 대응).
"""
import os
import numpy as np
import torch


@torch.no_grad()
def build_pseudo_orbits(traj, labels, allow_cross_class=False,
                        depth_consistent=True, chunk=2048, device=None):
    """모든 샘플에서 시작하는 최근접-이웃 pseudo-orbit 체인을 벡터화 생성.

    Returns
    -------
    seq      : torch.LongTensor (N, T)   — 체인을 구성하는 샘플 인덱스 (traj 기준)
    step_err : torch.FloatTensor (N, T-1) — 스텝별 오차 (d_g 공간의 delta 후보)
    """
    N, T, _ = traj.shape
    if device is not None:
        traj = traj.to(device)
        labels = labels.to(device)

    seq = torch.empty((N, T), dtype=torch.long)
    step_err = torch.empty((N, T - 1))
    inf = float('inf')

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        C = e - s
        rows = torch.arange(s, e, device=traj.device)
        ar = torch.arange(C, device=traj.device)

        # invalid[c, j] = True → 체인 c의 다음 멤버 후보에서 j 제외
        if allow_cross_class:
            invalid = torch.zeros((C, N), dtype=torch.bool, device=traj.device)
        else:
            invalid = labels[rows].unsqueeze(1) != labels.unsqueeze(0)
        invalid[ar, rows] = True                       # 자기 자신 제외

        cur = rows.clone()
        seq[s:e, 0] = rows.cpu()
        for t in range(1, T):
            q_t = t if depth_consistent else 1         # 현재 상태(phi 적용 후)의 블록
            k_t = t if depth_consistent else 0         # 후보 비교 블록
            Dm = torch.cdist(traj[cur, q_t], traj[:, k_t])      # (C, N)
            Dm = torch.where(invalid, torch.full_like(Dm, inf), Dm)
            v, nxt = Dm.min(dim=1)
            seq[s:e, t] = nxt.cpu()
            step_err[s:e, t - 1] = v.cpu()
            invalid[ar, nxt] = True                    # 방문한 샘플 제외
            cur = nxt

    return seq, step_err


@torch.no_grad()
def trace_orbits(traj, seq, labels, same_class_only=True, chunk=512, device=None):
    """각 pseudo-orbit 체인을 가장 잘 추적하는 진짜 궤도를 탐색.

    체인 i의 추적 오차:  eps_i = min_x  max_t  d(traj[x, t], traj[s_{i,t}, t])
    (논문 결론부의 '같은 클래스로 추적' 가정은 same_class_only=True로 반영 —
     후보 x를 체인 시작 샘플과 같은 클래스로 제한한다.)

    Returns
    -------
    eps    : torch.FloatTensor (N,) — 체인별 최적 추적 오차
    tracer : torch.LongTensor (N,)  — 그 오차를 달성한 진짜 궤도의 샘플 인덱스
    """
    N, T, _ = traj.shape
    if device is not None:
        traj = traj.to(device)
        labels = labels.to(device)
    seq_d = seq.to(traj.device)

    eps = torch.empty(N)
    tracer = torch.empty(N, dtype=torch.long)
    inf = float('inf')

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        E = torch.zeros((N, e - s), device=traj.device)         # 후보 x × 체인
        for t in range(T):
            Dm = torch.cdist(traj[:, t], traj[seq_d[s:e, t], t])  # (N, C)
            E = torch.maximum(E, Dm)
        if same_class_only:
            bad = labels.unsqueeze(1) != labels[seq_d[s:e, 0]].unsqueeze(0)
            E = torch.where(bad, torch.full_like(E, inf), E)
        v, arg = E.min(dim=0)
        eps[s:e] = v.cpu()
        tracer[s:e] = arg.cpu()

    return eps, tracer


def shadowing_constant(step_err, eps, quantiles=(0.0, 0.01, 0.05, 0.10, 0.25, 0.50)):
    """(delta_i, eps_i) 쌍들로부터 Sh_g 추정 곡선과 대표값을 계산.

    Parameters
    ----------
    step_err  : (N, T-1) — build_pseudo_orbits의 스텝 오차
    eps       : (N,)     — trace_orbits의 체인별 추적 오차
    quantiles : eps 그리드로 사용할 분위수들 (0.0 = 최소 eps)

    Returns
    -------
    dict:
      Sh_g       : float — 가장 작은 eps 그리드점에서의 delta*/eps (대표 추정치)
                   관측 공간이 붕괴하면 nan (degenerate=True)
      eps0       : float — 그 eps 값
      delta_star : float — 그 지점의 delta*
      curve      : list of (eps, delta_star, ratio)
      delta_per_chain, eps_per_chain : np.ndarray — 원자료
      degenerate : bool  — True면 모든 추적 오차가 0이라 Sh_g 정의 불가
      note       : str 또는 None — 그리드 폴백 등 참고 사항
    """
    step_err = np.asarray(step_err, dtype=float)
    eps = np.asarray(eps, dtype=float)
    delta = step_err.max(axis=1)                       # 체인별 최대 스텝 오차

    def _curve_from(grid):
        curve = []
        for e in grid:
            if e <= 0:
                continue                               # 0으로 나눌 수 없는 지점 제외
            not_shadowed = eps > e
            if not_shadowed.any():
                d_star = float(delta[not_shadowed].min())
            else:
                d_star = float(delta.max())            # 전부 추적됨 → 관측 최대 delta
            curve.append((e, d_star, d_star / e))
        return curve

    qs = sorted(quantiles)
    curve = _curve_from(float(np.quantile(eps, q)) for q in qs)

    # 폴백: 지정 분위수가 전부 0에 걸렸지만 양수 추적 오차가 존재하면
    # (부분 붕괴 — 0이 과반인 경우 등) 양수 부분집합의 분위수로 그리드 재구성
    note = None
    if not curve:
        pos = eps[eps > 0]
        if pos.size:
            curve = _curve_from(float(np.quantile(pos, q)) for q in qs)
            zero_pct = 100.0 * (1 - pos.size / eps.size)
            note = (f"추적 오차의 {zero_pct:.1f}%가 0 — 양수 부분집합"
                    f"({pos.size}/{eps.size}개 체인) 분위수로 그리드 대체")

    # 완전 붕괴: 모든 체인의 추적 오차가 0 → Sh_g 정의 불가.
    # 예외 대신 nan을 반환해 호출측(run_all 등)이 다음 조합으로 진행하게 한다.
    if not curve:
        return {
            "Sh_g": float('nan'), "eps0": float('nan'),
            "delta_star": float('nan'), "curve": [],
            "delta_per_chain": delta, "eps_per_chain": eps,
            "degenerate": True,
            "note": ("모든 추적 오차가 0 — 관측 공간 붕괴. "
                     "--space logit으로 재시도하거나 block_fc probe를 재학습하세요."),
        }

    e0, d0, r0 = curve[0]
    return {
        "Sh_g": r0,
        "eps0": e0,
        "delta_star": d0,
        "curve": curve,
        "delta_per_chain": delta,
        "eps_per_chain": eps,
        "degenerate": False,
        "note": note,
    }


def save_orbit_files(d_name, model_tag, seq, step_err, labels, eps, orig_idx,
                     path='task2'):
    """inspect_examples.py / print_results.py 호환 포맷으로 저장.

    seq는 traj(서브샘플) 기준 인덱스이므로 orig_idx로 원본 테스트셋 인덱스로
    되돌려 저장한다 (전체 샘플 사용 시 동일).
    """
    os.makedirs(path, exist_ok=True)
    seq_np = np.asarray(seq)
    orig = np.asarray(orig_idx)
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.asarray(labels)

    np.save(f"{path}/{d_name}_{model_tag}_SeqInfo.npy", orig[seq_np])
    np.save(f"{path}/{d_name}_{model_tag}_MaxList.npy", np.asarray(step_err))
    np.save(f"{path}/{d_name}_{model_tag}_ClassInfo.npy", labels_np[seq_np])
    np.save(f"{path}/{d_name}_{model_tag}_TraceEps.npy", np.asarray(eps))
