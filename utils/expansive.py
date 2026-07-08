"""
g-expansive 상수 (논문 Definition 1).

정의: g(x) != g(x')이면 어떤 블록 n >= 0에서 d_g(phi^n x, phi^n x') >= eps.

따라서 유효한 "최대" eps는

    eps = min_{서로 다른 클래스 쌍 (x, x')}   max_{블록 t}   d(g_t(x), g_t(x'))

즉 min-max 구조다. (구버전의 전역 min은 min-min이라 정의와 달랐고
eps를 크게 과소평가했다 — 쌍마다 "적어도 하나의 블록에서" 벌어지면 되므로
블록에 대해서는 max를 취해야 한다.)

전 과정이 torch.cdist 기반으로 벡터화되어 있으며, (chunk x N) 크기의
running-max 행렬만 유지하므로 N=10,000에서도 수백 MB 이내로 동작한다.
"""
import torch


@torch.no_grad()
def expansive_constant(traj, labels, chunk=1024, device=None, verbose=True):
    """min-max g-expansive 상수와 해당 쌍/블록 정보를 반환.

    Parameters
    ----------
    traj   : torch.FloatTensor (N, T, D) — 관측 공간 궤적 (utils.trajectory 참조)
    labels : torch.LongTensor (N,)
    chunk  : int — 행 청크 크기 (메모리 상한 조절)
    device : str 또는 None — 'cuda' 지정 시 GPU에서 계산

    Returns
    -------
    dict:
      epsilon  : float — min-max 상수
      sample_a : int   — (traj 기준) 샘플 인덱스 A
      sample_b : int   — 샘플 인덱스 B
      class_a  : int
      class_b  : int
      block    : int   — 해당 쌍의 거리가 최대가 된 블록 (argmax 블록)
    """
    N, T, _ = traj.shape
    if device is not None:
        traj = traj.to(device)
        labels = labels.to(device)

    inf = float('inf')
    best = {"epsilon": inf, "sample_a": None, "sample_b": None,
            "class_a": None, "class_b": None, "block": None}

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        q = traj[s:e]                                  # (C, T, D)
        run_max = None                                 # (C, N) 블록별 거리의 running max
        run_arg = None                                 # (C, N) 그 max가 발생한 블록
        for t in range(T):
            Dm = torch.cdist(q[:, t], traj[:, t])      # (C, N)
            if run_max is None:
                run_max = Dm
                run_arg = torch.zeros_like(Dm, dtype=torch.int8)
            else:
                upd = Dm > run_max
                run_max = torch.where(upd, Dm, run_max)
                run_arg = torch.where(upd, torch.full_like(run_arg, t), run_arg)

        # 서로 다른 클래스 쌍만 남김 (자기 자신은 같은 클래스이므로 자동 제외)
        cross = labels[s:e].unsqueeze(1) != labels.unsqueeze(0)   # (C, N)
        run_max = torch.where(cross, run_max, torch.full_like(run_max, inf))

        v, flat = run_max.flatten().min(dim=0)
        v = float(v)
        if v < best["epsilon"]:
            i_local, j = divmod(int(flat), N)
            i = s + i_local
            best = {
                "epsilon": v,
                "sample_a": int(i),
                "sample_b": int(j),
                "class_a": int(labels[i]),
                "class_b": int(labels[j]),
                "block": int(run_arg[i_local, j]),
            }
        if verbose and (s // chunk) % 4 == 0:
            print(f"  [eps] {e}/{N} rows...", end='\r')

    if verbose:
        print(f"  [eps] {N}/{N} rows — 완료          ")
    return best
