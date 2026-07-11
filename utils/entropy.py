"""
FTTE (Finite-Time Trajectory Entropy) — 논문 Appendix "Topological entropy".

이론 요약
---------
g-궤적 pseudometric:
    d_g^T(x, y) = max_t d( g_t(phi^t x), g_t(phi^t y) )
E ⊂ X 가 (T, eps)-separated  ⇔  E의 서로 다른 모든 x, y에 대해 d_g^T(x, y) > eps
    s_g^T(eps) = 최대 separated 집합의 크기
    h_g^T(eps) = (1/T) log s_g^T(eps)                       (FTTE)
    h_class    = (1/T) log m                                (Definition 4)
    Δh_g^T(eps) = h_g^T(eps) − h_class                      (Definition 5)
Proposition 2:  Δh > 0 ⇔ s > m,   Δh = 0 ⇔ s = m,   Δh < 0 ⇔ s < m.
Proposition 1:  클래스 내 d_g^T ≤ eps 이고 클래스 간 d_g^T > eps 이면 s = m.

수치 추정
---------
최대 (T, eps)-separated 집합 크기는 NP-hard이므로 greedy maximal packing으로
추정한다 (동역학 수치에서 표준). greedy 결과는 maximal separated set이므로
s_g^T(eps)의 유효한 하한이고, 순회 순서를 고정하면 재현 가능하다.

class_scale_diagnostics는 Proposition 1의 두 조건을 그대로 계산한다:
    intra_max = max_i max_{x,y ∈ X_i} d_g^T(x, y)   (클래스 내 최대)
    cross_min = min_{i≠j} min d_g^T(X_i, X_j)       (클래스 간 최소)
cross_min은 정의상 g-expansive 상수(min-max)와 동일하다.
intra_max ≤ eps < cross_min 인 eps가 존재하면 그 창(window)에서 s = m,
즉 Δh = 0이 이론적으로 보장된다.
"""
import math

import numpy as np
import torch


@torch.no_grad()
def _dT(traj_a, traj_b):
    """궤적 pseudometric 행렬 (A, B): max over t of 블록별 l2 거리."""
    out = None
    for t in range(traj_a.shape[1]):
        Dm = torch.cdist(traj_a[:, t], traj_b[:, t])
        out = Dm if out is None else torch.maximum(out, Dm)
    return out


@torch.no_grad()
def class_scale_diagnostics(traj, labels, chunk=1024, device=None):
    """Proposition 1 진단: intra_max / cross_min / s=m 보장 창."""
    N = traj.shape[0]
    if device is not None:
        traj = traj.to(device)
        labels = labels.to(device)

    intra_max = 0.0
    cross_min = float('inf')
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        R = _dT(traj[s:e], traj)                               # (C, N)
        same = labels[s:e].unsqueeze(1) == labels.unsqueeze(0)
        # 자기 자신(거리 0)은 same에 포함되지만 max에 영향 없음
        if same.any():
            intra_max = max(intra_max, float(R[same].max()))
        if (~same).any():
            cross_min = min(cross_min, float(R[~same].min()))

    window = (intra_max, cross_min) if cross_min > intra_max else None
    return {"intra_max": intra_max, "cross_min": cross_min,
            "prop1_window": window}


@torch.no_grad()
def separated_set_size(traj, eps, chunk=1024, device=None):
    """greedy maximal (T, eps)-separated set의 크기 (s_g^T(eps)의 하한).

    배치 단위 처리: 이미 선택된 집합과 eps 이내인 후보를 걸러낸 뒤,
    배치 내부에서 순차 greedy로 상호 분리를 보장한다.
    """
    N = traj.shape[0]
    if device is not None:
        traj = traj.to(device)

    sel_traj = None            # (S, T, D) 선택된 대표들의 궤적
    count = 0

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        q = traj[s:e]
        ok = torch.ones(e - s, dtype=torch.bool, device=traj.device)
        if sel_traj is not None:
            Dsel = _dT(q, sel_traj)                     # (C, S)
            ok = (Dsel > eps).all(dim=1)
        if not ok.any():
            continue

        sub = q[ok]                                     # 후보만
        Dcc = _dT(sub, sub)                             # 후보 간 상호 거리
        b = sub.shape[0]
        alive = torch.ones(b, dtype=torch.bool, device=traj.device)
        keep = []
        for i in range(b):
            if not alive[i]:
                continue
            keep.append(i)
            alive &= Dcc[i] > eps                       # i와 가까운 후보 제거
        new = sub[keep]
        sel_traj = new if sel_traj is None else torch.cat((sel_traj, new))
        count += len(keep)

    return count


@torch.no_grad()
def _sample_eps_grid(traj, quantiles, n_probe=512, seed=13):
    """무작위 표본 쌍의 d_g^T 분위수로 eps 그리드를 자동 생성."""
    N = traj.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:min(n_probe, N)]
    R = _dT(traj[idx], traj[idx])
    off = R[~torch.eye(len(idx), dtype=torch.bool, device=R.device)]
    vals = [float(torch.quantile(off, q)) for q in quantiles]
    return [v for v in vals if v > 0]


@torch.no_grad()
def find_sm_band(traj, m, rows=None, chunk=1024, device=None,
                 iters=12, rtol=0.005, verbose=True):
    """s_T(eps) = m 이 유지되는 eps 대역 [eps_lo, eps_hi)를 이분탐색으로 추정.

    s_T(eps)는 eps에 대해 단조 비증가이므로 두 경계를 각각 이분탐색한다:
      eps_lo = inf{ eps : s_T(eps) <= m }     (s > m  →  s = m 진입점)
      eps_hi = inf{ eps : s_T(eps) <  m }     (s = m  →  s < m 이탈점)
    eps_lo < eps_hi 이면 그 사이에서 s_T = m — Proposition 1 창
    [intra_max, cross_min)의 경험적 대응물이다 (Prop.1은 충분조건이라
    창이 없어도 이 대역은 존재할 수 있다).

    greedy packing이 s_g^T의 하한 추정이라 경계는 근사값이며, 대역이
    비어 있으면 s_T가 m을 건너뛴 것 (클래스 스케일 구조가 겹침).

    Parameters
    ----------
    rows : ftte_report()["rows"] — 있으면 (eps, s) 캐시/초기 브래킷으로 재사용

    Returns
    -------
    dict:
      exists  : bool  — s_T = m 대역이 발견되었는가
      eps_lo  : float — 대역 시작 (s가 m으로 떨어지는 지점)
      eps_hi  : float — 대역 끝   (s가 m 아래로 떨어지는 지점)
      width   : float — eps_hi - eps_lo (exists=False면 0)
      s_mid   : int 또는 None — 대역 중점에서 검증한 s_T (exists 시 m이어야 정상)
      s_edges : (int, int) — 두 경계 바깥쪽 s 값 (건너뛴 경우 진단용)
      n_evals : int — 추가로 수행한 greedy 평가 횟수
    """
    if device is not None:
        traj = traj.to(device)

    cache = {}
    if rows:
        for r in rows:
            cache[round(float(r["eps"]), 12)] = int(r["s"])
    n_evals = [0]

    def s_at(eps):
        key = round(float(eps), 12)
        if key not in cache:
            cache[key] = separated_set_size(traj, key, chunk=chunk)
            n_evals[0] += 1
        return cache[key]

    def boundary(thresh):
        """inf{ eps : s_T(eps) <= thresh } 의 브래킷 (lo, hi)을 반환.

        s(lo) > thresh, s(hi) <= thresh 를 유지하며 좁힌다.
        """
        known = sorted(cache.items())
        lo = max((e for e, s in known if s > thresh), default=0.0)
        hi = min((e for e, s in known if s <= thresh), default=None)
        if hi is None:                          # 그리드가 위쪽을 못 덮음 → 배가 탐색
            hi = max((e for e, _ in known), default=1.0) or 1.0
            for _ in range(40):
                hi *= 2.0
                if s_at(hi) <= thresh:
                    break
            else:
                return None                     # 이 스케일까지도 s > thresh
        for _ in range(iters):
            if hi - lo <= rtol * hi:
                break
            mid = 0.5 * (lo + hi)
            if s_at(mid) <= thresh:
                hi = mid
            else:
                lo = mid
        return lo, hi

    b_enter = boundary(m)                       # s > m → s <= m
    if b_enter is None:                         # 모든 스케일에서 s > m (비정상)
        return {"exists": False, "eps_lo": float('nan'), "eps_hi": float('nan'),
                "width": 0.0, "s_mid": None, "s_edges": None,
                "n_evals": n_evals[0]}
    b_exit = boundary(m - 1)                    # s >= m → s < m

    eps_lo = b_enter[1]                         # s(eps_lo) <= m 확인된 지점
    eps_hi = b_exit[0] if b_exit is not None else float('inf')
    s_lo, s_hi = s_at(eps_lo), (s_at(eps_hi) if eps_hi != float('inf') else None)

    exists = (eps_hi > eps_lo) and s_lo == m and (s_hi is None or s_hi >= m)
    s_mid = None
    if exists and eps_hi != float('inf'):
        s_mid = s_at(0.5 * (eps_lo + eps_hi))
        exists = s_mid == m

    if verbose:
        if exists:
            w = eps_hi - eps_lo
            print(f"  [FTTE] s_T = m 대역: eps ∈ [{eps_lo:.6f}, {eps_hi:.6f})  "
                  f"폭 {w:.6f}  (중점 s_T = {s_mid if s_mid is not None else m}, "
                  f"greedy 평가 {n_evals[0]}회)")
        else:
            around = (s_at(b_enter[0]), s_at(b_enter[1]))
            print(f"  [FTTE] s_T = m 대역 없음 — s_T가 m={m} 을 건너뛰거나 "
                  f"대역이 탐색 분해능보다 좁음 (경계 부근 s: "
                  f"{around[0]} → {around[1]}, greedy 평가 {n_evals[0]}회)")

    return {"exists": exists,
            "eps_lo": eps_lo if exists else float('nan'),
            "eps_hi": eps_hi if exists else float('nan'),
            "width": (eps_hi - eps_lo) if exists else 0.0,
            "s_mid": s_mid,
            "s_edges": (s_at(b_enter[0]), s_at(b_enter[1])),
            "n_evals": n_evals[0]}


def ftte_report(traj, labels, eps_list=None, chunk=1024, device=None,
                quantiles=(0.001, 0.01, 0.05, 0.10, 0.25, 0.50), verbose=True):
    """FTTE 전체 리포트: eps 그리드별 s_T / h_T / Δh_T + Proposition 1 진단.

    Returns
    -------
    dict:
      T, m, h_class
      diagnostics : class_scale_diagnostics 결과
      rows        : list of dict {eps, s, h, gap, cmp}  (cmp ∈ '>', '=', '<')
      recommended : Prop.1 창의 중점(존재 시) 또는 s가 m에 가장 가까운 그리드점
    """
    N, T, D = traj.shape
    labels_t = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
    m = int(torch.unique(labels_t).numel())
    h_class = math.log(m) / T

    if verbose:
        print(f"  [FTTE] N={N}  T={T}  m={m}  h_class = log(m)/T = {h_class:.4f}")

    diag = class_scale_diagnostics(traj, labels_t, chunk=chunk, device=device)
    if verbose:
        print(f"  [FTTE] intra_max = {diag['intra_max']:.6f}   "
              f"cross_min = {diag['cross_min']:.6e}  (= g-expansive 상수)")
        if diag["prop1_window"]:
            a, b = diag["prop1_window"]
            print(f"  [FTTE] Proposition 1 창: eps ∈ [{a:.6f}, {b:.6f}) → s = m 보장")
        else:
            print("  [FTTE] Proposition 1 창 없음 (intra_max >= cross_min): "
                  "이 스케일 범위에서는 클래스 구조와 궤적 구조가 겹칩니다.")

    if eps_list is None:
        eps_list = _sample_eps_grid(traj, quantiles)
        if diag["prop1_window"]:
            a, b = diag["prop1_window"]
            eps_list.append((a + b) / 2.0)
        eps_list = sorted(set(round(e, 12) for e in eps_list))

    rows = []
    for eps in eps_list:
        sT = separated_set_size(traj, eps, chunk=chunk, device=device)
        if sT < 1:
            continue
        h = math.log(sT) / T
        gap = h - h_class
        cmp = '>' if sT > m else ('=' if sT == m else '<')
        rows.append({"eps": eps, "s": sT, "h": h, "gap": gap, "cmp": cmp})
        if verbose:
            print(f"  [FTTE] eps={eps:>12.6f}  s_T={sT:>6d}  h_T={h:>8.4f}  "
                  f"Δh_T={gap:>+8.4f}  (s {cmp} m)")

    # 대표값: Prop.1 창 중점(이론 보장 구간) 우선, 없으면 s가 m에 가장 가까운 점
    recommended = None
    if rows:
        if diag["prop1_window"]:
            a, b = diag["prop1_window"]
            mid = (a + b) / 2.0
            in_win = [r for r in rows if a <= r["eps"] < b]
            recommended = min(in_win, key=lambda r: abs(r["eps"] - mid)) \
                if in_win else min(rows, key=lambda r: abs(r["s"] - m))
        else:
            recommended = min(rows, key=lambda r: abs(r["s"] - m))

    return {"T": T, "m": m, "h_class": h_class, "diagnostics": diag,
            "rows": rows, "recommended": recommended}
