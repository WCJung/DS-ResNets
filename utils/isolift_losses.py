"""
IsoLift-ResNeXt 학습 손실 — L = sum_d CE^(d) + λ_geo·L_geo + λ_lip·L_lip.

L_geo (local geometry loss):
    가까운 입력 쌍 (x, x') 에 대해 d_x = ||x-x'||, d_z = ||u_0-u_0'||
    (u_0 = A_d(E_d(x)) — adapter 통과 후) 로 두고
        L_geo = max(0, m·d_x - d_z)^2 + max(0, d_z - M·d_x)^2.
    E_d 는 정확한 등거리이므로 이 손실은 adapter(와 그 이후)가 거리를
    과도하게 축소/확대하는 것을 억제한다.

L_lip (soft spectral penalty):
    각 residual branch 의 Lipschitz 추정치 |alpha|·prod_i sigma_i 가
    rho(<1) 를 넘으면 벌점. sigma_i 는 conv 의 reshape 행렬
    (out, in/groups*kh*kw) 최대 특이값의 power-iteration 추정 —
    hard 제약(SNConv2d) 대신 쓰는 근사이며, provable 모드 모델에서는
    branch 가 이미 제약돼 있어 벌점이 자연히 0 근처가 된다.
"""
import torch
import torch.nn.functional as F


def geometry_loss(x, x2, u0, u02, m=0.5, M=2.0):
    """거리비 hinge:  m·d_x <= d_z <= M·d_x 를 벗어난 만큼 제곱 벌점."""
    dx = (x - x2).flatten(1).norm(dim=1)
    dz = (u0 - u02).flatten(1).norm(dim=1)
    lo = F.relu(m * dx - dz)
    hi = F.relu(dz - M * dx)
    return (lo ** 2 + hi ** 2).mean()


def _conv_sigma(conv, n_iter=1):
    """conv reshape 행렬의 최대 특이값 power-iteration 추정 (미분 가능).

    u 벡터는 conv 에 비영속 속성으로 캐시해 반복 호출에서 워밍업된다.
    """
    w = conv.weight.flatten(1)                     # (out, in/g*kh*kw)
    u = getattr(conv, "_pi_u", None)
    if u is None or u.shape[0] != w.shape[0] or u.device != w.device:
        u = torch.randn(w.shape[0], device=w.device)
    with torch.no_grad():
        for _ in range(n_iter):
            v = F.normalize(w.t() @ u, dim=0)
            u = F.normalize(w @ v, dim=0)
        conv._pi_u = u
    v = F.normalize((w.t() @ u).detach(), dim=0)
    return torch.dot(u, w @ v)                     # weight 에 미분 가능


def branch_lipschitz_estimates(model):
    """블록별 Lip 추정치 |alpha|·prod sigma_i 의 리스트 (미분 가능)."""
    out = []
    for blk in model.residual_blocks():
        lip = blk.alpha.abs() if torch.is_tensor(blk.alpha) else abs(blk.alpha)
        for conv in blk.branch_convs():
            lip = lip * _conv_sigma(conv)
        out.append(lip)
    return out


def lipschitz_penalty(model, rho=0.9):
    """L_lip = mean_l max(0, Lip_l - rho)^2."""
    lips = branch_lipschitz_estimates(model)
    total = sum(F.relu(l - rho) ** 2 for l in lips)
    return total / len(lips)
