"""
test_isolift.py — IsoLift-ResNeXt 스모크 테스트 (학습/데이터 다운로드 불필요).

검증 항목:
  1. E_d 등거리:  ||E_d(x)-E_d(x')|| = ||x-x'||  (세 데이터셋 모두, 수치 검증)
  2. 차원 보존:  48*56^2 = 192*28^2 = 768*14^2 = 150,528 (stage 전이 후에도)
  3. provable 모드: 각 residual branch 의 경험적 Lipschitz 비율 < rho
     (무작위 방향 유한차분) -> 블록 가역성의 충분조건
  4. geometry / lipschitz loss 정의역·기울기 정상
  5. 두 모드 모두 forward/backward 1 step 실행 가능

실행: python test_isolift.py
"""
import torch

from models.isolift import COMMON_DIM, LIFTS, IsoLiftNet
from utils.isolift_losses import (branch_lipschitz_estimates, geometry_loss,
                                  lipschitz_penalty)

torch.manual_seed(0)
DOMAINS = ("MNIST", "CIFAR10", "IMAGENET10")


def main():
    # ── 1. E_d 등거리 ────────────────────────────────────────────────────
    print("[1] E_d 등거리 검증")
    for name, cls in LIFTS.items():
        lift = cls().double()
        x = torch.randn(4, *cls.IN_SHAPE, dtype=torch.float64)
        x2 = torch.randn_like(x)
        din = (x - x2).flatten(1).norm(dim=1)
        dout = (lift(x) - lift(x2)).flatten(1).norm(dim=1)
        err = float((din - dout).abs().max())
        assert err < 1e-9, (name, err)
        z = lift(x)
        assert z.shape[1:] == (48, 56, 56) and z[0].numel() == COMMON_DIM
        print(f"    {name:<11} ||dx-dz||_max = {err:.2e}  ->  48x56x56  OK")

    # ── 2. stage 차원 보존 ───────────────────────────────────────────────
    print("[2] stage 차원 보존")
    model = IsoLiftNet(domains=DOMAINS, layers=(1, 1, 1), mode="performance")
    u = model.lift_and_adapt(torch.randn(2, 3, 224, 224), "IMAGENET10")
    shapes = []
    for s, stage in enumerate(model.stages):
        for blk in stage:
            u = blk(u, "IMAGENET10")
        shapes.append(tuple(u.shape[1:]))
        assert u[0].numel() == COMMON_DIM, shapes
        if s < 2:
            u = model.shuffle(u)
    print(f"    {shapes[0]} -> {shapes[1]} -> {shapes[2]}  (모두 {COMMON_DIM:,})  OK")

    # ── 3. provable 모드 branch Lipschitz < rho ─────────────────────────
    print("[3] provable 모드 Lip(alpha*F) < rho")
    rho = 0.9
    pmodel = IsoLiftNet(domains=("MNIST",), layers=(2, 1, 1),
                        mode="provable", rho=rho)
    pmodel.eval()
    with torch.no_grad():
        for _ in range(30):                     # power iteration 워밍업
            pmodel(torch.randn(2, 1, 28, 28), "MNIST")
        worst = 0.0
        for blk in pmodel.residual_blocks():
            c = blk.conv1.in_channels
            hw = {48: 56, 192: 28, 768: 14}[c]
            u0 = torch.randn(1, c, hw, hw)
            for _ in range(10):                 # 경험적 비율 (유한차분)
                d = torch.randn_like(u0) * 1e-3
                num = (blk.alpha * (blk.branch(u0 + d, "MNIST")
                                    - blk.branch(u0, "MNIST"))).norm()
                worst = max(worst, float(num / d.norm()))
        assert worst < rho + 1e-3, worst
    print(f"    경험적 최대 비율 = {worst:.4f} < rho = {rho}  OK")

    # ── 4. 손실 함수 ─────────────────────────────────────────────────────
    print("[4] geometry / lipschitz loss")
    x = torch.randn(8, 3, 32, 32)
    x2 = x + 0.05 * torch.randn_like(x)
    model_c = IsoLiftNet(domains=("CIFAR10",), layers=(1, 1, 1))
    u0, u02 = (model_c.lift_and_adapt(v, "CIFAR10") for v in (x, x2))
    g = geometry_loss(x, x2, u0, u02)
    # 완전 등거리(adapter 항등에 가깝게 beta->0)면 m<=1<=M 이라 벌점 0
    z0, z02 = (model_c.lift(v, "CIFAR10") for v in (x, x2))
    g_iso = geometry_loss(x, x2, z0, z02)
    lp = lipschitz_penalty(model_c, rho=0.9)
    assert torch.isfinite(g) and float(g_iso) < 1e-10 and torch.isfinite(lp)
    lips = branch_lipschitz_estimates(pmodel)
    assert all(float(l.detach()) < 1.0 for l in lips)
    print(f"    geo(adapter)={float(g):.4f}  geo(등거리)={float(g_iso):.1e}  "
          f"lip_penalty={float(lp):.5f}  provable branch Lip<1: OK")

    # ── 5. 공동 학습 1 step ─────────────────────────────────────────────
    print("[5] 두 모드 forward/backward 1 step")
    fake = {"MNIST": torch.randn(4, 1, 28, 28),
            "CIFAR10": torch.randn(4, 3, 32, 32),
            "IMAGENET10": torch.randn(2, 3, 224, 224)}
    for mode in ("performance", "provable"):
        m = IsoLiftNet(domains=DOMAINS, layers=(1, 1, 1), mode=mode)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                               lr=1e-4)
        loss = 0.0
        for d, x in fake.items():
            logits, u0 = m(x, d, return_u0=True)
            assert logits.shape == (x.shape[0], 10)
            y = torch.randint(0, 10, (x.shape[0],))
            loss = loss + torch.nn.functional.cross_entropy(logits, y)
        loss = loss + 0.01 * lipschitz_penalty(m)
        loss.backward()
        opt.step()
        print(f"    {mode:<12} loss={float(loss):.4f}  OK")

    print("\n=== IsoLift-ResNeXt 스모크 테스트 성공 ===")


if __name__ == "__main__":
    main()
