import math
import unittest
import torch

from flow_matching.utils.manifolds import SO3


def _rand_unit_quaternion(n: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    # Sample from normal and normalize; projx will also canonicalize.
    q = torch.randn(n, 4, dtype=dtype, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q


def _rand_unit_axis(n: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    a = torch.randn(n, 3, dtype=dtype, device=device)
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _canon_equiv(
    so3: SO3, q1: torch.Tensor, q2: torch.Tensor, atol=1e-8, rtol=1e-6
) -> bool:
    # Compare rotations up to antipodal sign by canonicalizing both.
    return torch.allclose(so3.projx(q1), so3.projx(q2), atol=atol, rtol=rtol)


class TestSO3(unittest.TestCase):
    def setUp(self):
        self.so3 = SO3()

    def test_projx_normalizes_and_is_sign_invariant(self):
        torch.manual_seed(0)
        q = torch.randn(64, 4, dtype=torch.float64)
        p = self.so3.projx(q)

        # unit norm
        self.assertTrue(
            torch.allclose(
                p.norm(dim=-1), torch.ones(64, dtype=torch.float64), atol=1e-10
            )
        )

        # antipodal invariance: projx(q) == projx(-q)
        p2 = self.so3.projx(-q)
        self.assertTrue(torch.allclose(p, p2, atol=1e-10))

    def test_product_hamilton_associativity(self):
        torch.manual_seed(1)
        a = self.so3.projx(_rand_unit_quaternion(32))
        b = self.so3.projx(_rand_unit_quaternion(32))
        c = self.so3.projx(_rand_unit_quaternion(32))

        left = SO3.product(SO3.product(a, b), c)
        right = SO3.product(a, SO3.product(b, c))
        # Associativity holds exactly in reals; allow numerical tolerance.
        self.assertTrue(_canon_equiv(self.so3, left, right, atol=1e-10, rtol=1e-8))

    def test_expmap_zero_is_identity(self):
        torch.manual_seed(2)
        x = self.so3.projx(_rand_unit_quaternion(16))
        omega = torch.zeros(16, 3, dtype=torch.float64)
        y = self.so3.expmap(x, omega)
        self.assertTrue(_canon_equiv(self.so3, y, x, atol=1e-12, rtol=1e-10))

    def test_logmap_identity_is_zero(self):
        torch.manual_seed(3)
        x = self.so3.projx(_rand_unit_quaternion(16))
        omega = self.so3.logmap(x, x)
        self.assertTrue(
            torch.allclose(omega, torch.zeros_like(omega), atol=1e-12, rtol=0.0)
        )

    def test_exp_log_inverse_small_angles(self):
        """
        For small rotation vectors, log(exp(x, omega)) should recover omega closely.
        """
        torch.manual_seed(4)
        x = self.so3.projx(_rand_unit_quaternion(128))
        axis = _rand_unit_axis(128)
        angles = torch.rand(128, 1, dtype=torch.float64) * 1e-3  # tiny angles
        omega = axis * angles

        y = self.so3.expmap(x, omega)
        omega_rec = self.so3.logmap(x, y)

        self.assertTrue(torch.allclose(omega_rec, omega, atol=1e-9, rtol=1e-6))

    def test_log_exp_inverse_general(self):
        """
        Verify exp(x, log(x, y)) == y (up to antipodal equivalence/canonicalization).
        This is the correct invariant to test even when log is not unique.
        """
        torch.manual_seed(5)
        x = self.so3.projx(_rand_unit_quaternion(64))

        # Build y via expmap with moderate angles in (0, pi)
        axis = _rand_unit_axis(64)
        angles = torch.rand(64, 1, dtype=torch.float64) * (
            math.pi - 1e-3
        )  # avoid exact pi
        omega = axis * angles
        y = self.so3.expmap(x, omega)

        omega_xy = self.so3.logmap(x, y)
        y_rec = self.so3.expmap(x, omega_xy)

        self.assertTrue(_canon_equiv(self.so3, y_rec, y, atol=1e-9, rtol=1e-6))

    def test_near_pi_behavior(self):
        """
        Stability near pi: log/exp should round-trip and distance should be near the angle.
        """
        torch.manual_seed(6)
        x = self.so3.projx(_rand_unit_quaternion(64))
        axis = _rand_unit_axis(64)
        angles = torch.full((64, 1), math.pi - 1e-5, dtype=torch.float64)
        omega = axis * angles
        y = self.so3.expmap(x, omega)

        omega_rec = self.so3.logmap(x, y)
        y_rec = self.so3.expmap(x, omega_rec)

        # Round trip
        self.assertTrue(_canon_equiv(self.so3, y_rec, y, atol=1e-8, rtol=1e-5))
        # Norm close to pi-1e-5 (axis sign may vary, but norm should match)
        self.assertTrue(
            torch.allclose(
                omega_rec.norm(dim=-1), angles.squeeze(-1), atol=1e-6, rtol=1e-5
            )
        )

    def test_dist_properties_and_matches_inner_product_formula(self):
        torch.manual_seed(7)
        x = self.so3.projx(_rand_unit_quaternion(128))
        y = self.so3.projx(_rand_unit_quaternion(128))
        z = self.so3.projx(_rand_unit_quaternion(128))

        d_xy = self.so3.dist(x, y)  # (...,1)
        d_yx = self.so3.dist(y, x)
        d_xx = self.so3.dist(x, x)
        d_xz = self.so3.dist(x, z)
        d_yz = self.so3.dist(y, z)

        # symmetry
        self.assertTrue(torch.allclose(d_xy, d_yx, atol=1e-12, rtol=1e-10))
        # identity
        self.assertTrue(
            torch.allclose(d_xx, torch.zeros_like(d_xx), atol=1e-12, rtol=0.0)
        )
        # triangle inequality (allow small numeric slack)
        self.assertTrue(torch.all(d_xz <= d_xy + d_yz + 1e-10))

        # Compare against 2*acos(|<q1,q2>|)
        dot = (x * y).sum(dim=-1, keepdim=True).abs().clamp(-1.0, 1.0)
        d_ref = 2.0 * torch.acos(dot)
        self.assertTrue(torch.allclose(d_xy, d_ref, atol=1e-10, rtol=1e-8))


if __name__ == "__main__":
    unittest.main()
