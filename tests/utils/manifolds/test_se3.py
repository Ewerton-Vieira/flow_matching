import math
import unittest
import torch

from flow_matching.utils.manifolds.so3 import SO3
from flow_matching.utils.manifolds.se3 import SE3


def _rand_unit_quaternion(n: int, dtype=torch.float64) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=dtype)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _rand_se3(n: int, dtype=torch.float64) -> torch.Tensor:
    """Random SE3 element: [tx, ty, tz, qw, qx, qy, qz]."""
    t = torch.randn(n, 3, dtype=dtype)
    q = _rand_unit_quaternion(n, dtype=dtype)
    return torch.cat([t, q], dim=-1)


class TestSE3(unittest.TestCase):
    def setUp(self):
        self.se3 = SE3()
        self.so3 = SO3()

    def test_projx_normalizes_quaternion(self):
        """projx must normalize the quaternion part and leave translation unchanged."""
        torch.manual_seed(400)
        x = torch.randn(32, 7, dtype=torch.float64)
        x[..., 3:] *= 3.0
        p = self.se3.projx(x)
        self.assertTrue(torch.allclose(p[..., :3], x[..., :3], atol=1e-15))
        q_norms = p[..., 3:].norm(dim=-1)
        self.assertTrue(
            torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-12),
        )

    def test_projx_idempotent(self):
        """projx(projx(x)) == projx(x)."""
        torch.manual_seed(401)
        x = torch.randn(32, 7, dtype=torch.float64)
        p1 = self.se3.projx(x)
        p2 = self.se3.projx(p1)
        self.assertTrue(torch.allclose(p1, p2, atol=1e-15))

    def test_proju_is_identity(self):
        """proju must return u unchanged."""
        torch.manual_seed(402)
        x = _rand_se3(16)
        u = torch.randn(16, 6, dtype=torch.float64)
        self.assertTrue(torch.allclose(self.se3.proju(x, u), u, atol=1e-15))

    def test_canon_normalizes_and_canonicalizes_quaternion(self):
        """canon must canonicalize quaternion (w >= 0) and leave translation unchanged."""
        torch.manual_seed(403)
        x = _rand_se3(32)
        c = self.se3.canon(x)
        self.assertTrue(torch.allclose(c[..., :3], x[..., :3], atol=1e-15))
        q_canon = SO3.canon(SO3.normalize(x[..., 3:]))
        self.assertTrue(torch.allclose(c[..., 3:], q_canon, atol=1e-12))

    def test_dist_shape(self):
        """dist must return (..., 4): 3 Euclidean + 1 angular."""
        torch.manual_seed(404)
        x = _rand_se3(16)
        y = _rand_se3(16)
        d = self.se3.dist(x, y)
        self.assertEqual(d.shape, (16, 4))

    def test_dist_identity(self):
        """dist(x, x) == 0."""
        torch.manual_seed(405)
        x = self.se3.projx(_rand_se3(16))
        d = self.se3.dist(x, x)
        self.assertTrue(torch.allclose(d, torch.zeros_like(d), atol=1e-12))

    def test_dist_symmetry(self):
        """dist(x, y) == dist(y, x)."""
        torch.manual_seed(406)
        x = self.se3.projx(_rand_se3(32))
        y = self.se3.projx(_rand_se3(32))
        self.assertTrue(torch.allclose(self.se3.dist(x, y), self.se3.dist(y, x), atol=1e-12))

    def test_dist_components(self):
        """First 3 components are Euclidean, last is SO3 geodesic."""
        torch.manual_seed(407)
        x = self.se3.projx(_rand_se3(32))
        y = self.se3.projx(_rand_se3(32))
        d = self.se3.dist(x, y)
        t_dist = torch.abs(x[..., :3] - y[..., :3])
        self.assertTrue(torch.allclose(d[..., :3], t_dist, atol=1e-14))
        rot_dist = self.so3.dist(x[..., 3:], y[..., 3:])
        self.assertTrue(torch.allclose(d[..., 3:], rot_dist, atol=1e-12))


    def test_expmap_zero_is_identity(self):
        """expmap(x, 0) must return x."""
        torch.manual_seed(410)
        x = self.se3.projx(_rand_se3(32))
        u = torch.zeros(32, 6, dtype=torch.float64)
        y = self.se3.expmap(x, u)
        self.assertTrue(torch.allclose(y[..., :3], x[..., :3], atol=1e-12))
        q_diff = self.so3.dist(y[..., 3:], x[..., 3:])
        self.assertTrue(torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-12))

    def test_expmap_pure_translation(self):
        """With ω=0, expmap should add translation directly (V=I when θ=0)."""
        torch.manual_seed(411)
        x = self.se3.projx(_rand_se3(32))
        v = torch.randn(32, 3, dtype=torch.float64) * 0.5
        u = torch.cat([v, torch.zeros(32, 3, dtype=torch.float64)], dim=-1)
        y = self.se3.expmap(x, u)
        q_diff = self.so3.dist(y[..., 3:], x[..., 3:])
        self.assertTrue(torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-12))
        R_x_v = SO3.quat_action(x[..., 3:], v)
        expected_t = x[..., :3] + R_x_v
        self.assertTrue(torch.allclose(y[..., :3], expected_t, atol=1e-10))

    def test_expmap_pure_rotation(self):
        """With v=0, expmap should only rotate (translation unchanged)."""
        torch.manual_seed(412)
        x = self.se3.projx(_rand_se3(32))
        omega = torch.randn(32, 3, dtype=torch.float64) * 0.5
        u = torch.cat([torch.zeros(32, 3, dtype=torch.float64), omega], dim=-1)
        y = self.se3.expmap(x, u)
        self.assertTrue(torch.allclose(y[..., :3], x[..., :3], atol=1e-10))
        q_expected = self.so3.expmap(x[..., 3:], omega)
        q_diff = self.so3.dist(y[..., 3:], q_expected)
        self.assertTrue(torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-10))

    def test_expmap_differs_from_product_manifold(self):
        """SE3 expmap must NOT equal naive R³ + SO3 expmap (the coupling matters)."""
        torch.manual_seed(413)
        x = self.se3.projx(_rand_se3(32))
        u = torch.randn(32, 6, dtype=torch.float64) * 0.5
        y_se3 = self.se3.expmap(x, u)
        v, omega = u[..., :3], u[..., 3:]
        t_naive = x[..., :3] + SO3.quat_action(x[..., 3:], v)
        q_naive = self.so3.expmap(x[..., 3:], omega)
        y_naive = torch.cat([t_naive, q_naive], dim=-1)
        t_diff = (y_se3[..., :3] - y_naive[..., :3]).abs().max()
        self.assertTrue(t_diff > 1e-6, f"SE3 expmap must differ from naive, but max diff = {t_diff}")


if __name__ == "__main__":
    unittest.main()
