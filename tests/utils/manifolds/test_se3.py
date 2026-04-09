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


if __name__ == "__main__":
    unittest.main()
