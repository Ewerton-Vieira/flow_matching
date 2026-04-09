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


    def test_logmap_identity(self):
        """logmap(x, x) must return zero twist."""
        torch.manual_seed(420)
        x = self.se3.projx(_rand_se3(32))
        u = self.se3.logmap(x, x)
        self.assertTrue(torch.allclose(u, torch.zeros_like(u), atol=1e-10))

    def test_exp_log_roundtrip(self):
        """expmap(x, logmap(x, y)) must recover y."""
        torch.manual_seed(421)
        x = self.se3.projx(_rand_se3(64))
        y = self.se3.projx(_rand_se3(64))
        u = self.se3.logmap(x, y)
        y_rec = self.se3.expmap(x, u)
        self.assertTrue(
            torch.allclose(y_rec[..., :3], y[..., :3], atol=1e-8),
            f"Translation roundtrip failed, max diff: {(y_rec[..., :3] - y[..., :3]).abs().max()}",
        )
        q_diff = self.so3.dist(y_rec[..., 3:], y[..., 3:])
        self.assertTrue(
            torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-8),
            f"Rotation roundtrip failed, max diff: {q_diff.max()}",
        )

    def test_log_exp_roundtrip_small_twist(self):
        """logmap(x, expmap(x, u)) must recover u for small u."""
        torch.manual_seed(422)
        x = self.se3.projx(_rand_se3(64))
        u = torch.randn(64, 6, dtype=torch.float64) * 0.1
        y = self.se3.expmap(x, u)
        u_rec = self.se3.logmap(x, y)
        self.assertTrue(
            torch.allclose(u_rec, u, atol=1e-8),
            f"Small twist roundtrip failed, max diff: {(u_rec - u).abs().max()}",
        )

    def test_log_exp_roundtrip_large_twist(self):
        """logmap(x, expmap(x, u)) must recover u for moderate u."""
        torch.manual_seed(423)
        x = self.se3.projx(_rand_se3(64))
        v = torch.randn(64, 3, dtype=torch.float64)
        axis = torch.randn(64, 3, dtype=torch.float64)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        angle = torch.rand(64, 1, dtype=torch.float64) * (math.pi - 0.2)
        omega = axis * angle
        u = torch.cat([v, omega], dim=-1)
        y = self.se3.expmap(x, u)
        u_rec = self.se3.logmap(x, y)
        self.assertTrue(
            torch.allclose(u_rec, u, atol=1e-7),
            f"Large twist roundtrip failed, max diff: {(u_rec - u).abs().max()}",
        )

    def test_logmap_differs_from_product_manifold(self):
        """SE3 logmap must NOT equal naive R³ + SO3 logmap (the coupling matters)."""
        torch.manual_seed(424)
        x = self.se3.projx(_rand_se3(32))
        u = torch.randn(32, 6, dtype=torch.float64) * 0.5
        y = self.se3.expmap(x, u)
        u_se3 = self.se3.logmap(x, y)
        omega_naive = self.so3.logmap(x[..., 3:], y[..., 3:])
        dt = y[..., :3] - x[..., :3]
        v_naive = SO3.quat_action(SO3._qconj(x[..., 3:]), dt)
        u_naive = torch.cat([v_naive, omega_naive], dim=-1)
        self.assertTrue(
            torch.allclose(u_se3[..., 3:], u_naive[..., 3:], atol=1e-10),
        )
        v_diff = (u_se3[..., :3] - u_naive[..., :3]).abs().max()
        self.assertTrue(v_diff > 1e-6, f"SE3 logmap must differ from naive, but max diff = {v_diff}")

    def _se3_compose(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """SE3 group composition: (R_a, t_a) * (R_b, t_b) = (R_a R_b, t_a + R_a t_b).
        Reference: se3_reference.hpp:302-305"""
        t_a, q_a = a[..., :3], a[..., 3:]
        t_b, q_b = b[..., :3], b[..., 3:]
        q_ab = SO3.product(q_a, q_b)
        t_ab = t_a + SO3.quat_action(q_a, t_b)
        return torch.cat([t_ab, SO3.normalize(q_ab)], dim=-1)

    def _se3_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """SE3 inverse: (R, t)⁻¹ = (Rᵀ, -Rᵀt).
        Reference: se3_reference.hpp:222-224"""
        t, q = x[..., :3], x[..., 3:]
        q_inv = SO3._qconj(q)
        t_inv = SO3.quat_action(q_inv, -t)
        return torch.cat([t_inv, SO3.normalize(q_inv)], dim=-1)

    def _se3_identity(self, n: int, dtype=torch.float64) -> torch.Tensor:
        """SE3 identity element: translation=0, quaternion=[1,0,0,0]."""
        t = torch.zeros(n, 3, dtype=dtype)
        q = torch.zeros(n, 4, dtype=dtype)
        q[..., 0] = 1.0
        return torch.cat([t, q], dim=-1)

    def test_identity_axioms(self):
        """X * Id = Id * X = X, and X * X⁻¹ = Id."""
        torch.manual_seed(430)
        identity = self._se3_identity(1)
        for _ in range(100):
            x = self.se3.projx(_rand_se3(1))
            x_inv = self._se3_inverse(x)
            x_id = self._se3_compose(x, identity)
            self.assertTrue(torch.allclose(x_id[..., :3], x[..., :3], atol=1e-10))
            q_diff = self.so3.dist(x_id[..., 3:], x[..., 3:])
            self.assertTrue(torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-10))
            x_xinv = self._se3_compose(x, x_inv)
            self.assertTrue(torch.allclose(x_xinv[..., :3], torch.zeros(1, 3, dtype=torch.float64), atol=1e-10))
            q_diff_id = self.so3.dist(x_xinv[..., 3:], identity[..., 3:])
            self.assertTrue(torch.allclose(q_diff_id, torch.zeros_like(q_diff_id), atol=1e-10))

    def test_associativity(self):
        """(X * Y) * Z = X * (Y * Z)."""
        torch.manual_seed(431)
        for _ in range(100):
            x = self.se3.projx(_rand_se3(1))
            y = self.se3.projx(_rand_se3(1))
            z = self.se3.projx(_rand_se3(1))
            left = self._se3_compose(self._se3_compose(x, y), z)
            right = self._se3_compose(x, self._se3_compose(y, z))
            self.assertTrue(torch.allclose(left[..., :3], right[..., :3], atol=1e-9))
            q_diff = self.so3.dist(left[..., 3:], right[..., 3:])
            self.assertTrue(torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-9))

    def test_expmap_inverse_property(self):
        """exp(-u) = exp(u)⁻¹: se3_reference.hpp exp/inverse."""
        torch.manual_seed(432)
        identity = self._se3_identity(64)
        u = torch.randn(64, 6, dtype=torch.float64) * 0.5
        exp_neg = self.se3.expmap(identity, -u)
        exp_pos = self.se3.expmap(identity, u)
        exp_pos_inv = self._se3_inverse(exp_pos)
        self.assertTrue(torch.allclose(exp_neg[..., :3], exp_pos_inv[..., :3], atol=1e-9))
        q_diff = self.so3.dist(exp_neg[..., 3:], exp_pos_inv[..., 3:])
        self.assertTrue(torch.allclose(q_diff, torch.zeros_like(q_diff), atol=1e-9))

    def test_logmap_norm_and_dist_relationship(self):
        """||logmap(x,y)|| should be related to dist but NOT equal (dist is decoupled)."""
        torch.manual_seed(433)
        x = self.se3.projx(_rand_se3(32))
        y = self.se3.projx(_rand_se3(32))
        u = self.se3.logmap(x, y)
        d = self.se3.dist(x, y)
        self.assertEqual(d.shape[-1], 4)
        self.assertEqual(u.shape[-1], 6)
        omega_norm = u[..., 3:].norm(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(omega_norm, d[..., 3:], atol=1e-9))


if __name__ == "__main__":
    unittest.main()
