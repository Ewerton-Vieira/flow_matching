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


class TestSE3SophusVectors(unittest.TestCase):
    """Tests using the exact test vectors from Sophus test_ceres_se3.cpp.

    Reproduces the se3_vec and point_vec fixtures from Sophus, then runs the
    same manifold invariants that Sophus::LieGroupCeresTests::testManifold
    checks on every pair (i, j).

    Reference: https://github.com/strasdat/Sophus/blob/main/test/ceres/test_ceres_se3.cpp
    """

    def setUp(self):
        self.se3 = SE3()
        self.so3 = SO3()
        self.kPi = math.pi

        # Build SE3 elements matching Sophus test_ceres_se3.cpp
        self.se3_vec = self._build_sophus_se3_vec()
        self.point_vec = self._build_sophus_point_vec()

    def _make_se3(self, omega, t):
        """Construct SE3 element from rotation vector omega and translation t.
        Equivalent to SE3d(SO3d::exp(omega), t) in Sophus."""
        omega_t = torch.tensor([omega], dtype=torch.float64)
        t_t = torch.tensor([t], dtype=torch.float64)
        identity_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        q = self.so3.expmap(identity_q, omega_t)
        return torch.cat([t_t, q], dim=-1)  # [tx, ty, tz, qw, qx, qy, qz]

    def _se3_compose(self, a, b):
        """SE3 group composition: (R_a, t_a) * (R_b, t_b) = (R_a R_b, t_a + R_a t_b)."""
        t_a, q_a = a[..., :3], a[..., 3:]
        t_b, q_b = b[..., :3], b[..., 3:]
        q_ab = SO3.product(q_a, q_b)
        t_ab = t_a + SO3.quat_action(q_a, t_b)
        return torch.cat([t_ab, SO3.normalize(q_ab)], dim=-1)

    def _se3_inverse(self, x):
        """SE3 inverse: (R, t)⁻¹ = (Rᵀ, -Rᵀt)."""
        t, q = x[..., :3], x[..., 3:]
        q_inv = SO3._qconj(q)
        t_inv = SO3.quat_action(q_inv, -t)
        return torch.cat([t_inv, SO3.normalize(q_inv)], dim=-1)

    def _build_sophus_se3_vec(self):
        """Build the exact se3_vec from test_ceres_se3.cpp."""
        vec = []
        # 0: SE3(SO3::exp([0.2, 0.5, 0.0]), [0, 0, 0])
        vec.append(self._make_se3([0.2, 0.5, 0.0], [0, 0, 0]))
        # 1: SE3(SO3::exp([0.2, 0.5, -1.0]), [10, 0, 0])
        vec.append(self._make_se3([0.2, 0.5, -1.0], [10, 0, 0]))
        # 2: SE3(SO3::exp([0, 0, 0]), [0, 100, 5])
        vec.append(self._make_se3([0.0, 0.0, 0.0], [0, 100, 5]))
        # 3: SE3(SO3::exp([0, 0, 0.00001]), [0, 0, 0]) — near identity
        vec.append(self._make_se3([0.0, 0.0, 0.00001], [0, 0, 0]))
        # 4: SE3(SO3::exp([0, 0, 0.00001]), [0, -1e-8, 1e-10]) — near identity, tiny translation
        vec.append(self._make_se3([0.0, 0.0, 0.00001], [0, -0.00000001, 0.0000000001]))
        # 5: SE3(SO3::exp([0, 0, 0.00001]), [0.01, 0, 0])
        vec.append(self._make_se3([0.0, 0.0, 0.00001], [0.01, 0, 0]))
        # 6: SE3(SO3::exp([π, 0, 0]), [4, -5, 0]) — π rotation
        vec.append(self._make_se3([self.kPi, 0, 0], [4, -5, 0]))
        # 7: Conjugation near π:
        #    SE3(exp([0.2,0.5,0]),0) * SE3(exp([π,0,0]),0) * SE3(exp([-0.2,-0.5,0]),0)
        a = self._make_se3([0.2, 0.5, 0.0], [0, 0, 0])
        b = self._make_se3([self.kPi, 0, 0], [0, 0, 0])
        c = self._make_se3([-0.2, -0.5, -0.0], [0, 0, 0])
        vec.append(self._se3_compose(self._se3_compose(a, b), c))
        # 8: Full conjugation near π:
        #    SE3(exp([0.3,0.5,0.1]),[2,0,-7]) * SE3(exp([π,0,0]),0) * SE3(exp([-0.3,-0.5,-0.1]),[0,6,0])
        a2 = self._make_se3([0.3, 0.5, 0.1], [2, 0, -7])
        b2 = self._make_se3([self.kPi, 0, 0], [0, 0, 0])
        c2 = self._make_se3([-0.3, -0.5, -0.1], [0, 6, 0])
        vec.append(self._se3_compose(self._se3_compose(a2, b2), c2))
        return vec

    def _build_sophus_point_vec(self):
        """Build the exact point_vec from test_ceres_se3.cpp."""
        points = [
            [1.012, 2.73, -1.4],
            [9.2, -7.3, -4.4],
            [2.5, 0.1, 9.1],
            [12.3, 1.9, 3.8],
            [-3.21, 3.42, 2.3],
            [-8.0, 6.1, -1.1],
            [0.0, 2.5, 5.9],
            [7.1, 7.8, -14],
            [5.8, 9.2, 0.0],
        ]
        return [torch.tensor([p], dtype=torch.float64) for p in points]

    def _rotational_norm(self, tangent):
        """RotationalPart<SE3d>::Norm — norm of the rotational (last 3) components."""
        return tangent[..., 3:].norm(dim=-1)

    def test_x_plus_zero_is_x(self):
        """Sophus xPlusZeroIsXAt: expmap(x, 0) == x for all test vectors."""
        for i, x in enumerate(self.se3_vec):
            zero = torch.zeros(1, 6, dtype=torch.float64)
            y = self.se3.expmap(x, zero)
            error = self.se3.logmap(x, y).square().sum()
            self.assertTrue(
                error < 1e-15,
                f"xPlusZeroIsX failed for se3_vec[{i}], error={error.item():.2e}",
            )

    def test_x_minus_x_is_zero(self):
        """Sophus xMinusXIsZeroAt: logmap(x, x) == 0 for all test vectors."""
        for i, x in enumerate(self.se3_vec):
            tangent = self.se3.logmap(x, x)
            error = tangent.square().sum()
            self.assertTrue(
                error < 1e-15,
                f"xMinusXIsZero failed for se3_vec[{i}], error={error.item():.2e}",
            )

    def test_minus_plus_is_identity(self):
        """Sophus minusPlusIsIdentityAt: logmap(x, expmap(x, delta)) == delta.

        Tested for all pairs (i, j) where delta = log(x_i⁻¹ * x_j).
        Skips when rotational norm of delta > π(1-ε), matching Sophus behavior."""
        eps = 1e-9
        for i, x in enumerate(self.se3_vec):
            for j, y in enumerate(self.se3_vec):
                delta = self.se3.logmap(x, y)
                rot_norm = self._rotational_norm(delta)
                if rot_norm > self.kPi * (1.0 - eps):
                    continue  # Skip near-π, same as Sophus
                y_rec = self.se3.expmap(x, delta)
                delta_rec = self.se3.logmap(x, y_rec)
                diff = delta_rec - delta
                error = diff.square().sum()
                self.assertTrue(
                    error < 1e-12,
                    f"minusPlusIsIdentity failed for pair ({i},{j}), error={error.item():.2e}",
                )

    def test_minus_plus_is_identity_at_zero(self):
        """Sophus minusPlusIsIdentityAt with delta=0: logmap(x, expmap(x, 0)) == 0."""
        for i, x in enumerate(self.se3_vec):
            zero = torch.zeros(1, 6, dtype=torch.float64)
            y = self.se3.expmap(x, zero)
            delta_rec = self.se3.logmap(x, y)
            error = delta_rec.square().sum()
            self.assertTrue(
                error < 1e-15,
                f"minusPlusIsIdentity(zero) failed for se3_vec[{i}], error={error.item():.2e}",
            )

    def test_plus_minus_is_identity(self):
        """Sophus plusMinusIsIdentityAt: expmap(x, logmap(x, y)) == y.

        Tested for all pairs (i, j), including (i, i)."""
        for i, x in enumerate(self.se3_vec):
            for j, y in enumerate(self.se3_vec):
                delta = self.se3.logmap(x, y)
                y_rec = self.se3.expmap(x, delta)
                error = self.se3.logmap(y, y_rec).square().sum()
                self.assertTrue(
                    error < 1e-12,
                    f"plusMinusIsIdentity failed for pair ({i},{j}), error={error.item():.2e}",
                )

    def test_group_action_on_points(self):
        """SE3 action on points: T * p = R * p + t.

        Tests all (se3_vec[i], point_vec[j]) pairs."""
        for i, T in enumerate(self.se3_vec):
            t, q = T[..., :3], T[..., 3:]
            for j, p in enumerate(self.point_vec):
                # SE3 action: R * p + t
                result = SO3.quat_action(q, p) + t

                # Verify via compose with identity-rotation point
                # T * (I, p) should give (R*p + t, R)
                p_se3 = torch.cat([p, torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)], dim=-1)
                composed = self._se3_compose(T, p_se3)
                self.assertTrue(
                    torch.allclose(result, composed[..., :3], atol=1e-10),
                    f"Point action inconsistency for se3_vec[{i}], point_vec[{j}]",
                )

    def test_inverse_action_roundtrip(self):
        """T⁻¹ * (T * p) == p for all test vector pairs."""
        for i, T in enumerate(self.se3_vec):
            t, q = T[..., :3], T[..., 3:]
            T_inv = self._se3_inverse(T)
            t_inv, q_inv = T_inv[..., :3], T_inv[..., 3:]
            for j, p in enumerate(self.point_vec):
                Tp = SO3.quat_action(q, p) + t
                p_rec = SO3.quat_action(q_inv, Tp) + t_inv
                self.assertTrue(
                    torch.allclose(p_rec, p, atol=1e-10),
                    f"Inverse action roundtrip failed for se3_vec[{i}], point_vec[{j}]",
                )

    def test_exp_log_roundtrip_at_identity(self):
        """exp(log(T)) == T at identity base point for all test vectors.

        Skips when rotational norm ≥ π (log is not unique there)."""
        eps = 1e-9
        identity = torch.tensor([[0, 0, 0, 1.0, 0, 0, 0]], dtype=torch.float64)
        for i, T in enumerate(self.se3_vec):
            u = self.se3.logmap(identity, T)
            rot_norm = self._rotational_norm(u)
            if rot_norm > self.kPi * (1.0 - eps):
                continue
            T_rec = self.se3.expmap(identity, u)
            error = self.se3.logmap(T, T_rec).square().sum()
            self.assertTrue(
                error < 1e-12,
                f"exp(log(T)) roundtrip failed for se3_vec[{i}], error={error.item():.2e}",
            )


from flow_matching.utils.manifolds import SE3 as SE3_imported, Product, Euclidean


class TestSE3Integration(unittest.TestCase):
    def test_se3_importable_from_package(self):
        """SE3 must be importable from flow_matching.utils.manifolds."""
        self.assertIs(SE3_imported, SE3)

    def test_product_with_se3(self):
        """Product(Euclidean(2), SE3()) must work with state_dim=7, tangent_dim=6."""
        prod = Product(input_dim=9, manifolds=[
            (Euclidean(), 2),
            (SE3(), 7, 6),
        ])
        self.assertEqual(prod.total_state_dim, 9)
        self.assertEqual(prod.total_tangent_dim, 8)

    def test_product_se3_expmap(self):
        """Product expmap must delegate correctly to SE3."""
        torch.manual_seed(440)
        prod = Product(input_dim=9, manifolds=[
            (Euclidean(), 2),
            (SE3(), 7, 6),
        ])
        x = torch.randn(8, 9, dtype=torch.float64)
        x[..., 5:9] = x[..., 5:9] / x[..., 5:9].norm(dim=-1, keepdim=True)
        u = torch.randn(8, 8, dtype=torch.float64) * 0.3
        y = prod.expmap(x, u)
        self.assertEqual(y.shape, (8, 9))
        self.assertTrue(torch.allclose(y[..., :2], x[..., :2] + u[..., :2], atol=1e-12))
        se3 = SE3()
        y_se3 = se3.expmap(x[..., 2:], u[..., 2:])
        self.assertTrue(torch.allclose(y[..., 2:], y_se3, atol=1e-12))

    def test_product_se3_validation(self):
        """Product must reject SE3 with wrong state_dim or tangent_dim."""
        with self.assertRaises(ValueError):
            Product(input_dim=6, manifolds=[(SE3(), 6, 6)])
        with self.assertRaises(ValueError):
            Product(input_dim=7, manifolds=[(SE3(), 7, 7)])


if __name__ == "__main__":
    unittest.main()
