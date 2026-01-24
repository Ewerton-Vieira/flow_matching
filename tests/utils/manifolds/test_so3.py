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


    # =========================================================================
    # Tests corresponding to C++ SE3 Lie group tests (adapted for SO3)
    # Based on Micro-Lie theory equations
    # =========================================================================

    def test_identity_quaternion(self):
        """Test identity quaternion (corresponds to constructor_test in C++)."""
        # Identity quaternion: [1, 0, 0, 0]
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        projected = self.so3.projx(identity)

        self.assertTrue(
            torch.allclose(projected, identity, atol=1e-10),
            f"Identity quaternion should remain identity after projx",
        )

    def test_known_rotation_composition(self):
        """Test composition with known rotations (corresponds to composition_test in C++)."""
        # 90° rotation about z-axis: q = [cos(45°), 0, 0, sin(45°)]
        q_z90 = torch.tensor(
            [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]],
            dtype=torch.float64,
        )
        # 90° rotation about x-axis: q = [cos(45°), sin(45°), 0, 0]
        q_x90 = torch.tensor(
            [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]],
            dtype=torch.float64,
        )

        # Compose: first z, then x (in body frame)
        q_composed = SO3.product(q_z90, q_x90)
        q_composed = self.so3.projx(q_composed)

        # Verify it's still unit quaternion
        self.assertTrue(
            torch.allclose(
                q_composed.norm(dim=-1),
                torch.ones(1, dtype=torch.float64),
                atol=1e-10,
            )
        )

        # Verify round-trip: apply inverse should give identity
        q_inverse = SO3._qconj(q_composed)
        q_identity = SO3.product(q_composed, q_inverse)
        q_identity = self.so3.projx(q_identity)

        expected_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        self.assertTrue(
            _canon_equiv(self.so3, q_identity, expected_identity, atol=1e-10)
        )

    def test_lie_group_identity_axioms(self):
        """
        Test Lie group identity axioms (corresponds to expmap_compose_test Eqs 2,3 in C++).
        Eq 2: X ∘ Id = Id ∘ X = X
        Eq 3: X ∘ X^(-1) = X^(-1) ∘ X = Id
        """
        torch.manual_seed(100)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        for _ in range(100):
            x = self.so3.projx(_rand_unit_quaternion(1))
            x_inv = SO3._qconj(x)

            # Eq 2: X ∘ Id = X
            x_compose_id = SO3.product(x, identity)
            self.assertTrue(_canon_equiv(self.so3, x_compose_id, x, atol=1e-10))

            # Eq 2: Id ∘ X = X
            id_compose_x = SO3.product(identity, x)
            self.assertTrue(_canon_equiv(self.so3, id_compose_x, x, atol=1e-10))

            # Eq 3: X ∘ X^(-1) = Id
            x_compose_xinv = SO3.product(x, x_inv)
            self.assertTrue(
                _canon_equiv(self.so3, x_compose_xinv, identity, atol=1e-10)
            )

            # Eq 3: X^(-1) ∘ X = Id
            xinv_compose_x = SO3.product(x_inv, x)
            self.assertTrue(
                _canon_equiv(self.so3, xinv_compose_x, identity, atol=1e-10)
            )

    def test_expmap_additivity(self):
        """
        Eq. 17 from Micro-Lie theory: exp((t+s)ω) = exp(tω) ∘ exp(sω)
        (corresponds to expmap_eq17_test in C++)

        This property holds for SO3 because rotations about the same axis commute.
        """
        torch.manual_seed(101)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        for _ in range(100):
            t = torch.rand(1, dtype=torch.float64).item() * 0.5
            s = torch.rand(1, dtype=torch.float64).item() * 0.5

            # Random rotation vector (axis-angle)
            omega = torch.randn(1, 3, dtype=torch.float64)
            # Scale to reasonable magnitude
            omega = omega / omega.norm() * torch.rand(1).item() * math.pi * 0.5

            # exp((t+s)ω)
            x0 = self.so3.expmap(identity, (t + s) * omega)

            # exp(tω) ∘ exp(sω)
            exp_t = self.so3.expmap(identity, t * omega)
            exp_s = self.so3.expmap(identity, s * omega)
            x1 = SO3.product(exp_t, exp_s)
            x1 = self.so3.projx(x1)

            self.assertTrue(
                _canon_equiv(self.so3, x0, x1, atol=1e-9, rtol=1e-7),
                f"exp((t+s)ω) should equal exp(tω)∘exp(sω)",
            )

    def test_expmap_inverse_property(self):
        """
        Eq. 19 from Micro-Lie theory: exp(-ω) = exp(ω)^(-1)
        (corresponds to expmap_eq19_test in C++)
        """
        torch.manual_seed(102)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        for _ in range(100):
            # Random rotation vector
            omega = torch.randn(1, 3, dtype=torch.float64)
            omega = omega / omega.norm() * torch.rand(1).item() * math.pi * 0.9

            # exp(-ω)
            x0 = self.so3.expmap(identity, -omega)

            # exp(ω)^(-1)
            exp_omega = self.so3.expmap(identity, omega)
            x1 = SO3._qconj(exp_omega)
            x1 = self.so3.projx(x1)

            self.assertTrue(
                _canon_equiv(self.so3, x0, x1, atol=1e-9, rtol=1e-7),
                f"exp(-ω) should equal exp(ω)^(-1)",
            )

    def test_logmap_expmap_roundtrip(self):
        """
        Test that Logmap(Expmap(ω)) = ω for general rotation vectors.
        (corresponds to logmap_test in C++)
        """
        torch.manual_seed(103)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        for _ in range(100):
            # Random rotation vector with angle in [0, π - ε] to avoid singularity
            axis = _rand_unit_axis(1)
            angle = torch.rand(1, 1, dtype=torch.float64) * (math.pi - 0.1)
            omega = axis * angle

            # Expmap then Logmap
            q = self.so3.expmap(identity, omega)
            omega_recovered = self.so3.logmap(identity, q)

            diff = omega - omega_recovered
            self.assertTrue(
                torch.allclose(diff, torch.zeros_like(diff), atol=1e-8),
                f"Logmap(Expmap(ω)) should recover ω, diff norm: {diff.norm()}",
            )

    def test_compose_associativity_extensive(self):
        """
        Eq. 4 from Micro-Lie theory: (X ∘ Y) ∘ Z = X ∘ (Y ∘ Z)
        (corresponds to expmap_compose_test Eq 4 in C++)

        More extensive than test_product_hamilton_associativity.
        """
        torch.manual_seed(104)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        for _ in range(100):
            # Generate random rotations via expmap
            omega_x = torch.randn(1, 3, dtype=torch.float64)
            omega_y = torch.randn(1, 3, dtype=torch.float64)
            omega_z = torch.randn(1, 3, dtype=torch.float64)

            x = self.so3.expmap(identity, omega_x)
            y = self.so3.expmap(identity, omega_y)
            z = self.so3.expmap(identity, omega_z)

            # (X ∘ Y) ∘ Z
            xy = SO3.product(x, y)
            left = SO3.product(xy, z)

            # X ∘ (Y ∘ Z)
            yz = SO3.product(y, z)
            right = SO3.product(x, yz)

            self.assertTrue(
                _canon_equiv(self.so3, left, right, atol=1e-10, rtol=1e-8),
                f"Associativity (X∘Y)∘Z = X∘(Y∘Z) should hold",
            )

    def test_expmap_at_base_point(self):
        """
        Test expmap at non-identity base points.
        Verifies: expmap(x, logmap(x, y)) = y
        """
        torch.manual_seed(105)

        for _ in range(100):
            x = self.so3.projx(_rand_unit_quaternion(1))
            y = self.so3.projx(_rand_unit_quaternion(1))

            # Get rotation vector from x to y
            omega_xy = self.so3.logmap(x, y)

            # Apply expmap at x
            y_recovered = self.so3.expmap(x, omega_xy)

            self.assertTrue(
                _canon_equiv(self.so3, y_recovered, y, atol=1e-9, rtol=1e-7),
                f"expmap(x, logmap(x, y)) should equal y",
            )

    def test_logmap_norm_equals_distance(self):
        """
        Test that ||logmap(x, y)|| = dist(x, y).
        This is a fundamental property of the Riemannian exponential map.
        """
        torch.manual_seed(106)

        for _ in range(100):
            x = self.so3.projx(_rand_unit_quaternion(1))
            y = self.so3.projx(_rand_unit_quaternion(1))

            omega_xy = self.so3.logmap(x, y)
            log_norm = omega_xy.norm(dim=-1)

            dist_xy = self.so3.dist(x, y).squeeze(-1)

            self.assertTrue(
                torch.allclose(log_norm, dist_xy, atol=1e-9, rtol=1e-7),
                f"||logmap(x,y)|| should equal dist(x,y)",
            )


if __name__ == "__main__":
    unittest.main()
