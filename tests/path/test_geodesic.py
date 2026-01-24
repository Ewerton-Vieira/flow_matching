# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Test suite for GeodesicProbPath with manifolds including SO(3) and Product manifolds."""

import pytest
import torch
from torch import Tensor

from flow_matching.path.geodesic import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import Euclidean, Product
from flow_matching.utils.manifolds.so3 import SO3
from flow_matching.utils.manifolds.torus import FlatTorus


class TestGeodesicProbPathEuclidean:
    """Tests for GeodesicProbPath with Euclidean manifold."""

    @pytest.fixture
    def setup_euclidean(self):
        """Setup Euclidean manifold and path."""
        manifold = Euclidean()
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)
        return manifold, scheduler, path

    def test_sample_shapes_euclidean(self, setup_euclidean):
        """Test that sample returns correct shapes for Euclidean manifold."""
        manifold, scheduler, path = setup_euclidean
        batch_size = 8
        dim = 6

        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        assert sample.x_t.shape == (
            batch_size,
            dim,
        ), f"Expected x_t shape {(batch_size, dim)}, got {sample.x_t.shape}"
        assert sample.dx_t.shape == (
            batch_size,
            dim,
        ), f"Expected dx_t shape {(batch_size, dim)}, got {sample.dx_t.shape}"
        assert sample.x_0.shape == (batch_size, dim)
        assert sample.x_1.shape == (batch_size, dim)
        assert sample.t.shape == (batch_size,)

    def test_boundary_t0_euclidean(self, setup_euclidean):
        """Test that at t=0, x_t = x_0 for Euclidean manifold."""
        manifold, scheduler, path = setup_euclidean
        batch_size = 4
        dim = 5

        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.zeros(batch_size)

        sample = path.sample(x_0, x_1, t)

        torch.testing.assert_close(sample.x_t, x_0, atol=1e-5, rtol=1e-5)

    def test_boundary_t1_euclidean(self, setup_euclidean):
        """Test that at t=1, x_t = x_1 for Euclidean manifold."""
        manifold, scheduler, path = setup_euclidean
        batch_size = 4
        dim = 5

        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.ones(batch_size)

        sample = path.sample(x_0, x_1, t)

        torch.testing.assert_close(sample.x_t, x_1, atol=1e-5, rtol=1e-5)

    def test_midpoint_euclidean(self, setup_euclidean):
        """Test that at t=0.5, x_t is the midpoint for Euclidean manifold with CondOT scheduler."""
        manifold, scheduler, path = setup_euclidean
        batch_size = 4
        dim = 5

        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.full((batch_size,), 0.5)

        sample = path.sample(x_0, x_1, t)

        # For CondOT scheduler, alpha_t = t, so midpoint should be (x_0 + x_1) / 2
        expected_midpoint = (x_0 + x_1) / 2
        torch.testing.assert_close(sample.x_t, expected_midpoint, atol=1e-5, rtol=1e-5)

    def test_velocity_direction_euclidean(self, setup_euclidean):
        """Test that velocity points from x_0 to x_1 for Euclidean manifold."""
        manifold, scheduler, path = setup_euclidean
        batch_size = 4
        dim = 5

        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # For CondOT scheduler with Euclidean, dx_t = d(alpha_t)/dt * (x_1 - x_0) = x_1 - x_0
        expected_velocity = x_1 - x_0
        torch.testing.assert_close(sample.dx_t, expected_velocity, atol=1e-5, rtol=1e-5)


class TestGeodesicProbPathSO3:
    """Tests for GeodesicProbPath with SO(3) manifold (quaternions)."""

    @pytest.fixture
    def setup_so3(self):
        """Setup SO(3) manifold and path."""
        manifold = SO3()
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)
        return manifold, scheduler, path

    def _random_unit_quaternions(self, batch_size: int):
        """Generate random unit quaternions."""
        q = torch.randn(batch_size, 4)
        q = q / q.norm(dim=1, keepdim=True)
        return q

    def test_sample_shapes_so3(self, setup_so3):
        """Test that sample returns correct shapes for SO(3) manifold."""
        manifold, scheduler, path = setup_so3
        batch_size = 8

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # State space is 4D (quaternion), tangent space is 3D (rotation vector)
        assert sample.x_t.shape == (batch_size, 4), f"Expected x_t shape {(batch_size, 4)}, got {sample.x_t.shape}"
        assert sample.dx_t.shape == (batch_size, 3), f"Expected dx_t shape {(batch_size, 3)}, got {sample.dx_t.shape}"

    def test_boundary_t0_so3(self, setup_so3):
        """Test that at t=0, x_t is equivalent to x_0 on SO(3)."""
        manifold, scheduler, path = setup_so3
        batch_size = 4

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.zeros(batch_size)

        sample = path.sample(x_0, x_1, t)

        # Use manifold distance to check equivalence (handles q ≈ -q)
        dist0 = manifold.dist(sample.x_t, x_0)
        assert (dist0 < 1e-5).all(), f"x_t does not match x_0 at t=0, max dist: {dist0.max()}"

    def test_boundary_t1_so3(self, setup_so3):
        """Test that at t=1, x_t is equivalent to x_1 on SO(3)."""
        manifold, scheduler, path = setup_so3
        batch_size = 4

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.ones(batch_size)

        sample = path.sample(x_0, x_1, t)

        # Use manifold distance to check equivalence (handles q ≈ -q)
        dist1 = manifold.dist(sample.x_t, x_1)
        assert (dist1 < 1e-4).all(), f"x_t does not match x_1 at t=1, max dist: {dist1.max()}"

    def test_quaternion_unit_norm_so3(self, setup_so3):
        """Test that x_t remains on the unit sphere (valid quaternion)."""
        manifold, scheduler, path = setup_so3
        batch_size = 8

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        norms = sample.x_t.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

    def test_velocity_tangent_space_so3(self, setup_so3):
        """Test that dx_t is in tangent space (3D rotation vector)."""
        manifold, scheduler, path = setup_so3
        batch_size = 8

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # dx_t should be 3D (rotation vector in tangent space of SO(3))
        assert sample.dx_t.shape[-1] == 3, f"Expected tangent dim 3, got {sample.dx_t.shape[-1]}"

    def test_velocity_magnitude_so3(self, setup_so3):
        """Test that velocity magnitude equals geodesic distance for CondOT scheduler."""
        manifold, scheduler, path = setup_so3
        batch_size = 8

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # For CondOT scheduler, d(alpha_t)/dt = 1, so ||dx_t|| = geodesic distance
        velocity_magnitude = sample.dx_t.norm(dim=-1)
        geodesic_dist = manifold.dist(x_0, x_1).squeeze(-1)
        torch.testing.assert_close(velocity_magnitude, geodesic_dist, atol=1e-4, rtol=1e-4)

    def test_intermediate_distance_so3(self, setup_so3):
        """Test that dist(x_0, x_t) = t * dist(x_0, x_1) for geodesic path."""
        manifold, scheduler, path = setup_so3
        batch_size = 8

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.rand(batch_size) * 0.8 + 0.1  # Avoid boundaries

        sample = path.sample(x_0, x_1, t)

        # For a geodesic with CondOT scheduler: dist(x_0, x_t) = t * dist(x_0, x_1)
        total_dist = manifold.dist(x_0, x_1).squeeze(-1)
        partial_dist = manifold.dist(x_0, sample.x_t).squeeze(-1)
        expected_partial_dist = t * total_dist

        torch.testing.assert_close(partial_dist, expected_partial_dist, atol=1e-4, rtol=1e-4)

    def test_midpoint_equidistant_so3(self, setup_so3):
        """Test that at t=0.5, x_t is equidistant from x_0 and x_1."""
        manifold, scheduler, path = setup_so3
        batch_size = 8

        x_0 = self._random_unit_quaternions(batch_size)
        x_1 = self._random_unit_quaternions(batch_size)
        t = torch.full((batch_size,), 0.5)

        sample = path.sample(x_0, x_1, t)

        dist_to_x0 = manifold.dist(sample.x_t, x_0).squeeze(-1)
        dist_to_x1 = manifold.dist(sample.x_t, x_1).squeeze(-1)

        torch.testing.assert_close(dist_to_x0, dist_to_x1, atol=1e-4, rtol=1e-4)

    def test_integration_consistency_so3(self, setup_so3):
        """Test that integrating dx_t from 0 to 1 recovers x_1."""
        manifold, scheduler, path = setup_so3

        x_0 = self._random_unit_quaternions(1)
        x_1 = self._random_unit_quaternions(1)

        # Euler integration with small steps
        n_steps = 100
        dt = 1.0 / n_steps
        x_current = x_0.clone()

        for i in range(n_steps):
            t = torch.tensor([i * dt])
            sample = path.sample(x_0, x_1, t)
            # Integrate: x_{t+dt} = expmap(x_t, dx_t * dt)
            x_current = manifold.expmap(x_current, sample.dx_t * dt)

        # After integration, x_current should be close to x_1
        final_dist = manifold.dist(x_current, x_1).squeeze()
        assert final_dist < 0.02, f"Integration did not recover x_1, final dist: {final_dist}"


class TestGeodesicProbPathProduct:
    """Tests for GeodesicProbPath with Product manifold (R^3 x SO(3) x R^6)."""

    @pytest.fixture
    def setup_product(self):
        """Setup Product manifold (Quadrotor3D: R^3 x SO(3) x R^6) and path."""
        manifold = Product(
            input_dim=13,
            manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)],
        )
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)
        return manifold, scheduler, path

    def _random_quadrotor_state(self, batch_size: int):
        """Generate random valid quadrotor states with unit quaternions."""
        x = torch.randn(batch_size, 13)
        # Normalize quaternion (indices 3-6)
        x[:, 3:7] = x[:, 3:7] / x[:, 3:7].norm(dim=1, keepdim=True)
        return x

    def test_sample_shapes_product(self, setup_product):
        """Test that sample returns correct shapes for Product manifold."""
        manifold, scheduler, path = setup_product
        batch_size = 8

        x_0 = self._random_quadrotor_state(batch_size)
        x_1 = self._random_quadrotor_state(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # State space is 13D, tangent space is 12D (3 + 3 + 6)
        assert sample.x_t.shape == (batch_size, 13), f"Expected x_t shape {(batch_size, 13)}, got {sample.x_t.shape}"
        assert sample.dx_t.shape == (batch_size, 12), f"Expected dx_t shape {(batch_size, 12)}, got {sample.dx_t.shape}"

    def test_boundary_t0_product(self, setup_product):
        """Test that at t=0, x_t is equivalent to x_0 for Product manifold."""
        manifold, scheduler, path = setup_product
        batch_size = 4

        x_0 = self._random_quadrotor_state(batch_size)
        x_1 = self._random_quadrotor_state(batch_size)
        t = torch.zeros(batch_size)

        sample = path.sample(x_0, x_1, t)

        # Use manifold distance (returns per-component distances)
        dist0 = manifold.dist(sample.x_t, x_0)
        assert (dist0 < 1e-5).all(), f"x_t does not match x_0 at t=0, max dist: {dist0.max()}"

    def test_boundary_t1_product(self, setup_product):
        """Test that at t=1, x_t is equivalent to x_1 for Product manifold."""
        manifold, scheduler, path = setup_product
        batch_size = 4

        x_0 = self._random_quadrotor_state(batch_size)
        x_1 = self._random_quadrotor_state(batch_size)
        t = torch.ones(batch_size)

        sample = path.sample(x_0, x_1, t)

        # Use manifold distance (returns per-component distances)
        dist1 = manifold.dist(sample.x_t, x_1)
        assert (dist1 < 1e-4).all(), f"x_t does not match x_1 at t=1, max dist: {dist1.max()}"

    def test_quaternion_unit_norm_product(self, setup_product):
        """Test that quaternion component remains normalized in Product manifold."""
        manifold, scheduler, path = setup_product
        batch_size = 8

        x_0 = self._random_quadrotor_state(batch_size)
        x_1 = self._random_quadrotor_state(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # Extract quaternion and check norm
        quat = sample.x_t[:, 3:7]
        norms = quat.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

    def test_tangent_space_decomposition_product(self, setup_product):
        """Test that dx_t correctly decomposes into tangent components."""
        manifold, scheduler, path = setup_product
        batch_size = 4

        x_0 = self._random_quadrotor_state(batch_size)
        x_1 = self._random_quadrotor_state(batch_size)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        # dx_t should be 12D: 3 (position) + 3 (rotation) + 6 (velocity)
        assert sample.dx_t.shape == (batch_size, 12)

        # Position tangent (indices 0-2) should match Euclidean velocity
        pos_tangent = sample.dx_t[:, :3]
        expected_pos_tangent = x_1[:, :3] - x_0[:, :3]  # For CondOT scheduler
        torch.testing.assert_close(pos_tangent, expected_pos_tangent, atol=1e-5, rtol=1e-5)

        # Linear velocity tangent (indices 6-11 in tangent space = state indices 7-12)
        vel_tangent = sample.dx_t[:, 6:]  # 6 = 3 (pos) + 3 (rotation)
        expected_vel_tangent = x_1[:, 7:] - x_0[:, 7:]  # For CondOT scheduler
        torch.testing.assert_close(vel_tangent, expected_vel_tangent, atol=1e-5, rtol=1e-5)

    def test_intermediate_distance_product(self, setup_product):
        """Test that component distances scale linearly with t for geodesic path."""
        manifold, scheduler, path = setup_product
        batch_size = 8

        x_0 = self._random_quadrotor_state(batch_size)
        x_1 = self._random_quadrotor_state(batch_size)
        t = torch.rand(batch_size) * 0.8 + 0.1  # Avoid boundaries

        sample = path.sample(x_0, x_1, t)

        # For a geodesic: dist(x_0, x_t) = t * dist(x_0, x_1) for each component
        total_dist = manifold.dist(x_0, x_1)  # (batch, 10) for R3+SO3+R6
        partial_dist = manifold.dist(x_0, sample.x_t)
        expected_partial_dist = t.unsqueeze(-1) * total_dist

        torch.testing.assert_close(partial_dist, expected_partial_dist, atol=1e-4, rtol=1e-4)

    def test_interpolation_smoothness_product(self, setup_product):
        """Test that interpolation is smooth (monotonic distance decrease)."""
        manifold, scheduler, path = setup_product

        x_0 = self._random_quadrotor_state(1)
        x_1 = self._random_quadrotor_state(1)

        # Sample at multiple time points
        times = torch.linspace(0, 1, 11)
        positions = []

        for t_val in times:
            t = torch.tensor([t_val])
            sample = path.sample(x_0, x_1, t)
            positions.append(sample.x_t[:, :3].squeeze())  # Just position for simplicity

        # Check that position moves monotonically from x_0 to x_1
        positions = torch.stack(positions)
        dists_to_end = (positions - x_1[:, :3]).norm(dim=1)

        # Distance to endpoint should generally decrease (allowing small numerical errors)
        for i in range(len(dists_to_end) - 1):
            assert (
                dists_to_end[i] >= dists_to_end[i + 1] - 1e-4
            ), f"Distance increased from t={times[i]:.2f} to t={times[i+1]:.2f}"


class TestGeodesicProbPathFlatTorus:
    """Tests for GeodesicProbPath with FlatTorus manifold."""

    @pytest.fixture
    def setup_torus(self):
        """Setup FlatTorus manifold and path."""
        manifold = FlatTorus()
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)
        return manifold, scheduler, path

    def test_sample_shapes_torus(self, setup_torus):
        """Test that sample returns correct shapes for FlatTorus manifold."""
        manifold, scheduler, path = setup_torus
        batch_size = 8
        dim = 4

        # Angles in [-pi, pi]
        x_0 = torch.rand(batch_size, dim) * 2 * torch.pi - torch.pi
        x_1 = torch.rand(batch_size, dim) * 2 * torch.pi - torch.pi
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        assert sample.x_t.shape == (batch_size, dim)
        assert sample.dx_t.shape == (batch_size, dim)

    def test_boundary_conditions_torus(self, setup_torus):
        """Test boundary conditions for FlatTorus manifold."""
        manifold, scheduler, path = setup_torus
        batch_size = 4
        dim = 3

        x_0 = torch.rand(batch_size, dim) * 2 * torch.pi - torch.pi
        x_1 = torch.rand(batch_size, dim) * 2 * torch.pi - torch.pi

        # At t=0, x_t should be equivalent to x_0 on the torus
        t0 = torch.zeros(batch_size)
        sample0 = path.sample(x_0, x_1, t0)
        dist0 = manifold.dist(sample0.x_t, x_0)
        assert (
            dist0 < 1e-5
        ).all(), f"x_t does not match x_0 at t=0, max dist: {dist0.max()}"

        # At t=1, x_t should be equivalent to x_1 on the torus
        t1 = torch.ones(batch_size)
        sample1 = path.sample(x_0, x_1, t1)
        dist1 = manifold.dist(sample1.x_t, x_1)
        assert (
            dist1 < 1e-4
        ).all(), f"x_t does not match x_1 at t=1, max dist: {dist1.max()}"


class TestGeodesicProbPathNumericalGradients:
    """Numerical gradient tests to verify dx_t is correct derivative of x_t."""

    def test_numerical_gradient_euclidean(self):
        """Verify dx_t matches numerical gradient for Euclidean manifold."""
        manifold = Euclidean()
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)

        batch_size = 4
        dim = 5
        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)

        # Get analytical derivative
        sample = path.sample(x_0, x_1, t)
        dx_t_analytical = sample.dx_t

        # Compute numerical derivative
        eps = 1e-5
        t_plus = t + eps
        t_minus = t - eps
        sample_plus = path.sample(x_0, x_1, t_plus)
        sample_minus = path.sample(x_0, x_1, t_minus)
        dx_t_numerical = (sample_plus.x_t - sample_minus.x_t) / (2 * eps)

        # Use looser tolerance for numerical differentiation
        torch.testing.assert_close(
            dx_t_analytical, dx_t_numerical, atol=1e-2, rtol=1e-2
        )

    def test_numerical_gradient_product_euclidean_components(self):
        """Verify dx_t matches numerical gradient for Euclidean components of Product manifold."""
        manifold = Product(
            input_dim=13, manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)]
        )
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)

        batch_size = 4

        def random_state():
            x = torch.randn(batch_size, 13)
            x[:, 3:7] = x[:, 3:7] / x[:, 3:7].norm(dim=1, keepdim=True)
            return x

        x_0 = random_state()
        x_1 = random_state()
        t = torch.rand(batch_size) * 0.8 + 0.1  # Avoid boundary

        # Get analytical derivative
        sample = path.sample(x_0, x_1, t)

        # Compute numerical derivative
        eps = 1e-5
        t_plus = t + eps
        t_minus = t - eps
        sample_plus = path.sample(x_0, x_1, t_plus)
        sample_minus = path.sample(x_0, x_1, t_minus)

        # Check position derivative (state indices 0-2, tangent indices 0-2)
        dx_pos_numerical = (sample_plus.x_t[:, :3] - sample_minus.x_t[:, :3]) / (
            2 * eps
        )
        dx_pos_analytical = sample.dx_t[:, :3]
        torch.testing.assert_close(
            dx_pos_analytical, dx_pos_numerical, atol=1e-2, rtol=1e-2
        )

        # Check linear velocity derivative (state indices 7-12, tangent indices 6-11)
        dx_vel_numerical = (sample_plus.x_t[:, 7:] - sample_minus.x_t[:, 7:]) / (
            2 * eps
        )
        dx_vel_analytical = sample.dx_t[:, 6:]  # 6 = 3 (pos) + 3 (rotation)
        torch.testing.assert_close(
            dx_vel_analytical, dx_vel_numerical, atol=1e-2, rtol=1e-2
        )


class TestGeodesicProbPathEdgeCases:
    """Edge case tests."""

    def test_same_start_end_euclidean(self):
        """Test when x_0 == x_1 (should stay at same point with zero velocity)."""
        manifold = Euclidean()
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)

        batch_size = 4
        dim = 5
        x = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)

        sample = path.sample(x, x, t)

        torch.testing.assert_close(sample.x_t, x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            sample.dx_t, torch.zeros_like(x), atol=1e-5, rtol=1e-5
        )

    def test_same_start_end_product(self):
        """Test when x_0 == x_1 for Product manifold."""
        manifold = Product(
            input_dim=13, manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)]
        )
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)

        batch_size = 4
        x = torch.randn(batch_size, 13)
        x[:, 3:7] = x[:, 3:7] / x[:, 3:7].norm(dim=1, keepdim=True)
        t = torch.rand(batch_size)

        sample = path.sample(x, x, t)

        # Use manifold distance (handles q ≈ -q equivalence for SO3 component)
        dist = manifold.dist(sample.x_t, x)
        assert (dist < 1e-5).all(), f"x_t does not match x at same start/end, max dist: {dist.max()}"
        torch.testing.assert_close(
            sample.dx_t, torch.zeros(batch_size, 12), atol=1e-5, rtol=1e-5
        )

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        manifold = Product(
            input_dim=13, manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)]
        )
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)

        x_0 = torch.randn(1, 13)
        x_0[:, 3:7] = x_0[:, 3:7] / x_0[:, 3:7].norm(dim=1, keepdim=True)
        x_1 = torch.randn(1, 13)
        x_1[:, 3:7] = x_1[:, 3:7] / x_1[:, 3:7].norm(dim=1, keepdim=True)
        t = torch.rand(1)

        sample = path.sample(x_0, x_1, t)

        assert sample.x_t.shape == (1, 13)
        assert sample.dx_t.shape == (1, 12)

    def test_large_batch(self):
        """Test with large batch size."""
        manifold = Product(
            input_dim=13, manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)]
        )
        scheduler = CondOTScheduler()
        path = GeodesicProbPath(scheduler, manifold)

        batch_size = 256
        x_0 = torch.randn(batch_size, 13)
        x_0[:, 3:7] = x_0[:, 3:7] / x_0[:, 3:7].norm(dim=1, keepdim=True)
        x_1 = torch.randn(batch_size, 13)
        x_1[:, 3:7] = x_1[:, 3:7] / x_1[:, 3:7].norm(dim=1, keepdim=True)
        t = torch.rand(batch_size)

        sample = path.sample(x_0, x_1, t)

        assert sample.x_t.shape == (batch_size, 13)
        assert sample.dx_t.shape == (batch_size, 12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
