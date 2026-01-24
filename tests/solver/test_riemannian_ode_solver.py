# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import time
import unittest

import torch
from flow_matching.solver.riemannian_ode_solver import RiemannianODESolver
from flow_matching.utils.manifolds import Sphere, FlatTorus, Euclidean


class HundredVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return torch.ones_like(x) * 100.0


class ZeroVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return torch.zeros_like(x)


class ExtraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t, must_be_true=False):
        assert must_be_true
        return torch.zeros_like(x)


class TestRiemannianODESolver(unittest.TestCase):
    def setUp(self):
        self.manifold = Sphere()
        self.velocity_model = HundredVelocityModel()
        self.solver = RiemannianODESolver(self.manifold, self.velocity_model)
        self.extra_model = ExtraModel()
        self.extra_solver = RiemannianODESolver(self.manifold, self.extra_model)

    def test_init(self):
        self.assertEqual(self.solver.manifold, self.manifold)
        self.assertEqual(self.solver.velocity_model, self.velocity_model)

    def test_sample_euler(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid
        )
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_sample_midpoint(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="midpoint", time_grid=time_grid
        )
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_sample_rk4(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="rk4", time_grid=time_grid
        )
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_zero_velocity_euler(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        zero_velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(self.manifold, zero_velocity_model)
        result = solver.sample(x_init, step_size, method="euler", time_grid=time_grid)
        self.assertTrue(torch.allclose(result, x_init))

    def test_zero_velocity_midpoint(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        zero_velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(self.manifold, zero_velocity_model)
        result = solver.sample(
            x_init, step_size, method="midpoint", time_grid=time_grid
        )
        self.assertTrue(torch.allclose(result, x_init))

    def test_zero_velocity_rk4(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        zero_velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(self.manifold, zero_velocity_model)
        result = solver.sample(x_init, step_size, method="rk4", time_grid=time_grid)
        self.assertTrue(torch.allclose(result, x_init))

    def test_sample_euler_step_size_none(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        time_grid = torch.linspace(0.0, 1.0, steps=100)
        result = self.solver.sample(x_init, None, method="euler", time_grid=time_grid)
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_sample_euler_verbose(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid, verbose=True
        )
        self.assertTrue(isinstance(result, torch.Tensor))

    def test_sample_return_intermediates_euler(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 0.5, 1.0])
        result = self.solver.sample(
            x_init,
            step_size,
            method="euler",
            time_grid=time_grid,
            return_intermediates=True,
        )
        self.assertEqual(result.shape, (3, 1, 3))  # Two intermediate points

    def test_model_extras(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 0.5, 1.0])
        result = self.extra_solver.sample(
            x_init,
            step_size,
            method="euler",
            time_grid=time_grid,
            return_intermediates=True,
            must_be_true=True,
        )
        self.assertEqual(result.shape, (3, 1, 3))

        with self.assertRaises(AssertionError):
            result = self.extra_solver.sample(
                x_init,
                step_size,
                method="euler",
                time_grid=time_grid,
                return_intermediates=True,
            )

    def test_gradient(self):
        x_init = torch.tensor(
            self.manifold.projx(torch.randn(1, 3)), requires_grad=True
        )
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid, enable_grad=True
        )
        result.sum().backward()
        self.assertIsInstance(x_init.grad, torch.Tensor)

    def test_no_gradient(self):
        x_init = torch.tensor(
            self.manifold.projx(torch.randn(1, 3)), requires_grad=True
        )
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid, enable_grad=False
        )
        with self.assertRaises(RuntimeError):
            result.sum().backward()


class TestEulerRiemannianFlatManifolds(unittest.TestCase):
    """Test that euler_riemannian and euler produce the same results on flat manifolds."""

    def test_euler_riemannian_vs_euler_euclidean(self):
        """Test euler_riemannian matches euler on Euclidean manifold."""
        manifold = Euclidean()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.randn(5, 3)
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        result_euler = solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid
        )
        result_euler_riemannian = solver.sample(
            x_init, step_size, method="euler_riemannian", time_grid=time_grid
        )

        self.assertTrue(
            torch.allclose(result_euler, result_euler_riemannian, rtol=1e-5, atol=1e-5),
            f"euler and euler_riemannian should match on Euclidean manifold. "
            f"Max diff: {(result_euler - result_euler_riemannian).abs().max()}",
        )

    def test_euler_riemannian_vs_euler_euclidean_zero_velocity(self):
        """Test euler_riemannian matches euler on Euclidean with zero velocity."""
        manifold = Euclidean()
        velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.randn(5, 3)
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        result_euler = solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid
        )
        result_euler_riemannian = solver.sample(
            x_init, step_size, method="euler_riemannian", time_grid=time_grid
        )

        self.assertTrue(torch.allclose(result_euler, x_init))
        self.assertTrue(torch.allclose(result_euler_riemannian, x_init))
        self.assertTrue(
            torch.allclose(result_euler, result_euler_riemannian, rtol=1e-5, atol=1e-5)
        )

    def test_euler_riemannian_vs_euler_flat_torus(self):
        """Test euler_riemannian matches euler on FlatTorus manifold."""
        manifold = FlatTorus()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        # Initialize points within [0, 2*pi) range for the torus
        x_init = torch.rand(5, 3) * 2 * torch.pi
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        result_euler = solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid
        )
        result_euler_riemannian = solver.sample(
            x_init, step_size, method="euler_riemannian", time_grid=time_grid
        )

        self.assertTrue(
            torch.allclose(result_euler, result_euler_riemannian, rtol=1e-5, atol=1e-5),
            f"euler and euler_riemannian should match on FlatTorus manifold. "
            f"Max diff: {(result_euler - result_euler_riemannian).abs().max()}",
        )

    def test_euler_riemannian_vs_euler_flat_torus_zero_velocity(self):
        """Test euler_riemannian matches euler on FlatTorus with zero velocity."""
        manifold = FlatTorus()
        velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.rand(5, 3) * 2 * torch.pi
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        result_euler = solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid
        )
        result_euler_riemannian = solver.sample(
            x_init, step_size, method="euler_riemannian", time_grid=time_grid
        )

        # With zero velocity, both should return the projected initial point
        x_init_proj = manifold.projx(x_init)
        self.assertTrue(torch.allclose(result_euler, x_init_proj, rtol=1e-5, atol=1e-5))
        self.assertTrue(
            torch.allclose(result_euler_riemannian, x_init_proj, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(result_euler, result_euler_riemannian, rtol=1e-5, atol=1e-5)
        )

    def test_euler_riemannian_vs_euler_flat_torus_return_intermediates(self):
        """Test euler_riemannian matches euler on FlatTorus with intermediate returns."""
        manifold = FlatTorus()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.rand(5, 3) * 2 * torch.pi
        step_size = 0.01
        time_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        result_euler = solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid, return_intermediates=True
        )
        result_euler_riemannian = solver.sample(
            x_init, step_size, method="euler_riemannian", time_grid=time_grid, return_intermediates=True
        )

        self.assertEqual(result_euler.shape, result_euler_riemannian.shape)
        self.assertTrue(
            torch.allclose(result_euler, result_euler_riemannian, rtol=1e-4, atol=1e-4),
            f"euler and euler_riemannian intermediates should match on FlatTorus. "
            f"Max diff: {(result_euler - result_euler_riemannian).abs().max()}",
        )

    def test_euler_riemannian_vs_euler_euclidean_return_intermediates(self):
        """Test euler_riemannian matches euler on Euclidean with intermediate returns."""
        manifold = Euclidean()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.randn(5, 3)
        step_size = 0.01
        time_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        result_euler = solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid, return_intermediates=True
        )
        result_euler_riemannian = solver.sample(
            x_init, step_size, method="euler_riemannian", time_grid=time_grid, return_intermediates=True
        )

        self.assertEqual(result_euler.shape, result_euler_riemannian.shape)
        self.assertTrue(
            torch.allclose(result_euler, result_euler_riemannian, rtol=1e-4, atol=1e-4),
            f"euler and euler_riemannian intermediates should match on Euclidean. "
            f"Max diff: {(result_euler - result_euler_riemannian).abs().max()}",
        )


class TestEulerRiemannianLatency(unittest.TestCase):
    """Test to compare latency between euler and euler_riemannian methods."""

    def _measure_latency(self, solver, x_init, step_size, time_grid, method, num_runs=10):
        """Measure average latency for a given method over multiple runs."""
        # Warmup run
        solver.sample(x_init, step_size, method=method, time_grid=time_grid)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            solver.sample(x_init, step_size, method=method, time_grid=time_grid)
            end = time.perf_counter()
            times.append(end - start)

        return sum(times) / len(times)

    def test_latency_comparison_euclidean(self):
        """Compare latency of euler vs euler_riemannian on Euclidean manifold."""
        manifold = Euclidean()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.randn(100, 3)
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        euler_latency = self._measure_latency(
            solver, x_init, step_size, time_grid, "euler"
        )
        euler_riemannian_latency = self._measure_latency(
            solver, x_init, step_size, time_grid, "euler_riemannian"
        )

        print(f"\n[Euclidean] euler latency: {euler_latency * 1000:.3f} ms")
        print(
            f"[Euclidean] euler_riemannian latency: {euler_riemannian_latency * 1000:.3f} ms"
        )
        print(
            f"[Euclidean] Difference: {(euler_riemannian_latency - euler_latency) * 1000:.3f} ms "
            f"({(euler_riemannian_latency / euler_latency - 1) * 100:.1f}% {'slower' if euler_riemannian_latency > euler_latency else 'faster'})"
        )

    def test_latency_comparison_flat_torus(self):
        """Compare latency of euler vs euler_riemannian on FlatTorus manifold."""
        manifold = FlatTorus()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = torch.rand(100, 3) * 2 * torch.pi
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        euler_latency = self._measure_latency(
            solver, x_init, step_size, time_grid, "euler"
        )
        euler_riemannian_latency = self._measure_latency(
            solver, x_init, step_size, time_grid, "euler_riemannian"
        )

        print(f"\n[FlatTorus] euler latency: {euler_latency * 1000:.3f} ms")
        print(
            f"[FlatTorus] euler_riemannian latency: {euler_riemannian_latency * 1000:.3f} ms"
        )
        print(
            f"[FlatTorus] Difference: {(euler_riemannian_latency - euler_latency) * 1000:.3f} ms "
            f"({(euler_riemannian_latency / euler_latency - 1) * 100:.1f}% {'slower' if euler_riemannian_latency > euler_latency else 'faster'})"
        )

    def test_latency_comparison_sphere(self):
        """Compare latency of euler vs euler_riemannian on Sphere manifold."""
        manifold = Sphere()
        velocity_model = HundredVelocityModel()
        solver = RiemannianODESolver(manifold, velocity_model)

        x_init = manifold.projx(torch.randn(100, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])

        euler_latency = self._measure_latency(
            solver, x_init, step_size, time_grid, "euler"
        )
        euler_riemannian_latency = self._measure_latency(
            solver, x_init, step_size, time_grid, "euler_riemannian"
        )

        print(f"\n[Sphere] euler latency: {euler_latency * 1000:.3f} ms")
        print(
            f"[Sphere] euler_riemannian latency: {euler_riemannian_latency * 1000:.3f} ms"
        )
        print(
            f"[Sphere] Difference: {(euler_riemannian_latency - euler_latency) * 1000:.3f} ms "
            f"({(euler_riemannian_latency / euler_latency - 1) * 100:.1f}% {'slower' if euler_riemannian_latency > euler_latency else 'faster'})"
        )


if __name__ == "__main__":
    unittest.main()
