import math
import unittest

import torch

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean


class TestProductManifold(unittest.TestCase):
    def setUp(self):
        self.manifold = Product(
            input_dim=2,
            manifolds=[
                (FlatTorus(), 1),
                (Euclidean(), 1),
            ],
        )

    def test_projx_wraps_torus_and_keeps_euclidean(self):
        x = torch.tensor([[3 * math.pi, 5.0], [-math.pi / 2, -2.0]])
        projected = self.manifold.projx(x)

        expected_angle = torch.tensor([[math.pi], [3 * math.pi / 2]])
        expected_vel = torch.tensor([[5.0], [-2.0]])
        self.assertTrue(torch.allclose(projected[:, :1], expected_angle))
        self.assertTrue(torch.allclose(projected[:, 1:], expected_vel))

    def test_proju_is_identity_componentwise(self):
        x = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        u = torch.tensor([[0.3, -2.5], [-0.7, 4.2]])
        projected_u = self.manifold.proju(x, u)
        self.assertTrue(torch.allclose(projected_u, u))

    def test_expmap_componentwise_behavior(self):
        x = torch.tensor([[1.0, 10.0], [2 * math.pi - 0.2, -1.0]])
        u = torch.tensor([[0.5, -3.0], [0.3, 2.0]])
        out = self.manifold.expmap(x, u)

        expected_angle = torch.remainder(x[:, :1] + u[:, :1], 2 * math.pi)
        expected_vel = x[:, 1:] + u[:, 1:]
        self.assertTrue(torch.allclose(out[:, :1], expected_angle))
        self.assertTrue(torch.allclose(out[:, 1:], expected_vel))

    def test_logmap_componentwise_behavior(self):
        x = torch.tensor([[0.1, 0.0], [2 * math.pi - 0.1, 1.0]])
        y = torch.tensor([[2 * math.pi - 0.1, 2.0], [0.1, -2.0]])
        u = self.manifold.logmap(x, y)

        expected_angle = torch.tensor([[-0.2], [0.2]])
        expected_vel = torch.tensor([[2.0], [-3.0]])
        self.assertTrue(torch.allclose(u[:, :1], expected_angle, atol=1e-6))
        self.assertTrue(torch.allclose(u[:, 1:], expected_vel, atol=1e-6))

    def test_dist_returns_per_component_distances(self):
        x = torch.tensor([[0.1, 0.0], [2 * math.pi - 0.1, 1.0]])
        y = torch.tensor([[2 * math.pi - 0.1, 2.0], [0.1, -2.0]])
        d = self.manifold.dist(x, y)

        expected = torch.tensor([[0.2, 2.0], [0.2, 3.0]])
        self.assertEqual(d.shape, expected.shape)
        self.assertTrue(torch.allclose(d, expected, atol=1e-6))

    def test_expmap_logmap_inverse_relationship(self):
        """Verify that exp and log maps are inverses: y = exp(x, u) => u = log(x, y)"""
        x = torch.tensor([[0.5, 1.0], [math.pi, -2.0], [1.5, 0.0]])
        u = torch.tensor([[0.3, 2.0], [-0.5, 1.0], [math.pi/2, -1.0]])
        
        # Apply expmap then logmap
        y = self.manifold.expmap(x, u)
        u_recovered = self.manifold.logmap(x, y)
        
        # The recovered tangent vector should match the original
        # Note: for the torus component, we need to handle the periodic nature
        self.assertTrue(torch.allclose(u_recovered, u, atol=1e-6))
        
        # Also test the reverse: x = exp(y, -log(y, x))
        neg_u = self.manifold.logmap(y, x)
        x_recovered = self.manifold.expmap(y, neg_u)
        x_projected = self.manifold.projx(x)  # Ensure x is on manifold first
        self.assertTrue(torch.allclose(x_recovered, x_projected, atol=1e-6))

    def test_torus_boundary_cases(self):
        """Test behavior at and across the 0/2π boundary of the torus"""
        # Points very close to the boundary
        x = torch.tensor([
            [0.01, 0.0],           # Just after 0
            [2*math.pi - 0.01, 0.0],  # Just before 2π
            [0.0, 0.0],            # Exactly at 0
            [2*math.pi, 0.0]       # Exactly at 2π (should wrap to 0)
        ])
        
        # Project should handle the wrap correctly
        projected = self.manifold.projx(x)
        self.assertTrue(torch.allclose(projected[2, 0], projected[3, 0]))  # 0 and 2π are same
        
        # Distance across boundary should be minimal
        d = self.manifold.dist(x[:1], x[1:2])  # Distance between 0.01 and 2π-0.01
        expected_torus_dist = 0.02  # Should go across the boundary, not around
        self.assertAlmostEqual(d[0, 0].item(), expected_torus_dist, places=5)
        
        # Logarithmic map should find shortest path across boundary
        u = self.manifold.logmap(x[1:2], x[:1])  # From 2π-0.01 to 0.01
        expected_u_torus = 0.02  # Should be positive (forward across boundary)
        self.assertAlmostEqual(u[0, 0].item(), expected_u_torus, places=5)

    def test_distance_metric_properties(self):
        """Verify that distance satisfies metric space properties"""
        x = torch.tensor([[0.5, 1.0], [math.pi, -2.0], [1.5, 3.0]])
        y = torch.tensor([[2.0, 0.0], [0.3, 1.5], [math.pi + 0.5, -1.0]])
        z = torch.tensor([[1.0, 2.0], [math.pi/2, 0.0], [0.2, 1.0]])
        
        # Test symmetry: d(x,y) = d(y,x)
        d_xy = self.manifold.dist(x, y)
        d_yx = self.manifold.dist(y, x)
        self.assertTrue(torch.allclose(d_xy, d_yx, atol=1e-6))
        
        # Test identity: d(x,x) = 0
        d_xx = self.manifold.dist(x, x)
        self.assertTrue(torch.allclose(d_xx, torch.zeros_like(d_xx), atol=1e-6))
        
        # Test triangle inequality for each component
        # d(x,z) <= d(x,y) + d(y,z)
        d_xz = self.manifold.dist(x, z)
        d_yz = self.manifold.dist(y, z)
        
        # Check component-wise (since dist returns per-component distances)
        for component in range(2):
            triangle_sum = d_xy[:, component] + d_yz[:, component]
            self.assertTrue(torch.all(d_xz[:, component] <= triangle_sum + 1e-6))

    def test_numerical_stability_near_points(self):
        """Test stability when points are very close together"""
        base = torch.tensor([[1.0, 2.0], [math.pi, -1.0]])
        epsilon = 1e-8
        
        # Create points very close to base
        perturbed = base + torch.tensor([[epsilon, -epsilon], [-epsilon, epsilon]])
        
        # Logarithmic map should give tiny tangent vectors
        u = self.manifold.logmap(base, perturbed)
        self.assertTrue(torch.all(torch.abs(u) < 1e-6))
        
        # Distance should be approximately epsilon
        d = self.manifold.dist(base, perturbed)
        self.assertTrue(torch.all(d < 1e-6))
        
        # Exponential map with tiny tangent vectors should barely move
        tiny_u = torch.full_like(base, epsilon)
        y = self.manifold.expmap(base, tiny_u)
        self.assertTrue(torch.allclose(y, base, atol=1e-6))

    def test_geodesic_interpolation(self):
        """Test that we can smoothly interpolate along geodesics"""
        x0 = torch.tensor([[0.5, 1.0], [3.0, -2.0]])
        x1 = torch.tensor([[2.0, 3.0], [0.5, 1.0]])
        
        # Get the tangent vector from x0 to x1
        v = self.manifold.logmap(x0, x1)
        
        # Interpolate at different points along the geodesic
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            xt = self.manifold.expmap(x0, t * v)
            
            # Check that distances scale linearly with t
            d0t = self.manifold.dist(x0, xt)
            dt1 = self.manifold.dist(xt, x1)
            d01 = self.manifold.dist(x0, x1)
            
            # For each component, d(x0,xt) ≈ t*d(x0,x1)
            expected_d0t = t * d01
            expected_dt1 = (1-t) * d01
            
            self.assertTrue(torch.allclose(d0t, expected_d0t, atol=1e-5))
            self.assertTrue(torch.allclose(dt1, expected_dt1, atol=1e-5))


if __name__ == "__main__":
    unittest.main()


