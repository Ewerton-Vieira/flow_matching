import math
import unittest
import torch

from flow_matching.utils.manifolds import Sphere


def _rand_unit_sphere(
    n: int, D: int, dtype=torch.float64, device="cpu"
) -> torch.Tensor:
    x = torch.randn(n, D, dtype=dtype, device=device)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.sphere = Sphere()

    def test_projx_outputs_unit_norm(self):
        torch.manual_seed(0)
        x = torch.randn(64, 7, dtype=torch.float64)
        y = self.sphere.projx(x)
        self.assertTrue(
            torch.allclose(
                y.norm(dim=-1), torch.ones(64, dtype=torch.float64), atol=1e-10
            )
        )

    def test_proju_is_orthogonal_to_base(self):
        torch.manual_seed(1)
        x = _rand_unit_sphere(128, 5)
        u = torch.randn(128, 5, dtype=torch.float64)

        pu = self.sphere.proju(x, u)
        inner = (x * pu).sum(dim=-1)
        self.assertTrue(
            torch.allclose(inner, torch.zeros_like(inner), atol=1e-10, rtol=0.0)
        )

    def test_expmap_stays_on_sphere(self):
        torch.manual_seed(2)
        x = _rand_unit_sphere(128, 6)

        # random tangent, scaled moderate
        u = torch.randn(128, 6, dtype=torch.float64)
        u = self.sphere.proju(x, u)
        u = u * 0.4  # moderate step

        y = self.sphere.expmap(x, u)
        self.assertTrue(
            torch.allclose(
                y.norm(dim=-1), torch.ones(128, dtype=torch.float64), atol=1e-9
            )
        )

    def test_logmap_is_tangent(self):
        torch.manual_seed(3)
        x = _rand_unit_sphere(128, 8)
        y = _rand_unit_sphere(128, 8)

        v = self.sphere.logmap(x, y)
        inner = (x * v).sum(dim=-1)
        self.assertTrue(
            torch.allclose(inner, torch.zeros_like(inner), atol=1e-9, rtol=0.0)
        )

    def test_exp_log_inverse_general(self):
        """
        exp(x, log(x, y)) == y for generic pairs, including antipodal, because
        the implementation chooses a deterministic tangent at theta ~ pi.
        """
        torch.manual_seed(4)
        x = _rand_unit_sphere(256, 7)

        # Construct y via exp to avoid pathological numeric issues and to ensure y on sphere
        u = torch.randn(256, 7, dtype=torch.float64)
        u = self.sphere.proju(x, u)
        # Use a range of angles up to near pi (avoid exact pi here; antipodal tested separately)
        angles = torch.rand(256, 1, dtype=torch.float64) * (math.pi - 1e-3)
        u_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = u_hat * angles

        y = self.sphere.expmap(x, u)
        v = self.sphere.logmap(x, y)
        y_rec = self.sphere.expmap(x, v)

        self.assertTrue(torch.allclose(y_rec, y, atol=1e-8, rtol=1e-6))

    def test_small_angle_log_exp_inverse(self):
        """
        For small steps, log(exp(x,u)) ~ u.
        """
        torch.manual_seed(5)
        x = _rand_unit_sphere(512, 9)
        u = torch.randn(512, 9, dtype=torch.float64)
        u = self.sphere.proju(x, u)
        u = u * 1e-4  # tiny

        y = self.sphere.expmap(x, u)
        u_rec = self.sphere.logmap(x, y)
        self.assertTrue(torch.allclose(u_rec, u, atol=1e-8, rtol=1e-4))

    def test_antipodal_case(self):
        """
        y = -x has non-unique log direction; implementation should return a pi-length tangent
        orthogonal to x, and exp should map back to -x.
        """
        torch.manual_seed(6)
        x = _rand_unit_sphere(128, 6)
        y = -x

        v = self.sphere.logmap(x, y)
        # Tangent: orthogonal to x
        inner = (x * v).sum(dim=-1)
        self.assertTrue(
            torch.allclose(inner, torch.zeros_like(inner), atol=1e-8, rtol=0.0)
        )
        # Norm should be close to pi
        self.assertTrue(
            torch.allclose(
                v.norm(dim=-1),
                torch.full((128,), math.pi, dtype=torch.float64),
                atol=1e-6,
                rtol=1e-6,
            )
        )

        y_rec = self.sphere.expmap(x, v)
        self.assertTrue(torch.allclose(y_rec, y, atol=1e-8, rtol=1e-6))

    def test_dist_properties_and_matches_arccos(self):
        torch.manual_seed(7)
        x = _rand_unit_sphere(256, 5)
        y = _rand_unit_sphere(256, 5)
        z = _rand_unit_sphere(256, 5)

        d_xy = self.sphere.dist(x, y)  # (...,1)
        d_yx = self.sphere.dist(y, x)
        d_xx = self.sphere.dist(x, x)
        d_xz = self.sphere.dist(x, z)
        d_yz = self.sphere.dist(y, z)

        # symmetry
        self.assertTrue(torch.allclose(d_xy, d_yx, atol=1e-12, rtol=1e-10))
        # identity
        self.assertTrue(
            torch.allclose(d_xx, torch.zeros_like(d_xx), atol=1e-12, rtol=0.0)
        )
        # triangle inequality (numeric slack)
        self.assertTrue(torch.all(d_xz <= d_xy + d_yz + 1e-10))

        # Compare with acos(<x,y>) (clamped)
        dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        d_ref = torch.acos(dot)
        self.assertTrue(torch.allclose(d_xy, d_ref, atol=1e-10, rtol=1e-8))

    def test_geodesic_interpolation(self):
        """
        Along the minimal geodesic, distances should scale approximately linearly with t.
        """
        torch.manual_seed(8)
        x0 = _rand_unit_sphere(128, 4)
        # Build x1 via expmap to ensure unique minimal geodesic (avoid antipodal)
        u = torch.randn(128, 4, dtype=torch.float64)
        u = self.sphere.proju(x0, u)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = u * (0.9 * math.pi)  # still < pi, but not too close
        x1 = self.sphere.expmap(x0, u)

        v = self.sphere.logmap(x0, x1)
        d01 = self.sphere.dist(x0, x1)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            xt = self.sphere.expmap(x0, t * v)
            d0t = self.sphere.dist(x0, xt)
            dt1 = self.sphere.dist(xt, x1)

            self.assertTrue(torch.allclose(d0t, t * d01, atol=1e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(dt1, (1.0 - t) * d01, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
