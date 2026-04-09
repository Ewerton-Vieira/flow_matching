import torch
from torch import Tensor

from collections import defaultdict
from flow_matching.utils.manifolds.manifold import Manifold
from flow_matching.utils.manifolds.so3 import SO3


class SE3(Manifold):
    r"""
    Special Euclidean group in 3D, :math:`\mathrm{SE}(3)`, represented by
    translation + unit quaternion.

    Representation
    -------------
    - Points are 7D vectors :math:`[t_x, t_y, t_z, q_w, q_x, q_y, q_z]`
      with translation first, then scalar-first unit quaternion.
    - Tangent vectors are 6D twists :math:`[v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]`
      with translational velocity first (Sophus convention).

    Reference: docs/sophus_se3_reference.hpp
    """

    EPS = defaultdict(lambda: 1e-6, {
        torch.float16: 1e-4,
        torch.bfloat16: 1e-4,
        torch.float32: 1e-6,
        torch.float64: 1e-9,
    })

    _so3 = SO3()

    def projx(self, x: Tensor) -> Tensor:
        """Project to SE(3): translation unchanged, quaternion normalized."""
        return torch.cat([x[..., :3], SO3.normalize(x[..., 3:])], dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project to tangent space — identity (Lie algebra coordinates)."""
        return u

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Decoupled distance: returns (..., 4) = [|dt_0|, |dt_1|, |dt_2|, theta].
        Translation: per-component absolute difference.
        Rotation: SO(3) geodesic angle.
        """
        t_dist = torch.abs(x[..., :3] - y[..., :3])
        rot_dist = self._so3.dist(x[..., 3:], y[..., 3:])
        return torch.cat([t_dist, rot_dist], dim=-1)

    def canon(self, x: Tensor) -> Tensor:
        """Canonical representative: translation unchanged, quaternion canonicalized."""
        return torch.cat([x[..., :3], SO3.canon(SO3.normalize(x[..., 3:]))], dim=-1)

    def _so3_expmap_with_theta(self, q: Tensor, omega: Tensor, theta: Tensor) -> Tensor:
        """
        SO3 exponential map reusing pre-computed theta = ||omega||.
        Reference: so3_reference.hpp:716-752 (expAndTheta)

        Computes: normalize(q * delta) where delta = [cos(theta/2), sin(theta/2)/theta * omega]
        """
        eps = self.EPS[q.dtype]
        q = SO3.normalize(q)
        theta_sq = theta * theta
        half = 0.5 * theta

        # Small-angle: so3_reference.hpp lines 731-733
        # imag_factor = 0.5 - theta^2/48 + theta^4/3840
        # real_factor = 1 - theta^2/8 + theta^4/384
        theta_po4 = theta_sq * theta_sq
        imag_factor_small = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4
        real_factor_small = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4

        # General: so3_reference.hpp lines 739-740
        imag_factor_general = torch.where(
            theta > eps, torch.sin(half) / theta, torch.zeros_like(theta)
        )
        real_factor_general = torch.cos(half)

        small = theta_sq < eps * eps
        imag_factor = torch.where(small, imag_factor_small, imag_factor_general)
        real_factor = torch.where(small, real_factor_small, real_factor_general)

        delta = torch.cat([real_factor, imag_factor * omega], dim=-1)
        return SO3.normalize(SO3.product(q, delta))

    def _left_jacobian_act(self, omega: Tensor, v: Tensor, theta: Tensor) -> Tensor:
        """
        Compute V*v via cross products (no 3x3 matrix).
        Reference: so3_reference.hpp:570-592 (leftJacobian)

        V*v = v + a*(omega x v) + b*(omega x (omega x v))
        where a = (1-cos theta)/theta^2, b = (theta-sin theta)/theta^3
        Small-angle: a = 0.5, b ~ 0 (so3_reference.hpp line 584: V = I + 0.5*Omega)
        """
        eps = self.EPS[omega.dtype]
        theta_sq = theta * theta

        # General coefficients: so3_reference.hpp lines 588-589
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        a_general = (1.0 - cos_theta) / theta_sq
        b_general = (theta - sin_theta) / (theta_sq * theta)

        # Small-angle coefficients: so3_reference.hpp line 584
        a_small = torch.full_like(theta, 0.5)
        b_small = torch.zeros_like(theta)

        small = theta_sq < eps * eps
        a = torch.where(small, a_small, a_general)
        b = torch.where(small, b_small, b_general)

        cross1 = torch.cross(omega, v, dim=-1)
        cross2 = torch.cross(omega, cross1, dim=-1)
        return v + a * cross1 + b * cross2

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        SE(3) exponential map at base point: x * exp(u).
        Reference: se3_reference.hpp:850-858 (exp), lines 302-305 (operator*)

        Given x = (t_x, q_x) and u = (v, omega):
            theta = ||omega||
            Vv = leftJacobian(omega) * v          (se3 line 858)
            t_new = t_x + R(q_x) * Vv             (se3 line 305)
            q_new = SO3.exp(q_x, omega)
        """
        t_x, q_x = x[..., :3], x[..., 3:]
        v, omega = u[..., :3], u[..., 3:]

        # Compute theta ONCE
        theta = omega.norm(dim=-1, keepdim=True)

        # V * v via cross products (so3 lines 570-592)
        Vv = self._left_jacobian_act(omega, v, theta)

        # R_x * (V * v) via quaternion rotation (no rotation matrix)
        t_delta = SO3.quat_action(q_x, Vv)

        # SO3 expmap reusing theta (so3 lines 716-752)
        q_new = self._so3_expmap_with_theta(q_x, omega, theta)

        t_new = t_x + t_delta
        return torch.cat([t_new, q_new], dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError("SE3.logmap will be implemented in Task 6")
