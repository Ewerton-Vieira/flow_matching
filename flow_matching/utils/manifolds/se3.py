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

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        raise NotImplementedError("SE3.expmap will be implemented in Task 5")

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError("SE3.logmap will be implemented in Task 6")
