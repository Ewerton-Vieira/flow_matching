# so3.py
# SO(3) manifold represented with unit quaternions (S^3 / {±1}).
# Mirrors the API used by Sphere/Torus/Manifold.

import torch
from torch import Tensor

from flow_matching.utils.manifolds import Manifold


class SO3(Manifold):
    r"""Special orthogonal group in 3D, :math:`\mathrm{SO}(3)`, represented by unit
    quaternions :math:`q \in \mathbb{S}^3` with the antipodal identification
    :math:`q \sim -q`. Points are 4D tensors whose last dimension is 4.

    Tangent vectors live in the 3D tangent space but are represented in
    :math:`\mathbb{R}^4` and must be orthogonal to the base point :math:`q`.
    This class implements exponential/logarithm maps, projections, and the
    geodesic distance consistent with the quotient metric.

    Conventions
    -----------
    - Canonical representative: we enforce the scalar part q_w >= 0 in `projx`.
    - Distance: d(R1, R2) = 2 * acos(|<q1, q2>|) ∈ [0, π].
    """

    EPS = {torch.float32: 1e-6, torch.float64: 1e-9}

    @staticmethod
    def from_euler(angles: Tensor, order: str = "zyx", degrees: bool = False) -> Tensor:
        """
        Convert Euler angles to a unit quaternion [qw, qx, qy, qz].

        Parameters
        ----------
        angles : Tensor
            Tensor of shape (..., 3). Interpretation depends on `order`:
            - order='zyx': angles = [yaw(z), pitch(y), roll(x)]
            - order='xyz': angles = [roll(x), pitch(y), yaw(z)]
        order : str
            Either 'zyx' (yaw–pitch–roll) or 'xyz' (roll–pitch–yaw).
        degrees : bool
            If True, `angles` are in degrees. Otherwise radians.

        Returns
        -------
        q : Tensor
            Unit quaternion of shape (..., 4) with canonical sign (qw ≥ 0).
        """
        if angles.shape[-1] != 3:
            raise ValueError("angles must have shape (..., 3)")

        if degrees:
            angles = angles * (torch.pi / 180.0)

        order = order.lower()

        # Unpack to (roll, pitch, yaw) regardless of input order
        if order == "zyx":
            yaw, pitch, roll = angles.unbind(dim=-1)   # [ψ, θ, φ]
        elif order == "xyz":  # 'xyz'
            roll, pitch, yaw = angles.unbind(dim=-1)   # [φ, θ, ψ]
        else:
            raise ValueError("order must be 'zyx' or 'xyz'")

        half = 0.5
        hr = roll * half
        hp = pitch * half
        hy = yaw * half

        cr = torch.cos(hr)
        sr = torch.sin(hr)
        cp = torch.cos(hp)
        sp = torch.sin(hp)
        cy = torch.cos(hy)
        sy = torch.sin(hy)

        # Standard yaw–pitch–roll (Z-Y-X) closed form (works given (roll,pitch,yaw) above)
        qw = cy*cp*cr + sy*sp*sr
        qx = cy*cp*sr - sy*sp*cr
        qy = cy*sp*cr + sy*cp*sr
        qz = sy*cp*cr - cy*sp*sr

        q = torch.stack((qw, qx, qy, qz), dim=-1)

        # Normalize and canonicalize (qw ≥ 0)
        eps = 1e-9 if q.dtype == torch.float64 else 1e-6
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
        flip = (q[..., :1] < 0).to(q.dtype) * -2.0 + 1.0  # -1 if qw<0 else +1
        q = q * torch.cat([flip, flip, flip, flip], dim=-1)
        return q

    @staticmethod
    def _canon(q: Tensor) -> Tensor:
        """Canonical representative with non-negative scalar part (qw >= 0)."""
        # q shape (..., 4) with scalar part first: [qw, qx, qy, qz]
        flip = (q[..., :1] < 0).to(q.dtype) * -2.0 + 1.0  # -1 if qw<0 else +1
        return q * torch.cat([flip, flip, flip, flip], dim=-1)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Exponential map on SO(3) at quaternion x along tangent u (both (...,4))."""
        eps = self.EPS[x.dtype]
        # Ensure valid base point and tangent
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
        u = self.proju(x, u)

        theta = u.norm(dim=-1, keepdim=True).clamp_min(eps)
        s = torch.sin(theta) / theta
        y = torch.cos(theta) * x + s * u
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
        return self._canon(y)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithm map on SO(3) at quaternion x toward quaternion y (both (...,4))."""
        eps = self.EPS[x.dtype]
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)

        # Choose y (or -y) giving the shortest geodesic from x
        inner = (x * y).sum(dim=-1, keepdim=True)
        y = torch.where(inner < 0, -y, y)
        inner = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        # Tangent direction on S^3 orthogonal to x
        v = y - inner * x
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)

        # Angle on S^3 is alpha ∈ [0, π/2]; SO(3) angle is 2*alpha
        alpha = torch.acos(inner)  # (...,1)
        u = v / v_norm * (2.0 * alpha)

        # First-order fallback for near-coincident points
        u = torch.where((alpha < 1e-6).expand_as(u), 2.0 * (y - x), u)

        # Ensure it's in the tangent space numerically
        return self.proju(x, u)

    def projx(self, x: Tensor) -> Tensor:
        """Project a 4D vector to a unit quaternion with canonical sign (qw ≥ 0)."""
        eps = self.EPS[x.dtype]
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return self._canon(x)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project a 4D vector to the tangent space at x (orthogonal component)."""
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance on SO(3): 2*acos(|<x,y>|) ∈ [0, π]."""
        eps = self.EPS[x.dtype]
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
        inner = (x * y).sum(dim=-1, keepdim=True).abs().clamp_max(1.0)
        return 2.0 * torch.acos(inner)