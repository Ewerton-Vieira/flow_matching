# so3.py
# SO(3) manifold represented with unit quaternions (S^3 / {±1}).

import torch
from torch import Tensor

from collections import defaultdict
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

    EPS = defaultdict(lambda: 1e-6, {
        torch.float16: 1e-4,
        torch.bfloat16: 1e-4,
        torch.float32: 1e-6,
        torch.float64: 1e-9,
    })

    @staticmethod
    def product(q1: Tensor, q2: Tensor) -> Tensor:
        """Hamilton product, scalar-first [w, x, y, z], vectorized."""
        w1 = q1[..., :1]
        v1 = q1[..., 1:]
        w2 = q2[..., :1]
        v2 = q2[..., 1:]
        w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdim=True)
        v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)
        return torch.cat([w, v], dim=-1)


    @staticmethod
    def _qconj(q: Tensor) -> Tensor:
        """Quaternion conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
        return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

    @staticmethod
    def from_euler(angles: Tensor, order: str = "zyx", degrees: bool = False) -> Tensor:
        """
        Convert Euler angles to a unit quaternion [qw, qx, qy, qz].

        angles: (..., 3)
          - order='zyx': [yaw(z), pitch(y), roll(x)]
          - order='xyz': [roll(x), pitch(y), yaw(z)]
        """
        if angles.shape[-1] != 3:
            raise ValueError("angles must have shape (..., 3)")

        if degrees:
            angles = angles * (torch.pi / 180.0)

        order = order.lower()
        if order == "zyx":
            yaw, pitch, roll = angles.unbind(-1)
        elif order == "xyz":
            roll, pitch, yaw = angles.unbind(-1)
        else:
            raise ValueError("order must be 'zyx' or 'xyz'")

        half = 0.5
        hr, hp, hy = roll*half, pitch*half, yaw*half

        sr, cr = torch.sin(hr), torch.cos(hr)
        sp, cp = torch.sin(hp), torch.cos(hp)
        sy, cy = torch.sin(hy), torch.cos(hy)
        zeros_like_hr, zeros_like_hp, zeros_like_hy = torch.zeros_like(hr), torch.zeros_like(hp), torch.zeros_like(hy)


        # Axis half-angle quaternions
        qx = torch.stack((cr, sr, zeros_like_hr, zeros_like_hr), dim=-1)
        qy = torch.stack((cp, zeros_like_hp, sp, zeros_like_hp), dim=-1)
        qz = torch.stack((cy, zeros_like_hy, zeros_like_hy, sy), dim=-1)

        # Compose in the requested order: q_total = q_axisN ⊗ ... ⊗ q_axis1
        if order == "zyx":
            q = SO3.product(SO3.product(qz, qy), qx)  # R = Rz * Ry * Rx
        else:  # 'xyz'
            q = SO3.product(SO3.product(qx, qy), qz)  # R = Rx * Ry * Rz

        # Normalize and canonicalize
        eps = SO3.EPS[q.dtype]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
        return SO3.canon(q)
        
    @staticmethod
    def canon(q: Tensor) -> Tensor:
        """
        Canonical representative with stable tie-break:
        1) If |qw| > eps or qw < 0: ensure qw >= 0
        2) Else (qw ≈ 0 and qw > 0): ensure the largest-magnitude component among (qx,qy,qz) is >= 0
        This guarantees projx(q) == projx(-q).
        """
        eps = SO3.EPS[q.dtype]
        qw = q[..., :1]
        qv = q[..., 1:]
        use_qw = (qw.abs() > eps)
        flip_qw = torch.where(qw < 0, -1.0, 1.0).to(q.dtype)
        idx = qv.abs().argmax(dim=-1, keepdim=True)
        max_comp = qv.gather(-1, idx)
        flip_v = torch.where(max_comp < 0, -1.0, 1.0).to(q.dtype)
        flip = torch.where(use_qw, flip_qw, flip_v)             # (...,1)
        return q * flip 


    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Lie-algebra (so(3)) log map with small-angle series for stability.
        Returns rotation vector ω with ||ω|| = angle ∈ [0, π].
        """
        eps = self.EPS[x.dtype]

        # Normalize + canonicalize inputs
        x = self.projx(x)
        y = self.projx(y)

        # Relative quaternion Δ = x* ⊗ y, flip to shortest arc (Δ.w ≥ 0)
        Delta = self.product(self._qconj(x), y)
        Delta = torch.where(Delta[..., :1] < 0, -Delta, Delta)

        w = Delta[..., 0:1]           # scalar part
        v = Delta[..., 1:]            # vector part (...,3)
        s = v.norm(dim=-1, keepdim=True)  # ||v|| = sin(φ/2)

        # ---- Small-angle series branch (about s = 0) ----
        # φ/s = 2 + (1/3)s^2 + (3/20)s^4 + (5/56)s^6 + ...
        s2 = s * s
        scale_series = (2.0
                        + (1.0/3.0) * s2
                        + (3.0/20.0) * s2 * s2
                        + (5.0/56.0) * s2 * s2 * s2)

        # ---- General branch (stable everywhere) ----
        # φ = 2 * atan2(s, w); ω = (φ/s) v, with safe s→0 handling
        phi = 2.0 * torch.atan2(s, w.clamp_min(eps))
        scale_general = torch.where(s > eps, phi / s, torch.zeros_like(s))

        # Threshold for switching to series (tuned per dtype)
        # s ≈ sin(φ/2); s < 1e-4 is very small rotation for fp32/bfloat16.
        s_thresh = min(eps * 1e3, 1e-4)

        scale = torch.where(s < s_thresh, scale_series, scale_general)

        # ω = scale * v (note: if s=0 then v=0 → ω=0)
        omega = scale * v
        return omega

    def expmap(self, x: Tensor, omega: Tensor) -> Tensor:
        """
        Lie-algebra (so(3)) exp map: apply rotation vector ω at base x.
        Input: x (...,4), ω (...,3) with ||ω|| = angle.
        Output: y (...,4) quaternion (canonicalized).
        """
        eps = self.EPS[x.dtype]
        x = self.projx(x)
        phi = omega.norm(dim=-1, keepdim=True)   # angle
        half = 0.5 * phi

        # Unit axis (where defined)
        a = torch.where(phi > eps, omega / phi, torch.zeros_like(omega))

        # Increment quaternion δ = [cos(φ/2), a*sin(φ/2)]
        c = torch.cos(half)
        s = torch.sin(half)
        delta = torch.cat([c, s * a], dim=-1)    # (...,4)

        y = self.product(x, delta)
        return self.projx(y)

    def projx(self, x: Tensor) -> Tensor:
        """Unit + canonical sign (qw ≥ 0) with rsqrt normalization."""
        eps = self.EPS[x.dtype]
        inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True).clamp_min(eps * eps))
        q = x * inv_norm
        return self.canon(q)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Projects a 3D vector to the tangent space at x (orthogonal component).
        But here the projection at x is the same as the input vector.
        """
        return u

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Stable geodesic distance on SO(3) using relative quaternion and atan2."""
        eps = self.EPS[x.dtype]
        x = self.projx(x)
        y = self.projx(y)

        Delta = self.product(self._qconj(x), y)                  # Δ = x* ⊗ y
        Delta = torch.where(Delta[..., :1] < 0, -Delta, Delta) # enforce shortest: Δ.w ≥ 0

        w = Delta[..., 0:1].clamp(-1.0, 1.0)
        v = Delta[..., 1:]
        s = v.norm(dim=-1, keepdim=True)

        # φ = 2 * atan2(‖v‖, w) ∈ [0, π]
        phi = 2.0 * torch.atan2(s, w.clamp_min(eps))
        return phi