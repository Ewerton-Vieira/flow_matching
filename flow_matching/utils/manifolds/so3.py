# so3.py
# SO(3) manifold represented with unit quaternions (S^3 / {±1}).

import torch
from torch import Tensor

from collections import defaultdict
from flow_matching.utils.manifolds import Manifold


class SO3(Manifold):
    r"""
    Special orthogonal group in 3D, :math:`\mathrm{SO}(3)`, represented by unit
    quaternions with antipodal identification.

    Representation
    -------------
    - Points are unit quaternions :math:`q \in \mathbb{S}^3 \subset \mathbb{R}^4`
      in scalar-first convention :math:`q = [w, x, y, z]`, with the equivalence
      :math:`q \sim -q` (same rotation).
    - This implementation uses a *Lie algebra coordinate* tangent representation:
      tangent vectors are represented as rotation vectors
      :math:`\omega \in \mathbb{R}^3` (axis-angle / exponential coordinates),
      with :math:`\|\omega\|` equal to the rotation angle.

      In this representation, `proju(x, u)` is the identity because tangents are
      already expressed in a fixed :math:`\mathbb{R}^3` coordinate chart rather
      than as ambient :math:`\mathbb{R}^4` vectors orthogonal to :math:`q`.

    Conventions
    -----------
    - Canonical representative: SO(3) has the identification `q ~ -q`, so there is
      no globally smooth choice of representative. This file therefore separates:
        - `projx`: smooth projection to the unit sphere (normalization only).
        - `canon`: deterministic (but discontinuous) representative selection.
        - `projx_canon`: convenience = normalize then canonicalize.
    - Relative rotation: :math:`\Delta = q_x^* \otimes q_y` (Hamilton product,
      scalar-first), flipped to the shortest arc by enforcing :math:`\Delta_w \ge 0`.
    - Returned angles are in :math:`[0,\pi]`.

    Maps
    ----
    - Exponential map (apply a rotation vector at a base quaternion):

      Let :math:`\phi = \|\omega\|`, :math:`a = \omega/\phi` (when :math:`\phi>0`),
      and

      .. math::
          \delta(\omega) = [\cos(\phi/2),\; a \sin(\phi/2)].

      Then

      .. math::
          \operatorname{Exp}_q(\omega) = \mathrm{projx}(q \otimes \delta(\omega)).

    - Logarithm map (rotation vector taking `x` to `y`):

      Compute the relative quaternion :math:`\Delta = q_x^* \otimes q_y` (shortest arc).
      Let :math:`s = \|\Delta_{xyz}\| = \sin(\phi/2)` and :math:`w = \Delta_w`.
      Then

      .. math::
          \phi = 2\,\mathrm{atan2}(s, w), \quad
          \omega = \frac{\phi}{s}\,\Delta_{xyz},

      with a small-angle series used when :math:`s \approx 0` for numerical stability.

    Distance
    --------
    Geodesic distance:

    .. math::
        d(x,y) = \phi = 2\,\mathrm{atan2}(\|\Delta_{xyz}\|, \Delta_w) \in [0,\pi],

    where :math:`\Delta = q_x^* \otimes q_y` is taken in the shortest-arc
    representative (:math:`\Delta_w \ge 0`).

    Notes
    -----
    - All operations are vectorized over batch dimensions.
    - This implementation uses rotation-vector tangents in :math:`\mathbb{R}^3`,
      not ambient :math:`\mathbb{R}^4` tangent vectors orthogonal to the base
      quaternion.
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

    def _relative_quat_shortest_with_pi_tiebreak(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute Delta = x* ⊗ y, enforce shortest arc (Delta.w >= 0),
        and additionally make Delta deterministic when Delta.w ≈ 0 (near π).

        The π tie-break picks the sign so that the largest-magnitude component of v is >= 0.
        """
        eps = self.EPS[x.dtype]

        # Normalize only (do not canon here)
        x = self.normalize(x)
        y = self.normalize(y)

        Delta = self.product(self._qconj(x), y)  # (...,4)

        # Shortest arc: flip to make w >= 0
        Delta = torch.where(Delta[..., :1] < 0, -Delta, Delta)

        w = Delta[..., :1]      # (...,1)
        v = Delta[..., 1:]      # (...,3)

        # ----- π tie-break region -----
        w_thresh = 1e-4 if x.dtype in (torch.float32, torch.float64) else 1e-3

        near_pi = (w.abs() < w_thresh)

        # If v is extremely tiny, don't flip
        v_norm = v.norm(dim=-1, keepdim=True)
        safe = (v_norm > eps)

        # Pick the index of the largest-magnitude component of v
        idx = v.abs().argmax(dim=-1, keepdim=True)        # (...,1) in {0,1,2}
        max_comp = v.gather(-1, idx)                      # (...,1)

        # Flip so that that component is >= 0
        flip = torch.where(max_comp < 0, -1.0, 1.0).to(Delta.dtype)  # (...,1)

        # Apply only in the near-pi region (and only if v is nonzero)
        Delta = torch.where(near_pi & safe, Delta * flip, Delta)

        return Delta

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

    @staticmethod
    def normalize(q: Tensor) -> Tensor:
        """Normalize quaternion(s) to unit norm (no canonicalization)."""
        eps = SO3.EPS[q.dtype]
        inv_norm = torch.rsqrt((q * q).sum(dim=-1, keepdim=True).clamp_min(eps * eps))
        return q * inv_norm

    def projx_canon(self, x: Tensor) -> Tensor:
        """Normalize then apply canonical sign convention (discontinuous)."""
        return self.canon(self.normalize(x))

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Lie-algebra (so(3)) log map with small-angle series for stability.
        Returns rotation vector ω with ||ω|| = angle ∈ [0, π].
        """
        eps = self.EPS[x.dtype]

        Delta = self._relative_quat_shortest_with_pi_tiebreak(x, y)

        w = Delta[..., 0:1]
        v = Delta[..., 1:]
        s = v.norm(dim=-1, keepdim=True)

        # ---- Small-angle series branch (about s = 0) ----
        # φ/s = 2 + (1/3)s^2 + (3/20)s^4 + (5/56)s^6 + ...
        s2 = s * s
        scale_series = (2.0
                        + (1.0/3.0) * s2
                        + (3.0/20.0) * s2 * s2
                        + (5.0/56.0) * s2 * s2 * s2)

        # ---- General branch ----
        phi = 2.0 * torch.atan2(s, w)
        scale_general = torch.where(s > eps, phi / s, torch.zeros_like(s))

        s_thresh = min(eps * 1e3, 1e-4)
        scale = torch.where(s < s_thresh, scale_series, scale_general)

        return scale * v

    def expmap(self, x: Tensor, omega: Tensor) -> Tensor:
        """
        Lie-algebra (so(3)) exp map: apply rotation vector ω at base x.
        Input: x (...,4), ω (...,3) with ||ω|| = angle.
        Output: y (...,4) quaternion (canonicalized).
        """
        eps = self.EPS[x.dtype]
        # Normalize base point (avoid canonicalization-induced discontinuities)
        x = self.normalize(x)
        phi = omega.norm(dim=-1, keepdim=True)   # angle
        half = 0.5 * phi

        # Unit axis (where defined)
        a = torch.where(phi > eps, omega / phi, torch.zeros_like(omega))

        # Increment quaternion δ = [cos(φ/2), a*sin(φ/2)]
        c = torch.cos(half)
        s = torch.sin(half)
        delta = torch.cat([c, s * a], dim=-1)    # (...,4)

        y = self.product(x, delta)
        # Normalize output (do not canonicalize here; representation is defined up to sign)
        return self.normalize(y)

    def projx(self, x: Tensor) -> Tensor:
        """Smooth projection to the unit quaternion sphere (normalization only)."""
        return self.normalize(x)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Projects a 3D vector to the tangent space at x (orthogonal component).
        But here the projection at x is the same as the input vector.
        """
        return u

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Stable geodesic distance on SO(3) using relative quaternion and atan2."""
        Delta = self._relative_quat_shortest_with_pi_tiebreak(x, y)

        w = Delta[..., 0:1].clamp(-1.0, 1.0)
        v = Delta[..., 1:]
        s = v.norm(dim=-1, keepdim=True)

        # φ = 2 * atan2(‖v‖, w) ∈ [0, π]
        return 2.0 * torch.atan2(s, w)
