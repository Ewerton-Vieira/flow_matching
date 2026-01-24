# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import torch
from torch import Tensor

from flow_matching.utils.manifolds import Manifold


class Sphere(Manifold):
    r"""
    Unit hypersphere :math:`\mathbb{S}^{D-1} \subset \mathbb{R}^{D}` with the
    standard (round) Riemannian metric, using an *embedded/ambient* tangent
    representation.

    Representation
    -------------
    - Points are ambient vectors :math:`x \in \mathbb{R}^{D}` constrained to unit
      norm: :math:`\|x\| = 1`.
    - Tangent vectors at :math:`x` are also represented in the ambient space
      :math:`\mathbb{R}^{D}` as vectors orthogonal to :math:`x`:

      .. math::
          T_x \mathbb{S}^{D-1} = \{u \in \mathbb{R}^{D} : \langle x, u \rangle = 0\}.

      The projection `proju(x, u)` implements the orthogonal projection onto
      this subspace.

    Maps
    ----
    - Exponential map (geodesic) for :math:`u \in T_x \mathbb{S}^{D-1}`:

      .. math::
          \operatorname{Exp}_x(u) = x \cos(\|u\|) + \frac{u}{\|u\|}\sin(\|u\|).

    - Logarithm map returning :math:`u \in T_x \mathbb{S}^{D-1}` such that
      :math:`\operatorname{Exp}_x(u)=y` (for non-antipodal pairs), computed via
      the great-circle angle :math:`\theta = \arccos(\langle x,y\rangle)` and the
      tangent direction obtained by removing the component of :math:`y` along
      :math:`x`.

      Near :math:`\theta \approx 0` the log map returns the zero vector.
      Near antipodal pairs (:math:`\theta \approx \pi`) the log map is not unique;
      this implementation deterministically selects an arbitrary unit tangent
      direction orthogonal to :math:`x`.

    Distance
    --------
    Geodesic distance (great-circle distance):

    .. math::
        d(x,y) = \arccos(\langle x, y \rangle) \in [0,\pi],

    with inner products clamped for numerical stability.

    Notes
    -----
    - `projx` normalizes inputs to the unit sphere.
    - All operations are vectorized over batch dimensions.
    """

    # Epsilon values for different data types to handle numerical stability
    EPS = defaultdict(lambda: 1e-6, {torch.float32: 1e-6, torch.float64: 1e-12})

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        # x on sphere, u in T_x S^(D-1)
        # Formula: exp_x(u) = x * cos(||u||) + (u / ||u||) * sin(||u||)
        # For small ||u||: sin(||u||) → 0, so (u / ||u||) * sin(||u||) → 0
        # regardless of the clamped division, giving exp_x(u) → x (correct)
        eps = self.EPS[u.dtype]
        u_norm = u.norm(dim=-1, keepdim=True)
        u_hat = u / u_norm.clamp_min(eps)
        cos_t = torch.cos(u_norm)
        sin_t = torch.sin(u_norm)
        return x * cos_t + u_hat * sin_t

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        # Returns v in T_x S^(D-1) such that Exp_x(v) = y
        eps = self.EPS[x.dtype]
        # Ensure unit inputs
        x = self.projx(x)
        y = self.projx(y)

        cos_th = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        # Handle near-antipodal (cos_th ~ -1) first, before standard computation
        # which becomes unstable when y ≈ -x
        anti = cos_th < -1.0 + 1e-6  # shape (..., 1)

        # For antipodal case: pick arbitrary v ⟂ x, scale by π
        k = torch.argmin(torch.abs(x), dim=-1, keepdim=True)  # (..., 1)
        e = torch.zeros_like(x).scatter(-1, k, 1.0)  # basis vector
        v_alt = self.proju(x, e)
        v_alt = v_alt / v_alt.norm(dim=-1, keepdim=True).clamp_min(eps)
        log_anti = v_alt * torch.pi

        # Standard case
        th = torch.acos(cos_th)  # θ ∈ [0, π]
        # Tangent direction toward y
        u = y - cos_th * x  # = y - ⟨x,y⟩ x
        sin_th = u.norm(dim=-1, keepdim=True).clamp_min(eps)
        v = u / sin_th  # unit tangent
        log_std = v * th  # θ * v

        # Handle near-identical (θ ~ 0): return zero vector in T_x
        small = (th < 1e-6).expand_as(log_std)
        log_std = torch.where(small, torch.zeros_like(log_std), log_std)

        # Combine: use antipodal result where cos_th ≈ -1
        log = torch.where(anti.expand_as(log_std), log_anti, log_std)
        return log

    def projx(self, x: Tensor) -> Tensor:
        eps = self.EPS[x.dtype]
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        # Use the stable formula: d = 2 * arcsin(||y - x|| / 2)
        # This is equivalent to acos(<x,y>) for unit vectors but more stable
        # near d=0 (identical points) and d=π (antipodal points)
        diff = y - x
        half_chord = diff.norm(dim=-1, keepdim=True) / 2.0
        half_chord = half_chord.clamp(max=1.0)  # numerical stability
        d = 2.0 * torch.asin(half_chord)
        return d
