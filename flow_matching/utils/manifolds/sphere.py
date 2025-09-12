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
    r"""Unit hypersphere S^{D-1} ⊂ ℝ^D."""

    # Epsilon values for different data types to handle numerical stability
    EPS = defaultdict(lambda: 1e-6, {torch.float32: 1e-6, torch.float64: 1e-12})

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        # x on sphere, u in T_x S^(D-1)
        eps = self.EPS[u.dtype]
        u_norm = u.norm(dim=-1, keepdim=True)
        # Safe unit direction in tangent
        u_hat = u / u_norm.clamp_min(eps)
        cos_t = torch.cos(u_norm)
        sin_t = torch.sin(u_norm)
        exp = x * cos_t + u_hat * sin_t
        # Retraction fallback for tiny steps
        retr = self.projx(x + u)
        cond = (u_norm > (10.0 * eps))  # slightly larger threshold
        return torch.where(cond, exp, retr)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        # Returns v in T_x S^(D-1) such that Exp_x(v) = y
        eps = self.EPS[x.dtype]
        # Ensure unit inputs
        x = self.projx(x)
        y = self.projx(y)

        cos_th = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
        th = torch.acos(cos_th)                            # θ ∈ (0, π)
        # Tangent direction toward y
        u = y - cos_th * x                                 # = y - ⟨x,y⟩ x
        sin_th = u.norm(dim=-1, keepdim=True).clamp_min(eps)
        v = u / sin_th                                     # unit tangent
        log = v * th                                       # θ * v

        # Handle near-identical (θ ~ 0): return zero vector in T_x
        small = (th < 1e-6).expand_as(log)
        log = torch.where(small, torch.zeros_like(log), log)

        # Handle near-antipodal (θ ~ π): direction is undefined; pick any v ⟂ x
        anti = (torch.abs(th - torch.tensor(torch.pi, dtype=th.dtype, device=th.device)) < 1e-6).expand_as(log)
        if anti.any():
            # choose a coordinate axis least aligned with x, then project & normalize
            D = x.shape[-1]
            # pick basis e_k with smallest |x_k|
            k = torch.argmin(torch.abs(x), dim=-1, keepdim=True)
            e = torch.zeros_like(x).scatter(-1, k, 1.0)
            v_alt = self.proju(x, e)
            v_alt = v_alt / v_alt.norm(dim=-1, keepdim=True).clamp_min(eps)
            log_alt = v_alt * th
            log = torch.where(anti, log_alt, log)

        return log

    def projx(self, x: Tensor) -> Tensor:
        eps = self.EPS[x.dtype]
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        eps = self.EPS[x.dtype]
        inner = (x * y).sum(dim=-1, keepdim=True)
        inner = inner.clamp(min=-1.0 + eps, max=1.0 - eps)
        d = torch.acos(inner)
        return d