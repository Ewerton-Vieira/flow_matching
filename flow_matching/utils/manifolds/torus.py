# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

from flow_matching.utils.manifolds import Manifold


class FlatTorus(Manifold):
    r"""Represents a flat torus on the :math:`[0, 2\pi]^D` subspace. Isometric to the product of 1-D spheres."""

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return (x + u) % (2 * torch.pi)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def projx(self, x: Tensor) -> Tensor:
        return x % (2 * torch.pi)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u
    
    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        delta = torch.abs(x - y)  # Euclidean difference
        delta_torus = torch.minimum(delta, 2 * torch.pi - delta)  # Apply periodic wrapping
        return delta_torus
