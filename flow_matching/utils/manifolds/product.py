# Ewerton R Vieira
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from typing import List, Tuple, Any
import warnings
from flow_matching.utils.manifolds import Euclidean, Sphere, FlatTorus, Manifold

class Product(Manifold):
    """The product of manifolds: Sphere, Torus and Euclidean."""

    def __init__(self, input_dim: int, manifolds: List[Tuple[Manifold, int]]):
        """
        Initialize a product manifold with arbitrary ordering and dimensions.
        
        Args:
            input_dim: Total dimensionality of the space
            manifolds: List of tuples (manifold, dimension) where manifold is a Manifold object and dimension is an integer
                       Default is None, which will use Euclidean space for all dimensions
        """
        super().__init__()

        assert sum(dim for _, dim in manifolds) == input_dim, "Sum of dimensions must match input_dim"

        m_list: List[Manifold] = []
        d_list: List[int] = []

        for manifold, dim in manifolds:
            m_list.append(manifold)
            d_list.append(int(dim))

        self.manifolds = tuple(m_list)
        self.dimensions = tuple(d_list)

        cum = [0]
        for d in self.dimensions:
            cum.append(cum[-1] + d)
        self._slices = tuple(slice(cum[i], cum[i + 1]) for i in range(len(self.dimensions)))

        if len(self.manifolds) > 50:
            warnings.warn("Product manifold has more than 50 manifolds. This may lead to performance issues.")

    def split(self, x: Tensor) -> List[Tensor]:
        """Split input tensor according to manifold dimensions"""
        return [x[..., s] for s in self._slices]

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        out = torch.empty_like(x)
        for manifold, s in zip(self.manifolds, self._slices):
            out[..., s] = manifold.expmap(x[..., s], u[..., s])
        return out

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        out = torch.empty_like(x)
        for manifold, s in zip(self.manifolds, self._slices):
            out[..., s] = manifold.logmap(x[..., s], y[..., s])
        return out

    def projx(self, x: Tensor) -> Tensor:
        out = torch.empty_like(x)
        for manifold, s in zip(self.manifolds, self._slices):
            out[..., s] = manifold.projx(x[..., s])
        return out

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        out = torch.empty_like(u)
        for manifold, s in zip(self.manifolds, self._slices):
            out[..., s] = manifold.proju(x[..., s], u[..., s])
        return out

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        outs: List[Tensor] = []
        for manifold, s in zip(self.manifolds, self._slices):
            outs.append(manifold.dist(x[..., s], y[..., s]))
        return torch.cat(outs, dim=-1)
