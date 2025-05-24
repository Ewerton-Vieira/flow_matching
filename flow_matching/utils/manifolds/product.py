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

    def __init__(self, input_dim: int, manifolds: List[Tuple[Manifold, int]] = None):
        """
        Initialize a product manifold with arbitrary ordering and dimensions.
        
        Args:
            input_dim: Total dimensionality of the space
            manifolds: List of tuples (manifold, dimension) where manifold is a Manifold object and dimension is an integer
                       Default is None, which will use Euclidean space for all dimensions
        """
        super().__init__()
        
        # Store manifolds and dimensions as lists
        self.manifolds: List[Manifold] = []
        self.dimensions: List[int] = []
        
        # If manifolds is None or empty, use Euclidean space for all dimensions
        if manifolds is None or len(manifolds) == 0:
            self.manifolds.append(Euclidean())
            self.dimensions.append(input_dim)
        else:
            # Otherwise use the provided manifolds
            for manifold, dim in manifolds:
                self.manifolds.append(manifold)
                self.dimensions.append(dim)

        if len(self.manifolds) > 50:
            warnings.warn("Product manifold has more than 50 manifolds. This may lead to performance issues.")

    def split(self, x: Tensor) -> List[Tensor]:
        """Split input tensor according to manifold dimensions"""
        return torch.split(x, self.dimensions, dim=-1)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_u = self.split(u)
        
        results = []
        for i, manifold in enumerate(self.manifolds):
            results.append(manifold.expmap(tensors_x[i], tensors_u[i]))
            
        return torch.cat(results, dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_y = self.split(y)
        
        results = []
        for i, manifold in enumerate(self.manifolds):
            results.append(manifold.logmap(tensors_x[i], tensors_y[i]))
            
        return torch.cat(results, dim=-1)

    def projx(self, x: Tensor) -> Tensor:
        tensors_x = self.split(x)
        
        results = []
        for i, manifold in enumerate(self.manifolds):
            results.append(manifold.projx(tensors_x[i]))
            
        return torch.cat(results, dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_u = self.split(u)
        
        results = []
        for i, manifold in enumerate(self.manifolds):
            results.append(manifold.proju(tensors_x[i], tensors_u[i]))
            
        return torch.cat(results, dim=-1)
    
    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_y = self.split(y)
        
        distances = []
        for i, manifold in enumerate(self.manifolds):
            distances.append(manifold.dist(tensors_x[i], tensors_y[i]))

        return torch.cat(distances, dim=-1)
