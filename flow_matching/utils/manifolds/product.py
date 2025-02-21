# Ewerton R Vieira
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

from flow_matching.utils.manifolds import Euclidean, Sphere, FlatTorus
from flow_matching.utils.manifolds import Manifold as Manifold1

class Product(Manifold1):
    """The product of manifolds, Sphere, Torus and Euclidean."""

    def __init__(self, sphere_dim = 0, torus_dim = 0, euclidian_dim = 0):
        super().__init__()
        self.sphere = Sphere()
        self.torus = FlatTorus()
        self.euclidian = Euclidean()
        self.sphere_dim = sphere_dim
        self.torus_dim = torus_dim
        self.euclidian_dim = euclidian_dim

    def split(self, x: Tensor) -> Tensor:
        return torch.split(x, [self.sphere_dim, self.torus_dim, self.euclidian_dim], dim=-1)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_u = self.split(u)
        exp = (self.sphere.expmap(tensors_x[0], tensors_u[0]), self.torus.expmap(tensors_x[1], tensors_u[1]), 
               self.euclidian.expmap(tensors_x[2], tensors_u[2]))
        return torch.cat(exp, dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_y = self.split(y)
        log = (self.sphere.logmap(tensors_x[0], tensors_y[0]), self.torus.logmap(tensors_x[1], tensors_y[1]), 
               self.euclidian.logmap(tensors_x[2], tensors_y[2]))
        return torch.cat(log, dim=-1)

    def projx(self, x: Tensor) -> Tensor:
        tensors_x = self.split(x)
        proj = (self.sphere.projx(tensors_x[0]), self.torus.projx(tensors_x[1]), 
               self.euclidian.projx(tensors_x[2]))
        return torch.cat(proj, dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_u = self.split(u)
        proj = (self.sphere.proju(tensors_x[0], tensors_u[0]), self.torus.proju(tensors_x[1], tensors_u[1]), 
               self.euclidian.proju(tensors_x[2], tensors_u[2]))
        return torch.cat(proj, dim=-1)
    
    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        tensors_x = self.split(x)
        tensors_y = self.split(y)
        d = (self.sphere.dist(tensors_x[0], tensors_y[0]), self.torus.dist(tensors_x[1], tensors_y[1]), 
               self.euclidian.dist(tensors_x[2], tensors_y[2]))
        return torch.norm(d)
