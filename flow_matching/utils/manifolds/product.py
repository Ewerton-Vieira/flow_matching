# Ewerton R Vieira
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from typing import List, Tuple
import warnings
from flow_matching.utils.manifolds import Euclidean, Sphere, FlatTorus, Manifold

class Product(Manifold):
    """The product of manifolds: Sphere, Torus and Euclidean."""

    def __init__(self, input_dim: int, manifolds: List[Tuple[Manifold, int]]):
        """
        Initialize a product manifold with arbitrary ordering and dimensions.

        Optimizes the operations for Euclidean and FlatTorus manifolds by vectorizing them over all dimensions corresponding to those manifolds.
        
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

        # Precompute indices that belong to Euclidean and FlatTorus to enable batched operations
        euclid_idx: List[int] = []
        torus_idx: List[int] = []
        for manifold, s in zip(self.manifolds, self._slices):
            if isinstance(manifold, Euclidean):
                euclid_idx.extend(range(s.start, s.stop))
            elif isinstance(manifold, FlatTorus):
                torus_idx.extend(range(s.start, s.stop))

        self._euclidean_indices = tuple(euclid_idx)
        self._flat_torus_indices = tuple(torus_idx)

        # Keep a reference instance for each type to call their vectorized ops
        self._euclidean_ref = next((m for m in self.manifolds if isinstance(m, Euclidean)), None)
        self._flat_torus_ref = next((m for m in self.manifolds if isinstance(m, FlatTorus)), None)

        if len(self.manifolds) > 50:
            warnings.warn("Product manifold has more than 50 manifolds. This may lead to performance issues.")

        self._cached_indices = {}  # {(device, dtype): (e_idx, t_idx)}

    def _get_indices(self, like):
        key = (like.device, torch.long)
        if key not in self._cached_indices:
            e = torch.as_tensor(self._euclidean_indices, device=like.device, dtype=torch.long) if self._euclidean_indices else None
            t = torch.as_tensor(self._flat_torus_indices, device=like.device, dtype=torch.long) if self._flat_torus_indices else None
            self._cached_indices[key] = (e, t)
        return self._cached_indices[key]

    def split(self, x: Tensor) -> List[Tensor]:
        """Split input tensor according to manifold dimensions"""
        return [x[..., s] for s in self._slices]

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        out = torch.empty_like(x)

        if self._euclidean_ref:
            idx = self._get_indices(x)[0]
            x_e = x.index_select(-1, idx)
            u_e = u.index_select(-1, idx)
            out[..., idx] = self._euclidean_ref.expmap(x_e, u_e)

        if self._flat_torus_ref:
            idx = self._get_indices(x)[1]
            x_t = x.index_select(-1, idx)
            u_t = u.index_select(-1, idx)
            out[..., idx] = self._flat_torus_ref.expmap(x_t, u_t)

        for manifold, s in zip(self.manifolds, self._slices):
            if isinstance(manifold, (Euclidean, FlatTorus)):
                continue
            out[..., s] = manifold.expmap(x[..., s], u[..., s])
        return out

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        out = torch.empty_like(x)

        if self._euclidean_ref:
            idx = self._get_indices(x)[0]
            x_e = x.index_select(-1, idx)
            y_e = y.index_select(-1, idx)
            out[..., idx] = self._euclidean_ref.logmap(x_e, y_e)

        if self._flat_torus_ref:
            idx = self._get_indices(x)[1]
            x_t = x.index_select(-1, idx)
            y_t = y.index_select(-1, idx)
            out[..., idx] = self._flat_torus_ref.logmap(x_t, y_t)

        for manifold, s in zip(self.manifolds, self._slices):
            if isinstance(manifold, (Euclidean, FlatTorus)):
                continue
            out[..., s] = manifold.logmap(x[..., s], y[..., s])
        return out

    def projx(self, x: Tensor) -> Tensor:
        # Skip identity work: Euclidean projx(x) == x; call others normally
        out = x.clone()
        for manifold, s in zip(self.manifolds, self._slices):
            if not isinstance(manifold, Euclidean):
                out[..., s] = manifold.projx(x[..., s])
        return out

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        # Skip identity work: Euclidean proju(x,u) == u and FlatTorus proju(x,u) == u
        out = u.clone()
        for manifold, s in zip(self.manifolds, self._slices):
            if not isinstance(manifold, (Euclidean, FlatTorus)):
                out[..., s] = manifold.proju(x[..., s], u[..., s])
        return out

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        outs: List[Tensor] = []

        # Precompute batched distances for Euclidean and FlatTorus
        e_idx_tensor = None
        t_idx_tensor = None
        d_e = None
        d_t = None

        if self._euclidean_ref:
            e_idx_tensor = self._get_indices(x)[0]
            x_e = x.index_select(-1, e_idx_tensor)
            y_e = y.index_select(-1, e_idx_tensor)
            d_e = self._euclidean_ref.dist(x_e, y_e)  # (..., Ne)

        if self._flat_torus_ref:
            t_idx_tensor = self._get_indices(x)[1]
            x_t = x.index_select(-1, t_idx_tensor)
            y_t = y.index_select(-1, t_idx_tensor)
            d_t = self._flat_torus_ref.dist(x_t, y_t)  # (..., Nt)

        # Index offsets for the euclidean and flat torus dimensions
        e_idx_off = 0
        t_idx_off = 0
        for manifold, s in zip(self.manifolds, self._slices):
            if isinstance(manifold, Euclidean):
                d = s.stop - s.start # number of dimensions for the current manifold
                if d_e is None:
                    # No euclidean dims (should not happen if isinstance), but guard anyway
                    outs.append(torch.empty(*x.shape[:-1], 0, device=x.device, dtype=x.dtype))
                else:
                    outs.append(d_e[..., e_idx_off:e_idx_off + d])
                    e_idx_off += d
            elif isinstance(manifold, FlatTorus):
                d = s.stop - s.start
                if d_t is None:
                    # No flat torus dims (should not happen if isinstance), but guard anyway
                    outs.append(torch.empty(*x.shape[:-1], 0, device=x.device, dtype=x.dtype))
                else:
                    outs.append(d_t[..., t_idx_off:t_idx_off + d])
                    t_idx_off += d
            else:
                outs.append(manifold.dist(x[..., s], y[..., s]))

        return torch.cat(outs, dim=-1)
