# Ewerton R Vieira
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from typing import List, Tuple, Sequence, Union
import warnings

# Import directly from modules to avoid circular imports with `flow_matching.utils.manifolds.__init__`.
from flow_matching.utils.manifolds.manifold import Euclidean, Manifold
from flow_matching.utils.manifolds.torus import FlatTorus


class Product(Manifold):
    """Product of component manifolds with potentially different state and tangent dimensions.

    Backward compatible with the original API where each component was specified as
    `(manifold, dim)` and assumed `state_dim == tangent_dim == dim`.

    Extended API additionally supports `(manifold, state_dim, tangent_dim)` which is
    required for manifolds like SO(3) represented by quaternions (state_dim=4) with
    3D tangent vectors (tangent_dim=3).
    """

    def __init__(
        self,
        input_dim: int,
        manifolds: Sequence[Union[Tuple[Manifold, int], Tuple[Manifold, int, int]]],
    ):
        """
        Initialize a product manifold with arbitrary ordering and dimensions.

        Optimizes the operations for Euclidean and FlatTorus manifolds by vectorizing them over all dimensions corresponding to those manifolds.

        Args:
            input_dim: Total dimensionality of the concatenated **state** vector.
            manifolds: Sequence of component manifold specs.
                - `(manifold, dim)` means `state_dim=dim` and `tangent_dim=dim`.
                - `(manifold, state_dim, tangent_dim)` explicitly sets both.
        """
        super().__init__()

        # Normalize specs to (manifold, state_dim, tangent_dim)
        norm_specs: List[Tuple[Manifold, int, int]] = []
        for spec in manifolds:
            if len(spec) == 2:
                m, d = spec  # type: ignore[misc]
                norm_specs.append((m, int(d), int(d)))
            elif len(spec) == 3:
                m, sd, td = spec  # type: ignore[misc]
                norm_specs.append((m, int(sd), int(td)))
            else:
                raise ValueError(
                    "Each manifold spec must be (manifold, dim) or (manifold, state_dim, tangent_dim)."
                )

        assert sum(sd for _, sd, _ in norm_specs) == int(
            input_dim
        ), "Sum of state_dim must match input_dim"

        m_list: List[Manifold] = []
        state_d_list: List[int] = []
        tangent_d_list: List[int] = []

        for manifold, sd, td in norm_specs:
            m_list.append(manifold)
            state_d_list.append(int(sd))
            tangent_d_list.append(int(td))

        self.manifolds = tuple(m_list)
        self.dimensions = tuple(state_d_list)
        self.tangent_dims = tuple(tangent_d_list)
        self.total_state_dim = int(sum(self.dimensions))
        self.total_tangent_dim = int(sum(self.tangent_dims))

        cum_state = [0]
        for d in self.dimensions:
            cum_state.append(cum_state[-1] + d)
        self._state_slices = tuple(
            slice(cum_state[i], cum_state[i + 1]) for i in range(len(self.dimensions))
        )

        cum_tan = [0]
        for d in self.tangent_dims:
            cum_tan.append(cum_tan[-1] + d)
        self._tangent_slices = tuple(
            slice(cum_tan[i], cum_tan[i + 1]) for i in range(len(self.tangent_dims))
        )

        # Precompute indices that belong to Euclidean and FlatTorus to enable batched operations
        euclid_state_idx: List[int] = []
        euclid_tan_idx: List[int] = []
        torus_state_idx: List[int] = []
        torus_tan_idx: List[int] = []

        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            # Vectorize only when the component uses the same dimension in state and tangent.
            if isinstance(manifold, Euclidean) and (s_x.stop - s_x.start) == (
                s_u.stop - s_u.start
            ):
                euclid_state_idx.extend(range(s_x.start, s_x.stop))
                euclid_tan_idx.extend(range(s_u.start, s_u.stop))
            elif isinstance(manifold, FlatTorus) and (s_x.stop - s_x.start) == (
                s_u.stop - s_u.start
            ):
                torus_state_idx.extend(range(s_x.start, s_x.stop))
                torus_tan_idx.extend(range(s_u.start, s_u.stop))

        self._euclidean_state_indices = tuple(euclid_state_idx)
        self._euclidean_tangent_indices = tuple(euclid_tan_idx)
        self._flat_torus_state_indices = tuple(torus_state_idx)
        self._flat_torus_tangent_indices = tuple(torus_tan_idx)

        # Keep a reference instance for each type to call their vectorized ops
        self._euclidean_ref = next(
            (m for m in self.manifolds if isinstance(m, Euclidean)), None
        )
        self._flat_torus_ref = next(
            (m for m in self.manifolds if isinstance(m, FlatTorus)), None
        )

        if len(self.manifolds) > 50:
            warnings.warn(
                "Product manifold has more than 50 manifolds. This may lead to performance issues."
            )

        self._cached_indices = {}  # {(device, torch.long): (e_x, e_u, t_x, t_u)}

    def _get_indices(self, like: Tensor):
        key = (like.device, torch.long)
        if key not in self._cached_indices:
            e_x = (
                torch.as_tensor(
                    self._euclidean_state_indices, device=like.device, dtype=torch.long
                )
                if self._euclidean_state_indices
                else None
            )
            e_u = (
                torch.as_tensor(
                    self._euclidean_tangent_indices,
                    device=like.device,
                    dtype=torch.long,
                )
                if self._euclidean_tangent_indices
                else None
            )
            t_x = (
                torch.as_tensor(
                    self._flat_torus_state_indices, device=like.device, dtype=torch.long
                )
                if self._flat_torus_state_indices
                else None
            )
            t_u = (
                torch.as_tensor(
                    self._flat_torus_tangent_indices,
                    device=like.device,
                    dtype=torch.long,
                )
                if self._flat_torus_tangent_indices
                else None
            )
            self._cached_indices[key] = (e_x, e_u, t_x, t_u)
        return self._cached_indices[key]

    @staticmethod
    def _broadcast_batch(*tensors: Tensor) -> Tuple[Tensor, ...]:
        """Broadcast only batch dims (all dims except the last), allowing different last-dim sizes."""
        if len(tensors) == 0:
            return tuple()
        batch_shapes = [t.shape[:-1] for t in tensors]
        batch_shape = torch.broadcast_shapes(*batch_shapes)
        return tuple(t.expand(batch_shape + (t.shape[-1],)) for t in tensors)

    def split(self, x: Tensor) -> List[Tensor]:
        """Split a concatenated **state** tensor according to component state dimensions."""
        return [x[..., s] for s in self._state_slices]

    def split_tangent(self, u: Tensor) -> List[Tensor]:
        """Split a concatenated **tangent** tensor according to component tangent dimensions."""
        return [u[..., s] for s in self._tangent_slices]

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        x, u = self._broadcast_batch(x, u)
        out = torch.empty_like(x)

        if self._euclidean_ref:
            idx_x, idx_u = self._get_indices(x)[0], self._get_indices(x)[1]
            if idx_x is not None and idx_u is not None and idx_x.numel() > 0:
                x_e = x.index_select(-1, idx_x)
                u_e = u.index_select(-1, idx_u)
                out[..., idx_x] = self._euclidean_ref.expmap(x_e, u_e)

        if self._flat_torus_ref:
            idx_x, idx_u = self._get_indices(x)[2], self._get_indices(x)[3]
            if idx_x is not None and idx_u is not None and idx_x.numel() > 0:
                x_t = x.index_select(-1, idx_x)
                u_t = u.index_select(-1, idx_u)
                out[..., idx_x] = self._flat_torus_ref.expmap(x_t, u_t)

        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            if isinstance(manifold, (Euclidean, FlatTorus)):
                continue
            out[..., s_x] = manifold.expmap(x[..., s_x], u[..., s_u])
        return out

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._broadcast_batch(x, y)

        if self.total_state_dim == self.total_tangent_dim:
            out = torch.empty_like(x)
        else:
            out_shape = list(x.shape)
            out_shape[-1] = self.total_tangent_dim
            out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        if self._euclidean_ref:
            idx_x, idx_u = self._get_indices(x)[0], self._get_indices(x)[1]
            if idx_x is not None and idx_u is not None and idx_x.numel() > 0:
                x_e = x.index_select(-1, idx_x)
                y_e = y.index_select(-1, idx_x)
                out[..., idx_u] = self._euclidean_ref.logmap(x_e, y_e)

        if self._flat_torus_ref:
            idx_x, idx_u = self._get_indices(x)[2], self._get_indices(x)[3]
            if idx_x is not None and idx_u is not None and idx_x.numel() > 0:
                x_t = x.index_select(-1, idx_x)
                y_t = y.index_select(-1, idx_x)
                out[..., idx_u] = self._flat_torus_ref.logmap(x_t, y_t)

        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            if isinstance(manifold, (Euclidean, FlatTorus)):
                continue
            out[..., s_u] = manifold.logmap(x[..., s_x], y[..., s_x])
        return out

    def projx(self, x: Tensor) -> Tensor:
        # Skip identity work: Euclidean projx(x) == x; call others normally
        out = x.clone()
        for manifold, s_x in zip(self.manifolds, self._state_slices):
            if not isinstance(manifold, Euclidean):
                out[..., s_x] = manifold.projx(x[..., s_x])
        return out

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        x, u = self._broadcast_batch(x, u)
        # Skip identity work: Euclidean proju(x,u) == u and FlatTorus proju(x,u) == u
        out = u.clone()
        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            if not isinstance(manifold, (Euclidean, FlatTorus)):
                out[..., s_u] = manifold.proju(x[..., s_x], u[..., s_u])
        return out

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._broadcast_batch(x, y)
        outs: List[Tensor] = []
        for manifold, s_x in zip(self.manifolds, self._state_slices):
            outs.append(manifold.dist(x[..., s_x], y[..., s_x]))
        return torch.cat(outs, dim=-1)
