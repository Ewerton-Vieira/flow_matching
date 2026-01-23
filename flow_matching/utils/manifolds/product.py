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
from flow_matching.utils.manifolds.so3 import SO3
from flow_matching.utils.manifolds.torus import FlatTorus
from flow_matching.utils.manifolds.sphere import Sphere


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

                # Check that the manifold is consistent with the state and tangent dimensions
                if isinstance(m, Euclidean) and sd != td:
                    raise ValueError("Euclidean manifold must have state_dim == tangent_dim")
                if isinstance(m, FlatTorus) and sd != td:
                    raise ValueError("FlatTorus manifold must have state_dim == tangent_dim")
                if isinstance(m, Sphere) and sd != td:
                    raise ValueError("Sphere manifold must have state_dim == tangent_dim")
                if isinstance(m, SO3):
                    if sd != 4:
                        raise ValueError("SO3 manifold must have state_dim == 4")
                    if td != 3:
                        raise ValueError("SO3 manifold must have tangent_dim == 3")
                
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

        self._cached_indices = {}  # {device: (e_x, e_u, t_x, t_u)}

    def _get_indices(self, like: Tensor):
        key = like.device
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

        # Build output by computing each manifold's expmap and concatenating
        # This is vmap-compatible (avoids in-place tensor-indexed assignment)
        parts = []
        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            x_part = x[..., s_x]
            u_part = u[..., s_u]
            exp_part = manifold.expmap(x_part, u_part)
            parts.append(exp_part)

        return torch.cat(parts, dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._broadcast_batch(x, y)

        # Build output by computing each manifold's logmap and concatenating
        # This is vmap-compatible (avoids in-place tensor-indexed assignment)
        parts = []
        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            x_part = x[..., s_x]
            y_part = y[..., s_x]
            log_part = manifold.logmap(x_part, y_part)
            parts.append(log_part)

        return torch.cat(parts, dim=-1)

    def projx(self, x: Tensor) -> Tensor:
        # Build output by computing each manifold's projx and concatenating
        # This is vmap-compatible (avoids in-place assignment)
        parts = []
        for manifold, s_x in zip(self.manifolds, self._state_slices):
            x_part = x[..., s_x]
            # Euclidean projx is identity, but we call it anyway for consistency
            proj_part = manifold.projx(x_part)
            parts.append(proj_part)
        return torch.cat(parts, dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        x, u = self._broadcast_batch(x, u)
        # Build output by computing each manifold's proju and concatenating
        # This is vmap-compatible (avoids in-place assignment)
        parts = []
        for manifold, s_x, s_u in zip(
            self.manifolds, self._state_slices, self._tangent_slices
        ):
            x_part = x[..., s_x]
            u_part = u[..., s_u]
            # Euclidean and FlatTorus proju is identity, but we call it anyway
            proj_part = manifold.proju(x_part, u_part)
            parts.append(proj_part)
        return torch.cat(parts, dim=-1)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._broadcast_batch(x, y)
        outs: List[Tensor] = []
        for manifold, s_x in zip(self.manifolds, self._state_slices):
            outs.append(manifold.dist(x[..., s_x], y[..., s_x]))
        return torch.cat(outs, dim=-1)
