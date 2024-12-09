from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit


if TYPE_CHECKING:
    from .tensor_data import Index, Shape, Storage, Strides


from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Wrapper for Numba's jit decorator with inline optimization."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                in_index: Index = np.empty(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        # for i in prange(len(out)):
        #     out_idx = np.zeros(len(out_shape), dtype=np.int32)
        #     to_index(i, out_shape, out_idx)

        #     in_idx = np.zeros(len(in_shape), dtype=np.int32)
        #     broadcast_index(out_idx, out_shape, in_shape, in_idx)

        #     out_pos = index_to_position(out_idx, out_strides)
        #     in_pos = index_to_position(in_idx, in_strides)

        #     out[out_pos] = fn(in_storage[in_pos])

        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                a_index: Index = np.empty(MAX_DIMS, np.int32)
                b_index: Index = np.empty(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_strides)
                k = index_to_position(b_index, b_index)
                out[o] = fn(a_storage[j], b_storage[k])
            else:
                for i in prange(len(out)):
                    out[i] = fn(a_storage[i], b_storage)
        # for i in prange(len(out)):
        #     out_idx = np.zeros(len(out_shape), dtype=np.int32)
        #     to_index(i, out_shape, out_idx)

        #     a_idx = np.zeros(len(a_shape), dtype=np.int32)
        #     broadcast_index(out_idx, out_shape, a_shape, a_idx)

        #     b_idx = np.zeros(len(b_shape), dtype=np.int32)
        #     broadcast_index(out_idx, out_shape, b_shape, b_idx)

        #     out_pos = index_to_position(out_idx, out_strides)
        #     a_pos = index_to_position(a_idx, a_strides)
        #     b_pos = index_to_position(b_idx, b_strides)

        #     out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_zip, parallel=True)  # type: ignore
    # return njit(parallel=True)(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            reduce_size = a_shape[reduce_dim]
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            accum = out[o]
            j = index_to_position(out_index, a_strides)
            step = a_strides[reduce_dim]
            for s in range(reduce_size):
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum
        # for i in prange(len(out)):
        #     out_idx = np.zeros(len(out_shape), dtype=np.int32)
        #     to_index(i, out_shape, out_idx)

        #     out_pos = index_to_position(out_idx, out_strides)

        #     for j in range(a_shape[reduce_dim]):
        #         a_idx = np.copy(out_idx)
        #         a_idx[reduce_dim] = j

        #         a_pos = index_to_position(a_idx, a_strides)
        #         out[out_pos] = fn(out[out_pos], a_storage[a_pos])
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_reduce, parallel=True)  # type: ignore
    # return njit(parallel=True)(_reduce)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for i1 in prange(out_shape[0]):
        for i2 in prange(out_shape[1]):
            for i3 in prange(out_shape[2]):
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]
                b_inner = i1 * b_batch_stride + 13 * b_strides[2]
                acc = 0.0
                for _ in range(a_shape[2]):
                    acc += a_storage[a_inner] * b_storage[b_inner]
                    a_inner += a_strides[2]
                    b_inner += b_strides[1]
                out_position = (
                    i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                )
                out[out_position] = acc
    # for n in prange(out_shape[0]):  # Batch dimension
    #     for i in range(out_shape[1]):  # Rows of the output matrix
    #         for j in range(out_shape[2]):  # Columns of the output matrix
    #             # Initialize accumulator
    #             acc = 0.0
    #             for k in range(a_shape[2]):  # Inner product dimension
    #                 # Compute flat indices directly using strides
    #                 a_pos = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]
    #                 b_pos = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]
    #                 acc += a_storage[a_pos] * b_storage[b_pos]

    #             # Compute the output position
    #             out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]

    #             # Write result to the output storage
    #             out[out_pos] = acc

    # TODO: Implement for Task 3.2.
    # raise NotImplementedError("Need to implement for Task 3.2")


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
# tensor_matrix_multiply = njit (parallel=True, fastmath=True)(_tensor_matrix_multiply)
assert tensor_matrix_multiply is not None
