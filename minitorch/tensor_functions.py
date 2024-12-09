"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation operation.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the backward pass of the Neg operation."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for inverse operation.

        Args:
        ----
            ctx (Context): Context for saving tensors for backward pass.
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Inverse of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for inverse operation.

        Args:
        ----
            ctx (Context): The context containing saved tensors.
            grad_output (Tensor): The gradient from the subsequent layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for element-wise addition of two tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition operation.

        Args:
        ----
            ctx (Context): Context object (unused in this case)
            grad_output (Tensor): Gradient from the subsequent layer

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients for both inputs

        """
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        # return t1.f.mul_zip(t1, t2)
        ctx.save_for_backward(t1, t2)  # Save both input tensors for the backward pass
        return t1.f.mul_zip(t1, t2)  # Perform the forward multiplication

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication."""
        t1, t2 = ctx.saved_values
        return grad_output * t2, grad_output * t1


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the sigmoid function element-wise."""
        result = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid."""
        (sigmoid,) = ctx.saved_values
        return grad_output * sigmoid * (1.0 - sigmoid)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the ReLU function element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU."""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the natural logarithm element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for logarithm."""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the exponential function element-wise."""
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exponentiation."""
        (exp_t1,) = ctx.saved_values
        return grad_output.f.mul_zip(exp_t1, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for sum operation."""
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the reshape operation.

        Args:
        ----
            ctx (Context): The context containing saved tensors from the forward pass.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to the inputs.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise less-than comparison between two tensors.

        This method computes the result of the element-wise less-than
        operation between the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the boolean results of the
            less-than comparisons.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)  # Using lt_map for less-than operation

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the less-than operation during backpropagation.

        This method returns zero tensors for the gradients of both input
        tensors since the less-than operation is non-differentiable.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Two zero tensors for the gradients
            of the input tensors.

        """
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise equality comparison between two tensors.

        This method computes the result of the element-wise equality
        operation between the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the boolean results of the
            equality comparisons.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)  # Using eq_map for equality operation

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the equality operation during backpropagation.

        This method returns zero tensors for the gradients of both input
        tensors since the equality operation is non-differentiable.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Two zero tensors for the gradients
            of the input tensors.

        """
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise is-close comparison (checks if elements are approximately equal)."""
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute tensor dimensions according to the given order."""
        order_list = order.to_numpy().astype(int).tolist()
        ctx.save_for_backward(order_list)
        return a._new(a._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permute operation.

        Args:
        ----
            ctx (Context): The context containing saved tensors from the forward pass.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient with respect to the input and None.

        """
        # Retrieve the order list saved in the context
        order = ctx.saved_values[0]

        # Reverse the permutation order
        order2 = [a[0] for a in sorted(enumerate(order), key=lambda a: a[1])]

        # Apply the reverse permutation to grad_output
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Change the shape of a tensor without changing its data.

        This method returns a new tensor with the same data as `a` but
        reshaped according to the dimensions specified in `shape`.
        The original tensor must be contiguous.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            a (Tensor): The input tensor to reshape.
            shape (Tensor): A tensor specifying the new shape.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the reshape operation.

        Args:
        ----
            ctx (Context): The context containing saved tensors from the forward pass.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to the inputs.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute gradient using central difference method.

    Args:
    ----
        f (Any): Function to compute gradient for
        *vals (Tensor): Input tensors
        arg (int): Argument to compute gradient with respect to
        epsilon (float): Small value for finite difference
        ind (UserIndex): Index for specific element

    Returns:
    -------
        float: Computed gradient

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
