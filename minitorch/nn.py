from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling.

    Args:
    ----
        input: Tensor of shape batch x channel x height x width
        kernel: Pooling size as (height, width)

    Returns:
    -------
        Reshaped tensor with added kernel dimensions and updated height and width.

    """
    batch, channels, height, width = input.shape
    kernel_h, kernel_w = kernel
    assert height % kernel_h == 0
    assert width % kernel_w == 0

    out_h = height // kernel_h
    out_w = width // kernel_w

    reshaped = input.contiguous()
    reshaped = (
        reshaped.view(batch, channels, out_h, kernel_h, out_w, kernel_w)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )

    reshaped = reshaped.view(batch, channels, out_h, out_w, kernel_h * kernel_w)

    return reshaped, out_h, out_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling to the input tensor.

    Args:
    ----
        input: Input tensor with shape (batch, channel, height, width)
        kernel: Pooling window size as (kernel_height, kernel_width)

    Returns:
    -------
        Output tensor after average pooling.

    """
    tiled_input, height_out, width_out = tile(input, kernel)
    return tiled_input.mean(dim=4).view(
        tiled_input.shape[0], tiled_input.shape[1], height_out, width_out
    )


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(x: Tensor, dim: int) -> Tensor:
    """Compute argmax as a one-hot tensor along the specified dimension.

    Args:
    ----
        x: Input tensor
        dim: Dimension along which argmax is computed

    Returns:
    -------
        One-hot tensor indicating the position of maximum values.

    """
    max_vals = max_reduce(x, dim)
    return x == max_vals


class Max(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation along a dimension.

        Args:
        ----
            ctx: Context object for backpropagation
            x: Input tensor
            dim: Dimension to reduce

        Returns:
        -------
            Tensor with maximum values along the specified dimension.

        """
        if isinstance(dim, Tensor):
            axis = int(dim._tensor._storage[0])
        else:
            axis = int(dim)

        ctx.save_for_backward(x, x._ensure_tensor(axis))
        return max_reduce(x, axis)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for max operation.

        Args:
        ----
            ctx: Context object from forward pass
            grad_out: Upstream gradient tensor

        Returns:
        -------
            Gradients with respect to inputs.

        """
        x, dim_tensor = ctx.saved_values
        axis = int(dim_tensor.item())
        grad_input = (argmax(x, axis) * grad_out).sum(dim=axis)
        return grad_input, x._ensure_tensor(0.0)


def max(x: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        x: Input tensor
        dim: Dimension to reduce

    Returns:
    -------
        Tensor with maximum values along the given dimension.

    """
    return Max.apply(x, x._ensure_tensor(dim))


def softmax(x: Tensor, dim: int) -> Tensor:
    """Apply softmax along the specified dimension.

    Args:
    ----
        x: Input tensor
        dim: Dimension to apply softmax

    Returns:
    -------
        Softmax-transformed tensor.

    """
    exp_values = x.exp()
    exp_sums = exp_values.sum(dim)
    shape = list(exp_values.shape)
    shape[dim] = 1
    return exp_values / exp_sums.contiguous().view(*shape)


def logsoftmax(x: Tensor, dim: int) -> Tensor:
    """Compute the log-softmax along a given dimension.

    Args:
    ----
        x: Input tensor
        dim: Dimension to apply log-softmax

    Returns:
    -------
        Tensor with log-softmax applied.

    """
    max_vals = max(x, dim)
    shifted = x - max_vals
    exp_vals = shifted.exp()
    exp_sum = exp_vals.sum(dim)
    return shifted - exp_sum.log().view(*max_vals.shape)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling to the input tensor.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width)
        kernel: Size of pooling window as (height, width)

    Returns:
    -------
        Tensor after max pooling operation.

    """
    batch, channels = input.shape[:2]
    tiled, out_h, out_w = tile(input, kernel)
    pooled = max(tiled, 4)
    return pooled.view(batch, channels, out_h, out_w)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout with specified probability.

    Args:
    ----
        input: Input tensor
        rate: Dropout probability
        ignore: If True, disables dropout

    Returns:
    -------
        Tensor after applying dropout.

    """
    if not ignore and rate > 0.0:
        mask = rand(input.shape) > rate
        return input * mask
    else:
        return input
