import torch
import triton
import triton.language as tl

__all__ = ['fm', 'fm_compile', 'fm_kernel']


def fm(x):
    square_of_sum = torch.sum(x, dim=1) ** 2
    sum_of_square = torch.sum(x ** 2, dim=1)
    ix = square_of_sum - sum_of_square
    return ix


@torch.compile
def fm_compile(x):
    return fm(x)


@triton.jit
def fm_fwd(
    x_ptr,
    y_ptr,
    f,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_ptr = x_ptr + pid * f
    offsets_f = tl.arange(0, BLOCK_SIZE)
    mask_f = offsets_f < f
    x = tl.load(start_ptr + offsets_f, mask=mask_f, other=0.)
    sum_ = tl.sum(x, axis=0)
    square_of_sum = sum_ * sum_
    sum_of_square = tl.sum(x * x, axis=0)
    ix = square_of_sum - sum_of_square
    tl.store(y_ptr + pid, ix)
    

@triton.jit
def fm_bwd(
    x_ptr,
    grad_y_ptr,
    f,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_ptr = x_ptr + pid * f
    offsets_f = tl.arange(0, BLOCK_SIZE)
    mask_f = offsets_f < f
    x = tl.load(start_ptr + offsets_f, mask=mask_f, other=0.)
    grad_y = tl.load(grad_y_ptr + pid)
    grad_x = 2 * (tl.sum(x, axis=0) - x) * grad_y
    tl.store(start_ptr + offsets_f, grad_x, mask=mask_f)


class FMKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        b, e, f = x.shape # batch size, embedding size, num fields
        y = torch.empty(b, e, dtype=x.dtype, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(f)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        fm_fwd[(b*e, )](x, y, f, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        
        return y


    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        b, e, f = x.shape
        BLOCK_SIZE = triton.next_power_of_2(f)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        fm_bwd[(b*e, )](x, grad_y, f, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
        return x


def fm_kernel(x:torch.Tensor):
    x = x.permute(0, 2, 1).contiguous()
    y = FMKernel.apply(x)
    return y
    