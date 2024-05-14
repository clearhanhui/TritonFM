import torch
from kernel import fm, fm_compile, fm_kernel


def test():
    x = torch.randn((2, 3, 4))
    x = x.cuda() if torch.cuda.is_available() else x
    x_compile = x.clone().requires_grad_(True)
    x_kernel = x.clone().requires_grad_(True)
    x = x.requires_grad_(True)
    y = fm(x)
    y_compiled = fm_compile(x_compile)
    y_kernel = fm_kernel(x_kernel)
    y.backward(torch.ones_like(y))
    y_compiled.backward(torch.ones_like(y_compiled))
    y_kernel.backward(torch.ones_like(y_kernel))

    assert torch.allclose(y, y_compiled)
    assert torch.allclose(y, y_kernel)
    assert torch.allclose(x.grad, x_compile.grad)
    assert torch.allclose(x.grad, x_kernel.grad)

