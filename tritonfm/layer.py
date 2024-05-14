import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernel import fm_kernel


class FactorizationMachine(nn.Module):
    """
    Factorization Machine Layer.
    Args:
        reduce_sum (bool): Whether to sum the output along the last dimension.
    Input:
        - x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
    Output:
        - Float tensor of size ``(batch_size, embed_dim)`` if ``reduce_sum=True`` else ``(batch_size, 1)``
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum


    def forward(self, x):
        ix = fm_kernel(x)
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
