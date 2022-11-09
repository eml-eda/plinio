
import torch
import torch.nn as nn


class PaddedModule(nn.Module):

    def __init__(self, inner_module: nn.Module, padding: int):
        super(PaddedModule, self).__init__()

        self.inner_module = inner_module
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_shape = list(input.shape)
        out_shape[1] = self.padding
        padding = torch.zeros((out_shape))

        y = self.inner_module(input)
        y = torch.cat((y, padding), 1)
        return y
