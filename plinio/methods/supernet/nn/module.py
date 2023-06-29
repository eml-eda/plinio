from typing import Any, Dict, Iterable, Iterator, Tuple
import torch
import torch.nn as nn
from .combiner import SuperNetCombiner


class SuperNetModule(nn.Module):
    """A nn.Module containing some different layer alternatives.
    One of these layers will be selected by the SuperNet NAS tool for the current layer.

    :param supernet_branches: iterable of possible alternative layers to be selected
    :type supernet_branches: Iterable[nn.Module]
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SofrMax
    :type gumbel_softmax: bool
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool
    """
    def __init__(self,
                 supernet_branches: Iterable[nn.Module],
                 gumbel_softmax: bool = False,
                 hard_softmax: bool = False):
        super(SuperNetModule, self).__init__()
        self.sn_branches = nn.ModuleList(list(supernet_branches))
        self.sn_combiner = SuperNetCombiner(len(self.sn_branches), gumbel_softmax, hard_softmax)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for the SuperNetModule that returns a weighted
        sum of all the outputs of the different input layers.

        :param input: the input tensor
        :type input: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """
        return self.sn_combiner([branch(input) for branch in self.sn_branches])

    def __getitem__(self, pos: int) -> nn.Module:
        """Get the layer at position pos in the list of all the possible
        layers for the SuperNetModule

        :param pos: position of the required module in the list input_layers
        :type pos: int
        :return: module at postion pos in the list input_layers
        :rtype: nn.Module
        """
        return self.sn_branches[pos]
