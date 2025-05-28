"""
Code adapted from:
https://github.com/aojunzz/NM-sparsity
"""

from typing import Dict, Any, cast
import torch
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.node import Node as Node
import torch.nn as nn
from torch import autograd, nn
from .module import NMPruningModule


class _PrunerConv2d(autograd.Function):
    """
    Prune by magnitude during the forward pass but keep the
    gradient dense.
    """

    @staticmethod
    def forward(ctx, weight, n, m, pruning_decay=0.0002):

        ctx.save_for_backward(weight)
        diff_group = m - n
        output = weight.clone()
        length = weight.numel()
        group = torch.div(length, m).to(torch.int)  # int(length / m)
        weight_temp = weight.detach().abs().permute(0, 2, 3, 1).reshape(group, m)

        index = torch.argsort(weight_temp, dim=1)[:, :diff_group]
        w_b = torch.ones_like(weight_temp)

        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(
            weight.permute(0, 2, 3, 1).shape
        )
        w_b = w_b.permute(0, 3, 1, 2)

        ctx.mask = w_b
        ctx.decay = pruning_decay

        return output * w_b

    @staticmethod
    def backward(ctx, grad_output):

        (weight,) = ctx.saved_tensors
        return grad_output + ctx.decay * (1 - ctx.mask) * weight, None, None, None


class NMPruningConv2d(nn.Conv2d, NMPruningModule):
    """A nn.Module implementing a Conv2d layer with N:M weight pruning support

    :param conv: the inner `nn.Conv2d` layer to be pruned
    :type conv: nn.Conv2d
    :param n: number of non-zero parameters
    :type n: int
    :param m: group of weights considered for pruning
    :type m: int
    :param pruning_decay: the decay factor for the pruning mask
    :type pruning_decay: float
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        n: int,
        m: int,
        pruning_decay: float = 0.0002,
    ):
        super(NMPruningConv2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
        )
        self.n = n
        self.m = m
        self.pruning_decay = pruning_decay

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        sparse_weights = self.get_sparse_weights()
        output = torch.nn.functional.conv2d(
            input,
            sparse_weights,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def get_sparse_weights(self) -> torch.Tensor:
        """
        :return: the sparse weights tensor, in NHWC format
        :rtype: torch.Tensor
        """
        return _PrunerConv2d.apply(self.weight, self.n, self.m, self.pruning_decay)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the layer hyperparameters

        :return: a dictionary containing the layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            "n": self.n,
            "m": self.m,
        }

    @staticmethod
    def autoimport(node: Node, mod: GraphModule, n: int, m: int, pruning_decay: float):
        """Create a new fx.Node relative to a NMPruningConv2d layer, starting from the fx.Node
        of a nn.Conv2d layer, and replace it into the parent fx.GraphModule

        :param node: a fx.Node corresponding to a nn.Conv2d layer
        :type node: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param n: number of non-zero parameters
        :type n: int
        :param m: group of weights considered for pruning
        :type m: int
        :param pruning_decay: the decay factor for the pruning mask
        :type pruning_decay: float
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(node.target))
        if type(submodule) != nn.Conv2d:
            msg = f"Trying to generate NMPruningConv2d from layer of type {type(submodule)}"
            raise TypeError(msg)
        submodule = cast(nn.Conv2d, submodule)

        # Support check
        if not NMPruningConv2d.is_prunable(submodule, n, m):
            print(
                f"Layer {node.target} cannot be pruned with N={n} and M={m}, skipping"
            )
            return
        new_submodule = NMPruningConv2d(submodule, n, m, pruning_decay)
        mod.add_submodule(str(node.target), new_submodule)

    @staticmethod
    def export(node: Node, mod: GraphModule):
        """Replaces a fx.Node corresponding to a NMPruning layer,
        with a nn.Conv2d layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(node.target))
        if type(submodule) != NMPruningConv2d:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")

        # Get the inner nn.Conv2d layer
        new_submodule = nn.Conv2d(
            submodule.in_channels,
            submodule.out_channels,
            submodule.kernel_size,
            submodule.stride,
            submodule.padding,
            submodule.dilation,
            submodule.groups,
            submodule.bias is not None,
            submodule.padding_mode,
        )
        # Set bias and weights
        new_submodule.bias = submodule.bias
        new_submodule.weight = torch.nn.Parameter(
            submodule.get_sparse_weights()
        )
        mod.add_submodule(str(node.target), new_submodule)

    @staticmethod
    def is_prunable(layer: nn.Conv2d, n: int, m: int) -> bool:
        """Check if a layer can be pruned with N and M given"""
        # Support check
        is_pointwise = layer.kernel_size == (1, 1) and layer.stride == (1, 1)
        is_depthwise = layer.groups == layer.in_channels
        if is_pointwise or is_depthwise:
            return False
        # Compatible dimensions check
        if (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]) % m != 0:
            return False
        return True
