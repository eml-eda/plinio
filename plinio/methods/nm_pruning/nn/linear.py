"""
Code adapted from:
https://github.com/aojunzz/NM-sparsity
"""

import torch
import torch.nn as nn
from torch import autograd, nn
from .module import NMPruningModule
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.node import Node as Node
from typing import Dict, Any, cast


class _PrunerLinear(autograd.Function):
    """ " Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, n, m, pruning_decay=0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = torch.div(length, m).to(torch.int)  # int(length / m)

        weight_temp = weight.detach().abs().reshape(group, m)
        index = torch.argsort(weight_temp, dim=1)[:, : m - n]

        w_b = torch.ones_like(weight_temp)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = pruning_decay

        return output * w_b

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        return grad_output + ctx.decay * (1 - ctx.mask) * weight, None, None, None


class NMPruningLinear(nn.Linear, NMPruningModule):
    """A nn.Module implementing a Linear layer with pruning support

    :param linear: the inner `nn.Linear` layer to be pruned
    :type linear: nn.Linear
    :param n: number of non-zero parameters
    :type n: int
    :param m: group of weights considered for pruning
    :type m: int
    """

    def __init__(
        self, linear: nn.Linear, n: int, m: int, pruning_decay: float = 0.0002
    ):
        super(NMPruningLinear, self).__init__(
            linear.in_features, linear.out_features, linear.bias is not None
        )

        self.n = n
        self.m = m
        self.pruning_decay = pruning_decay

    def get_sparse_weights(self):
        return _PrunerLinear.apply(self.weight, self.n, self.m, self.pruning_decay)

    def forward(self, x):
        w = self.get_sparse_weights()
        x = torch.nn.functional.linear(x, w, self.bias)
        return x

    @staticmethod
    def autoimport(node: Node, mod: GraphModule, n: int, m: int, pruning_decay: float):
        """Create a new fx.Node relative to a NMPruningLinear layer, starting from the fx.Node
        of a nn.Linear layer, and replace it into the parent fx.GraphModule

        :param node: a fx.Node corresponding to a nn.Linear layer
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
        if type(submodule) != nn.Linear:
            msg = f"Trying to generate NMPruningLinear from layer of type {type(submodule)}"
            raise TypeError(msg)
        submodule = cast(nn.Linear, submodule)
        # Support check
        # Compatible dimensions check
        if (submodule.in_features) % m != 0:
            return
        new_submodule = NMPruningLinear(submodule, n, m, pruning_decay)
        mod.add_submodule(str(node.target), new_submodule)

    @staticmethod
    def export(node: Node, mod: GraphModule):
        """Replaces a fx.Node corresponding to a NMPruning layer,
        with a nn.Linear layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(node.target))
        if type(submodule) != NMPruningLinear:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")

        # Get the inner nn.Linear layer
        new_submodule = nn.Linear(
            submodule.in_features,
            submodule.out_features,
            submodule.bias is not None,
        )
        mod.add_submodule(str(node.target), new_submodule)

    @staticmethod
    def is_prunable(layer: nn.Linear, n: int, m: int) -> bool:
        """Check if a layer can be pruned with N and M given"""
        # Support check
        # Compatible dimensions check
        if (layer.in_features) % m != 0:
            return False
        return True
