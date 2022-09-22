
from typing import Iterable, Iterator, Tuple
import torch
import torch.nn as nn
from torchinfo import summary


class SuperNetModule(nn.Module):

    def __init__(self, input_layers: Iterable[nn.Module], input_shape: Tuple[int, ...]):
        super().__init__()

        self.input_layers = list(input_layers)
        self.input_shape = input_shape
        self.n_layers = len(self.input_layers)
        self.n_parameters = torch.zeros(1)
        self.macs = torch.zeros(1)

        self.alpha = nn.Parameter(
            (1 / self.n_layers) * torch.ones(self.n_layers, dtype=torch.float), requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for the SuperNetModule that returns a weighted
        sum of all the outputs of the different input layers.

        :param input: the input tensor
        :type input: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """
        softmax = nn.Softmax(dim=0)
        soft_alpha = softmax(self.alpha)

        y = []
        for i, layer in enumerate(self.input_layers):
            y.append(soft_alpha[i] * layer(input))
        y = torch.stack(y, dim=0).sum(dim=0)

        return y

    def export(self) -> nn.Module:
        """It returns a single layer within the ones
        given as input parameter to the SuperNetModule.
        The chosen layer will be the one with the highest alpha value (highest probability).

        :return: single nn classic layer
        :rtype: nn.Module
        """
        index = torch.argmax(self.alpha).item()
        return self.input_layers[int(index)]

    def get_size(self) -> torch.Tensor:
        """Method that returns the number of weights for the module
        computed as a weighted sum of the number of weights of each layer.

        :return: number of weights of the module (weighted sum)
        :rtype: torch.Tensor
        """
        for i, layer in enumerate(self.input_layers):
            for param in layer.parameters():
                prod = torch.prod(torch.tensor(param.shape))
                self.n_parameters += self.alpha[i] * prod

        return self.n_parameters

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the module

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        softmax = nn.Softmax(dim=0)
        soft_alpha = softmax(self.alpha)

        for i, layer in enumerate(self.input_layers):
            stats = summary(layer, self.input_shape, verbose=0)
            self.macs += soft_alpha[i] * stats.total_mult_adds

        return self.macs

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this module, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names, defaults to ''
        :type prefix: str, optional
        :param recurse: kept for uniformity with pytorch API, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[Tuple[str, nn.Parameter]]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        prfx += "alpha"

        yield prfx, self.alpha

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of this module

        :param recurse: kept for uniformity with pytorch API, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[nn.Parameter]
        """
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param
