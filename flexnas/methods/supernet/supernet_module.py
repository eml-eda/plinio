
from typing import Iterable, Iterator, Tuple
import torch
import torch.nn as nn


class SuperNetModule(nn.Module):

    def __init__(self, input_layers: Iterable[nn.Module]):
        super().__init__()

        self.input_layers = list(input_layers)
        self.n_layers = len(self.input_layers)
        self.n_parameters = torch.zeros(1)

        # self.alpha = nn.Parameter()
        self.alpha = (1 / self.n_layers) * torch.ones(self.n_layers, dtype=torch.float)

        for i, layer in enumerate(self.input_layers):
            for param in layer.parameters():
                prod = torch.prod(torch.tensor(param.shape))
                self.n_parameters += self.alpha[i] * prod

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for the SuperNetModule that returns a weighted
        sum of all the outputs of the different input layers.

        :param input: the input tensor
        :type input: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """

        y = []
        for i, layer in enumerate(self.input_layers):
            y.append(self.alpha[i] * layer(input))
        y = sum(y)
        return y

    def export(self) -> nn.Module:
        """It returns a single classic layer within the ones
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

        return self.n_parameters

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of all layers in the module, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names, defaults to ''
        :type prefix: str, optional
        :param recurse: _description_, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[Tuple[str, nn.Parameter]]
        """

        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""

        for layer in self.input_layers:
            for name, param in layer.named_parameters(
                    prfx + layer.__class__.__name__, recurse):
                yield name, param

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of all layers in the module

        :param recurse: _description_, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[nn.Parameter]
        """

        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param
