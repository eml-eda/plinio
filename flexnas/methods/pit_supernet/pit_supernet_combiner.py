from typing import List, cast, Iterator, Tuple
import torch
import torch.nn as nn
import torch.fx as fx
from torchinfo import summary
from flexnas.methods.pit import PITModule


class PITSuperNetCombiner(nn.Module):

    def __init__(self, input_layers: List[nn.Module]):
        super(PITSuperNetCombiner, self).__init__()

        self.sn_input_layers = input_layers
        self.input_shape = None
        self.n_layers = len(input_layers)
        self.alpha = nn.Parameter(
            (1 / self.n_layers) * torch.ones(self.n_layers, dtype=torch.float), requires_grad=True)

        self._pit_layers = list()
        for _ in range(self.n_layers):
            self._pit_layers.append(list())

        self.layers_sizes = []
        self.layers_macs = []

    def forward(self, layers_outputs: List[torch.Tensor]):
        soft_alpha = nn.functional.softmax(self.alpha, dim=0)
        y = []
        for i, yi in enumerate(layers_outputs):
            y.append(soft_alpha[i] * yi)
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    def export(self, n: fx.Node, mod: fx.GraphModule):
        index = torch.argmax(self.alpha).item()
        submodule = self.sn_input_layers[int(index)]
        target = str(n.target)
        mod.add_submodule(target, submodule)

    def compute_layers_sizes(self):
        for layer in self.sn_input_layers:
            layer_size = 0
            for param in layer.parameters():
                layer_size += torch.prod(torch.tensor(param.shape))
            self.layers_sizes.append(layer_size)

        for i, layer in enumerate(self.sn_input_layers):
            for module in layer.modules():
                if isinstance(module, PITModule):
                    cast(List[PITModule], self._pit_layers[i]).append(module)
                    # self._pit_layers[i].append(module)
                    size = 0
                    for param in module.parameters():
                        size += torch.prod(torch.tensor(param.shape))
                    self.layers_sizes[i] = self.layers_sizes[i] - size

    def compute_layers_macs(self):
        for layer in self.sn_input_layers:
            stats = summary(layer, self.input_shape, verbose=0, mode='eval')
            self.layers_macs.append(stats.total_mult_adds)

        for i, layer in enumerate(self.sn_input_layers):
            for module in layer.modules():
                if isinstance(module, PITModule):
                    # cast(List[PITModule], self._pit_layers[i]).append(module)
                    # already done in compute_layers_sizes
                    stats = summary(module, self.input_shape, verbose=0, mode='eval')
                    self.layers_macs[i] = self.layers_macs[i] - stats.total_mult_adds

    def get_size(self) -> torch.Tensor:
        soft_alpha = nn.functional.softmax(self.alpha, dim=0)

        size = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_layers):
            var_size = torch.tensor(0, dtype=torch.float32)
            for pl in cast(List[PITModule], self._pit_layers[i]):
                var_size += pl.get_size()
            size = size + (soft_alpha[i] * (self.layers_sizes[i] + var_size))
        return size

    def get_macs(self) -> torch.Tensor:
        soft_alpha = nn.functional.softmax(self.alpha, dim=0)

        macs = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_layers):
            var_macs = torch.tensor(0, dtype=torch.float32)
            for pl in cast(List[PITModule], self._pit_layers[i]):
                var_macs += pl.get_macs()
            macs = macs + (soft_alpha[i] * (self.layers_macs[i] + var_macs))
        return macs

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        prfx += "alpha"
        yield prfx, self.alpha

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param
