from typing import List, cast, Iterator, Tuple, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from plinio.methods.pit.nn import PITModule


class PITSuperNetCombiner(nn.Module):
    """This nn.Module is included in the PITSuperNetModule and contains most of the logic of
    the module and the NAS paramaters.

    :param input_layers: list of possible alternative layers to be selected
    :type input_layers: List[nn.Module]
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SofrMax
    :type gumbel_softmax: bool
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool
    """
    def __init__(self, input_layers: nn.ModuleList, gumbel_softmax: bool, hard_softmax: bool):
        super(PITSuperNetCombiner, self).__init__()
        self.sn_input_layers = [_ for _ in input_layers]
        self.n_layers = len(input_layers)
        # initially alpha set non-trainable to allow warmup of the "user-defined" model
        self.alpha = nn.Parameter(
            (1 / self.n_layers) * torch.ones(self.n_layers, dtype=torch.float), requires_grad=False)
        self.register_buffer('theta_alpha', torch.tensor(self.n_layers, dtype=torch.float32))
        self.theta_alpha.data = self.alpha  # uniform initialization
        self._softmax_temperature = 1
        self._pit_layers = list()
        for _ in range(self.n_layers):
            self._pit_layers.append(list())
        self.layers_sizes = []
        self.layers_macs = []
        self.hard_softmax = hard_softmax
        if gumbel_softmax:
            self.sample_alpha = self.sample_alpha_gs
        else:
            self.sample_alpha = self.sample_alpha_sm

    def update_input_layers(self, input_layers: nn.Module):
        """Updates the list of input layers after torch.fx tracing, which "explodes" nn.Sequential
        and nn.ModuleList, causing the combiner to wrongly reference to the pre-tracing version.
        """
        il = [cast(nn.Module, input_layers.__getattr__(str(_))) for _ in range(self.n_layers)]
        self.sn_input_layers = il

        # update the size and macs removing PIT Modules from the "static" baseline
        for i, layer in enumerate(self.sn_input_layers):
            for module in layer.modules():
                if isinstance(module, PITModule):
                    cast(List[PITModule], self._pit_layers[i]).append(module)
                    with torch.no_grad():
                        self.layers_sizes[i] = self.layers_sizes[i] - module.get_size()

        for i, layer in enumerate(self.sn_input_layers):
            for module in layer.modules():
                if isinstance(module, PITModule):
                    with torch.no_grad():
                        self.layers_macs[i] = self.layers_macs[i] - module.get_macs()

    def sample_alpha_sm(self):
        """
        Samples the alpha architectural coefficients using a standard SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        self.theta_alpha = F.softmax(self.alpha / self.softmax_temperature, dim=0)
        if self.hard_softmax:
            self.theta_alpha = F.one_hot(
                    torch.argmax(self.theta_alpha, dim=0), num_classes=len(self.theta_alpha)
                ).to(torch.float32)

    def sample_alpha_gs(self):
        """
        Samples the alpha architectural coefficients using a Gumbel SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        if self.training:
            self.theta_alpha = nn.functional.gumbel_softmax(
                    self.alpha, self.softmax_temperature, self.hard_softmax, dim=0)
        else:
            self.sample_alpha_sm()

    def forward(self, layers_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for the PITSuperNetCombiner that returns a weighted
        sum of all the outputs of the different alternative layers.

        :param layers_outputs: outputs of all different modules
        :type layers_outputs: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """
        self.sample_alpha()
        y = []
        for i, yi in enumerate(layers_outputs):
            y.append(self.theta_alpha[i] * yi)
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    def best_layer_index(self) -> int:
        """Returns the index of the layer with the largest architectural coefficient
        """
        return int(torch.argmax(self.alpha).item())

    def compute_layers_sizes(self):
        """Computes the size of each possible layer of the PITSuperNetModule
        and stores the values in a list.
        It removes the size of the PIT modules contained in each layer because
        these sizes will be computed and re-added at training time.
        """
        for layer in self.sn_input_layers:
            layer_size = 0
            for param in layer.parameters():
                layer_size = layer_size + torch.prod(torch.tensor(param.shape))
            self.layers_sizes.append(layer_size)

    def compute_layers_macs(self, input_shape: Tuple[int, ...]):
        """Computes the MACs of each possible layer of the PITSuperNetModule
        and stores the values in a list.
        It removes the MACs of the PIT modules contained in each layer because
        these MACs will be computed and re-added at training time.
        """
        for layer in self.sn_input_layers:
            stats = summary(layer, input_shape, verbose=0, mode='eval')
            self.layers_macs.append(stats.total_mult_adds)

    @property
    def softmax_temperature(self) -> float:
        """Value of the temperature that divide the alpha for layer choice

        :return: Value of softmax_temperature
        :rtype: float
        """
        return self._softmax_temperature

    @softmax_temperature.setter
    def softmax_temperature(self, value: float):
        """Set the value of the temperature that divide the alpha for layer choice

        :param value: value
        :type value: float
        """
        self._softmax_temperature = value

    def get_size(self) -> torch.Tensor:
        """Method that returns the number of weights for the module
        computed as a weighted sum of the number of weights of each layer.

        :return: number of weights of the module (weighted sum)
        :rtype: torch.Tensor
        """
        size = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_layers):
            var_size = torch.tensor(0, dtype=torch.float32)
            for pl in cast(List[PITModule], self._pit_layers[i]):
                var_size = var_size + pl.get_size()
            size = size + (self.theta_alpha[i] * (self.layers_sizes[i] + var_size))
        return size

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the module

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_layers):
            var_macs = torch.tensor(0, dtype=torch.float32)
            for pl in cast(List[PITModule], self._pit_layers[i]):
                var_macs = var_macs + pl.get_macs()
            macs = macs + (self.theta_alpha[i] * (self.layers_macs[i] + var_macs))
        return macs

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized SN hyperparameters
        TODO: cleanup

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        with torch.no_grad():
            self.sample_alpha()

        res = {"supernet_branches": {}}
        for i, branch in enumerate(self.sn_input_layers):
            if hasattr(branch, "summary") and callable(branch.summary):
                branch_arch = branch.summary()
            else:
                branch_arch = {}
            branch_arch['type'] = branch.__class__.__name__
            branch_arch['alpha'] = self.theta_alpha[i].item()
            branch_layers = branch._modules
            for layer_name in branch_layers:
                layer = cast(nn.Module, branch_layers[layer_name])
                if hasattr(layer, "summary") and callable(layer.summary):
                    layer_arch = branch_layers[layer_name].summary()
                    layer_arch['type'] = branch_layers[layer_name].__class__.__name__
                    branch_arch[layer_name] = layer_arch
            res["supernet_branches"][f"branch_{i}"] = branch_arch
        return res

    @property
    def train_selection(self) -> bool:
        """True if the choice of layers is being optimized by the Combiner

        :return: True if the choice of layers is being optimized by the Combiner
        :rtype: bool
        """
        return self.alpha.requires_grad

    @train_selection.setter
    def train_selection(self, value: bool):
        """Set to True in order to let the Combiner optimize the choice of layers

        :param value: set to True in order to let the Combine optimize the choice of layers
        :type value: bool
        """
        self.alpha.requires_grad = value

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
