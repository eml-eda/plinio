from typing import List, cast, Iterator, Tuple, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from plinio.graph.utils import NamedLeafModules
from plinio.graph.inspection import shapes_dict
from plinio.cost import CostSpec, CostFn


class SuperNetCombiner(nn.Module):
    """This nn.Module is included in the SuperNetModule and contains most of the logic of
    the module and the NAS paramaters.

    :param n_branches: number of SuperNet branches
    :type n_branches: int
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SofrMax
    :type gumbel_softmax: bool
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool
    """
    def __init__(self, n_branches: int, gumbel_softmax: bool, hard_softmax: bool):
        super(SuperNetCombiner, self).__init__()
        # uniform initialization of NAS parameters
        # initially set non-trainable to allow warmup of the "user-defined" model
        self.n_branches = n_branches
        self.alpha = nn.Parameter(
            (1 / n_branches) * torch.ones(n_branches, dtype=torch.float), requires_grad=False)
        self.theta_alpha = torch.tensor(self.n_branches, dtype=torch.float32)
        self.theta_alpha.data = self.alpha
        self._softmax_temperature = 1
        self.hard_softmax = hard_softmax
        if gumbel_softmax:
            self.sample_alpha = self.sample_alpha_gs
        else:
            self.sample_alpha = self.sample_alpha_sm
        self._unique_leaf_modules = [[]] * self.n_branches
        self._cost_fn_map = None

    def set_sn_branch(self, i: int, ulf: NamedLeafModules):
        """Associates the lists of all unique leaf modules in each SuperNet branch
        to the combiner
        """
        self._unique_leaf_modules[i] = ulf

    def get_cost(self, cost_spec: CostSpec, cost_fn_map: Dict[str, CostFn]) -> torch.Tensor:
        """Links the SuperNet branches to the Combiner after torch.fx tracing, which "explodes"
        nn.Sequential and nn.ModuleList.
        """
        cost = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_branches):
            cost_i = torch.tensor(0, dtype=torch.float32)
            for lname, node, layer in self._unique_leaf_modules[i]:
                # TODO: this is constant and can be pre-computed for efficiency
                v = vars(layer)
                v.update(shapes_dict(node))
                cost_i = cost_i + cost_fn_map[lname](v)
            cost = cost + (cost_i * self.theta_alpha[i])
        return cost

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

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized SN hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        with torch.no_grad():
            self.sample_alpha()
        res = {"supernet_branches": {}}
        for i in range(self.n_branches):
            res["supernet_branches"][f"branch_{i}"] = {}
            res["supernet_branches"][f"branch_{i}"]['alpha'] = self.theta_alpha[i].item()
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
