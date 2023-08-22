from typing import Union, Tuple, Any, Iterator, Dict, Optional, cast
import torch
import torch.nn as nn
from plinio.methods.dnas_base import DNAS
from .nn.combiner import SuperNetCombiner
from plinio.cost import CostSpec, CostFn, params
from plinio.graph.inspection import shapes_dict
from .graph import convert


class SuperNet(DNAS):
    """A class that wraps a nn.Module with the functionality of the PITSuperNet NAS tool

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param cost: the cost models(s) used by the NAS, defaults to the number of params.
    :type cost: Union[CostSpec, Dict[str, CostSpec]]
    :param input_example: an input with the same shape and type of the seed's input, used
    for symbolic tracing (default: None)
    :type input_example: Optional[Any]
    :param input_shape: the shape of an input tensor, without batch size, used as an
    alternative to input_example to generate a random input for symbolic tracing (default: None)
    :type input_shape: Optional[Tuple[int, ...]]
    :param full_cost: True is the cost model should be applied to the entire network, rather
    than just to the NAS-able layers, defaults to False
    :type full_cost: bool, optional
    """
    def __init__(
            self,
            model: nn.Module,
            cost: Union[CostSpec, Dict[str, CostSpec]] = params,
            input_example: Optional[Any] = None,
            input_shape: Optional[Tuple[int, ...]] = None,
            full_cost: bool = False):
        super(SuperNet, self).__init__(model, cost, input_example, input_shape)
        self.seed, self._leaf_modules, self._unique_leaf_modules = convert(
            model, self._input_example, 'import')
        self._cost_fn_map = self._create_cost_fn_map()
        self.train_selection = True
        self.full_cost = full_cost

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self.seed.forward(*args)

    @property
    def cost_specification(self) -> Union[CostSpec, Dict[str, CostSpec]]:
        return self._cost_specification

    @cost_specification.setter
    def cost_specification(self, cs: Union[CostSpec, Dict[str, CostSpec]]):
        self._cost_specification = cs
        self._cost_fn_map = self._create_cost_fn_map()

    @property
    def train_selection(self):
        return self._train_selection

    @train_selection.setter
    def train_selection(self, value: bool):
        for _, _, layer in self._unique_leaf_modules:
            if isinstance(layer, SuperNetCombiner):
                layer.train_selection = value
        self._train_selection = value

    def get_total_icv(self, eps: float = 1e-3) -> torch.Tensor:
        """Computes the total inverse coefficient of variation of the SuperNet architectural
        parameters

        :param eps: stability term, limiting the max value of icv
        :type eps: float
        :return: the total ICV
        :rtype: torch.Tensor
        """
        tot_icv = torch.tensor(0, dtype=torch.float32)
        for _, _, layer in self._unique_leaf_modules:
            if isinstance(layer, SuperNetCombiner):
                theta_alpha = layer.theta_alpha
                tot_icv = tot_icv + (torch.mean(theta_alpha)**2) / (torch.var(theta_alpha) + eps)
        return tot_icv

    def update_softmax_options(
            self,
            temperature: Optional[float] = None,
            hard: Optional[bool] = None):
        """Update softmax options of all SuperNet blocks

        :param temperature: SoftMax temperature
        :type temperature: Optional[float]
        :param hard: Hard vs Soft sampling
        :type hard: Optional[bool]
        """
        for _, _, layer in self._unique_leaf_modules:
            if isinstance(layer, SuperNetCombiner):
                if temperature is not None:
                    layer.softmax_temperature = temperature
                if hard is not None:
                    layer.hard_softmax = hard

    def arch_export(self) -> nn.Module:
        """Export the architecture found by the NAS as a 'nn.Module'
        It replaces each PITSuperNetModule found in the model with a single layer.

        :return: the architecture found by the NAS
        :rtype: nn.Module
        """
        model = self.seed
        model, _, _ = convert(model, self._input_example, 'export')
        return model

    def arch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architecture found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for name, _, layer in self._unique_leaf_modules:
            if isinstance(layer, SuperNetCombiner):
                arch[name] = layer.summary()
                arch[name]['type'] = layer.__class__.__name__
        return arch

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of the NAS, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API
        :type recurse: bool
        :return: an iterator over the architectural parameters of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        included = set()
        for lname, layer in self.named_modules():
            if isinstance(layer, SuperNetCombiner):
                layer = cast(SuperNetCombiner, layer)
                prfx = prefix
                prfx += "." if len(prefix) > 0 else ""
                prfx += lname
                for name, param in layer.named_nas_parameters(prefix=prfx, recurse=recurse):
                    # avoid duplicates (e.g. shared channels masks)
                    if param not in included:
                        included.add(param)
                        yield name, param

    def named_net_parameters(
            self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the inner network parameters, EXCEPT the NAS architectural
        parameters, yielding both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, not actually used
        :type recurse: bool
        :return: an iterator over the inner network parameters
        :rtype: Iterator[nn.Parameter]
        """
        exclude = set(_[1] for _ in self.named_nas_parameters())
        for name, param in self.named_parameters(prefix=prefix, recurse=recurse):
            if param not in exclude:
                yield name, param

    def _get_single_cost(self, cost_spec: CostSpec,
                         cost_fn_map: Dict[str, CostFn]) -> torch.Tensor:
        """Private method to compute a single cost value"""
        # TODO: relies on the attribute name sn_branches. Not nice, but didn't find
        # a better solution that remains flexible.
        cost = torch.tensor(0, dtype=torch.float32)
        target_list = self._unique_leaf_modules if cost_spec.shared else self._leaf_modules
        for lname, node, layer in target_list:
            if isinstance(layer, SuperNetCombiner):
                cost = cost + layer.get_cost(cost_spec, cost_fn_map)
            elif 'sn_branches' not in str(node.target) and self.full_cost:
                # TODO: this is constant and can be pre-computed for efficiency
                v = vars(layer)
                v.update(shapes_dict(node))
                cost = cost + cost_fn_map[lname](v)
        return cost

    def _single_cost_fn_map(self, c: CostSpec) -> Dict[str, CostFn]:
        """SuperNet-specific creator of {layertype, cost_fn} maps based on a CostSpec."""
        # simply computes cost of all layers that are not Combiners
        cost_fn_map = {}
        for lname, _, layer in self._unique_leaf_modules:
            if not isinstance(layer, SuperNetCombiner):
                t = type(layer)
                cost_fn_map[lname] = c[(t, vars(layer))]
        return cost_fn_map

    def __str__(self):
        """Prints the architecture found by the NAS to screen

        :return: a str representation of the current architecture
        :rtype: str
        """
        arch = self.arch_summary()
        return str(arch)
