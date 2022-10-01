
from typing import Tuple, Iterable, List, Any, Iterator
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from flexnas.utils import model_graph
from flexnas.methods.dnas_base import DNAS
from flexnas.methods.supernet.supernet_module import SuperNetModule


class SuperNetTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, SuperNetModule):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


class SuperNet(DNAS):

    def __init__(
            self,
            model: nn.Module,
            input_shape: Tuple[int, ...],
            regularizer: str = 'size',
            exclude_names: Iterable[str] = ()):

        super(SuperNet, self).__init__(regularizer, exclude_names)

        self._input_shape = input_shape
        self._regularizer = regularizer
        self.seed = model
        self.exclude_names = exclude_names

        target_modules = []
        self._target_modules = self.get_supernetModules(target_modules, model, exclude_names)

        tracer = SuperNetTracer()
        graph = tracer.trace(model.eval())
        name = model.__class__.__name__
        self.mod = fx.GraphModule(tracer.root, graph, name)
        # create a "fake" minibatch of 1 inputs for shape prop
        batch_example = torch.stack([torch.rand(self._input_shape)] * 1, 0)

        # TODO: this is not very robust. Find a better way
        device = next(model.parameters()).device
        ShapeProp(self.mod).propagate(batch_example.to(device))

    def get_supernetModules(
            self,
            target_modules: List,
            model: nn.Module,
            exclude_names: Iterable[str]) -> List[Tuple[str, SuperNetModule]]:
        """This function spots each SuperNetModule contained in the model received
        and saves them in a list.

        :param target_modules: list where the function saves the SuperNetModules
        :type target_modules: List
        :param model: seed model
        :type model: nn.Module
        :param exclude_names: the names of `model` submodules that should be ignored by the NAS
        :type exclude_names: Iterable[str]
        :return: list of target modules for the NAS
        :rtype: List[Tuple[str, SuperNetModule]]
        """
        for named_module in model.named_modules():
            if(named_module[0] != ''):
                submodules = list(named_module[1].children())
                if(named_module[1].__class__.__name__ == "SuperNetModule" and
                        named_module[0] not in exclude_names):
                    target_modules.append(named_module)
                elif(submodules):
                    for child in submodules:
                        self.get_supernetModules(target_modules, child, exclude_names)
            elif(named_module[1].__class__.__name__ == "SuperNetModule"):
                target_modules.append(named_module)

        return target_modules

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self.seed.forward(*args)

    def supported_regularizers(self) -> Tuple[str, ...]:
        """Returns a list of names of supported regularizers

        :return: a tuple of strings with the name of supported regularizers
        :rtype: Tuple[str, ...]
        """
        return ('size', 'macs')

    def get_size(self) -> torch.Tensor:
        """Computes the total number of parameters of all NAS-able modules

        :return: the total number of parameters
        :rtype: torch.Tensor
        """
        size = torch.tensor(0, dtype=torch.float32)
        for named_module in self._target_modules:
            size = size + named_module[1].get_size()
        return size

    def get_macs(self) -> torch.Tensor:
        """Computes the total number of MACs in all NAS-able modules

        :return: the total number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        if (self._target_modules):
            g = self.mod.graph
            queue = model_graph.get_output_nodes(g)
            target_nodes = []
            while queue:
                n = queue.pop(0)
                if(n.name in self._target_modules[0]):
                    target_nodes.append(n)
                for pred in n.all_input_nodes:
                    queue.append(pred)

            for i, named_module in enumerate(self._target_modules):
                shape = target_nodes[i].all_input_nodes[0].meta['tensor_meta'].shape
                named_module[1].input_shape = shape
                macs = macs + named_module[1].get_macs()

        return macs

    @property
    def regularizer(self) -> str:
        """Returns the regularizer type

        :raises ValueError: for unsupported conversion types
        :return: the string identifying the regularizer type
        :rtype: str
        """
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: str):
        if value == 'size':
            self.get_regularization_loss = self.get_size
        elif value == 'macs':
            self.get_regularization_loss = self.get_macs
        else:
            raise ValueError(f"Invalid regularizer {value}")
        self._regularizer = value

    def arch_export(self) -> nn.Module:
        """Export the architecture found by the NAS as a 'nn.Module'
        It replaces each SuperNetModule found in the model with a single layer.

        :return: the architecture found by the NAS
        :rtype: nn.Module
        """
        for module in self._target_modules:
            name = module[0]
            if name == '':
                self.seed = module[1].export()
            else:
                setattr(self.seed, module[0], module[1].export())

        return self.seed

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
        for sn_module in self._target_modules:
            prfx = prefix
            prfx += "." if len(prefix) > 0 else ""
            prfx += "alpha"

            yield prfx, sn_module[1].alpha

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
        exclude = set(_[0] for _ in self.named_nas_parameters())

        for name, param in self.seed.named_parameters():
            last_name = name.split(".")[-1]
            if last_name not in exclude:
                yield name, param
