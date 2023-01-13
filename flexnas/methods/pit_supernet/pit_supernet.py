from typing import Tuple, Any, Iterator
import torch
import torch.nn as nn
from flexnas.methods.dnas_base import DNAS
from .pit_supernet_graph import convert


class PITSuperNet(DNAS):
    """A class that wraps a nn.Module with the functionality of the SuperNet NAS tool

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param regularizer: a string defining the type of cost regularizer, defaults to 'size'
    :type regularizer: Optional[str], optional
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    """
    def __init__(
            self,
            model: nn.Module,
            input_shape: Tuple[int, ...],
            regularizer: str = 'size'):

        super(PITSuperNet, self).__init__(regularizer)

        self._input_shape = input_shape
        self._regularizer = regularizer
        self.seed, self._target_modules = convert(model, self._input_shape, 'import')

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
        for module in self._target_modules:
            size = size + module[1].get_size()
        return size

    def get_macs(self) -> torch.Tensor:
        """Computes the total number of MACs in all NAS-able modules

        :return: the total number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        for t in self._target_modules:
            macs = macs + t[1].get_macs()
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

    def arch_export(self, add_bn=True) -> nn.Module:
        """Export the architecture found by the NAS as a 'nn.Module'
        It replaces each SuperNetModule found in the model with a single layer.

        :return: the architecture found by the NAS
        :rtype: nn.Module
        """
        model = self.seed
        '''
        if not add_bn:
            for layer in self._target_layers:
                if hasattr(layer, 'following_bn_args'):
                    layer.following_bn_args = None
        '''
        model, _ = convert(model, self._input_shape, 'export')

        return model

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
        for module in self._target_modules:
            prfx = prefix
            prfx += "." if len(prefix) > 0 else ""
            prfx += module[0]
            prfx += "." if len(prfx) > 0 else ""
            for name, param in module[1].named_nas_parameters():
                prfx = prfx + name
                yield prfx, param

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
            if name not in exclude:
                yield name, param

    '''
    def arch_summary(self) -> Dict[str, str]:
        """Generates a dictionary representation of the architecture found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}

        for module in self._target_modules:
            mod = module[1].export()
            name = mod.__class__.__name__
            if (name == "Conv2d"):
                kernel_size = mod.kernel_size
                t = (name, kernel_size)
                arch[module[0]] = t
            elif (name == "ConvBlock"):
                children = mod.children()
                child = next(children)
                name_child = child.__class__.__name__
                kernel_size = child.kernel_size
                t = (name_child, kernel_size)
                arch[module[0]] = t
            elif (name == "Sequential"):
                children = mod.children()
                child1 = next(children)
                child2 = next(children)
                if (child2.__class__.__name__ == "Conv2d"):
                    arch[module[0]] = "Depthwise Separable"
                elif (child2.__class__.__name__ == "ConvBlock"):
                    arch[module[0]] = "Depthwise Separable"
                else:
                    name_child = child1.__class__.__name__
                    kernel_size = child1.kernel_size
                    t = (name_child, kernel_size)
                    arch[module[0]] = t
            else:
                arch[module[0]] = name
        return arch

    def __str__(self):
        """Prints the architecture found by the NAS to screen

        :return: a str representation of the current architecture
        :rtype: str
        """
        arch = self.arch_summary()
        return str(arch)
    '''
