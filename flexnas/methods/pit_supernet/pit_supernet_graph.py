from typing import Tuple, Iterable, List, Type, cast, Optional
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from flexnas.utils import model_graph
from flexnas.utils.features_calculator import SoftMaxFeaturesCalculator
from .pit_supernet_combiner import PITSuperNetCombiner
from flexnas.methods.pit import pit_graph
from flexnas.methods.pit import PITModule


class PITSuperNetTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, PITSuperNetCombiner):
            return True
        if isinstance(m, PITModule):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def add_combiner_properties(mod: fx.GraphModule):
    g = mod.graph
    nx_graph = model_graph.fx_to_nx_graph(g)
    queue = model_graph.get_input_nodes(g)

    while queue:
        n = queue.pop(0)

        if n.op == 'call_module':
            sub_mod = mod.get_submodule(str(n.target))
            if isinstance(sub_mod, PITSuperNetCombiner):
                n.meta['shared_input_features'] = True

        for succ in nx_graph.successors(n):
            queue.append(succ)


def convert(model: nn.Module, input_shape: Tuple[int, ...], conversion_type: str,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = ()
            ) -> Tuple[nn.Module, List]:

    if conversion_type not in ('import', 'autoimport', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    tracer = PITSuperNetTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    batch_example = torch.stack([torch.rand(input_shape)] * 32, 0)
    device = next(model.parameters()).device
    ShapeProp(mod).propagate(batch_example.to(device))
    model_graph.add_node_properties(mod)
    add_combiner_properties(mod)
    target_layers = pit_graph.convert_layers(mod, conversion_type, exclude_names, exclude_types)
    convert_layers(mod, conversion_type)
    if conversion_type in ('autoimport', 'import'):
        # pit_graph.fuse_conv_bn(mod)
        model_graph.add_features_calculator(mod,
                                            [pit_graph.pit_features_calc, combiner_features_calc])
        model_graph.associate_input_features(mod)
        pit_graph.register_input_features(mod)
    # mod.graph.lint()
    mod.recompile()
    return mod, target_layers


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str) -> List[nn.Module]:
    """Replaces target layers with their NAS-able version, or vice versa, while also
    recording the list of NAS-able layers for speeding up later regularization loss
    computations. Layer conversion is implemented as a reverse BFS on the model graph.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the sizes of PIT masks.
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :return: the list of target layers that will be optimized by the NAS
    :rtype: List[nn.Module]
    """
    g = mod.graph
    queue = model_graph.get_output_nodes(g)

    # the list of target layers is only used in 'import' and 'autoimport' modes. Empty for export
    target_layers = []
    visited = []
    exported_layers = []
    while queue:
        n = queue.pop(0)

        if n not in visited:
            new_target = str(n.target).split('.')[0]
            if new_target not in exported_layers:
                if conversion_type == 'export':
                    target = export_node(n, mod)
                    if target:
                        exported_layers.append(target)
                # if conversion_type in ('import', 'autoimport'):
                    # add_to_targets(n, mod, target_layers, exclude_names, exclude_types)

            for pred in n.all_input_nodes:
                queue.append(pred)

            visited.append(n)
    return target_layers


def export_node(n: fx.Node, mod: fx.GraphModule) -> Optional[str]:
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a NAS-able
    layer with its original nn.Module counterpart

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    if n.op == 'call_module':
        sub_mod = mod.get_submodule(str(n.target))
        if isinstance(sub_mod, PITSuperNetCombiner):
            layer = cast(PITSuperNetCombiner, sub_mod)
            target = layer.export(n, mod)
            return target


def combiner_features_calc(n, mod):
    if model_graph.is_inherited_layer(n, mod, (PITSuperNetCombiner,)):
        sub_mod = mod.get_submodule(str(n.target))
        prev_features = [_.meta['features_calculator'] for _ in n.all_input_nodes]
        return SoftMaxFeaturesCalculator(sub_mod, 'alpha', prev_features)
    else:
        return None
