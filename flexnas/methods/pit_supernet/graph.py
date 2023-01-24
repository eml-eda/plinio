from typing import Tuple, Iterable, List, Type, cast, Optional
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from flexnas.graph.annotation import add_node_properties, add_features_calculator, \
        associate_input_features
from flexnas.graph.inspection import is_layer, get_graph_outputs, get_graph_inputs, \
        all_output_nodes
from flexnas.graph.features_calculation import SoftMaxFeaturesCalculator
from .nn.combiner import PITSuperNetCombiner
from flexnas.methods.pit import graph as pit_graph
from flexnas.methods.pit.nn import PITModule


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


def convert(model: nn.Module, input_shape: Tuple[int, ...], conversion_type: str,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = ()
            ) -> Tuple[nn.Module, List]:
    """Converts a nn.Module, to/from "NAS-able" PIT and SuperNet format

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'autoimport', 'export')
    :type conversion_type: str
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: for unsupported conversion types
    :return: the converted model, and the list of target layers for the NAS (only for imports)
    :rtype: Tuple[nn.Module, List]
    """

    if conversion_type not in ('import', 'autoimport', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    tracer = PITSuperNetTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    batch_example = torch.stack([torch.rand(input_shape)] * 32, 0)
    device = next(model.parameters()).device
    ShapeProp(mod).propagate(batch_example.to(device))
    add_node_properties(mod)
    add_combiner_properties(mod)
    # first convert Conv/FC layers to/from PIT versions first
    pit_graph.convert_layers(mod, conversion_type, exclude_names, exclude_types)
    # then import/export SuperNet selection layers
    sn_target_layers = convert_layers(mod, conversion_type)
    sn_target_layers.reverse()
    if conversion_type in ('autoimport', 'import'):
        pit_graph.fuse_conv_bn(mod)
        add_features_calculator(mod, [pit_graph.pit_features_calc, combiner_features_calc])
        associate_input_features(mod)
        pit_graph.register_input_features(mod)

    if conversion_type == 'export':
        clean_graph(mod)

    mod.graph.lint()
    mod.recompile()
    return mod, sn_target_layers


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str) -> List[Tuple[str, PITSuperNetCombiner]]:
    """Exports the PITSuperNetModule layers if in 'export' mode or creates a list of the
    PITSuperNetModules as target_layers if in 'import'/'autoimport' mode.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the sizes of PIT masks.
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :return: the list of target layers that will be optimized by the NAS
    :rtype: List[Tuple[str, PITSuperNetCombiner]]
    """
    g = mod.graph
    queue = get_graph_outputs(g)

    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)

        if n not in visited:
            if conversion_type == 'export':
                export_node(n, mod)
            if conversion_type in ('import', 'autoimport'):
                import_node(n, mod, target_layers)

            for pred in n.all_input_nodes:
                queue.append(pred)

            visited.append(n)
    return target_layers


def export_node(n: fx.Node, mod: fx.GraphModule):
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a NAS-able
    layer with its PITSuperNet choice

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    """
    if is_layer(n, mod, (PITSuperNetCombiner,)):
        sub_mod = mod.get_submodule(str(n.target))
        layer = cast(PITSuperNetCombiner, sub_mod)
        layer.export(n, mod)


def import_node(n: fx.Node, mod: fx.GraphModule,
                target_layers: List[Tuple[str, PITSuperNetCombiner]]):
    """Optionally adds the layer corresponding to a torch.fx.Node to the list of NAS target
    layers and computes the number of MACs and parameters for each "fixed" (i.e., not-prunable)
    branch

    :param n: the node to be added
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :param target_layers: the list of current target layers
    :type target_layers: List[Tuple[str, PITSuperNetCombiner]]
    """
    if is_layer(n, mod, (PITSuperNetCombiner,)):
        submod = cast(PITSuperNetCombiner, mod.get_submodule(str(n.target)))
        submod.compute_layers_sizes()
        submod.compute_layers_macs()
        target_layers.append((str(n.target), submod))


def clean_graph(mod: fx.GraphModule):
    """This function cleans the mod from unused nodes and modules after the export

    :param mod: the module
    :type mod: fx.GraphModule
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    visited = []
    prev_args = None
    while queue:
        n = queue.pop(0)
        # skip nodes for which predecessors have not yet been processed completely, we'll come
        # back to them later
        skip_flag = False
        if len(n.all_input_nodes) > 0:
            for i in n.all_input_nodes:
                if i not in visited:
                    skip_flag = True
        if skip_flag:
            continue

        if n.op == 'call_module':
            if 'sn_input_layers' in str(n.target):
                if prev_args is None:
                    prev_args = n.args
                n.args = ()
            if 'sn_combiner' in str(n.target):
                n.args = cast(Tuple, prev_args)
                prev_args = None

        visited.append(n)
        for succ in all_output_nodes(n):
            queue.append(succ)

    for node in mod.graph.nodes:
        if node.op == 'call_module':
            if 'sn_input_layers' in node.target:
                mod.graph.erase_node(node)
    mod.delete_all_unused_submodules()


def add_combiner_properties(mod: fx.GraphModule):
    """Searches for the combiner nodes in the graph and adds their properties

    :param mod: module
    :type mod: fx.GraphModule
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    while queue:
        n = queue.pop(0)

        if is_layer(n, mod, (PITSuperNetCombiner,)):
            n.meta['shared_input_features'] = True
            n.meta['features_defining'] = True

        for succ in all_output_nodes(n):
            queue.append(succ)


def combiner_features_calc(n: fx.Node, mod: fx.GraphModule) -> Optional[SoftMaxFeaturesCalculator]:
    """Sets the feature calculator for a PITSuperNetCombiner node

    :param n: node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :return: optional feature calculator object for the combiner node
    :rtype: SoftMaxFeaturesCalculator
    """
    if is_layer(n, mod, (PITSuperNetCombiner,)):
        sub_mod = mod.get_submodule(str(n.target))
        prev_features = [_.meta['features_calculator'] for _ in n.all_input_nodes]
        return SoftMaxFeaturesCalculator(sub_mod, 'alpha', prev_features)
    else:
        return None
