from typing import Tuple, Iterable, List, Type, cast
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
    # pit_target_layers
    pit_graph.convert_layers(mod, conversion_type, exclude_names, exclude_types)
    sn_target_layers = convert_layers(mod, conversion_type)
    sn_target_layers.reverse()
    compute_shapes(mod, sn_target_layers)
    if conversion_type in ('autoimport', 'import'):
        # pit_graph.fuse_conv_bn(mod)
        model_graph.add_features_calculator(mod,
                                            [pit_graph.pit_features_calc, combiner_features_calc])
        model_graph.associate_input_features(mod)
        pit_graph.register_input_features(mod)

    if conversion_type == 'export':
        clean_graph(mod)

    mod.graph.lint()
    mod.recompile()
    return mod, sn_target_layers


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str) -> List[Tuple[str, PITSuperNetCombiner]]:
    g = mod.graph
    queue = model_graph.get_output_nodes(g)

    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)

        if n not in visited:
            if conversion_type == 'export':
                export_node(n, mod)
            if conversion_type == 'import':
                add_to_targets(n, mod, target_layers)

            for pred in n.all_input_nodes:
                queue.append(pred)

            visited.append(n)
    return target_layers


def export_node(n: fx.Node, mod: fx.GraphModule):
    if model_graph.is_layer(n, mod, (PITSuperNetCombiner,)):
        sub_mod = mod.get_submodule(str(n.target))
        layer = cast(PITSuperNetCombiner, sub_mod)
        layer.export(n, mod)


def add_to_targets(n: fx.Node, mod: fx.GraphModule,
                   target_layers: List[Tuple[str, PITSuperNetCombiner]]):
    if model_graph.is_layer(n, mod, (PITSuperNetCombiner,)):
        target_layers.append((str(n.target),
                             cast(PITSuperNetCombiner, mod.get_submodule(str(n.target)))))


def compute_shapes(mod: fx.GraphModule, target_modules: List[Tuple[str, PITSuperNetCombiner]]):
    """This function computes the input shape for each SuperNetModule in the target modules
    """
    if (target_modules):
        g = mod.graph

        for t in target_modules:
            for n in g.nodes:
                if t[0] == n.target:
                    t[1].input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape

        for target in target_modules:
            target[1].compute_layers_sizes()
            target[1].compute_layers_macs()


def clean_graph(mod: fx.GraphModule):
    g = mod.graph
    nx_graph = model_graph.fx_to_nx_graph(g)
    queue = model_graph.get_input_nodes(g)
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
                n.args = prev_args
                prev_args = None

        visited.append(n)
        for succ in nx_graph.successors(n):
            queue.append(succ)

    for node in mod.graph.nodes:
        if node.op == 'call_module':
            if '_input_layers' in node.target:
                mod.graph.erase_node(node)


def combiner_features_calc(n, mod):
    if model_graph.is_inherited_layer(n, mod, (PITSuperNetCombiner,)):
        sub_mod = mod.get_submodule(str(n.target))
        prev_features = [_.meta['features_calculator'] for _ in n.all_input_nodes]
        return SoftMaxFeaturesCalculator(sub_mod, 'alpha', prev_features)
    else:
        return None
