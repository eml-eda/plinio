from typing import Tuple, Iterable, List, Type, cast, Optional
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from flexnas.graph.annotation import add_node_properties, add_features_calculator, \
        associate_input_features
from flexnas.graph.inspection import is_layer, is_inherited_layer, get_graph_outputs, \
        get_graph_inputs, all_output_nodes
from flexnas.graph.features_calculation import SoftMaxFeaturesCalculator
from flexnas.graph.utils import fx_to_nx_graph
from .nn import PITSuperNetCombiner, PITSuperNetModule
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
            ) -> Tuple[nn.Module, List, List]:
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
    batch_example = torch.stack([torch.rand(input_shape)] * 1, 0)
    device = next(model.parameters()).device
    ShapeProp(mod).propagate(batch_example.to(device))
    add_node_properties(mod)
    set_combiner_properties(mod, add=['shared_input_features', 'features_propagating'])
    # first convert Conv/FC layers to/from PIT versions first
    pit_graph.convert_layers(mod, conversion_type, exclude_names, exclude_types)
    if conversion_type in ('autoimport', 'import'):
        # then, for import, perform additional graph passes needed mostly for PIT
        pit_graph.fuse_conv_bn(mod)
        add_features_calculator(mod, [pit_graph.pit_features_calc, combiner_features_calc])
        set_combiner_properties(mod, add=['features_defining'], remove=['features_propagating'])
        associate_input_features(mod)
        pit_graph.register_input_features(mod)
        # lastly, import SuperNet selection layers
        sn_target_layers = import_layers(mod)
        pit_only_layers = find_pit_only_layers(mod)
    else:
        export_graph(mod)
        sn_target_layers = []
        pit_only_layers = []

    mod.graph.lint()
    mod.recompile()
    return mod, sn_target_layers, pit_only_layers


def find_pit_only_layers(mod: fx.GraphModule) -> List[nn.Module]:
    pit_only_layers = []

    for n in mod.graph.nodes:
        if '.sn_input_layers' not in str(n.target):
            if is_inherited_layer(n, mod, (PITModule,)):
                sub_mod = mod.get_submodule(str(n.target))
                pit_only_layers.append(sub_mod)
    return pit_only_layers


def import_layers(mod: fx.GraphModule) -> List[Tuple[str, PITSuperNetCombiner]]:
    """TODO

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the sizes of PIT masks.
    :type mod: fx.GraphModule
    :return: the list of target layers that will be optimized by the NAS
    :rtype: List[Tuple[str, PITSuperNetCombiner]]
    """
    target_layers = []
    for n in mod.graph.nodes:
        if is_layer(n, mod, (PITSuperNetCombiner,)):
            # parent_name = n.target.removesuffix('.sn_combiner')
            parent_name = n.target.replace('.sn_combiner', '')
            sub_mod = cast(PITSuperNetCombiner, mod.get_submodule(str(n.target)))
            parent_mod = cast(PITSuperNetModule, mod.get_submodule(parent_name))
            # TODO: fix this mess
            prev = n.all_input_nodes[0]
            while '.sn_input_layers' in prev.target:
                prev = prev.all_input_nodes[0]
            input_shape = prev.meta['tensor_meta'].shape
            sub_mod.compute_layers_sizes()
            sub_mod.compute_layers_macs(input_shape)
            sub_mod.update_input_layers(parent_mod.sn_input_layers)
            sub_mod.train_selection = True
            target_layers.append((str(n.target), sub_mod))
    return target_layers


def export_graph(mod: fx.GraphModule):
    """TODO
    """
    for n in mod.graph.nodes:
        if 'sn_combiner' in str(n.target):
            sub_mod = cast(PITSuperNetCombiner, mod.get_submodule(n.target))
            best_idx = sub_mod.best_layer_index()
            best_branch_name = 'sn_input_layers.' + str(best_idx)
            to_erase = []
            for ni in n.all_input_nodes:
                if best_branch_name in str(ni.target):
                    n.replace_all_uses_with(ni)
                else:
                    to_erase.append(ni)
            n.args = ()
            mod.graph.erase_node(n)
            for ni in to_erase:
                ni.args = ()
                mod.graph.erase_node(ni)
    mod.graph.eliminate_dead_code()
    mod.delete_all_unused_submodules()


def set_combiner_properties(
        mod: fx.GraphModule,
        add: List[str] = [],
        remove: List[str] = []):
    """Searches for the combiner nodes in the graph and sets their properties

    :param mod: module
    :type mod: fx.GraphModule
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue

        if is_layer(n, mod, (PITSuperNetCombiner,)):
            for p in add:
                n.meta[p] = True
            for p in remove:
                n.meta[p] = False

        for succ in all_output_nodes(n):
            queue.append(succ)
        visited.append(n)


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
