import torch.fx as fx
from flexnas.utils import model_graph


def set_input_features(mod: fx.GraphModule):
    g = mod.graph
    # convert to networkx graph to have successors information, fx only gives predecessors
    # unfortunately
    nx_graph = model_graph.fx_to_nx_graph(g)
    queue = model_graph.get_input_nodes(g)

    while queue:
        n = queue.pop(0)

        input_nodes = n.all_input_nodes
        if len(input_nodes) > 0:
            # check_prev_nodes(n, input_nodes, mod)
            check_prev_node(n, input_nodes[0], mod)
        else:  # input node
            n.meta['input_features_set_by'] = n

        for succ in nx_graph.successors(n):
            queue.append(succ)


def check_prev_node(n: fx.Node, prev: fx.Node, mod: fx.GraphModule):
    if model_graph.is_features_defining_op(prev, mod):
        n.meta['input_features_set_by'] = prev
    elif model_graph.is_features_propagating_op(prev, mod):
        n.meta['input_features_set_by'] = prev.meta['input_features_set_by']
    else:
        n.meta['input_features_set_by'] = prev  # something not included in the checks
