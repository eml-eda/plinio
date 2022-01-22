import torch.nn as nn
import networkx as nx
from torch.fx import symbolic_trace


def model_to_nx_graph(model: nn.Module) -> nx.Graph:
    fx_graph = symbolic_trace(model).graph
    nx_graph = nx.DiGraph()
    for n in fx_graph.nodes:
        for i in n.all_input_nodes:
            nx_graph.add_edge(i.name, n.name)
    return nx_graph
