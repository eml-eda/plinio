import torch.fx as fx
import networkx as nx
from torch.fx import symbolic_trace


def fx_to_nx_graph(fx_graph: fx.Graph) -> nx.Graph:
    nx_graph = nx.DiGraph()
    for n in fx_graph.nodes:
        for i in n.all_input_nodes:
            nx_graph.add_edge(i.name, n.name)
    return nx_graph
