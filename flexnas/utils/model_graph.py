import torch.fx as fx
import networkx as nx


def fx_to_nx_graph(fx_graph: fx.Graph) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    for n in fx_graph.nodes:
        for i in n.all_input_nodes:
            nx_graph.add_edge(i, n)
    return nx_graph
