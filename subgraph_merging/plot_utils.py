from itertools import count, combinations
import networkx as nx
import matplotlib.pyplot as plt
from .common import *

def plot_compatibility_graph(g1, g1_map, g2, g2_map, gb, gc):
    plt.subplot(1, 4, 1)
    plt.margins(0.2)
    g = g1
    groups1 = set(nx.get_node_attributes(g1, 'op').values())
    groups2 = set(nx.get_node_attributes(g2, 'op').values())
    groups = groups1.union(groups2)
    mapping = dict(zip(sorted(groups), count()))
    nodes = g.nodes()
    colors = [plt.cm.Pastel1(mapping[g.nodes[n]['op']]) for n in nodes]

    labels = {}
    for n in nodes:
        labels[n] = g1_map[n] + "\n" + op_types[g.nodes[n]['op']]
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    ec = nx.draw_networkx_edges(
        g,
        pos,
        alpha=1,
        width=3,
        node_size=1500,
        arrows=True,
        arrowsize=15)
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        # node_list=nodes, 
        node_color=colors,
        # with_labels=False,
        node_size=1500,
        alpha=1)
    nx.draw_networkx_labels(g, pos, labels)

    plt.subplot(1, 4, 2)
    plt.margins(0.2)
    g = g2
    nodes = g.nodes()
    colors = [plt.cm.Pastel1(mapping[g.nodes[n]['op']]) for n in nodes]

    labels = {}
    for n in nodes:
        labels[n] = g2_map[n] + "\n" + op_types[g.nodes[n]['op']]
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    ec = nx.draw_networkx_edges(
        g,
        pos,
        alpha=1,
        width=3,
        node_size=1500,
        arrows=True,
        arrowsize=15)
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        # node_list=nodes,
        node_color=colors,
        # with_labels=False,
        node_size=1500,
        alpha=1)
    nx.draw_networkx_labels(g, pos, labels)

    plot_gb = gb.copy()


    plt.subplot(1, 4, 3)

    left = [n for n, d in plot_gb.nodes(data=True) if d['bipartite'] == 0]
    right = [n for n, d in plot_gb.nodes(data=True) if d['bipartite'] == 1]
    pos = {}

    # Update position for node from each group
    pos.update((node, (1, index)) for index, node in enumerate(left))
    pos.update((node, (2, index)) for index, node in enumerate(right))

    nx.draw_networkx(
        plot_gb,
        node_color=plt.cm.Pastel1(2),
        node_size=1500,
        pos=pos,
        width=3)
    plt.margins(0.2)

    plot_gc = gc.copy()

    # for n, d in plot_gc.copy().nodes.data(True):
    #     if d["in_or_out"]:
    #         plot_gc.remove_node(n)

    plt.subplot(1, 4, 4)
    starts = nx.get_node_attributes(plot_gc, 'start')
    ends = nx.get_node_attributes(plot_gc, 'end')
    weights = nx.get_node_attributes(plot_gc, 'weight')
    labels = {
        n: d + "/" + ends[n] + "\n" + str(weights[n])
        for n, d in starts.items()
    }
    nx.draw_networkx(
        plot_gc,
        node_color=plt.cm.Pastel1(2),
        node_size=1500,
        width=3,
        labels=labels)
    plt.margins(0.2)
    plt.show()



def plot_max_weight_clique(gc, widths):
    plot_gc = gc.copy()

    for n, d in plot_gc.copy().nodes.data(True):
        if d["in_or_out"]:
            plot_gc.remove_node(n)

    plt.subplot(1, 2, 1)
    starts = nx.get_node_attributes(plot_gc, 'start')
    ends = nx.get_node_attributes(plot_gc, 'end')
    weights = nx.get_node_attributes(plot_gc, 'weight')
    labels = {
        n: d + "/" + ends[n] + "\n" + str(weights[n])
        for n, d in starts.items()
    }
    pos = nx.drawing.layout.spring_layout(plot_gc)
    nx.draw_networkx(
        plot_gc,
        pos,
        node_color=plt.cm.Pastel1(2),
        node_size=1500,
        width=3,
        labels=labels)
    plt.margins(0.2)

    plt.subplot(1, 2, 2)
    starts = nx.get_node_attributes(plot_gc, 'start')
    ends = nx.get_node_attributes(plot_gc, 'end')
    weights = nx.get_node_attributes(plot_gc, 'weight')
    labels = {
        n: d + "/" + ends[n] + "\n" + str(weights[n])
        for n, d in starts.items()
    }

    colors = [plt.cm.Pastel1(widths[n]) for n in plot_gc.nodes()]

    nx.draw_networkx(
        plot_gc,
        pos,
        node_color=colors,
        node_size=1500,
        width=3,
        labels=labels)
    plt.margins(0.2)
    plt.show()


def plot_reconstructed_graph(g1, g2, g):
    graphs = [g1, g2, g]
    for i, g in enumerate(graphs):

        ret_g = g.copy()

        plt.subplot(1, 3, i + 1)
        groups = set(nx.get_node_attributes(ret_g, 'op').values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = ret_g.nodes()
        colors = [mapping[ret_g.nodes[n]['op']] for n in nodes]
        labels = {}
        for n in nodes:
            labels[n] = op_types[ret_g.nodes[n]['op']] + "\n" + n

        pos = nx.nx_agraph.graphviz_layout(ret_g, prog='dot')
        ec = nx.draw_networkx_edges(
            ret_g,
            pos,
            alpha=1,
            width=3,
            node_size=1500,
            arrows=True,
            arrowsize=15)
        nc = nx.draw_networkx_nodes(
            ret_g,
            pos,
            # node_list=nodes,
            node_color=colors,
            # with_labels=False,
            node_size=1500,
            cmap=plt.cm.Pastel1,
            alpha=1)
        nx.draw_networkx_labels(ret_g, pos, labels)

    plt.show()


def plot_graph(g):


    ret_g = g.copy()

    groups = set(nx.get_node_attributes(ret_g, 'op').values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = ret_g.nodes()
    colors = [mapping[ret_g.nodes[n]['op']] for n in nodes]
    labels = {}
    for n in nodes:
        labels[n] = op_types[ret_g.nodes[n]['op']] + "\n" + n

    pos = nx.nx_agraph.graphviz_layout(ret_g, prog='dot')
    ec = nx.draw_networkx_edges(
        ret_g,
        pos,
        alpha=1,
        width=3,
        node_size=1500,
        arrows=True,
        arrowsize=15)
    nc = nx.draw_networkx_nodes(
        ret_g,
        pos,
        # node_list=nodes,
        node_color=colors,
        # with_labels=False,
        node_size=1500,
        cmap=plt.cm.Pastel1,
        alpha=1)
    nx.draw_networkx_labels(ret_g, pos, labels)

    plt.show()