#!/usr/bin/env python
# coding: utf-8

import itertools
import base64

import numpy as np
import pandas as pd
import plotly
import scipy.spatial
from plotly import express as px
from plotly import graph_objs as go
from umap import UMAP
from tqdm.notebook import tqdm
import collections
import pickle
import string
import os

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from pyphi import relations as rel
from pyphi.utils import powerset
from pyphi import direction

import tkinter as tk
from matplotlib.image import imread



def get_screen_size():
    havedisplay = "DISPLAY" in os.environ

    if not havedisplay:
        return 1920, 1080
    else:
        root = tk.Tk()

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        return screen_width, screen_height


def flatten(iterable):
    return itertools.chain.from_iterable(iterable)

def strip_punct(s):
    return str(s.translate(str.maketrans({key: None for key in string.punctuation})).replace(' ', ''))

def i2n(subsystem,mech):
    return strip_punct(subsystem.indices2nodes(mech))

def feature_matrix(ces, relations):
    """Return a matrix representing each cause and effect in the CES.

    .. note::
        Assumes that causes and effects have been separated.
    """
    N = len(ces)
    M = len(relations)
    # Create a mapping from causes and effects to indices in the feature matrix
    index_map = {purview: i for i, purview in enumerate(ces)}
    # Initialize the feature vector
    features = np.zeros([N, M])
    # Assign features
    for j, relation in enumerate(relations):
        indices = [index_map[relatum] for relatum in relation.relata]
        # Create the column corresponding to the relation
        relation_features = np.zeros(N)
        # Assign 1s where the cause/effect purview is involved in the relation
        relation_features[indices] = 1
        # Assign the feature column to the feature matrix
        features[:, j] = relation_features
    return features


def get_coords(data, y=None, n_components=3, **params):

    if len(data) <= 3:
        coords = np.zeros((len(data), 2))
        for i in range(len(data)):
            coords[i][0] = i * 0.5
            coords[i][1] = i * 0.5

    else:

        if n_components >= data.shape[0]:
            params["init"] = "random"

        umap = UMAP(
            n_components=n_components,
            metric="euclidean",
            n_neighbors=30,
            min_dist=0.5,
            **params,
        )
        coords = umap.fit_transform(data, y=y)

    return coords


def relation_vertex_indices(features, j):
    """Return the indices of the vertices for relation ``j``."""
    return features[:, j].nonzero()[0]


def all_triangles(N):
    combos = list(itertools.combinations(list(range(N)), 3))
    return (
        [i[0] for i in [list(l) for l in combos]],
        [i[1] for i in [list(l) for l in combos]],
        [i[2] for i in [list(l) for l in combos]],
    )


def all_edges(vertices):
    """Return all edges within a set of vertices."""
    return itertools.combinations(vertices, 2)


def make_label(node_indices, node_labels=None, bold=False, state=False):

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in node_indices]
    else:
        node_labels = node_labels.indices2labels(node_indices)

    if state:
        nl = []
        # capitalizing labels of mechs that are on
        for n, i in zip(node_labels, node_indices):
            if state[i] == 0:
                nl.append(n.lower())
            else:
                nl.append(n.upper())
        node_labels = nl

    return "<b>" + "".join(node_labels) + "</b>" if bold else "".join(node_labels)


def label_mechanism(mice, bold=True, state=False):
    return make_label(
        mice.mechanism, node_labels=mice.node_labels, bold=bold, state=state
    )


def label_purview(mice, state=False):
    return make_label(mice.purview, node_labels=mice.node_labels, state=state)


def label_state(mice):
    return [rel.maximal_state(mice)[0][node] for node in mice.purview]


def label_mechanism_state(subsystem, distinction):
    mechanism_state = [subsystem.state[node] for node in distinction.mechanism]
    return "".join(str(node) for node in mechanism_state)


def label_purview_state(mice):
    return "".join(str(x) for x in label_state(mice))


def label_relation(relation):
    relata = relation.relata

    relata_info = "<br>".join(
        [
            f"{label_mechanism(mice)} / {label_purview(mice)} [{mice.direction.name}]"
            for n, mice in enumerate(relata)
        ]
    )

    relation_info = f"<br>Relation purview: {make_label(relation.purview, relation.subsystem.node_labels)}<br>Relation φ = {phi_round(relation.phi)}<br>"

    return relata_info + relation_info


def hovertext_mechanism(distinction):
    return f"Distinction: {label_mechanism(distinction.cause)}<br>Cause: {label_purview(distinction.cause)}<br>Cause φ = {phi_round(distinction.cause.phi)}<br>Cause state: {[rel.maximal_state(distinction.cause)[0][i] for i in distinction.cause.purview]}<br>Effect: {label_purview(distinction.effect)}<br>Effect φ = {phi_round(distinction.effect.phi)}<br>Effect state: {[rel.maximal_state(distinction.effect)[0][i] for i in distinction.effect.purview]}"


def hovertext_purview(mice):
    return f"Distinction: {label_mechanism(mice)}<br>Direction: {mice.direction.name}<br>Purview: {label_purview(mice)}<br>φ = {phi_round(mice.phi)}<br>State: {[rel.maximal_state(mice)[0][i] for i in mice.purview]}"


def hovertext_relation(relation):
    relata = relation.relata

    relata_info = "".join(
        [
            f"<br>Distinction {n}: {label_mechanism(mice)}<br>Direction: {mice.direction.name}<br>Purview: {label_purview(mice)}<br>φ = {phi_round(mice.phi)}<br>State: {[rel.maximal_state(mice)[0][i] for i in mice.purview]}<br>"
            for n, mice in enumerate(relata)
        ]
    )

    relation_info = f"<br>Relation purview: {make_label(relation.purview, relation.subsystem.node_labels)}<br>Relation φ = {phi_round(relation.phi)}<br>"

    return f"<br>={len(relata)}-Relation=<br>" + relata_info + relation_info


def grounded_position(
    mechanism_indices,
    element_positions,
    jitter=0.0,
    x_offset=0.0,
    y_offset=0.0,
    z_offset=0.0,
):
    x_pos = (
        np.mean([element_positions[x][0] for x in mechanism_indices])
        + np.random.random() * jitter
        + x_offset
    )
    y_pos = (
        np.mean([element_positions[y][1] for y in mechanism_indices])
        + np.random.random() * jitter
        + y_offset
    )
    z_pos = len(mechanism_indices) + np.random.random() * jitter + z_offset
    return [x_pos, y_pos, z_pos]


def normalize_sizes(min_size, max_size, elements):
    phis = np.array([element.phi for element in elements])
    min_phi = phis.min()
    max_phi = phis.max()
    # Add exception in case all purviews have the same phi (e.g. monad case)
    if max_phi == min_phi:
        return [(min_size + max_size) / 2 for x in phis]
    else:
        return min_size + (
            ((phis - min_phi) * (max_size - min_size)) / (max_phi - min_phi)
        )


def phi_round(phi):
    return np.round(phi, 4)


def chunk_list(my_list, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(my_list), n):
        yield my_list[i : i + n]


def format_node(n, subsystem):
    node_format = {
        "label": subsystem.node_labels[n],
        "style": "filled" if subsystem.state[n] == 1 else "",
        "fillcolor": "black" if subsystem.state[n] == 1 else "",
        "fontcolor": "white" if subsystem.state[n] == 1 else "black",
    }
    return node_format


def save_digraph(
    subsystem,
    element_positions,
    digraph_filename="digraph.png",
    plot_digraph=False,
    layout="dot",
):
    # network = subsystem.network
    G = nx.DiGraph()

    for n in range(subsystem.size):
        node_info = format_node(n, subsystem)
        G.add_node(
            node_info["label"],
            style=node_info["style"],
            fillcolor=node_info["fillcolor"],
            fontcolor=node_info["fontcolor"],
            pos=element_positions[n, :],
        )

    edges = [
        [format_node(i, subsystem)["label"] for i in n]
        for n in np.argwhere(subsystem.cm)
    ]

    G.add_edges_from(edges)
    G.graph["node"] = {"shape": "circle"}
    pos = {
        subsystem.node_labels[n]: tuple(element_positions[n, :])
        for n in range(subsystem.size)
    }
    nx.draw(G, pos)
    # A = to_agraph(G)
    # A.layout(layout)
    # A.draw(digraph_filename)
    if plot_digraph:
        return Image(digraph_filename)


def get_edge_color(relation, colorcode_2_relations):
    if colorcode_2_relations:
        purview0 = list(relation.relata.purviews)[0]
        purview1 = list(relation.relata.purviews)[1]
        relation_purview = relation.purview
        # Isotext (mutual full-overlap)
        if purview0 == purview1 == relation_purview:
            return "fuchsia"
        # Sub/Supertext (inclusion / full-overlap)
        elif purview0 != purview1 and (
            all(n in purview1 for n in purview0) or all(n in purview0 for n in purview1)
        ):
            return "indigo"
        # Paratext (connection / partial-overlap)
        elif (purview0 == purview1 != relation_purview) or (
            any(n in purview1 for n in purview0)
            and not all(n in purview1 for n in purview0)
        ):
            return "cyan"
        else:
            raise ValueError(
                "Unexpected relation type, check function to cover all cases"
            )
    else:
        return "teal"


# This seperates cause and effect parts of features of purviews. For example, it separates labels, states, z-coordinates in the
# original combined list. Normally purview lists are in the shape of [featureOfCause1,featureOfEffect1,featureOfCause2...]
# by using this we can separate all those features into two lists, so that we can use them to show cause and effect purviews
# separately in the CES plot.

# WARNING: This doesn't work for coordinates.


def separate_cause_and_effect_purviews_for(given_list):
    causes_in_given_list = []
    effects_in_given_list = []

    for i in range(len(given_list)):
        if i % 2 == 0:
            causes_in_given_list.append(given_list[i])
        else:
            effects_in_given_list.append(given_list[i])

    return causes_in_given_list, effects_in_given_list


# This separates the xy coordinates of purviews.Similar to above the features treated in the above function,
# the coordinates are given in a "[x of c1, y of c1],
# [x of e1,y of e1],..." fashion and  with this function we can separate them to x and y of cause and effect purviews.


def separate_cause_and_effect_for(coords):

    causes_x = []
    effects_x = []
    causes_y = []
    effects_y = []

    for i in range(len(coords)):
        if i % 2 == 0:
            causes_x.append(coords[i][0])
            causes_y.append(coords[i][1])
        else:
            effects_x.append(coords[i][0])
            effects_y.append(coords[i][1])

    return causes_x, causes_y, effects_x, effects_y


def label_to_mechanisms(labels, node_labels):
    mechanisms = []
    for label in labels:
        isString = isinstance(label, str)
        isTuple = isinstance(label, tuple)
        if isString:
            distinction = tuple()
            for letter in label:
                if letter.isdigit():
                    distinction = distinction + ((int(letter)),)
                elif letter == ",":
                    pass
                else:
                    distinction = distinction + (node_labels.index(letter),)
        if isTuple:
            distinction = label
        mechanisms.append(distinction)
    return mechanisms


def is_there_higher_relation(show_intersection_of, higher_relations, node_labels):
    mechanisms = label_to_mechanisms(show_intersection_of, node_labels)
    higher_relation_exists = False
    mechanisms_are_unique = len(set(mechanisms)) == len(mechanisms)
    if mechanisms_are_unique:
        for relation in higher_relations:
            count = 0
            for mechanism in mechanisms:
                if mechanism in relation.mechanisms:
                    count += 1
            if count == len(mechanisms):
                higher_relation_exists = True
    else:
        print(
            "The mechanisms you provided are not unique. Intersection will not be checked. Please provide unique mechanisms."
        )
    print("There is high")
    return higher_relation_exists


def intersection_indices_to_labels(show_intersection_of, node_labels):
    labels = []
    for mechanism in show_intersection_of:
        isString = isinstance(mechanism, str)
        isTuple = isinstance(mechanism, tuple)

        if isString:
            node_list = []
            for node in mechanism:
                if node == ",":
                    pass
                else:
                    node_list.append(int(node))
            node_tuple = tuple(node_list)
            labels.append(make_label(node_tuple, node_labels))
        elif isTuple:
            labels.append(make_label(mechanism, node_labels))
        else:
            print("Please provide integer tuples or strings of mechanisms.")
    return labels


def plot_node_qfolds2D(
    r,
    relation,
    node_indices,
    node_labels,
    go,
    fig,
    show_edges,
    legend_nodes,
    two_relations_coords,
    two_relations_sizes,
    relation_color,
    relation_nodes,
):

    for node in node_indices:
        node_label = make_label([node], node_labels)
        if node in relation_nodes:

            edge_two_relation_trace = go.Scatter3d(
                visible=show_edges,
                legendgroup=f"Node {node_label} q-fold",
                showlegend=True if node not in legend_nodes else False,
                x=two_relations_coords[0][r],
                y=two_relations_coords[1][r],
                z=two_relations_coords[2][r],
                mode="lines",
                name=f"Node {node_label} q-fold",
                line_width=two_relations_sizes[r],
                line_color=relation_color,
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )
            fig.add_trace(edge_two_relation_trace)

            if node not in legend_nodes:

                legend_nodes.append(node)
            return legend_nodes


def plot_node_qfolds3D(
    r,
    relation,
    show_mesh,
    node_indices,
    node_labels,
    go,
    fig,
    legend_nodes,
    legend_mechanisms,
    x,
    y,
    z,
    i,
    j,
    k,
    three_relations_sizes,
    relation_nodes,
):

    for node in node_indices:
        node_label = make_label([node], node_labels)
        if node in relation_nodes:
            triangle_three_relation_trace = go.Mesh3d(
                visible=show_mesh,
                legendgroup=f"Node {node_label} q-fold",
                showlegend=True if node not in legend_nodes else False,
                # x, y, and z are the coordinates of vertices
                x=x,
                y=y,
                z=z,
                # i, j, and k are the vertices of triangles
                i=[i[r]],
                j=[j[r]],
                k=[k[r]],
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity=np.linspace(0, 1, len(x), endpoint=True),
                opacity=three_relations_sizes[r],
                colorscale="viridis",
                showscale=False,
                name=f"Node {node_label} q-fold",
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )
            fig.add_trace(triangle_three_relation_trace)

            if node not in legend_nodes:

                legend_nodes.append(node)
    return legend_nodes


def plot_mechanism_qfolds2D(
    r,
    relation,
    ces,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_mechanisms,
    relation_color,
):
    mechanisms_list = [distinction.mechanism for distinction in ces]
    for mechanism in mechanisms_list:
        mechanism_label = make_label(mechanism, node_labels)
        if mechanism in relation.mechanisms:

            edge_two_relation_trace = go.Scatter3d(
                visible=show_edges,
                legendgroup=f"Mechanism {mechanism_label} q-fold",
                showlegend=True if mechanism_label not in legend_mechanisms else False,
                x=two_relations_coords[0][r],
                y=two_relations_coords[1][r],
                z=two_relations_coords[2][r],
                mode="lines",
                name=f"Mechanism {mechanism_label} q-fold",
                line_width=two_relations_sizes[r],
                line_color=relation_color,
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )

            fig.add_trace(edge_two_relation_trace)

            if mechanism_label not in legend_mechanisms:

                legend_mechanisms.append(mechanism_label)

            return legend_mechanisms

def plot_selected_mechanism_qfolds2D(
    r,
    relation,
    mechanisms_list,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_mechanisms,
    relation_color,
):
    for mechanism in mechanisms_list:
        mechanism_label = make_label(mechanism, node_labels)
        if mechanism in relation.mechanisms:

            edge_two_relation_trace = go.Scatter3d(
                visible=True,
                legendgroup=f"Mechanism {mechanism_label} q-fold",
                showlegend=True if mechanism_label not in legend_mechanisms else False,
                x=two_relations_coords[0][r],
                y=two_relations_coords[1][r],
                z=two_relations_coords[2][r],
                mode="lines",
                name=f"Mechanism {mechanism_label} q-fold",
                line_width=two_relations_sizes[r],
                line_color=relation_color,
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )

            fig.add_trace(edge_two_relation_trace)

            if mechanism_label not in legend_mechanisms:

                legend_mechanisms.append(mechanism_label)

            return legend_mechanisms            


def plot_mechanism_qfolds3D(
    r,
    relation,
    ces,
    show_mesh,
    node_labels,
    go,
    fig,
    legend_mechanisms,
    x,
    y,
    z,
    i,
    j,
    k,
    three_relations_sizes,
):

    mechanisms_list = [distinction.mechanism for distinction in ces]
    for mechanism in mechanisms_list:
        mechanism_label = make_label(mechanism, node_labels)
        if mechanism in relation.mechanisms:
            triangle_three_relation_trace = go.Mesh3d(
                visible=show_mesh,
                legendgroup=f"Mechanism {mechanism_label} q-fold",
                showlegend=True if mechanism_label not in legend_mechanisms else False,
                # x, y, and z are the coordinates of vertices
                x=x,
                y=y,
                z=z,
                # i, j, and k are the vertices of triangles
                i=[i[r]],
                j=[j[r]],
                k=[k[r]],
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity=np.linspace(0, 1, len(x), endpoint=True),
                opacity=three_relations_sizes[r],
                colorscale="viridis",
                showscale=False,
                name=f"Mechanism {mechanism_label} q-fold",
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )
            fig.add_trace(triangle_three_relation_trace)
            if mechanism_label not in legend_mechanisms:
                legend_mechanisms.append(mechanism_label)
            return legend_mechanisms

def plot_selected_mechanism_qfolds3D(
    r,
    relation,
    mechanisms_list,
    show_mesh,
    node_labels,
    go,
    fig,
    legend_mechanisms,
    x,
    y,
    z,
    i,
    j,
    k,
    three_relations_sizes,
):

    for mechanism in mechanisms_list:
        mechanism_label = make_label(mechanism, node_labels)
        if mechanism in relation.mechanisms:
            triangle_three_relation_trace = go.Mesh3d(
                visible=True,
                legendgroup=f"Selected Mechanism {mechanism_label} q-fold",
                showlegend=True if mechanism_label not in legend_mechanisms or legend_mechanisms is None else False,
                # x, y, and z are the coordinates of vertices
                x=x,
                y=y,
                z=z,
                # i, j, and k are the vertices of triangles
                i=[i[r]],
                j=[j[r]],
                k=[k[r]],
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity=np.linspace(0, 1, len(x), endpoint=True),
                opacity=three_relations_sizes[r],
                colorscale="viridis",
                showscale=False,
                name=f"Mechanism {mechanism_label} q-fold",
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )
            fig.add_trace(triangle_three_relation_trace)
            if mechanism_label not in legend_mechanisms:
                legend_mechanisms.append(mechanism_label)
            return legend_mechanisms

def plot_relation_purview_qfolds2D(
    r,
    relation,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_relation_purviews,
    relation_color,
):
    purview = relation.purview
    purview_label = make_label(purview, node_labels)

    edge_relation_purview_two_relation_trace = go.Scatter3d(
        visible=show_edges,
        legendgroup=f"Relation Purview {purview_label} q-fold",
        showlegend=True if purview_label not in legend_relation_purviews else False,
        x=two_relations_coords[0][r],
        y=two_relations_coords[1][r],
        z=two_relations_coords[2][r],
        mode="lines",
        name=f"Relation Purview {purview_label} q-fold",
        line_width=two_relations_sizes[r],
        line_color=relation_color,
        hoverinfo="text",
        hovertext=hovertext_relation(relation),
    )

    fig.add_trace(edge_relation_purview_two_relation_trace)

    if purview_label not in legend_relation_purviews:
        legend_relation_purviews.append(purview_label)

    return legend_relation_purviews


def plot_relation_purview_qfolds3D(
    r,
    relation,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_relation_purviews,
    relation_color,
    x,
    y,
    z,
    i,
    j,
    k,
    three_relations_sizes,
):

    purview = relation.purview
    purview_label = make_label(purview, node_labels)

    relation_purview_three_relation_trace = go.Mesh3d(
        visible=show_edges,
        legendgroup=f"Relation Purview {purview_label} q-fold",
        showlegend=True if purview_label not in legend_relation_purviews else False,
        x=x,
        y=y,
        z=z,
        i=[i[r]],
        j=[j[r]],
        k=[k[r]],
        intensity=np.linspace(0, 1, len(x), endpoint=True),
        opacity=three_relations_sizes[r],
        colorscale="viridis",
        showscale=False,
        name=f"Relation Purview {purview_label} q-fold",
        hoverinfo="text",
        hovertext=hovertext_relation(relation),
    )

    fig.add_trace(relation_purview_three_relation_trace)

    if purview_label not in legend_relation_purviews:
        legend_relation_purviews.append(purview_label)

    return legend_relation_purviews


def plot_per_mechanism_purview_qfolds2D(
    r,
    relation,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_mechanism_purviews,
    relation_color,
):
    for relatum in relation.relata:
        purview = relatum.purview
        mechanism = relatum.mechanism
        direction = str(relatum.direction)
        purview_label = make_label(purview, node_labels)
        mechanism_label = make_label(mechanism, node_labels)
        mechanism_purview_label = (
            f"Mechanism {mechanism_label} {direction} Purview {purview_label} q-fold"
        )

        edge_purviews_with_mechanisms_two_relation_trace = go.Scatter3d(
            visible=show_edges,
            legendgroup=mechanism_purview_label,
            showlegend=True
            if mechanism_purview_label not in legend_mechanism_purviews
            else False,
            x=two_relations_coords[0][r],
            y=two_relations_coords[1][r],
            z=two_relations_coords[2][r],
            mode="lines",
            name=mechanism_purview_label,
            line_width=two_relations_sizes[r],
            line_color=relation_color,
            hoverinfo="text",
            hovertext=hovertext_relation(relation),
        )

        fig.add_trace(edge_purviews_with_mechanisms_two_relation_trace)

        if mechanism_purview_label not in legend_mechanism_purviews:
            legend_mechanism_purviews.append(mechanism_purview_label)

    return legend_mechanism_purviews


def plot_per_mechanism_purview_qfolds3D(
    r,
    relation,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_mechanism_purviews,
    relation_color,
    x,
    y,
    z,
    i,
    j,
    k,
    three_relations_sizes,
):

    for relatum in relation.relata:

        purview = relatum.purview
        mechanism = relatum.mechanism
        direction = str(relatum.direction)
        purview_label = make_label(purview, node_labels)
        mechanism_label = make_label(mechanism, node_labels)
        mechanism_purview_label = (
            f"Mechanism {mechanism_label} {direction} Purview {purview_label} q-fold"
        )

        purviews_with_mechanisms_three_relation_trace = go.Mesh3d(
            visible=show_edges,
            legendgroup=mechanism_purview_label,
            showlegend=True
            if mechanism_purview_label not in legend_mechanism_purviews
            else False,
            x=x,
            y=y,
            z=z,
            i=[i[r]],
            j=[j[r]],
            k=[k[r]],
            intensity=np.linspace(0, 1, len(x), endpoint=True),
            opacity=three_relations_sizes[r],
            colorscale="viridis",
            showscale=False,
            name=mechanism_purview_label,
            hoverinfo="text",
            hovertext=hovertext_relation(relation),
        )

        fig.add_trace(purviews_with_mechanisms_three_relation_trace)

        if mechanism_purview_label not in legend_mechanism_purviews:
            legend_mechanism_purviews.append(mechanism_purview_label)

    return legend_mechanism_purviews


def plot_compound_purview_qfolds2D(
    r,
    relation,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_compound_purviews,
    relation_color,
):

    purviews = list(relation.relata.purviews)

    for purview in purviews:

        purview_label = make_label(purview, node_labels)
        edge_compound_purview_two_relation_trace = go.Scatter3d(
            visible=show_edges,
            legendgroup=f"Compound Purview {purview_label} q-fold",
            showlegend=True if purview_label not in legend_compound_purviews else False,
            x=two_relations_coords[0][r],
            y=two_relations_coords[1][r],
            z=two_relations_coords[2][r],
            mode="lines",
            name=f"Compound Purview {purview_label} q-fold",
            line_width=two_relations_sizes[r],
            line_color=relation_color,
            hoverinfo="text",
            hovertext=hovertext_relation(relation),
        )

        fig.add_trace(edge_compound_purview_two_relation_trace)

        if purview_label not in legend_compound_purviews:
            legend_compound_purviews.append(purview_label)

        return legend_compound_purviews


def plot_compound_purview_qfolds3D(
    r,
    relation,
    show_edges,
    node_labels,
    go,
    fig,
    two_relations_coords,
    two_relations_sizes,
    legend_compound_purviews,
    relation_color,
    x,
    y,
    z,
    i,
    j,
    k,
    three_relations_sizes,
):

    purviews = list(relation.relata.purviews)

    for purview in purviews:

        purview_label = make_label(purview, node_labels)
        compound_purview_three_relation_trace = go.Mesh3d(
            visible=show_edges,
            legendgroup=f"Compound Purview {purview_label} q-fold",
            showlegend=True if purview_label not in legend_compound_purviews else False,
            x=x,
            y=y,
            z=z,
            i=[i[r]],
            j=[j[r]],
            k=[k[r]],
            intensity=np.linspace(0, 1, len(x), endpoint=True),
            opacity=three_relations_sizes[r],
            colorscale="viridis",
            showscale=False,
            name=f"Compound Purview {purview_label} q-fold",
            hoverinfo="text",
            hovertext=hovertext_relation(relation),
        )

        fig.add_trace(compound_purview_three_relation_trace)

        if purview_label not in legend_compound_purviews:
            legend_compound_purviews.append(purview_label)
        return legend_compound_purviews


from scipy.special import comb
import math


def grounded_position(
    mechanism_indices,
    element_positions,
    jitter=0.0,
    x_offset=0.0,
    y_offset=0.0,
    z_offset=0.0,
    floor_spacing=1,
    floor_scale=1,
    x=False,
    y=False,
    z=False,
    restrict_ground_floor=True,
):

    from scipy.special import comb

    N = len(element_positions)
    c = np.mean(element_positions, axis=0)
    n = len(mechanism_indices)
    factor = comb(N, n) * floor_scale if n > 1 and restrict_ground_floor else 0

    if not x:
        x_pos = (
            np.mean([element_positions[x, 0] for x in mechanism_indices])
            + (np.random.random() - 1 / 2) * jitter
            + x_offset
        )
        x_pos += (x_pos - c[0]) * np.abs(factor - 1)
    else:
        x_pos = x

    if not y:
        y_pos = (
            np.mean([element_positions[y, 1] for y in mechanism_indices])
            + (np.random.random() - 1 / 2) * jitter
            + y_offset
        )
        y_pos += (y_pos - c[1]) * np.abs(factor - 1)
    else:
        y_pos = y

    if not z:
        z_pos = n * floor_spacing + (np.random.random() - 1 / 2) * jitter + z_offset
    else:
        z_pos = z

    return [
        x_pos,
        y_pos,
        z_pos,
    ]


def regular_polygon(n, center=(0, 0), angle=0, z=0, radius=None, scale=1):
    if radius == None:
        radius = n / (2 * math.pi)

    radius = radius * scale

    if n == 1:
        return [[center[0], center[1], z]]
    else:
        angle -= math.pi / n
        coord_list = [
            [
                center[0] + radius * math.sin((2 * math.pi / n) * i - angle),
                center[1] + radius * math.cos((2 * math.pi / n) * i - angle),
                z,
            ]
            for i in range(n)
        ]
        return coord_list


def get_mechanism_state(mechanism, subsystem):
    return tuple(subsystem.state[i] for i in mechanism)

def get_mechanism_label_text_color(mechanism, subsystem):
    state = get_mechanism_state(mechanism, subsystem)
    return ['white' if s==1 else 'black' for s in state]

def get_mechanism_label_bg_color(mechanism, subsystem):
    state = get_mechanism_state(mechanism, subsystem)
    return ['black' if s==1 else 'white' for s in state]    

def get_purview_state(mice):
    return tuple([rel.maximal_state(mice)[0][n] for n in mice.purview])

def get_purview_label_text_color(mice, composition=False):
        direction = mice.direction.value
        state = get_purview_state(mice)
        if composition:
            if direction==0:
                return ['red' for s in state]
            else:
                return ['green' for s in state]    
        else:
            if direction==0:
                return ['white' if s==1 else 'red' for s in state]
            else:
                return ['white' if s==1 else 'green' for s in state]    

def get_purview_label_bg_color(mice):
        direction = mice.direction.value
        state = get_purview_state(mice)
        if direction==0:
            return ['red' if s==1 else 'white' for s in state]
        else:
            return ['green' if s==1 else 'white' for s in state]    

def get_purview_label_border_color(mice):
        direction = mice.direction.value
        if direction==0:
            return 'red'
        else:
            return 'green'


def relative_phi(ces,mini=0.1,maxi=1):
    max_phi = max(c.phi for c in ces)
    return [(maxi-mini)*c.phi/max_phi+mini for c in ces]


def plot_ces_epicycles(
    subsystem,
    ces,
    relations,
    network=None,
    title="",
    max_order=3,
    purview_x_offset=0.1,
    mechanism_z_offset=0,
    vertex_size_range=(10, 30),
    edge_size_range=(0.5, 3),
    surface_size_range=(0.001, 0.05),
    plot_dimentions=None,
    mechanism_labels_size=15,
    mechanism_label_position='top center',
    purview_label_position='middle left',
    mechanism_state_labels_size=12,
    labels_z_offset=0,
    states_z_offset=0.15,
    purview_labels_size=15,
    purview_state_labels_size=10,
    show_mechanism_labels=True,
    show_links=True,
    show_mechanism_state_labels=False,
    show_purview_labels=True,
    show_purview_state_labels=False,
    show_vertices_mechanisms=False,
    show_vertices_purviews=False,
    show_edges=True,
    show_mesh=True,
    show_node_qfolds=False,
    show_mechanism_qfolds=False,
    show_compound_purview_qfolds=False,
    show_relation_purview_qfolds=False,
    show_per_mechanism_purview_qfolds=False,
    show_grid=False,
    network_name="",
    eye_coordinates=(0.7, -0.6, -0.2),
    hovermode="x",
    digraph_filename="digraph.png",
    digraph_layout=None,
    save_plot_to_html=True,
    show_causal_model=False,
    order_on_z_axis=False,
    save_coords=False,
    link_width=1.5,
    link_width_range=(5, 10),
    colorcode_2_relations=True,
    left_margin=get_screen_size()[0] / 10,
    floor_center=(0, 0),
    floor_scale=1,
    floor_scales=None,
    floor_angles=None,
    ground_floor_height=0,
    epicycle_radius=0.4,
    base_center=(0, 0),
    base_scale=.5,
    base_floor_height=2,
    base_z_offset=-2/2.5,
    base_opacity=1,
    base_color='white',
    show_mechanism_base=False,
    base_intensity=.5,
    mechanism_label_bold=False,
    state_as_lettercase=True,
    mechanisms_as_annotations=False,
    purviews_as_annotations=False,
    annotation_z_spacing=0.175,
    annotation_z_spacing_mechanisms=.08,
    annotation_x_spacing=0,
    annotation_y_spacing=0,
    show_chains=True,
    show_chains_mesh=True,
    chain_width=3,
    chain_color="black",
    chain_dash = 'dash',
    annotation_alpha_from_mechanism_phi=False,
    annotation_alpha_from_purview_phi=False,
    annotations_alpha_mechanism_label=.8,
    annotations_alpha_purview_label=.8,
    intersect_mechanisms=None,
    paper_bgcolor='white',
    plot_bgcolor='white',
    composition=False,
    composition_color='black',
    composition_edge_size=.5,
    composition_link_width=3,
    composition_surface_opacity=.05,
    composition_surface_intensity=.05,
    integration_cut_elements=None,
    integration_color='gray',
    composition_text_color="#727272",
    autosize=True,
    image_center = (0,0),
    image_z_offset = 0,
    image_xy_scale = 1,
    image_downsample = 10,
    image_opacity = 0.9,
    image_file = 'brain.png',
    show_image = False,
    selected_mechanism_qfolds=None,
    img_background=False,
    distinctions_lost=None,
    relations_lost=None,
    
    distinctions_lost_mechanism_color='blue',
    distinctions_lost_mechanism_hoverlabel_color='blue',
    distinctions_lost_link_color='blue',
    relations_lost_edge_color='blue',
    relations_lost_surface_colorscale='blues',
    
    distinctions_new_mechanism_color='orange',
    distinctions_new_mechanism_hoverlabel_color='orange',
    distinctions_new_link_color='orange',
    relations_new_edge_color='orange',
    relations_new_surface_colorscale='oranges',
    
    show_distinctions_remained_mechanisms=False,
    distinctions_remained_mechanism_color='gray',
    distinctions_remained_mechanism_hoverlabel_color='gray',
    distinctions_remained_link_color='gray',    
    relations_remained_edge_color='gray',
    relations_remained_surface_color='greys',

    
):
   
    # if intersect_mechanisms or selected_mechanism_qfolds or distinctions_lost or relations_lost:
    #     show_chains='legendonly'
    #     show_chains_mesh='legendonly'
    #     show_links='legendonly'
    #     show_edges='legendonly'
    #     show_mesh='legendonly'

    # Select only relations <= max_order
    relations = list(filter(lambda r: len(r.relata) <= max_order, relations))

    # Separate CES into causes and effects
    separated_ces = rel.separate_ces(ces)

    # Initialize figure
    fig = go.Figure()

    # computing epicycles
    N = len(subsystem)

    # Things for the new function
    purviews = [c.purview for c in separated_ces]
    mechanisms = [c.mechanism for c in separated_ces[::2]]


    # generate floors
    floors = [
        np.array(
            regular_polygon(
                int(comb(N, k+1)),
                center=floor_center,
                angle=floor_angles[k] if floor_angles else 0,
                z=k + ground_floor_height,
                scale=floor_scales[k] if floor_scales else floor_scale,
            )
        )
        for k in range( N )
    ]
    floor_vertices = np.concatenate([f for f in floors])

    # getting a list of all possible purviews
    all_purviews = list(powerset(subsystem.node_indices, nonempty=True))

    # find number of times each purview appears
    vertex_purview = {
        p: fv for p, fv in zip(all_purviews, floor_vertices) if purviews.count(p) > 0
    }

    # Create epicycles
    num_purviews = [
        purviews.count(p) if purviews.count(p) > 0 else 0 for p in all_purviews
    ]
    epicycles = [
        regular_polygon(n, center=(e[0], e[1]), z=e[2], radius=epicycle_radius)
        for e, n in zip(floor_vertices, num_purviews)
        if n > 0
    ]

    # associating each purview with vertices in a regular polygon around the correct floor vertex
    purview_positions = [
        {v: e, "N": 0} for v, e in zip(vertex_purview.keys(), epicycles)
    ]

    # placing purview coordinates in the correct order
    purview_vertex_coordinates = []
    for p in purviews:
        for pp in purview_positions:
            if p in pp.keys():
                purview_vertex_coordinates.append(pp[p][pp["N"]])
                pp["N"] += 1

    coords = np.array(purview_vertex_coordinates)

    # Construct base
    base = [
        np.array(
            regular_polygon(
                int(comb(N, k)),
                center=base_center,
                z=((k / N) * base_floor_height) + base_z_offset,
                scale=base_scale,
            )
        )
        for k in range(1, N + 1)
    ]


    base_vertices = np.concatenate([f for f in base])
    i = 0
    base_coords = []
    for m, c, i in zip(all_purviews, base_vertices, range(len(all_purviews))):
        if m in mechanisms:
            base_coords.append(list(base_vertices[i]))

    xm = [p[0] for p in base_coords]
    ym = [p[1] for p in base_coords]
    zm = [p[2] for p in base_coords]



    # Dimensionality reduction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if not grounded:
    # Create the features for each cause/effect based on their relations
    features = feature_matrix(separated_ces, relations)

    # Now we get one set of coordinates for the CES; these will then be offset to
    # get coordinates for causes and effects separately, so that causes/effects
    # are always near each other in the embedding.

    # Collapse rows of cause/effect belonging to the same distinction
    # NOTE: This depends on the implementation of `separate_ces`; causes and
    #       effects are assumed to be adjacent in the returned list

    # Purviews
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract vertex indices for plotly
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    causes_x, causes_y, effects_x, effects_y = separate_cause_and_effect_for(coords)

    coords = np.stack((x, y, z), axis=-1)

    if save_coords:
        with open("coords.pkl", "wb") as f:
            pickle.dump(coords, f)

    # This separates z-coordinates of cause and effect purviews
    causes_z, effects_z = separate_cause_and_effect_purviews_for(z)

    # Get node labels and indices for future use:
    node_labels = subsystem.node_labels
    node_indices = subsystem.node_indices

    # Get mechanism and purview labels (Quickly!)
    # mechanism_labels = list(map(label_mechanism, ces))
    mechanism_labels = [
        label_mechanism(mice, bold=mechanism_label_bold, state=subsystem.state)
        for mice in separated_ces[::2]
    ]
    mechanism_state_labels = [
        label_mechanism_state(subsystem, distinction) for distinction in ces
    ]

    # Get indices and labels of intersected mechanisms (if specified)
    if intersect_mechanisms:
        intersected_mechanisms_indices = [i for i in range(len(ces)) if ces[i].mechanism in intersect_mechanisms]
        intersected_mechanisms_labels = [make_label(mechanism, node_labels=subsystem.node_labels, bold=False, state=False) for mechanism in intersect_mechanisms]

# Get indices and labels of selected mechanisms q-fold (if specified)
    if selected_mechanism_qfolds:
        selected_mechanisms_indices = [i for i in range(len(ces)) if ces[i].mechanism in selected_mechanism_qfolds]
        selected_mechanisms_labels = [make_label(mechanism, node_labels=subsystem.node_labels, bold=False, state=False) for mechanism in selected_mechanism_qfolds]        
        selected_qfold_relations = [r for r in relations if any([m in r.mechanisms for m in selected_mechanism_qfolds])]

        selected_qfold_distinctions = []
        for r in selected_qfold_relations:
            selected_qfold_distinctions.extend(r.mechanisms)
        selected_qfold_distinctions = sorted(sorted(list(set(selected_qfold_distinctions))),key=len)

        # selected_qfold_purviews = [mice.purview if mice.mechanism in selected_qfold_distinctions else None for mice in rel.separate_ces(ces)]        
        selected_qfold_mices = list(set(flatten([[mice for relation in selected_qfold_relations if mice in relation.relata] for mice in separated_ces])))
        # selected_qfold_causes = selected_qfold_purviews[::2]
        # selected_qfold_effects = selected_qfold_purviews[1::2]
    # purview_labels = list(map(label_purview, separated_ces))

    if distinctions_lost and relations_lost:
        distinctions_lost_indices = [i for i in range(len(ces)) if ces[i] in distinctions_lost]
        distinctions_lost_mechanisms = [d.mechanism for d in distinctions_lost]
        distinctions_lost_labels = [make_label(mechanism, node_labels=subsystem.node_labels, bold=False, state=False) for mechanism in distinctions_lost_mechanisms]        
        distinctions_lost_mices = flatten([[d.cause,d.effect] for d in distinctions_lost])
        
        distinctions_remained = [d for d in ces if d not in distinctions_lost]
        distinctions_remained_indices = [i for i in range(len(ces)) if ces[i] in distinctions_remained]
        distinctions_remained_mechanisms = [d.mechanism for d in distinctions_remained]
        distinctions_remained_labels = [make_label(mechanism, node_labels=subsystem.node_labels, bold=False, state=False) for mechanism in distinctions_remained_mechanisms]        
        distinctions_remained_mices = flatten([[d.cause,d.effect] for d in distinctions_remained])

    purview_labels = [
        label_purview(mice, state=list(rel.maximal_state(mice)[0]))
        for mice in separated_ces
    ]

    purview_state_labels = list(map(label_purview_state, separated_ces))

    (
        cause_purview_labels,
        effect_purview_labels,
    ) = separate_cause_and_effect_purviews_for(purview_labels)
    (
        cause_purview_state_labels,
        effect_purview_state_labels,
    ) = separate_cause_and_effect_purviews_for(purview_state_labels)

    mechanism_hovertext = list(map(hovertext_mechanism, ces))
    vertices_hovertext = list(map(hovertext_purview, separated_ces))
    causes_hovertext, effects_hovertext = separate_cause_and_effect_purviews_for(
        vertices_hovertext
    )

    # Make selected mechanism labels trace
    if selected_mechanism_qfolds:        
        selected_mechanism_qfold_text=[mechanism_labels[i] if mechanisms[i] in selected_qfold_distinctions else '' for i in range(len(mechanisms))]
        labels_mechanisms_trace = go.Scatter3d(
            visible= 'legendonly' if mechanisms_as_annotations or intersect_mechanisms else show_mechanism_labels,
            x=xm,
            y=ym,
            z=[n + labels_z_offset + mechanism_z_offset for n in zm],
            mode="text",
            text=selected_mechanism_qfold_text if selected_mechanism_qfolds else mechanism_labels,
            name="Mechanism Labels",
            showlegend=True,
            textfont=dict(size=mechanism_labels_size, color="black"),
            textposition=mechanism_label_position,
            hoverinfo="text",
            hovertext=mechanism_hovertext,
            hoverlabel=dict(bgcolor="black", font_color="white"),
        )
        fig.add_trace(labels_mechanisms_trace)

    # Make lost mechanism labels trace
    elif distinctions_lost and relations_lost:
        distinctions_lost_text=[mechanism_labels[i] if mechanisms[i] in distinctions_lost_mechanisms else '' for i in range(len(mechanisms))]        
        labels_mechanisms_trace = go.Scatter3d(
            visible=show_mechanism_labels,
            x=xm,
            y=ym,
            z=[n + labels_z_offset + mechanism_z_offset for n in zm],
            mode="text",
            text=distinctions_lost_text,
            name="Lost Distinctions",
            showlegend=True,
            textfont=dict(size=mechanism_labels_size, color=distinctions_lost_mechanism_color),
            textposition=mechanism_label_position,
            hoverinfo="text",
            hovertext=mechanism_hovertext,
            hoverlabel=dict(bgcolor=distinctions_lost_mechanism_hoverlabel_color, font_color="white"),
        )
        fig.add_trace(labels_mechanisms_trace)

        distinctions_remained_text=[mechanism_labels[i] if mechanisms[i] in distinctions_remained_mechanisms else '' for i in range(len(mechanisms))]        
        if show_distinctions_remained_mechanisms:
            labels_mechanisms_trace = go.Scatter3d(
                visible=show_mechanism_labels,
                x=xm,
                y=ym,
                z=[n + labels_z_offset + mechanism_z_offset for n in zm],
                mode="text",
                text=distinctions_remained_text,
                name="Remained Distinctions",
                showlegend=True,
                textfont=dict(size=mechanism_labels_size, color=distinctions_remained_mechanism_color),
                textposition=mechanism_label_position,
                hoverinfo="text",
                hovertext=mechanism_hovertext,
                hoverlabel=dict(bgcolor=distinctions_remained_mechanism_hoverlabel_color, font_color="white"),
            )
            fig.add_trace(labels_mechanisms_trace)

    #Make intersected mechanisms labels trace
    elif intersect_mechanisms:
        intersected_labels_mechanisms_trace = go.Scatter3d(
            visible= show_mechanism_labels if not mechanisms_as_annotations else False,
            x=[xm[i] for i in intersected_mechanisms_indices],
            y=[ym[i] for i in intersected_mechanisms_indices],
            z=[zm[i] + labels_z_offset + mechanism_z_offset for i in intersected_mechanisms_indices],
            mode="text",
            text=[mechanism_labels[i] for i in intersected_mechanisms_indices],
            name="Intersected Mechanism Labels",
            showlegend=True,
            textfont=dict(size=mechanism_labels_size, color="black"),
            textposition=mechanism_label_position,
            hoverinfo="text",
            hovertext=[mechanism_hovertext[i] for i in intersected_mechanisms_indices],
            hoverlabel=dict(bgcolor="black", font_color="white"),
        )
        fig.add_trace(intersected_labels_mechanisms_trace)   

    #Make mechanisms labels trace
    else:
        labels_mechanisms_trace = go.Scatter3d(
            visible=show_mechanism_labels,
            x=xm,
            y=ym,
            z=[n + labels_z_offset + mechanism_z_offset for n in zm],
            mode="text",
            text=mechanism_labels,
            name="Mechanism Labels",
            showlegend=True,
            textfont=dict(size=mechanism_labels_size, color="black"),
            textposition=mechanism_label_position,
            hoverinfo="text",
            hovertext=mechanism_hovertext,
            hoverlabel=dict(bgcolor="black", font_color="white"),
        )
        fig.add_trace(labels_mechanisms_trace)

    # Make mechanism base
    i_base, j_base, k_base = all_triangles(len(xm))
    mechanism_base_trace = go.Mesh3d(
        x=xm,
        y=ym,
        z=zm,
        visible=show_mechanism_base,
        legendgroup="Mechanism base",
        showlegend=True,
        opacity=base_opacity,
        colorscale=[base_color for x in xm],
        intensity=[base_intensity] * len(i_base),
        i=i_base,
        j=j_base,
        k=k_base,
        name="Mechanism base",
        showscale=False,
    )
    fig.add_trace(mechanism_base_trace)

    # Make mechanism chains
    first_order_mechanisms = list(filter(lambda m: len(m) == 1, mechanisms))

    chained_mechanisms = [] 
    chain_counter=0
    for m1,mech1 in enumerate(mechanisms):
        for m2,mech2 in enumerate(first_order_mechanisms):
            if mech2[0] in mech1:
                chained_mechanisms.append((chain_counter,(m1,m2)))
                chain_counter+=1
    
    
    chains_xs = [(xm[c[0]],xm[c[1]]) for i,c in chained_mechanisms]
    chains_ys = [(ym[c[0]],ym[c[1]]) for i,c in chained_mechanisms]
    chains_zs = [(zm[c[0]],zm[c[1]]) for i,c in chained_mechanisms]

    if show_chains:

        if intersect_mechanisms:
            for m,mechanism in chained_mechanisms:
                if mechanism in intersect_mechanisms:
                    chains_trace = go.Scatter3d(
                        visible=show_chains,
                        legendgroup="Chains",
                        showlegend=True if m == 0 else False,
                        x=chains_xs[m],
                        y=chains_ys[m],
                        z=chains_zs[m],
                        mode="lines",
                        name="Chains",
                        line={'dash': chain_dash, 'color':chain_color,'width':chain_width},
                        hoverinfo="skip",
                        )
                    fig.add_trace(chains_trace)
        # elif distinctions_lost and relations_lost:
        #     for m,mechanism in chained_mechanisms:
        #         if mechanism in distinctions_lost_mechanisms:
        #             chains_trace = go.Scatter3d(
        #                 visible=True,
        #                 legendgroup="Chains",
        #                 showlegend=True,# if m == 0 else False,
        #                 x=chains_xs[m],
        #                 y=chains_ys[m],
        #                 z=chains_zs[m],
        #                 mode="lines",
        #                 name="Chains",
        #                 line={'dash': chain_dash, 'color':chain_color,'width':chain_width},
        #                 hoverinfo="skip",
        #                 )
        #             fig.add_trace(chains_trace)                    
        else:
            for m,mechanism in chained_mechanisms:

                chains_trace = go.Scatter3d(
                    visible=show_chains,
                    legendgroup="Chains",
                    showlegend=True if m == 0 else False,
                    x=chains_xs[m],
                    y=chains_ys[m],
                    z=chains_zs[m],
                    mode='lines',
                    name="Chains",
                    line={'dash': chain_dash, 'color':chain_color,'width':chain_width},
                    hoverinfo="skip",
                    )
                fig.add_trace(chains_trace)
    
    if show_chains_mesh:
        chained_mechanisms_pairs = [chain[1] for chain in chained_mechanisms]

        chained_mechanisms_triplets = []
        ss = []
        for a, b in itertools.combinations(chained_mechanisms_pairs, 2):
            s = set(a).union(b)
            if len(s) == 3:
                chained_mechanisms_triplets.append(tuple(sorted(s)))

        chained_mechanisms_triplets = sorted(list(set(chained_mechanisms_triplets)))

        chained_mechanisms_triangles = np.array([triplet for triplet in chained_mechanisms_triplets if len(triplet)==3 and len(list(filter(lambda mechanism_index: mechanism_index >len(first_order_mechanisms)-1, triplet)))==1])

        chains_mesh = go.Mesh3d(
                    visible=show_chains_mesh,
                    legendgroup="Chains mesh",
                    showlegend=True,
                    x=xm,
                    y=ym,
                    z=zm,
                    i=chained_mechanisms_triangles[:,0],
                    j=chained_mechanisms_triangles[:,1],
                    k=chained_mechanisms_triangles[:,2],
                    name="Chains mesh",
                    intensity=[base_intensity for x in xm],
                    opacity=base_opacity,
                    colorscale=[base_color for x in xm],
                    showscale=False,
                    )
        fig.add_trace(chains_mesh)

    # Make mechanism state labels trace
    if show_mechanism_state_labels:
        labels_mechanisms_state_trace = go.Scatter3d(
            visible=show_mechanism_state_labels,
            x=xm,
            y=ym,
            z=[
                n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset + states_z_offset)
                for n in zm
            ],
            mode="text",
            text=mechanism_state_labels,
            name="Mechanism State Labels",
            showlegend=True,
            textfont=dict(size=mechanism_state_labels_size, color="black"),
            hoverinfo="text",
            hovertext=mechanism_hovertext,
            hoverlabel=dict(bgcolor="black", font_color="white"),
        )
        fig.add_trace(labels_mechanisms_state_trace)

    # Compute purview and mechanism marker sizes
    purview_sizes = normalize_sizes(
        vertex_size_range[0], vertex_size_range[1], separated_ces
    )

    cause_purview_sizes, effect_purview_sizes = separate_cause_and_effect_purviews_for(
        purview_sizes
    )

    mechanism_sizes = [min(phis) for phis in chunk_list(purview_sizes, 2)]
    # Make mechanisms trace
    vertices_mechanisms_trace = go.Scatter3d(
        visible=show_vertices_mechanisms,
        x=xm,
        y=ym,
        z=zm,
        mode="markers",
        name="Mechanisms",
        text=mechanism_labels,
        showlegend=True,
        marker=dict(size=mechanism_sizes, color="black"),
        hoverinfo="text",
        hovertext=mechanism_hovertext,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(vertices_mechanisms_trace)

    # Make cause purview labels trace
    if selected_mechanism_qfolds:
        selected_qfold_causes_indices = [i for i,purview in enumerate(separated_ces[::2]) if purview in selected_qfold_mices]        
    
        selected_mechanism_labels_cause_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=[causes_x[i] for i in selected_qfold_causes_indices],
            y=[causes_y[i] for i in selected_qfold_causes_indices],
            z=[causes_z[i] + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for i in selected_qfold_causes_indices],
            mode="text",
            text=[cause_purview_labels[i] for i in selected_qfold_causes_indices],
            textposition=purview_label_position,
            name="Selected Mechanisms Cause Purview Labels",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="red"),
            hoverinfo="text",
            hovertext=causes_hovertext,
            hoverlabel=dict(bgcolor="red"),
        )
        fig.add_trace(selected_mechanism_labels_cause_purviews_trace)

    elif intersect_mechanisms:
        intersection_labels_cause_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=[causes_x[i] for i in intersected_mechanisms_indices],
            y=[causes_y[i] for i in intersected_mechanisms_indices],
            z=[causes_z[i] + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for i in intersected_mechanisms_indices],
            mode="text",
            text=[cause_purview_labels[i] for i in intersected_mechanisms_indices],
            textposition=purview_label_position,
            name="Intersection Cause Purview Labels",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="red"),
            hoverinfo="text",
            hovertext=[causes_hovertext[i] for i in intersected_mechanisms_indices],
            hoverlabel=dict(bgcolor="red"),
        )
        fig.add_trace(intersection_labels_cause_purviews_trace)

    elif distinctions_lost and relations_lost:
        lost_labels_cause_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=[causes_x[i] for i in distinctions_lost_indices],
            y=[causes_y[i] for i in distinctions_lost_indices],
            z=[causes_z[i] + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for i in distinctions_lost_indices],
            mode="text",
            text=[cause_purview_labels[i] for i in distinctions_lost_indices],
            textposition=purview_label_position,
            name="Lost Cause Purviews",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="red"),
            hoverinfo="text",
            hovertext=[causes_hovertext[i] for i in distinctions_lost_indices],
            hoverlabel=dict(bgcolor="red"),
        )
        fig.add_trace(lost_labels_cause_purviews_trace)        

    else:
        labels_cause_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=causes_x,
            y=causes_y,
            z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in causes_z],
            mode="text",
            text=cause_purview_labels,
            textposition=purview_label_position,
            name="Cause Purview Labels",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="red"),
            hoverinfo="text",
            hovertext=causes_hovertext,
            hoverlabel=dict(bgcolor="red"),
        )
        fig.add_trace(labels_cause_purviews_trace)

    # Make effect purview labels trace
    if selected_mechanism_qfolds:
        selected_qfold_effects_indices = [i for i,purview in enumerate(separated_ces[1::2]) if purview in selected_qfold_mices]        

        labels_effect_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=[effects_x[i] for i in selected_qfold_effects_indices],
            y=[effects_y[i] for i in selected_qfold_effects_indices],
            z=[effects_z[i] + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for i in selected_qfold_effects_indices],
            mode="text",
            text=[effect_purview_labels[i] for i in selected_qfold_effects_indices],
            textposition=purview_label_position,
            name="Effect Purview Labels",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="green"),
            hoverinfo="text",
            hovertext=effects_hovertext,
            hoverlabel=dict(bgcolor="green"),
        )
        fig.add_trace(labels_effect_purviews_trace)

    elif intersect_mechanisms:
        intersection_labels_effect_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=[effects_x[i] for i in intersected_mechanisms_indices],
            y=[effects_y[i] for i in intersected_mechanisms_indices],
            z=[effects_z[i] + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for i in intersected_mechanisms_indices],
            mode="text",
            text=[effect_purview_labels[i] for i in intersected_mechanisms_indices],
            textposition=purview_label_position,
            name="Intersection Effect Purview Labels",
            showlegend=True,
        textfont=dict(size=purview_labels_size, color="green"),
            hoverinfo="text",
            hovertext=[effects_hovertext[i] for i in intersected_mechanisms_indices],
            hoverlabel=dict(bgcolor="green"),
        )
        fig.add_trace(intersection_labels_effect_purviews_trace)

    elif distinctions_lost and relations_lost:
        lost_labels_effect_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=[effects_x[i] for i in distinctions_lost_indices],
            y=[effects_y[i] for i in distinctions_lost_indices],
            z=[effects_z[i] + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for i in distinctions_lost_indices],
            mode="text",
            text=[effect_purview_labels[i] for i in distinctions_lost_indices],
            textposition=purview_label_position,
            name="Lost Effect Purviews",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="green"),
            hoverinfo="text",
            hovertext=[effects_hovertext[i] for i in distinctions_lost_indices],
            hoverlabel=dict(bgcolor="green"),
        )
        fig.add_trace(lost_labels_effect_purviews_trace)        

    else:
        labels_effect_purviews_trace = go.Scatter3d(
            visible=show_purview_labels,
            x=effects_x,
            y=effects_y,
            z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in effects_z],
            mode="text",
            text=effect_purview_labels,
            textposition=purview_label_position,
            name="Effect Purview Labels",
            showlegend=True,
            textfont=dict(size=purview_labels_size, color="green"),
            hoverinfo="text",
            hovertext=effects_hovertext,
            hoverlabel=dict(bgcolor="green"),
        )
        fig.add_trace(labels_effect_purviews_trace)     

    # Make cause purviews state labels trace
    if show_purview_state_labels:
        labels_cause_purviews_state_trace = go.Scatter3d(
            visible=show_purview_state_labels,
            x=causes_x,
            y=causes_y,
            z=[
                n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset + states_z_offset)
                for n in causes_z
            ],
            mode="text",
            text=cause_purview_state_labels,
            name="Cause Purview State Labels",
            showlegend=True,
            textfont=dict(size=purview_state_labels_size, color="red"),
            hoverinfo="text",
            hovertext=causes_hovertext,
            hoverlabel=dict(bgcolor="red"),
        )
        fig.add_trace(labels_cause_purviews_state_trace)

    # Make effect purviews state labels trace
    if show_purview_state_labels:
        labels_effect_purviews_state_trace = go.Scatter3d(
            visible=show_purview_state_labels,
            x=effects_x,
            y=effects_y,
            z=[
                n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset + states_z_offset)
                for n in effects_z
            ],
            mode="text",
            text=effect_purview_state_labels,
            name="Effect Purview State Labels",
            showlegend=True,
            textfont=dict(size=purview_state_labels_size, color="green"),
            hoverinfo="text",
            hovertext=effects_hovertext,
            hoverlabel=dict(bgcolor="green"),
        )
        fig.add_trace(labels_effect_purviews_state_trace)

    # Separating purview traces

    purview_phis = [purview.phi for purview in separated_ces]
    cause_purview_phis, effect_purview_phis = separate_cause_and_effect_purviews_for(
        purview_phis
    )

    # direction_labels = list(flatten([["Cause", "Effect"] for c in ces]))
    vertices_cause_purviews_trace = go.Scatter3d(
        visible=show_vertices_purviews,
        x=causes_x,
        y=causes_y,
        z=causes_z,
        mode="markers",
        name="Cause Purviews",
        text=purview_labels,
        showlegend=True,
        marker=dict(size=cause_purview_sizes, color="red"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="red"),
    )
    fig.add_trace(vertices_cause_purviews_trace)

    vertices_effect_purviews_trace = go.Scatter3d(
        visible=show_vertices_purviews,
        x=effects_x,
        y=effects_y,
        z=effects_z,
        mode="markers",
        name="Effect Purviews",
        text=purview_labels,
        showlegend=True,
        marker=dict(size=effect_purview_sizes, color="green"),
        hoverinfo="text",
        hovertext=effects_hovertext,
        hoverlabel=dict(bgcolor="green"),
    )
    fig.add_trace(vertices_effect_purviews_trace)

    # Initialize lists for legend
    legend_nodes = []
    legend_mechanisms = []
    legend_compound_purviews = []
    legend_relation_purviews = []
    legend_mechanism_purviews = []
    legend_intersection = []

    # Counters for trace legend items.
    intersection_2_relations_counter = 0
    intersection_3_relations_counter = 0
    lost_2_relations_counter=0
    lost_3_relations_counter=0
    remained_2_relations_counter=0
    remained_3_relations_counter=0

    # Plot distinction links (edge connecting cause, mechanism, effect vertices)
    coords_links = (
        list(zip(x, flatten(list(zip(xm, xm))))),
        list(zip(y, flatten(list(zip(ym, ym))))),
        list(zip(z, flatten(list(zip(zm, zm))))),
    )
    
    links_widths = normalize_sizes(
        link_width_range[0], link_width_range[1], ces)

    links_widths = list(flatten(list(zip(links_widths,links_widths))))
    if composition:
        links_widths = [composition_link_width for l in links_widths]
  
    if show_links:
        intersection_links_counter = 0        
        selected_qfold_links_counter=0
        lost_links_counter=0
        remained_links_counter=0
        links_counter=0
        for i, mice in enumerate(separated_ces):

            if selected_mechanism_qfolds:                
                if mice in selected_qfold_mices:
                    link_trace = go.Scatter3d(
                        visible=True,
                        legendgroup="Links",
                        showlegend=True if selected_qfold_links_counter == 0 else False,
                        x=coords_links[0][i],
                        y=coords_links[1][i],
                        z=coords_links[2][i],
                        mode="lines",
                        name="Links",
                        line_width=links_widths[i],
                        line_color=composition_color if composition or (integration_cut_elements and any([m in purview.mechanism for m in integration_cut_elements])) else "brown",
                        hoverinfo="skip",
                        # hovertext=hovertext_relation(relation),
                )

                    fig.add_trace(link_trace)
                    selected_qfold_links_counter+=1
            
            # Make trace link for intersection only
            elif intersect_mechanisms and mice.mechanism in intersect_mechanisms:
                intersection_link_trace = go.Scatter3d(
                    visible=True,
                    legendgroup="intersection links",
                    showlegend=True if intersection_links_counter == 0 else False,
                    x=coords_links[0][i],
                    y=coords_links[1][i],
                    z=coords_links[2][i],
                    mode="lines",
                    name=f"{' ∩ '.join(intersected_mechanisms_labels)} Links",
                    line_width=links_widths[i],
                    line_color="brown",
                    hoverinfo="skip",
                    # hovertext=hovertext_relation(relation),
                )
                intersection_links_counter += 1
                fig.add_trace(intersection_link_trace)

            elif distinctions_lost and relations_lost:
                if mice.mechanism in distinctions_lost_mechanisms:
                    lost_link_trace = go.Scatter3d(
                        visible=True,
                        legendgroup="Lost Links",
                        showlegend=True if lost_links_counter == 0 else False,
                        x=coords_links[0][i],
                        y=coords_links[1][i],
                        z=coords_links[2][i],
                        mode="lines",
                        name="Lost Links",
                        line_width=links_widths[i],
                        line_color=distinctions_lost_link_color,
                        hoverinfo="skip",
                        # hovertext=hovertext_relation(relation),
                    )
                    lost_links_counter += 1
                    fig.add_trace(lost_link_trace)
                else:
                    remained_link_trace = go.Scatter3d(
                        visible=True,
                        legendgroup="Remained Links",
                        showlegend=True if remained_links_counter == 0 else False,
                        x=coords_links[0][i],
                        y=coords_links[1][i],
                        z=coords_links[2][i],
                        mode="lines",
                        name="Remained Links",
                        line_width=links_widths[i],
                        line_color=distinctions_remained_link_color,
                        hoverinfo="skip",
                        # hovertext=hovertext_relation(relation),
                    )
                    remained_links_counter += 1
                    fig.add_trace(remained_link_trace)

            else:
                link_trace = go.Scatter3d(
                    visible=show_links,
                    legendgroup="Links",
                    showlegend=True if links_counter==0 else False,
                    x=coords_links[0][i],
                    y=coords_links[1][i],
                    z=coords_links[2][i],
                    mode="lines",
                    name="Links",
                    line_width=links_widths[i],
                    line_color=composition_color if composition or (integration_cut_elements and any([m in mice.mechanism for m in integration_cut_elements])) else "brown",
                    hoverinfo="skip",
                    # hovertext=hovertext_relation(relation),
                )
                links_counter += 1
                fig.add_trace(link_trace)

    # 2-relations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if show_edges:
        # Get edges from all relations
        edges = list(
            flatten(
                relation_vertex_indices(features, j)
                for j in range(features.shape[1])
                if features[:, j].sum() == 2
            )
        )
        if edges:
            # Convert to DataFrame
            edges = pd.DataFrame(
                dict(
                    x=x[edges],
                    y=y[edges],
                    z=z[edges],
                    line_group=flatten(
                        zip(range(len(edges) // 2), range(len(edges) // 2))
                    ),
                )
            )

            # Plot edges separately:
            two_relations = list(filter(lambda r: len(r.relata) == 2, relations))

            two_relations_sizes = normalize_sizes(
                edge_size_range[0], edge_size_range[1], two_relations
            )

            two_relations_coords = [
                list(chunk_list(list(edges["x"]), 2)),
                list(chunk_list(list(edges["y"]), 2)),
                list(chunk_list(list(edges["z"]), 2)),
            ]

            for r, relation in tqdm(
                enumerate(two_relations),
                desc="Computing edges",
                total=len(two_relations),
            ):
                relation_nodes = list(flatten(relation.mechanisms))
                relation_color = get_edge_color(relation, colorcode_2_relations)



                # Make node contexts traces and legendgroups
                if show_node_qfolds:
                    legend_nodes = plot_node_qfolds2D(
                        r,
                        relation,
                        node_indices,
                        node_labels,
                        go,
                        fig,
                        show_edges,
                        legend_nodes,
                        two_relations_coords,
                        two_relations_sizes,
                        relation_color,
                        relation_nodes,
                    )

                # Make nechanism contexts traces and legendgroups
                if show_mechanism_qfolds:
                    legend_mechanisms = plot_mechanism_qfolds2D(
                        r,
                        relation,
                        ces,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_mechanisms,
                        relation_color,
                    )

                if selected_mechanism_qfolds:
                    
                    legend_mechanisms = plot_selected_mechanism_qfolds2D(
                        r,
                        relation,
                        selected_mechanism_qfolds,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_mechanisms,
                        relation_color,
                    )


                # Make compound purview contexts traces and legendgroups
                if show_compound_purview_qfolds:

                    legend_compound_purviews = plot_compound_purview_qfolds2D(
                        r,
                        relation,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_compound_purviews,
                        relation_color,
                    )

                # Make relation purview contexts traces and legendgroups

                # For plotting Relation Purview Q-Folds, which are the relations over a certain purview, regardless of the mechanism.
                if show_relation_purview_qfolds:

                    legend_relation_purviews = plot_relation_purview_qfolds2D(
                        r,
                        relation,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_relation_purviews,
                        relation_color,
                    )

                # Make cause/effect purview per mechanism contexts traces and legendgroups
                if show_per_mechanism_purview_qfolds:

                    legend_mechanism_purviews = plot_per_mechanism_purview_qfolds2D(
                        r,
                        relation,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_mechanism_purviews,
                        relation_color,
                    )
                
                else:
                    legend_mechanisms = []

                

                #Make intersection 2-relations traces and legendgroup
                if intersect_mechanisms:

                    if set(relation.mechanisms) == set(intersect_mechanisms):
     
                        intersection_two_relation_trace = go.Scatter3d(
                            visible=True,
                            legendgroup=f"{' ∩ '.join(intersected_mechanisms_labels)} 2-Relations",
                            showlegend=True if intersection_2_relations_counter == 0 else False,
                            x=two_relations_coords[0][r],
                            y=two_relations_coords[1][r],
                            z=two_relations_coords[2][r],
                            mode="lines",
                            # name=label_relation(relation),
                            name=f"{' ∩ '.join(intersected_mechanisms_labels)} 2-Relations",
                            line_width=two_relations_sizes[r],
                            line_color=composition_color if composition else relation_color,
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )

                        intersection_2_relations_counter += 1

                        fig.add_trace(intersection_two_relation_trace)

                if distinctions_lost and relations_lost:

                    if relation in relations_lost:
     
                        lost_two_relation_trace = go.Scatter3d(
                            visible=True,
                            legendgroup="Lost 2-Relations",
                            showlegend=True if lost_2_relations_counter == 0 else False,
                            x=two_relations_coords[0][r],
                            y=two_relations_coords[1][r],
                            z=two_relations_coords[2][r],
                            mode="lines",
                            # name=label_relation(relation),
                            name="Lost 2-Relations",
                            line_width=two_relations_sizes[r],
                            line_color=relations_lost_edge_color,
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        lost_2_relations_counter += 1

                        fig.add_trace(lost_two_relation_trace)
                    else:        
                        remained_two_relation_trace = go.Scatter3d(
                            visible=True,
                            legendgroup="Remained 2-Relations",
                            showlegend=True if remained_2_relations_counter == 0 else False,
                            x=two_relations_coords[0][r],
                            y=two_relations_coords[1][r],
                            z=two_relations_coords[2][r],
                            mode="lines",
                            # name=label_relation(relation),
                            name="Remained 2-Relations",
                            line_width=two_relations_sizes[r],
                            line_color=relations_remained_edge_color,
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        remained_2_relations_counter += 1

                        fig.add_trace(remained_two_relation_trace)

                
                # Make all 2-relations traces and legendgroup
                edge_two_relation_trace = go.Scatter3d(
                    visible='legendonly' if selected_mechanism_qfolds or intersect_mechanisms or distinctions_lost else show_edges,
                    legendgroup="All 2-Relations",
                    showlegend=True if r == 0 else False,
                    x=two_relations_coords[0][r],
                    y=two_relations_coords[1][r],
                    z=two_relations_coords[2][r],
                    mode="lines",
                    # name=label_relation(relation),
                    name="All 2-Relations",
                    line_width=composition_edge_size if composition or (integration_cut_elements and any([m in flatten(relation.mechanisms) for m in integration_cut_elements])) else two_relations_sizes[r],
                    line_color=composition_color if composition or (integration_cut_elements and any([m in flatten(relation.mechanisms) for m in integration_cut_elements])) else relation_color,
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )

                fig.add_trace(edge_two_relation_trace)

    # 3-relations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get triangles from all relations
    if show_mesh:
        triangles = [
            relation_vertex_indices(features, j)
            for j in range(features.shape[1])
            if features[:, j].sum() == 3
        ]

        if triangles:
            three_relations = list(filter(lambda r: len(r.relata) == 3, relations))
            three_relations_sizes = normalize_sizes(
                surface_size_range[0], surface_size_range[1], three_relations
            )
            # Extract triangle indices
            i, j, k = zip(*triangles)
            for r, triangle in tqdm(
                enumerate(triangles), desc="Computing triangles", total=len(triangles)
            ):
                relation = three_relations[r]
                relation_nodes = list(flatten(relation.mechanisms))

                if show_node_qfolds:

                    legend_nodes = plot_node_qfolds3D(
                        r,
                        relation,
                        show_mesh,
                        node_indices,
                        node_labels,
                        go,
                        fig,
                        legend_nodes,
                        legend_mechanisms,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
                        relation_nodes,
                    )

                if show_mechanism_qfolds:

                    legend_mechanisms = plot_mechanism_qfolds3D(
                        r,
                        relation,
                        ces,
                        show_mesh,
                        node_labels,
                        go,
                        fig,
                        legend_mechanisms,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
                    )

                if selected_mechanism_qfolds:

                    legend_mechanisms = plot_selected_mechanism_qfolds3D(
                        r,
                        relation,
                        selected_mechanism_qfolds,
                        show_mesh,
                        node_labels,
                        go,
                        fig,
                        legend_mechanisms,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
                    )


                if show_compound_purview_qfolds:

                    legend_compound_purviews = plot_compound_purview_qfolds3D(
                        r,
                        relation,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_compound_purviews,
                        relation_color,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
                    )

                if show_relation_purview_qfolds:

                    legend_relation_purviews = plot_relation_purview_qfolds3D(
                        r,
                        relation,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_relation_purviews,
                        relation_color,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
                    )

                if show_per_mechanism_purview_qfolds:
                    legend_mechanism_purviews = plot_per_mechanism_purview_qfolds3D(
                        r,
                        relation,
                        show_edges,
                        node_labels,
                        go,
                        fig,
                        two_relations_coords,
                        two_relations_sizes,
                        legend_mechanism_purviews,
                        relation_color,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
                    )
                else:
                    legend_mechanisms = []

                #Make intersection 3-relations traces and legendgroup
                if intersect_mechanisms:

                    if set(relation.mechanisms) == set(intersect_mechanisms):
     
                        intersection_three_relation_trace = go.Mesh3d(
                            visible=True,
                            legendgroup=f"{' ∩ '.join(intersected_mechanisms_labels)} 3-Relations",
                            showlegend=True if intersection_3_relations_counter == 0 else False,
                            # x, y, and z are the coordinates of vertices
                            x=x,
                            y=y,
                            z=z,
                            # i, j, and k are the vertices of triangles
                            i=[i[r]],
                            j=[j[r]],
                            k=[k[r]],
                            intensity=np.linspace(0, 1, len(x), endpoint=True),
                            opacity=three_relations_sizes[r],
                            colorscale="Greys" if composition else "viridis",
                            showscale=False,
                            name=f"{' ∩ '.join(intersected_mechanisms_labels)} 3-Relations",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )

                        intersection_3_relations_counter += 1

                        fig.add_trace(intersection_three_relation_trace)

                if distinctions_lost and relations_lost:

                    if relation in relations_lost:
     
                        lost_three_relation_trace = go.Mesh3d(
                            visible=True,
                            legendgroup="Lost 3-Relations",
                            showlegend=True if lost_3_relations_counter == 0 else False,
                            # x, y, and z are the coordinates of vertices
                            x=x,
                            y=y,
                            z=z,
                            # i, j, and k are the vertices of triangles
                            i=[i[r]],
                            j=[j[r]],
                            k=[k[r]],
                            intensity=np.linspace(0, 1, len(x), endpoint=True),
                            opacity=three_relations_sizes[r],
                            colorscale=relations_lost_surface_colorscale,
                            showscale=False,
                            name="Lost 3-Relations",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        lost_3_relations_counter += 1

                        fig.add_trace(lost_three_relation_trace)                        

                    else:
                        remained_three_relation_trace = go.Mesh3d(
                            visible=True,
                            legendgroup="Remained 3-Relations",
                            showlegend=True if remained_3_relations_counter == 0 else False,
                            # x, y, and z are the coordinates of vertices
                            x=x,
                            y=y,
                            z=z,
                            # i, j, and k are the vertices of triangles
                            i=[i[r]],
                            j=[j[r]],
                            k=[k[r]],
                            intensity=np.linspace(0, 1, len(x), endpoint=True),
                            opacity=three_relations_sizes[r],
                            colorscale=relations_remained_surface_color,
                            showscale=False,
                            name="Remained 3-Relations",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )                        

                        remained_3_relations_counter += 1

                        fig.add_trace(remained_three_relation_trace)                        

                triangle_three_relation_trace = go.Mesh3d(
                    visible='legendonly' if selected_mechanism_qfolds or intersect_mechanisms or distinctions_lost else show_mesh,
                    legendgroup="All 3-Relations",
                    showlegend=True if r == 0 else False,
                    # x, y, and z are the coordinates of vertices
                    x=x,
                    y=y,
                    z=z,
                    # i, j, and k are the vertices of triangles
                    i=[i[r]],
                    j=[j[r]],
                    k=[k[r]],
                    # Intensity of each vertex, which will be interpolated and color-coded
                    intensity=[composition_surface_intensity for x in range(len(x))] if composition or (integration_cut_elements and any([m in flatten(relation.mechanisms) for m in integration_cut_elements])) else np.linspace(0, 1, len(x), endpoint=True),
                    opacity=composition_surface_opacity if composition or (integration_cut_elements and any([m in flatten(relation.mechanisms) for m in integration_cut_elements])) else three_relations_sizes[r],
                    colorscale="Greys" if composition or (integration_cut_elements and any([m in flatten(relation.mechanisms) for m in integration_cut_elements])) else "viridis",
                    showscale=False,
                    name="All 3-Relations",
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )
                fig.add_trace(triangle_three_relation_trace)        

    
    # Add image to xy-plane
    if show_image:
        # loading image and subsampling to approximate resolution
        img = imread(image_file)
        
        # getting 1D color representation (NOTE: dont know how to do RBG code)
        img_shade = img[::image_downsample,::image_downsample,:3].mean(2)
        # forcing shade to use the whole scale (0-1)
        img_shade = 1-(img_shade-np.min(img_shade)+0.1)/(np.abs(np.max(img_shade)-np.min(img_shade))+0.2)

        # getting variables to specify xy-plane to cover
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        xcenter = np.mean([xmin,xmax])
        ycenter = np.mean([ymin,ymax])
        new_x = np.array([xcenter-np.abs(xcenter-xmin)*image_xy_scale, xcenter+np.abs(xcenter-xmax)*image_xy_scale]) + image_center[0]
        new_y = np.array([ycenter-np.abs(ycenter-ymin)*image_xy_scale, ycenter+np.abs(ycenter-ymax)*image_xy_scale]) + image_center[1]

        z_im_pos = min([min(z),min(zm)]) + image_z_offset

        x_im = np.linspace(-new_x[0], -new_x[1], img_shade.shape[1])
        y_im = np.linspace(new_y[0], new_y[1], img_shade.shape[0])
        z_im = np.ones((img_shade.shape[0],img_shade.shape[1]))*z_im_pos

        xy_plane = go.Surface(x=x_im, y=y_im, z=z_im, surfacecolor=img_shade, opacity=image_opacity,
                            colorscale="Greys",showscale=False,)
                  
        fig.add_trace(xy_plane)

    # Create figure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if show_image:
    #     axes_range = [
    #         (min(d) - 1, max(d) + 1)
    #         for d in (x_im, y_im, np.append(z, zm))
    #     ]
    # else:
    #     axes_range = [
    #         (min(d) - 1, max(d) + 1)
    #         for d in (np.append(x, xm), np.append(y, ym), np.append(z, zm))
    #     ]
    axes_range = [
        (min(d) - 1, max(d) + 1)
        for d in (np.append(x, xm), np.append(y, ym), np.append(z, zm))
    ]    

    axes = [
        dict(
            showbackground=img_background,
            showline=False,
            zeroline=False,
            showgrid=show_grid,
            gridcolor="lightgray",
            showticklabels=False,
            showspikes=True,
            autorange=False,
            range=axes_range[dimension],
            backgroundcolor="white",
            title="",
        )
        for dimension in range(3)
    ]

    layout = go.Layout(
        showlegend=True,
        scene_xaxis=axes[0],
        scene_yaxis=axes[1],
        scene_zaxis=axes[2],
        scene_camera=dict(
            eye=dict(x=eye_coordinates[0], y=eye_coordinates[1], z=eye_coordinates[2])
        ),
        hovermode=hovermode,
        title=title,
        title_font_size=30,
        legend=dict(
            title=dict(
                text="Trace legend (click trace to show/hide):",
                font=dict(color="black", size=15),
            )
        ),
        autosize=autosize,
        height=plot_dimentions[0] if plot_dimentions else None,
        width=plot_dimentions[1] if plot_dimentions else None,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        
    )

    # Apply layout
    fig.layout = layout

    # Make labels as annotations:
    if mechanisms_as_annotations:

        # Make mechanism labels as annotations:
        mechanism_annotation_labels = [
            label_mechanism(mice, bold=False, state=False) for mice in separated_ces[::2]
        ]
        mechanism_label_text_colors = [get_mechanism_label_text_color(mice.mechanism,subsystem) for mice in separated_ces[::2]]
        mechanism_label_bg_colors = [get_mechanism_label_bg_color(mice.mechanism,subsystem) for mice in separated_ces[::2]]
        
        mech_alpha = relative_phi(ces,mini=0.1,maxi=1) if annotation_alpha_from_mechanism_phi else [annotations_alpha_mechanism_label,]*len(ces)

        if intersect_mechanisms:
            #Plot only intersected mechanisms' labels if specified
            mechanism_annotation_indices_labels = [
            (i,label_mechanism(mice, state=False, bold=False)) for i,mice in enumerate(separated_ces[::2]) if mice.mechanism in intersect_mechanisms
        ]
            # print(mechanism_annotation_indices_labels)
            
            annotations_mechanisms = [
                [
                    dict(
                        visible=True,
                        showarrow=False,
                        x=xm[i]
                        - n * annotation_x_spacing
                        + ((len(label) - 1) / 2) * annotation_x_spacing,
                        y=ym[i]
                        - n * annotation_y_spacing
                        + ((len(label) - 1) / 2) * annotation_y_spacing,
                        z=zm[i]
                        - n * annotation_z_spacing
                        + ((len(label) - 1) / 2) * annotation_z_spacing_mechanisms,
                        text=node_label,
                        font=dict(
                            size=mechanism_labels_size,
                            color='black' if composition else mechanism_label_text_colors[i][n],
                        ),
                        opacity=mech_alpha[i],
                        # bordercolor='black',
                        # borderwidth=1,
                        # borderpad=2,
                        # bgcolor=composition_color if composition else mechanism_label_bg_colors[i][n],
                    )
                    for n, node_label in enumerate(label)
                ]
                for i, label in mechanism_annotation_indices_labels
            ]
            # print(annotations_mechanisms)

        else:
            #Plot all mechanisms' labels
            annotations_mechanisms = [
                [
                    dict(
                        visible=True,
                        showarrow=False,
                        x=xm[i]
                        - n * annotation_x_spacing
                        + ((len(label) - 1) / 2) * annotation_x_spacing,
                        y=ym[i]
                        - n * annotation_y_spacing
                        + ((len(label) - 1) / 2) * annotation_y_spacing,
                        z=zm[i]
                        - n * annotation_z_spacing_mechanisms
                        + ((len(label) - 1) / 2) * annotation_z_spacing_mechanisms,
                        text=node_label,
                        font=dict(
                            size=mechanism_labels_size,
                            color='black' if composition else mechanism_label_text_colors[i][n],
                        ),
                        opacity=mech_alpha[i],
                        # bordercolor='black',
                        # borderwidth=1,
                        # borderpad=2,
                        # bgcolor=composition_color if composition else mechanism_label_bg_colors[i][n],
                    )
                    for n, node_label in enumerate(label)
                ]
                for i, label in enumerate(mechanism_annotation_labels)
            ]
        annotations_mechanisms = list(flatten(annotations_mechanisms))

    if purviews_as_annotations:

        # Make purview labels as annotations:
        purview_annotation_labels = [
            label_purview(mice, state=False) for mice in separated_ces
        ]
        purview_label_text_colors = [get_purview_label_text_color(mice, composition) for mice in separated_ces]
        purview_label_bg_colors = [get_purview_label_bg_color(mice) for mice in separated_ces] 
        purview_label_border_colors = [get_purview_label_border_color(mice) for mice in separated_ces] 

        purview_alpha = relative_phi(separated_ces,mini=0.1,maxi=1) if annotation_alpha_from_purview_phi else [annotations_alpha_purview_label,]*len(separated_ces)

        if intersect_mechanisms:
            #Plot only intersected mechanisms' purview labels if specified
            purview_annotation_labels_and_indices = [
            (i,label_purview(mice, state=False)) for i,mice in enumerate(separated_ces) if mice.mechanism in intersect_mechanisms
        ]

            annotations_purviews = [
                [
                    dict(
                        visible=True,
                        showarrow=False,
                        x=x[i]
                        - n * annotation_x_spacing
                        + ((len(label) - 1) / 2) * annotation_x_spacing,
                        y=y[i]
                        - n * annotation_y_spacing
                        + ((len(label) - 1) / 2) * annotation_y_spacing,
                        z=z[i]
                        - n * annotation_z_spacing_mechanisms
                        + ((len(label) - 1) / 2) * annotation_z_spacing,
                        text=node_label,
                        font=dict(
                            size=purview_labels_size,
                            color=purview_label_text_colors[i][n],                        
                        ),
                        opacity=purview_alpha[i],
                        bordercolor=purview_label_border_colors[i],
                        borderwidth=1,
                        borderpad=2,
                        bgcolor=composition_color if composition else purview_label_bg_colors[i][n],
                    )
                    for n, node_label in enumerate(label)
                ]
                for i, label in purview_annotation_labels_and_indices
            ]

        elif integration_cut_elements:
            integration_cut_mechanisms = [tuple([m]) for m in integration_cut_elements]
            integration_cut_elements_labels = strip_punct(str([subsystem.indices2nodes(mech) for mech in integration_cut_mechanisms]))
        
            annotations_purviews = [
                [
                    dict(
                        visible=True,
                        showarrow=False,
                        x=x[i]
                        - n * annotation_x_spacing
                        + ((len(label) - 1) / 2) * annotation_x_spacing,
                        y=y[i]
                        - n * annotation_y_spacing
                        + ((len(label) - 1) / 2) * annotation_y_spacing,
                        z=z[i]
                        - n * annotation_z_spacing
                        + ((len(label) - 1) / 2) * annotation_z_spacing,
                        text=node_label,
                        font=dict(
                            size=purview_labels_size,
                            color=purview_label_text_colors[i][n],                        
                        ),
                        opacity=purview_alpha[i],
                        bordercolor=purview_label_border_colors[i],
                        borderwidth=1,
                        borderpad=2,
                        bgcolor=integration_color if any([m in label for m in integration_cut_elements_labels]) else purview_label_bg_colors[i][n],
                    )
                    for n, node_label in enumerate(label)
                ]
                for i, label in enumerate(purview_annotation_labels)
            ]

        else:
            #Plot all intersected mechanisms' purview labels
            annotations_purviews = [
                [
                    dict(
                        visible=True,
                        showarrow=False,
                        x=x[i]
                        - n * annotation_x_spacing
                        + ((len(label) - 1) / 2) * annotation_x_spacing,
                        y=y[i]
                        - n * annotation_y_spacing
                        + ((len(label) - 1) / 2) * annotation_y_spacing,
                        z=z[i]
                        - n * annotation_z_spacing
                        + ((len(label) - 1) / 2) * annotation_z_spacing,
                        text=node_label,
                        font=dict(
                            size=purview_labels_size,
                            color=composition_text_color if composition else purview_label_text_colors[i][n],                        
                        ),
                        opacity=purview_alpha[i],
                        bordercolor=purview_label_border_colors[i],
                        borderwidth=1,
                        borderpad=2,
                        bgcolor=composition_color if composition else purview_label_bg_colors[i][n],
                    )
                    for n, node_label in enumerate(label)
                ]
                for i, label in enumerate(purview_annotation_labels)
            ]
        
       


        annotations_purviews = list(flatten(annotations_purviews))

        annotations_all = annotations_mechanisms + annotations_purviews if mechanisms_as_annotations else annotations_purviews

        fig.update_layout(scene=dict(annotations=annotations_all))      

    if show_causal_model:
        # Create system image
        save_digraph(
            subsystem, element_positions, digraph_filename, layout=digraph_layout
        )

        fn = get_sample_data(digraph_filename, asfileobj=False)
        img = read_png(fn)
        x, y = ogrid[0 : img.shape[0], 0 : img.shape[1]]
        ax = gca(projection="3d")
        ax.plot_surface(x, y, axes_range[0][0], rstride=5, cstride=5, facecolors=img)

    if save_plot_to_html:
        plotly.io.write_html(fig, f"{network_name}_CES.html")
    return fig
