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

test = 1 + 2


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


def all_triangles(vertices):
    """Return all triangles within a set of vertices."""
    return itertools.combinations(vertices, 3)


def all_edges(vertices):
    """Return all edges within a set of vertices."""
    return itertools.combinations(vertices, 2)


def make_label(nodes, node_labels=None):
    if node_labels is not None:
        nodes = node_labels.indices2labels(nodes)
    return "".join(nodes)


def label_mechanism(mice):
    return make_label(mice.mechanism, node_labels=mice.node_labels)


def label_purview(mice):
    return make_label(mice.purview, node_labels=mice.node_labels)


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
    show_edges,
    legend_nodes,
    two_relations_coords,
    two_relations_sizes,
    relation_color,
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


def plot_ces(
    subsystem,
    ces,
    relations,
    coords=None,
    network=None,
    max_order=3,
    cause_effect_offset=(0.3, 0, 0),
    vertex_size_range=(10, 40),
    edge_size_range=(0.5, 4),
    surface_size_range=(0.005, 0.1),
    plot_dimentions=(800, 1000),
    mechanism_labels_size=14,
    mechanism_state_labels_size=12,
    labels_z_offset=0.15,
    states_z_offset=0.15,
    purview_labels_size=12,
    purview_state_labels_size=10,
    show_mechanism_labels="legendonly",
    show_links="legendonly",
    show_mechanism_state_labels="legendonly",
    show_purview_labels="legendonly",
    show_purview_state_labels="legendonly",
    show_vertices_mechanisms=True,
    show_vertices_purviews=True,
    show_edges="legendonly",
    show_mesh="legendonly",
    show_node_qfolds=False,
    show_mechanism_qfolds=False,
    show_compound_purview_qfolds=False,
    show_relation_purview_qfolds=False,
    show_per_mechanism_purview_qfolds=False,
    show_grid=False,
    network_name="",
    eye_coordinates=(0.3, 0.3, 0.3),
    hovermode="x",
    digraph_filename="digraph.png",
    digraph_layout="dot",
    save_plot_to_html=True,
    show_causal_model=False,
    order_on_z_axis=True,
    save_coords=False,
    link_width=1.5,
    colorcode_2_relations=True,
    left_margin=get_screen_size()[0] / 10,
):

    # Select only relations <= max_order
    relations = list(filter(lambda r: len(r.relata) <= max_order, relations))

    # Separate CES into causes and effects
    separated_ces = rel.separate_ces(ces)

    # Initialize figure
    fig = go.Figure()

    # Dimensionality reduction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create the features for each cause/effect based on their relations
    features = feature_matrix(separated_ces, relations)

    # Now we get one set of coordinates for the CES; these will then be offset to
    # get coordinates for causes and effects separately, so that causes/effects
    # are always near each other in the embedding.

    # Collapse rows of cause/effect belonging to the same distinction
    # NOTE: This depends on the implementation of `separate_ces`; causes and
    #       effects are assumed to be adjacent in the returned list
    umap_features = features[0::2] + features[1::2]
    if coords is None:
        if order_on_z_axis:
            distinction_coords = get_coords(umap_features, n_components=2)
            cause_effect_offset = cause_effect_offset[:2]

        else:
            distinction_coords = get_coords(umap_features)
        # Duplicate causes and effects so they can be plotted separately
        coords = np.empty(
            (distinction_coords.shape[0] * 2, distinction_coords.shape[1]),
            dtype=distinction_coords.dtype,
        )
        coords[0::2] = distinction_coords
        coords[1::2] = distinction_coords
        # Add a small offset to effects to separate them from causes
        coords[1::2] += cause_effect_offset

    # Purviews
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract vertex indices for plotly
    x, y = coords[:, 0], coords[:, 1]

    causes_x, causes_y, effects_x, effects_y = separate_cause_and_effect_for(coords)

    if order_on_z_axis:
        z = np.array([len(c.mechanism) for c in separated_ces])
    else:
        z = coords[:, 2]

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
    mechanism_labels = list(map(label_mechanism, ces))
    mechanism_state_labels = [
        label_mechanism_state(subsystem, distinction) for distinction in ces
    ]
    purview_labels = list(map(label_purview, separated_ces))
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

    # Make mechanism labels
    xm, ym, zm = (
        [c + cause_effect_offset[0] / 2 for c in x[::2]],
        y[::2],
        [z + 0.1 for z in z[::2]],
        # [n + (vertex_size_range[1] / 10 ** 3) for n in z[::2]],
    )

    labels_mechanisms_trace = go.Scatter3d(
        visible=show_mechanism_labels,
        x=xm,
        y=ym,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in zm],
        mode="text",
        text=mechanism_labels,
        name="Mechanism Labels",
        showlegend=True,
        textfont=dict(size=mechanism_labels_size, color="black"),
        hoverinfo="text",
        hovertext=mechanism_hovertext,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(labels_mechanisms_trace)

    # Make mechanism state labels trace
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
    labels_cause_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=causes_x,
        y=causes_y,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in causes_z],
        mode="text",
        text=cause_purview_labels,
        name="Cause Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color="red"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="red"),
    )
    fig.add_trace(labels_cause_purviews_trace)

    # Make effect purview labels trace
    labels_effect_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=effects_x,
        y=effects_y,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in effects_z],
        mode="text",
        text=effect_purview_labels,
        name="Effect Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color="green"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="green"),
    )
    fig.add_trace(labels_effect_purviews_trace)

    # Make cause purviews state labels trace
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

    intersectionCount = 0  # A flag and a counter for the times there is a check for intersection and it is found.
    # Plot distinction links (edge connecting cause, mechanism, effect vertices)
    coords_links = (
        list(zip(x, flatten(list(zip(xm, xm))))),
        list(zip(y, flatten(list(zip(ym, ym))))),
        list(zip(z, flatten(list(zip(zm, zm))))),
    )

    for i, distinction in enumerate(separated_ces):
        link_trace = go.Scatter3d(
            visible=show_links,
            legendgroup="Links",
            showlegend=True if i == 1 else False,
            x=coords_links[0][i],
            y=coords_links[1][i],
            z=coords_links[2][i],
            mode="lines",
            name="Links",
            line_width=link_width,
            line_color="brown",
            hoverinfo="skip",
            # hovertext=hovertext_relation(relation),
        )

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
                        show_edges,
                        legend_nodes,
                        two_relations_coords,
                        two_relations_sizes,
                        relation_color,
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

                # Make all 2-relations traces and legendgroup
                edge_two_relation_trace = go.Scatter3d(
                    visible=show_edges,
                    legendgroup="All 2-Relations",
                    showlegend=True if r == 0 else False,
                    x=two_relations_coords[0][r],
                    y=two_relations_coords[1][r],
                    z=two_relations_coords[2][r],
                    mode="lines",
                    # name=label_relation(relation),
                    name="All 2-Relations",
                    line_width=two_relations_sizes[r],
                    line_color=relation_color,
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
                        node_labels,
                        go,
                        fig,
                        legend_nodes,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
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

                triangle_three_relation_trace = go.Mesh3d(
                    visible=show_mesh,
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
                    intensity=np.linspace(0, 1, len(x), endpoint=True),
                    opacity=three_relations_sizes[r],
                    colorscale="viridis",
                    showscale=False,
                    name="All 3-Relations",
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )
                fig.add_trace(triangle_three_relation_trace)

        # Create figure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes_range = [(min(d) - 1, max(d) + 1) for d in (x, y, z)]

    axes = [
        dict(
            showbackground=False,
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
        title=f"{network_name} Q-STRUCTURE",
        title_font_size=30,
        legend=dict(
            title=dict(
                text="Trace legend (click trace to show/hide):",
                font=dict(color="black", size=15),
            )
        ),
        autosize=True,
        # height=plot_dimentions[0],
        # width=plot_dimentions[1],
    )

    # Apply layout
    fig.layout = layout

    if show_causal_model:
        # Create system image
        save_digraph(
            subsystem, element_positions, digraph_filename, layout=digraph_layout
        )
        encoded_image = base64.b64encode(open(digraph_filename, "rb").read())
        digraph_coords = (0, 0.75)
        digraph_size = (0.2, 0.3)

        fig.add_layout_image(
            dict(
                name="Causal model",
                source="data:image/png;base64,{}".format(encoded_image.decode()),
                #         xref="paper", yref="paper",
                x=digraph_coords[0],
                y=digraph_coords[1],
                sizex=digraph_size[0],
                sizey=digraph_size[1],
                xanchor="left",
                yanchor="top",
            )
        )

        draft_template = go.layout.Template()
        draft_template.layout.annotations = [
            dict(
                name="Causal model",
                text="Causal model",
                opacity=1,
                font=dict(color="black", size=20),
                xref="paper",
                yref="paper",
                x=digraph_coords[0],
                y=digraph_coords[1] + 0.05,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
            )
        ]

        fig.update_layout(
            margin_l=left_margin,
            template=draft_template,
            annotations=[dict(templateitemname="Causal model", visible=True)],
        )

    if save_plot_to_html:
        plotly.io.write_html(fig, f"{network_name}_CES.html")

    return fig


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


def plot_ces_epicycles(
    subsystem,
    ces,
    relations,
    network=None,
    max_order=3,
    purview_x_offset=0.1,
    mechanism_z_offset=0.1,
    vertex_size_range=(10, 40),
    edge_size_range=(0.5, 4),
    surface_size_range=(0.005, 0.1),
    plot_dimentions=(800, 1000),
    mechanism_labels_size=14,
    mechanism_state_labels_size=12,
    labels_z_offset=0.15,
    states_z_offset=0.15,
    purview_labels_size=12,
    purview_state_labels_size=10,
    show_mechanism_labels=True,
    show_links=True,
    show_mechanism_state_labels="legendonly",
    show_purview_labels=True,
    show_purview_state_labels="legendonly",
    show_vertices_mechanisms=True,
    show_vertices_purviews=True,
    show_edges=True,
    show_mesh=True,
    show_node_qfolds=False,
    show_mechanism_qfolds=False,
    show_compound_purview_qfolds=False,
    show_relation_purview_qfolds=False,
    show_per_mechanism_purview_qfolds=False,
    show_grid=False,
    network_name="",
    eye_coordinates=(1, 1, 1),
    hovermode="x",
    digraph_filename="digraph.png",
    digraph_layout=None,
    save_plot_to_html=True,
    show_causal_model=False,
    order_on_z_axis=False,
    save_coords=False,
    link_width=1.5,
    colorcode_2_relations=True,
    left_margin=get_screen_size()[0] / 10,
    floor_center=(0, 0),
    floor_scale=1,
    ground_floor_height=0,
    mezzanine_center=(0, 0),
    mezzanine_scale=0.2,
    mezzanine_floor_height=0,
):

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
                int(comb(N, k)),
                center=floor_center,
                z=k + ground_floor_height,
                scale=floor_scale,
            )
        )
        for k in range(1, N + 1)
    ]
    floor_vertices = np.concatenate([f for f in floors])

    # getting a list of all possible purviews
    all_purviews = list(powerset(range(N), nonempty=True))

    # find number of times each purview appears
    vertex_purview = {p:fv for p,fv in zip(all_purviews,floor_vertices) if purviews.count(p)>0} 

    # Create epicycles
    num_purviews = [purviews.count(p) for p in all_purviews if purviews.count(p)>0]
    epicycles = [
        regular_polygon(n, center=(e[0], e[1]), z=e[2], radius=0.2)
        for e, n in zip(floor_vertices, num_purviews)
        if n > 0
    ]

    # associating each purview with vertices in a regular polygon around the correct floor vertex
    purview_positions = [{v:e, 'N':0} for v,e in zip(vertex_purview.keys(),epicycles)]

    # placing purview coordinates in the correct order 
    purview_vertex_coordinates = []
    for p in purviews:
        for pp in purview_positions:
            if p in pp.keys():
                purview_vertex_coordinates.append(pp[p][pp['N']])
                pp['N']+=1


    coords = np.array(purview_vertex_coordinates)

    # Construct mezzanine
    mezzanine = [
        np.array(
            regular_polygon(
                int(comb(N, k)),
                center=mezzanine_center,
                z=k / N + mezzanine_floor_height,
                scale=mezzanine_scale,
            )
        )
        for k in range(1, N + 1)
    ]

    mezzanine_vertices = np.concatenate([f for f in mezzanine])
    i = 0
    mezzanine_coords = []
    for m,c,i in zip(all_purviews,mezzanine_vertices,range(len(all_purviews))):
        if m in mechanisms:
            mezzanine_coords.append(list(mezzanine_vertices[i]))

    xm = [p[0] for p in mezzanine_coords]
    ym = [p[1] for p in mezzanine_coords]
    zm = [p[2] for p in mezzanine_coords]

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
    mechanism_labels = list(map(label_mechanism, ces))
    mechanism_state_labels = [
        label_mechanism_state(subsystem, distinction) for distinction in ces
    ]
    purview_labels = list(map(label_purview, separated_ces))
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

    # Make mechanism labels
    labels_mechanisms_trace = go.Scatter3d(
        visible=show_mechanism_labels,
        x=xm,
        y=ym,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in zm],
        mode="text",
        text=mechanism_labels,
        name="Mechanism Labels",
        showlegend=True,
        textfont=dict(size=mechanism_labels_size, color="black"),
        hoverinfo="text",
        hovertext=mechanism_hovertext,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(labels_mechanisms_trace)

    # Make mechanism state labels trace
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
    labels_cause_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=causes_x,
        y=causes_y,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in causes_z],
        mode="text",
        text=cause_purview_labels,
        name="Cause Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color="red"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="red"),
    )
    fig.add_trace(labels_cause_purviews_trace)

    # Make effect purview labels trace
    labels_effect_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=effects_x,
        y=effects_y,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in effects_z],
        mode="text",
        text=effect_purview_labels,
        name="Effect Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color="green"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="green"),
    )
    fig.add_trace(labels_effect_purviews_trace)

    # Make cause purviews state labels trace
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

    intersectionCount = 0  # A flag and a counter for the times there is a check for intersection and it is found.
    # Plot distinction links (edge connecting cause, mechanism, effect vertices)
    coords_links = (
        list(zip(x, flatten(list(zip(xm, xm))))),
        list(zip(y, flatten(list(zip(ym, ym))))),
        list(zip(z, flatten(list(zip(zm, zm))))),
    )

    for i, distinction in enumerate(separated_ces):
        link_trace = go.Scatter3d(
            visible=show_links,
            legendgroup="Links",
            showlegend=True if i == 1 else False,
            x=coords_links[0][i],
            y=coords_links[1][i],
            z=coords_links[2][i],
            mode="lines",
            name="Links",
            line_width=link_width,
            line_color="brown",
            hoverinfo="skip",
            # hovertext=hovertext_relation(relation),
        )

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
                        show_edges,
                        legend_nodes,
                        two_relations_coords,
                        two_relations_sizes,
                        relation_color,
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

                # Make all 2-relations traces and legendgroup
                edge_two_relation_trace = go.Scatter3d(
                    visible=show_edges,
                    legendgroup="All 2-Relations",
                    showlegend=True if r == 0 else False,
                    x=two_relations_coords[0][r],
                    y=two_relations_coords[1][r],
                    z=two_relations_coords[2][r],
                    mode="lines",
                    # name=label_relation(relation),
                    name="All 2-Relations",
                    line_width=two_relations_sizes[r],
                    line_color=relation_color,
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
                        node_labels,
                        go,
                        fig,
                        legend_nodes,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
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

                triangle_three_relation_trace = go.Mesh3d(
                    visible=show_mesh,
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
                    intensity=np.linspace(0, 1, len(x), endpoint=True),
                    opacity=three_relations_sizes[r],
                    colorscale="viridis",
                    showscale=False,
                    name="All 3-Relations",
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )
                fig.add_trace(triangle_three_relation_trace)

        # Create figure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes_range = [
        (min(d) - 1, max(d) + 1)
        for d in (np.append(x, xm), np.append(y, ym), np.append(z, zm))
    ]

    axes = [
        dict(
            showbackground=False,
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
        title=f"{network_name} Q-STRUCTURE",
        title_font_size=30,
        legend=dict(
            title=dict(
                text="Trace legend (click trace to show/hide):",
                font=dict(color="black", size=15),
            )
        ),
        autosize=True,
        # height=plot_dimentions[0],
        # width=plot_dimentions[1],
    )

    # Apply layout
    fig.layout = layout
    #from matplotlib._png import read_png

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
    """
        encoded_image = base64.b64encode(open(digraph_filename, "rb").read())
        digraph_coords = (0, 0.75)
        digraph_size = (0.2, 0.3)

        fig.add_layout_image(
            dict(
                name="Causal model",
                source="data:image/png;base64,{}".format(encoded_image.decode()),
                #         xref="paper", yref="paper",
                x=digraph_coords[0],
                y=digraph_coords[1],
                sizex=digraph_size[0],
                sizey=digraph_size[1],
                xanchor="left",
                yanchor="top",
            )
        )

        draft_template = go.layout.Template()
        draft_template.layout.annotations = [
            dict(
                name="Causal model",
                text="Causal model",
                opacity=1,
                font=dict(color="black", size=20),
                xref="paper",
                yref="paper",
                x=digraph_coords[0],
                y=digraph_coords[1] + 0.05,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
            )
        ]

        fig.update_layout(
            margin_l=left_margin,
            template=draft_template,
            annotations=[dict(templateitemname="Causal model", visible=True)],
        )

"""


def plot_ces_on_being(
    subsystem,
    ces,
    relations,
    network=None,
    max_order=3,
    purview_x_offset=0.1,
    mechanism_z_offset=0.1,
    vertex_size_range=(10, 40),
    edge_size_range=(0.5, 4),
    surface_size_range=(0.005, 0.1),
    plot_dimentions=(800, 1000),
    mechanism_labels_size=14,
    mechanism_state_labels_size=12,
    labels_z_offset=0.15,
    states_z_offset=0.15,
    purview_labels_size=12,
    purview_state_labels_size=10,
    show_mechanism_labels=True,
    show_links=True,
    show_mechanism_state_labels="legendonly",
    show_purview_labels=True,
    show_purview_state_labels="legendonly",
    show_vertices_mechanisms=True,
    show_vertices_purviews=True,
    show_edges=True,
    show_mesh=True,
    show_node_qfolds=False,
    show_mechanism_qfolds=False,
    show_compound_purview_qfolds=False,
    show_relation_purview_qfolds=False,
    show_per_mechanism_purview_qfolds=False,
    show_grid=False,
    network_name="",
    eye_coordinates=(1, 1, 1),
    hovermode="x",
    digraph_filename="digraph.png",
    digraph_layout=None,
    save_plot_to_html=True,
    show_causal_model=False,
    order_on_z_axis=False,
    save_coords=False,
    link_width=1.5,
    colorcode_2_relations=True,
    left_margin=get_screen_size()[0] / 10,
    element_positions=None,
    jitter=0.0,
    floor_scale=1,
    floor_spacing=1,
    mezzanine=False,
    mezzanine_z_offset=None,
    mezzanine_floor_spacing=None,
    mezzanine_floor_scale=None,
    purview_in_place=False,
    purview_aligned_with_mechanism=False,
    restrict_ground_floor=True,
):

    # Select only relations <= max_order
    relations = list(filter(lambda r: len(r.relata) <= max_order, relations))

    # Separate CES into causes and effects
    separated_ces = rel.separate_ces(ces)

    # Initialize figure
    fig = go.Figure()

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

    # find positions of the mechanisms (if grounded)
    if mezzanine_z_offset == None:
        mezzanine_z_offset = floor_spacing / 2

    if mezzanine_floor_spacing == None:
        mezzanine_floor_spacing = mezzanine_z_offset / len(subsystem)

    if mezzanine_floor_scale == None:
        mezzanine_floor_scale = floor_scale / 2

    # mechanism positions
    pos = [
        grounded_position(
            m.mechanism,
            element_positions,
            jitter=0,
            z_offset=-mezzanine_z_offset if mezzanine else mechanism_z_offset,
            floor_scale=mezzanine_floor_scale if mezzanine else floor_scale,
            floor_spacing=mezzanine_floor_spacing if mezzanine else floor_spacing,
            restrict_ground_floor=False if mezzanine else restrict_ground_floor,
        )
        for m in separated_ces[::2]
    ]
    xm = [p[0] for p in pos]
    ym = [p[1] for p in pos]
    zm = [p[2] for p in pos]

    # get purview positions
    if mezzanine:
        params = [
            (c.purview, -1)
            if c.direction == direction.Direction.CAUSE
            else (c.purview, 1)
            for c in separated_ces
        ]

        coords = np.array(
            [
                grounded_position(
                    p[0],
                    element_positions,
                    x_offset=p[1] * purview_x_offset,
                    jitter=jitter,
                    floor_scale=floor_scale,
                    floor_spacing=floor_spacing,
                    restrict_ground_floor=restrict_ground_floor,
                )
                for p in params
            ]
        )
    # DANGER DO NOT USE THIS FUNCTION:
    elif purview_aligned_with_mechanism:
        params = [
            (c.purview, -1)
            if c.direction == direction.Direction.CAUSE
            else (c.purview, 1)
            for c in separated_ces
        ]

        coords = np.array(
            [
                grounded_position(
                    p[0],
                    element_positions,
                    jitter=jitter,
                    floor_scale=0,
                    floor_spacing=floor_spacing,
                    x=xm[int(i / 2)] + p[1] * purview_x_offset,
                    y=ym[int(i / 2)],
                    restrict_ground_floor=restrict_ground_floor,
                )
                for p, i in zip(params, range(len(params)))
            ]
        )

    elif purview_in_place:
        params = [
            (c.purview, -1)
            if c.direction == direction.Direction.CAUSE
            else (c.purview, 1)
            for c in separated_ces
        ]
        coords = np.array(
            [
                grounded_position(
                    p[0],
                    element_positions,
                    jitter=jitter,
                    x_offset=p[1] * purview_x_offset,
                    floor_scale=floor_scale,
                    floor_spacing=floor_spacing,
                    restrict_ground_floor=restrict_ground_floor,
                )
                for p in params
            ]
        )

    else:
        params = [
            (c.purview, -1)
            if c.direction == direction.Direction.CAUSE
            else (c.purview, 1)
            for c in separated_ces
        ]
        coords = np.array(
            [
                grounded_position(
                    p[0],
                    element_positions,
                    jitter=jitter,
                    floor_scale=floor_scale,
                    floor_spacing=floor_spacing,
                    x=xm[int(i / 2)] + p[1] * purview_x_offset,
                    y=ym[int(i / 2)],
                    z=zm[int(i / 2)] + mechanism_z_offset,
                    restrict_ground_floor=restrict_ground_floor,
                )
                for p, i in zip(params, range(len(params)))
            ]
        )

    # Purviews
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract vertex indices for plotly
    x, y = coords[:, 0], coords[:, 1]

    causes_x, causes_y, effects_x, effects_y = separate_cause_and_effect_for(coords)

    if order_on_z_axis:
        z = np.array([len(c.mechanism) for c in separated_ces])
    else:
        z = coords[:, 2]

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
    mechanism_labels = list(map(label_mechanism, ces))
    mechanism_state_labels = [
        label_mechanism_state(subsystem, distinction) for distinction in ces
    ]
    purview_labels = list(map(label_purview, separated_ces))
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

    # Make mechanism labels
    labels_mechanisms_trace = go.Scatter3d(
        visible=show_mechanism_labels,
        x=xm,
        y=ym,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in zm],
        mode="text",
        text=mechanism_labels,
        name="Mechanism Labels",
        showlegend=True,
        textfont=dict(size=mechanism_labels_size, color="black"),
        hoverinfo="text",
        hovertext=mechanism_hovertext,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(labels_mechanisms_trace)

    # Make mechanism state labels trace
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
    labels_cause_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=causes_x,
        y=causes_y,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in causes_z],
        mode="text",
        text=cause_purview_labels,
        name="Cause Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color="red"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="red"),
    )
    fig.add_trace(labels_cause_purviews_trace)

    # Make effect purview labels trace
    labels_effect_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=effects_x,
        y=effects_y,
        z=[n + (vertex_size_range[1] / 10 ** 3 + labels_z_offset) for n in effects_z],
        mode="text",
        text=effect_purview_labels,
        name="Effect Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color="green"),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="green"),
    )
    fig.add_trace(labels_effect_purviews_trace)

    # Make cause purviews state labels trace
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

    intersectionCount = 0  # A flag and a counter for the times there is a check for intersection and it is found.
    # Plot distinction links (edge connecting cause, mechanism, effect vertices)
    coords_links = (
        list(zip(x, flatten(list(zip(xm, xm))))),
        list(zip(y, flatten(list(zip(ym, ym))))),
        list(zip(z, flatten(list(zip(zm, zm))))),
    )

    for i, distinction in enumerate(separated_ces):
        link_trace = go.Scatter3d(
            visible=show_links,
            legendgroup="Links",
            showlegend=True if i == 1 else False,
            x=coords_links[0][i],
            y=coords_links[1][i],
            z=coords_links[2][i],
            mode="lines",
            name="Links",
            line_width=link_width,
            line_color="brown",
            hoverinfo="skip",
            # hovertext=hovertext_relation(relation),
        )

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
                        show_edges,
                        legend_nodes,
                        two_relations_coords,
                        two_relations_sizes,
                        relation_color,
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

                # Make all 2-relations traces and legendgroup
                edge_two_relation_trace = go.Scatter3d(
                    visible=show_edges,
                    legendgroup="All 2-Relations",
                    showlegend=True if r == 0 else False,
                    x=two_relations_coords[0][r],
                    y=two_relations_coords[1][r],
                    z=two_relations_coords[2][r],
                    mode="lines",
                    # name=label_relation(relation),
                    name="All 2-Relations",
                    line_width=two_relations_sizes[r],
                    line_color=relation_color,
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
                        node_labels,
                        go,
                        fig,
                        legend_nodes,
                        x,
                        y,
                        z,
                        i,
                        j,
                        k,
                        three_relations_sizes,
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

                triangle_three_relation_trace = go.Mesh3d(
                    visible=show_mesh,
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
                    intensity=np.linspace(0, 1, len(x), endpoint=True),
                    opacity=three_relations_sizes[r],
                    colorscale="viridis",
                    showscale=False,
                    name="All 3-Relations",
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )
                fig.add_trace(triangle_three_relation_trace)

        # Create figure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes_range = [
        (min(d) - 1, max(d) + 1)
        for d in (np.append(x, xm), np.append(y, ym), np.append(z, zm))
    ]

    axes = [
        dict(
            showbackground=False,
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
        title=f"{network_name} Q-STRUCTURE",
        title_font_size=30,
        legend=dict(
            title=dict(
                text="Trace legend (click trace to show/hide):",
                font=dict(color="black", size=15),
            )
        ),
        autosize=True,
        # height=plot_dimentions[0],
        # width=plot_dimentions[1],
    )

    # Apply layout
    fig.layout = layout
    from matplotlib._png import read_png

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
    """
        encoded_image = base64.b64encode(open(digraph_filename, "rb").read())
        digraph_coords = (0, 0.75)
        digraph_size = (0.2, 0.3)

        fig.add_layout_image(
            dict(
                name="Causal model",
                source="data:image/png;base64,{}".format(encoded_image.decode()),
                #         xref="paper", yref="paper",
                x=digraph_coords[0],
                y=digraph_coords[1],
                sizex=digraph_size[0],
                sizey=digraph_size[1],
                xanchor="left",
                yanchor="top",
            )
        )

        draft_template = go.layout.Template()
        draft_template.layout.annotations = [
            dict(
                name="Causal model",
                text="Causal model",
                opacity=1,
                font=dict(color="black", size=20),
                xref="paper",
                yref="paper",
                x=digraph_coords[0],
                y=digraph_coords[1] + 0.05,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
            )
        ]

        fig.update_layout(
            margin_l=left_margin,
            template=draft_template,
            annotations=[dict(templateitemname="Causal model", visible=True)],
        )

"""

