import numpy as np
import math
import pickle as pkl
from scipy.special import comb

from pyphi import visualize as viz
from pyphi import relations as rel

import matplotlib.pyplot as plt
import string

import random
import imageio

# TODO: make sure all packages are installed in the environment

### TOP LEVEL FUNCTIONS
def spring_force_simulation(
    ces,
    rels,
    distinction_relation_force_balance=1,
    friction_factor=5,
    repulsion_factor=1,
    spring_factor=1,
    T=300,
    delta=0.01,
    mass=None,
    immovable=(),
    initial_positions="singularity",
    return_path=False,
    adjust_force_by_phi=True,
):
    # Get the spring forces between all terms
    spring_constants = get_interactions(
        ces, rels, distinction_relation_force_balance, adjust_force_by_phi
    )

    # initialize positions based on input argument
    x, y, z = get_initial_positions(ces, initial_positions)

    # simulating the spring force system
    path = equilibrate(
        spring_constants,
        x,
        y,
        z,
        friction_factor=friction_factor,
        repulsion_factor=repulsion_factor,
        spring_factor=spring_factor,
        T=T,
        delta=delta,
        mass=mass,
        immovable=immovable,
    )

    if return_path:
        return path
    else:
        mech_coords = path[-1, :, : len(ces)]
        purv_coords = path[-1, :, len(ces) :]
        return mech_coords, purv_coords


### SIMULATION FUNCTIONS
def compute_forces(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    spring_constants,
    friction_factor=1,
    repulsion_factor=1,
    spring_factor=1,
):

    N = len(x)
    Fx = []
    Fy = []
    Fz = []
    for i in range(N):

        F = np.zeros(3)
        for j in range(N):
            if i != j:
                # computing distance
                distance_vector = [x[i] - x[j], y[i] - y[j], z[i] - z[j]]
                distance = np.sqrt(np.sum([d ** 2 for d in distance_vector]))

                # making sure distance is not 0
                if distance < 0.1:
                    distance_vector = 0.05 + np.random.rand(3) * 0.1
                    distance = np.sqrt(np.sum([d ** 2 for d in distance_vector]))
                unit_vector = [d / distance for d in distance_vector]

                # computing spring forces
                spring_force_magnitude = (
                    distance * spring_constants[i, j] * spring_factor
                )
                spring_force_vector = np.array(
                    [-spring_force_magnitude * u for u in unit_vector]
                )

                # computing repulsive force
                repulsive_force_magnitude = (
                    np.sum(spring_constants[j, :])
                    * repulsion_factor
                    / (distance ** 2)
                    / N
                )
                repulsive_force_vector = np.array(
                    [repulsive_force_magnitude * u for u in unit_vector]
                )

                # summing total forces
                F += spring_force_vector + repulsive_force_vector

        # computing friction force
        total_speed = np.sqrt(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2)
        friction_magnitude = total_speed * friction_factor
        friction_vector = np.array(
            [
                friction_magnitude * v / total_speed if total_speed > 0 else 0
                for v in [vx[i], vy[i], vz[i]]
            ]
        )

        Fx.append(F[0] - friction_vector[0])
        Fy.append(F[1] - friction_vector[1])
        Fz.append(F[2] - friction_vector[2])

    return Fx, Fy, Fz


def update_position_and_velocity(x, y, z, vx, vy, vz, Fx, Fy, Fz, delta, mass=None):
    if mass == None:
        mass = np.ones(len(x))

    for i in range(len(x)):
        ax = Fx[i] / mass[i]
        ay = Fy[i] / mass[i]
        az = Fz[i] / mass[i]
        vx[i] = vx[i] + (ax * delta)
        vy[i] = vy[i] + (ay * delta)
        vz[i] = vz[i] + (az * delta)
        x[i] = x[i] + (vx[i] * delta)
        y[i] = y[i] + (vy[i] * delta)
        z[i] = z[i] + (vz[i] * delta)

    return x, y, z, vx, vy, vz


def equilibrate(
    spring_constants,
    x,
    y,
    z=None,
    vx=None,
    vy=None,
    vz=None,
    friction_factor=2,
    repulsion_factor=1,
    spring_factor=1,
    T=100,
    delta=0.1,
    mass=None,
    immovable=[],
):
    if z is None:
        z = np.zeros(len(x))
    if vx is None:
        vx = np.zeros(len(x))
    if vy is None:
        vy = np.zeros(len(x))
    if vz is None:
        vz = np.zeros(len(x))

    path = []
    for t in range(T):
        Fx, Fy, Fz = compute_forces(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            spring_constants,
            friction_factor,
            repulsion_factor,
            spring_factor,
        )

        # removing forces and velocities from immovable components
        for i in immovable:
            Fx[i], Fy[i], Fz[i], vx[i], vy[i], vz[i] = 0, 0, 0, 0, 0, 0

        x, y, z, vx, vy, vz = update_position_and_velocity(
            x, y, z, vx, vy, vz, Fx, Fy, Fz, delta, mass
        )

        path.append([x.copy(), y.copy(), z.copy()])

    return np.array(path)


### SETUP FUNCTIONS
def initialize_positions(
    ces, center=(1, 1), z=0, radius=1, purview_scale=0.9, mech_base=True
):

    if mech_base:
        aux = construct_mechanism_base(len(ces[-1].mechanism), center, 0.1, 0.1)
        mechs = [m for mm in aux for m in mm]
    else:
        mechs = viz.regular_polygon(
            len(ces), center=center, angle=0, z=0, radius=radius, scale=1
        )

    purviews = viz.regular_polygon(
        2 * len(ces),
        center=center,
        angle=math.pi / (len(ces)),
        z=0,
        radius=radius,
        scale=purview_scale,
    )

    xyz = np.array(mechs + purviews)

    return (xyz[:, 0], xyz[:, 1], xyz[:, 2])


def get_interactions(ces, rels, adjustment_factor=1, adjust_force_by_phi=True):

    separated_ces = rel.separate_ces(ces)
    N = len(ces)
    M = len(separated_ces)

    # denote what components each purview will be affected by
    # first the mechanisms
    features_mechs = np.zeros((M, N))
    for i in range(N):
        features_mechs[2 * i : 2 * i + 2, i] = 1

    # next the relations
    features_rels = viz.feature_matrix(rel.separate_ces(ces), rels)
    features_rels = viz.feature_matrix(rel.separate_ces(ces), rels)
    # features now contains information about any distinction or relation each purview is associated to

    # next, we exchange the 1's with phi values
    if adjust_force_by_phi:
        # frist for distinctions
        for i in range(M):
            features_mechs[i, :] *= separated_ces[i].phi

        for i in range(len(rels)):
            features_rels[:, i] *= rels[i].phi / len(rels[i].relata)

    # getting the interactions between constituents
    interactions = np.zeros((N + M, N + M))

    # filling with mechanism-purview interactions
    interactions[N:, :N] = features_mechs
    interactions[:N, N:] = np.transpose(features_mechs)

    # now for the purview-purview interactions
    purview_purview = np.zeros((M, M))
    for i, feature in enumerate(features_rels):
        for r in feature.nonzero()[0]:
            for j in features_rels[:, r].nonzero()[0]:
                if not i == j:
                    purview_purview[i, j] += feature[r]
                    purview_purview[j, i] += feature[r]

    # renormalizing to have same max as mechanisms interactions
    purview_purview = (
        adjustment_factor
        * np.max(interactions[N:, :N])
        * purview_purview
        / (np.max(np.sum(purview_purview, axis=1)))
    )

    interactions[N:, N:] = purview_purview

    return interactions / interactions.mean()


def construct_mechanism_base(N, base_center, base_floor_height, base_scale):
    return [
        viz.regular_polygon(
            int(comb(N, k)),
            center=base_center,
            z=((k / N) * base_floor_height),
            scale=base_scale,
        )
        for k in range(1, N + 1)
    ]


def get_initial_positions(ces, initial_positions):

    if initial_positions == "singularity":
        x = np.zeros(3 * len(ces))
        y = np.zeros(3 * len(ces))
        z = np.zeros(3 * len(ces))

    elif initial_positions == "mechanisms inside":
        x, y, z = initialize_positions(
            ces, center=(0, 0), z=0, radius=0.1, purview_scale=3, mech_base=False
        )

    elif initial_positions == "mechanisms outside":
        x, y, z = initialize_positions(
            ces, center=(0, 0), z=0, radius=2, purview_scale=0.9, mech_base=False
        )

    elif initial_positions == "mechanism core":

        aux = construct_mechanism_base(len(ces[-1].mechanism), (0, 0), 0.1, 0.1)
        mechs = [m for mm in aux for m in mm]
        purviews = viz.regular_polygon(
            2 * len(ces),
            center=(0, 0),
            angle=math.pi / (len(ces)),
            z=0,
            radius=1,
            scale=1,
        )
        xyz = np.array(mechs + purviews)
        x, y, z = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    else:
        # assuming intial positions are given
        x = initial_positions[0, :]
        y = initial_positions[1, :]
        z = initial_positions[2, :]

    return x, y, z


### PLOTTING FUNCTIONS
def plot_simulated_ces(
    system, ces, rels, simulation_kwargs=dict(), plotting_kwargs=dict(),
):
    print("Running physics simulation on the components of the CES")
    user_mechanism_coords, user_purview_coords = spring_force_simulation(
        ces, rels, return_path=False, **simulation_kwargs,
    )

    print("Plotting the CES")
    fig = viz.plot_ces_epicycles(
        system,
        ces,
        rels,
        user_mechanism_coords=user_mechanism_coords,
        user_purview_coords=user_purview_coords,
        **plotting_kwargs,
    )
    return fig


def get_ces_gif(
    file_path,
    system,
    ces,
    rels,
    simulation_kwargs=dict(),
    plotting_kwargs=dict(),
    time_grain=5,
    plot_final=True,
):

    print("Running physics simulation on the components of the CES")
    path = spring_force_simulation(ces, rels, return_path=True, **simulation_kwargs,)

    print("Creating the gif")

    times = range(0, len(path), time_grain)
    filenames = []
    for t in times:
        mech_coords = path[t, :, : len(ces)]
        purv_coords = path[t, :, len(ces) :]

        name = file_path + "frame" + str(t) + ".png"
        fig = viz.plot_ces_epicycles(
            system,
            ces,
            rels,
            user_mechanism_coords=mech_coords,
            user_purview_coords=purv_coords,
            link_width_range=(1, 3),
            eye_coordinates=(0, 0, 1),
            mechanism_labels_size=8,
            purview_labels_size=8,
            mechanism_label_position="middle center",
            purview_label_position="middle center",
            show_purview_labels=False,
            save_plot_to_html=False,
            png_name=name,
            showlegend=False,
            show_mechanism_labels=False,
            show_mechanism_state_labels=False,
        )
        filenames.append(name)

    with imageio.get_writer(file_path + "new.gif", mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    if plot_final:
        print("Plotting the final CES")
        fig = viz.plot_ces_epicycles(
            system,
            ces,
            rels,
            network_name=file_path + "final",
            user_mechanism_coords=path[-1, :, : len(ces)],
            user_purview_coords=path[-1, :, len(ces) :],
            **plotting_kwargs,
        )

    return fig
