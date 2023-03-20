import matplotlib
import numpy as np
import pandas as pd

import copy

import seaborn as sns


from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import Patch


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

######### BlackJack Plots


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    argV = copy.deepcopy(V)

    V = defaultdict(float)
    for state, value in argV.items():
        if state[0] <= 21:
            V[state] = value

    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(
        lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y])
    )
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title, fig=None, i=0):
        if fig is None:
            fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, i, projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=matplotlib.cm.coolwarm,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_xlabel("Player Sum")
        ax.set_ylabel("Dealer Showing")
        ax.set_zlabel("Value")
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)

    fig = plt.figure(figsize=(10, 5))
    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title), fig, 1)
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title), fig, 2)
    plt.show()


## Random Walk Plots


def plot_RW(V_dict):
    """Param V is dictionary int[0..7]->float"""

    V_true = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0.0]

    V_arr = np.zeros(7)
    for st in range(7):
        V_arr[st] = V_dict[st]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        range(1, 6),
        V_true[1:-1],
        color="black",
        label="True V for random policy with gamma=1",
    )
    ax.plot(range(1, 6), V_arr[1:-1], label="V")

    ax.legend()
    plt.show()


## Generic Plots


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = (
        pd.Series(stats.episode_rewards)
        .rolling(smoothing_window, min_periods=smoothing_window)
        .mean()
    )
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title(
        "Episode Reward over Time (Smoothed over window size {})".format(
            smoothing_window
        )
    )
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3
