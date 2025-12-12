import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib.text import TextPath
from typing import Dict, List


import concurrent.futures
import os


def plot_results(
    N: int,
    mode_init: str,
    state_type: str,
    results: list,
    noisy_p: float = None,
    plot_cube: bool = False,
    use_symbols: bool = True,
    only_energy: bool = True,
) -> None:
    """
    Plots the results from multiple Simulated Annealing runs.
    Highlights the Mean, the Best Run (global minimum final energy), and the Worst Run.

    Args:
        results: A list of dictionaries, where each dict must contain
                "energy" and "attacked_queens" (lists of values per step).
    """

    # 1. Set Plot Style
    #plt.style.use("seaborn-v0_8-darkgrid")
    plt.style.use("default")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Aggregate Calculation and Winner Identification ---
    final_energies = np.array([r["energy"][-1] for r in results])

    # Identify the Global Winner and Loser Indices
    winner_index = np.argmin(final_energies)
    loser_index = np.argmax(final_energies)

    # Check if simulation lengths are consistent for MEAN/aggregate calculations
    has_aggregate = False
    try:
        all_energies = np.array([r["energy"] for r in results])
        all_queens = np.array([r["attacked_queens"] for r in results])

        mean_energy = np.mean(all_energies, axis=0)
        mean_queens = np.mean(all_queens, axis=0)

        has_aggregate = True
    except ValueError:
        print(
            "Warning: Simulation lengths are inconsistent; Mean trace will be skipped."
        )

    # --- Subplot 1: Energy over Steps ---
    ax_energy = axes[0]

    # Plot ALL individual runs (faint and transparent)
    for res in results:
        ax_energy.plot(res["energy"], color="gray", alpha=0.1, linewidth=0.8)

    # Plot the Mean Run
    if has_aggregate:
        ax_energy.plot(mean_energy, color="red", linewidth=2, label="Mean Energy")

    # Highlight the BEST RUN (Global Winner)
    winner_energy = results[winner_index]["energy"]
    ax_energy.plot(
        winner_energy,
        color="green",
        linewidth=0.5,
        linestyle="-",
        label=f"BEST Run (Seed {results[winner_index].get('seed', 'N/A')} - Final E={final_energies[winner_index]:.2f})",
    )

    # Highlight the WORST RUN (Global Loser)
    loser_energy = results[loser_index]["energy"]
    ax_energy.plot(
        loser_energy,
        color="purple",
        linewidth=0.5,
        linestyle="--",
        label=f"WORST Run (Seed {results[loser_index].get('seed', 'N/A')} - Final E={final_energies[loser_index]:.2f})",
    )

    ax_energy.set_title("Energy Evolution Across Simulations", fontsize=14)
    ax_energy.set_xlabel("Steps")
    ax_energy.set_ylabel("Energy")
    ax_energy.legend(loc="upper right", fontsize=8)
    ax_energy.grid(True, linestyle="--", alpha=0.7)

    # Subplot 2: Attacked Queens (Conflicts)
    ax_queens = axes[1]

    # Plot ALL individual runs (faint and transparent)
    for res in results:
        ax_queens.plot(
            res["attacked_queens"], color="tab:blue", alpha=0.1, linewidth=0.8
        )

    # Plot the Mean Run
    if has_aggregate:
        ax_queens.plot(mean_queens, color="navy", linewidth=2, label="Mean Conflicts")

    # Highlight the BEST RUN (Conflict Trajectory)
    winner_queens = results[winner_index]["attacked_queens"]
    ax_queens.plot(
        winner_queens,
        color="green",
        linewidth=0.5,
        linestyle="-",
        label=f"BEST Run Trajectory, final numbers: {winner_queens[-1]:.2f}",
    )

    # Highlight the WORST RUN (Conflict Trajectory)
    loser_queens = results[loser_index]["attacked_queens"]
    ax_queens.plot(
        loser_queens,
        color="purple",
        linewidth=0.5,
        linestyle="--",
        label=f"WORST Run Trajectory, final numbers: {loser_queens[-1]:.2f}",
    )

    ax_queens.set_title("Attacked Queens over Steps", fontsize=14)
    ax_queens.set_xlabel("Steps")
    ax_queens.set_ylabel("Number of Conflicts")
    ax_queens.legend(loc="upper right", fontsize=8)
    ax_queens.grid(True, linestyle="--", alpha=0.7)

    state_label = state_type if state_type != "stack" else ""

    if mode_init != "noisy_latin_square":
        init_label = f"'{mode_init}' Initialization"
    else:
        init_label = f"{mode_init} Initialization (p={noisy_p})"

    title = (
        f"Simulated Annealing Results for {N}-Stacks {state_label} "
        f"Problem, with {len(results)} simulations and {init_label}"
    )

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if only_energy :
        #! SAVE ONLY Energy vs Steps
        #! Minimal fix but very verbose
        fig_energy, ax = plt.subplots(figsize=(7, 5))

        for res in results:
            ax.plot(res["energy"], color="gray", alpha=0.08, linewidth=0.8)

        ax.plot(winner_energy, color="green", linewidth=0.8, label="Best Run", alpha=0.35)
        ax.plot(loser_energy, color="purple", linestyle="--", linewidth=0.8, label="Worst Run", alpha=0.35)

        if has_aggregate:
            ax.plot(mean_energy, color="firebrick", linewidth=2, label="Mean Energy")

        #?ax.set_title("Energy Evolution Across Simulations") #? Not for the report
        ax.set_xlabel("Steps")
        ax.set_ylabel("Energy")
        #ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        fig_energy.tight_layout()
        fig_energy.savefig("plots/energy_vs_steps.png", dpi=300)
        plt.close(fig_energy)


    plt.show()
    # Optionally plot the 3D cube of the best solution
    if plot_cube:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        queen_positions = results[winner_index]["positions"][-1]
        cube_size = N
        edges = [
            ([0, 0, 0], [1, 0, 0]),
            ([0, 0, 0], [0, 1, 0]),
            ([0, 0, 0], [0, 0, 1]),
            ([1, 1, 1], [0, 1, 1]),
            ([1, 1, 1], [1, 0, 1]),
            ([1, 1, 1], [1, 1, 0]),
            ([0, 1, 0], [0, 1, 1]),
            ([0, 1, 0], [1, 1, 0]),
            ([1, 0, 0], [1, 0, 1]),
            ([1, 0, 0], [1, 1, 0]),
            ([0, 0, 1], [0, 1, 1]),
            ([0, 0, 1], [1, 0, 1]),
        ]
        for s, e in edges:
            ax.plot3D(
                [s[0] * cube_size, e[0] * cube_size],
                [s[1] * cube_size, e[1] * cube_size],
                [s[2] * cube_size, e[2] * cube_size],
                color="black",
            )

        try:
            n_strates = int(cube_size)
            cmap = plt.cm.get_cmap("viridis", max(1, n_strates))
        except Exception:
            cmap = plt.cm.get_cmap("viridis")

        font_name = "DejaVu Sans"
        fp = FontProperties(family=font_name)

        def font_has_glyph(font_prop, glyph):
            try:
                _ = TextPath((0, 0), glyph, prop=font_prop)
                return True
            except Exception:
                return False

        glyph = "♛"
        glyph_supported = font_has_glyph(fp, glyph)

        use_textpath_fallback = False
        if use_symbols and glyph_supported:
            for x, y, z in queen_positions:
                color = (
                    cmap(int(z))
                    if isinstance(cmap, plt.cm.ScalarMappable)
                    else cmap(int(z))
                )
                ax.text(
                    x,
                    y,
                    z,
                    glyph,
                    fontproperties=fp,
                    fontsize=18,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                )
        elif use_symbols and not glyph_supported:
            use_textpath_fallback = True
            tp = TextPath((0, 0), glyph, prop=FontProperties(size=40))
            for x, y, z in queen_positions:
                color = (
                    cmap(int(z))
                    if isinstance(cmap, plt.cm.ScalarMappable)
                    else cmap(int(z))
                )
                ax.scatter([x], [y], [z], marker=tp, s=200, color=color)
        else:
            for x, y, z in queen_positions:
                color = (
                    cmap(int(z))
                    if isinstance(cmap, plt.cm.ScalarMappable)
                    else cmap(int(z))
                )
                ax.scatter([x], [y], [z], s=80, color=color, marker="o")

        # --- Ajustements ---
        ax.set_xlim(0, cube_size)
        ax.set_ylim(0, cube_size)
        ax.set_zlim(0, cube_size)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"3D Cube Representation of Best Solution for N = {N}", fontsize=14
        )

        if isinstance(cmap, plt.cm.ScalarMappable):
            handles = []
            labels = []
            for z in range(min(n_strates, 10)):
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap(z),
                        markersize=8,
                    )
                )
                labels.append(f"z={z}")
            ax.legend(
                handles,
                labels,
                title="z",
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
            )

        plt.tight_layout()
        plt.show()



def _aggregate_time_series(results, key="energy"):
    """
    Given a list of result dicts, aggregate a time series (e.g. energy)
    across runs: returns (steps, mean, std).

    We truncate all runs to the minimum common length to be safe.
    """
    # Keep only successful runs
    clean = [r for r in results if isinstance(r, dict) and key in r and "error" not in r]
    if not clean:
        raise ValueError(f"No valid runs found with key '{key}'.")

    lengths = [len(r[key]) for r in clean]
    T = min(lengths)

    data = np.stack([np.array(r[key][:T], dtype=float) for r in clean], axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    steps = np.arange(T)
    return steps, mean, std


def plot_sa_vs_constant(
    N: int,
    sa_results: list[dict],
    const_results: list[dict],
    mode_init: str,
    state_type: str,
    noisy_p: float | None = None,
    save_path: str | None = None,
) -> None:
    """
    Compare simulated annealing vs constant-temperature MCMC for fixed N.

    Parameters
    ----------
    N : int
        Board size.
    sa_results : list of dict
        Histories from runs using a cooling schedule (e.g. GeometricSchedule).
    const_results : list of dict
        Histories from runs using a constant-temperature schedule.
    mode_init : str
        Initialization mode (for title / annotation).
    state_type : str
        'stack' or 'constraint' (for title / annotation).
    noisy_p : float or None
        Noise parameter if using noisy_latin_square (for title / annotation).
    save_path : str or None
        If provided, the figure is saved to this path; otherwise plt.show() is called.
    """
    #plt.style.use("seaborn-v0_8-darkgrid")
    plt.style.use("default")


    # --- Aggregate energies
    steps_sa, mean_sa, std_sa = _aggregate_time_series(sa_results, key="energy")
    steps_ct, mean_ct, std_ct = _aggregate_time_series(const_results, key="energy")

    # Make sure x-axes are aligned by truncating to common length
    T = min(len(steps_sa), len(steps_ct))
    steps = steps_sa[:T]
    mean_sa, std_sa = mean_sa[:T], std_sa[:T]
    mean_ct, std_ct = mean_ct[:T], std_ct[:T]

    # --- Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, mean_sa, label="Simulated annealing (mean)", linewidth=2, color='mediumblue')
    ax.fill_between(
        steps,
        mean_sa - std_sa,
        mean_sa + std_sa,
        alpha=0.25,    
    )#!label="SA ±1 std",

    ax.plot(steps, mean_ct, label="Constant T (mean)", linewidth=2, linestyle="--", color='firebrick')
    ax.fill_between(
        steps,
        mean_ct - std_ct,
        mean_ct + std_ct,
        alpha=0.25,
    ) #!label="Const. T ±1 std",

    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy")

    subtitle = f"N={N}, init={mode_init}, state_type={state_type}"
    if mode_init == "noisy_latin_square" and noisy_p is not None:
        subtitle += f", p={noisy_p}"
    #ax.set_title(f"Energy vs steps with and without simulated annealing\n{subtitle}")

    ax.legend(loc="lower center")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_min_energy_vs_N(
    min_energy_by_N: dict[int, float],
    save_path: str | None = None,
) -> None:
    """
    Plot the minimal energy reached as a function of N.
    Assumes `min_energy_by_N` maps N -> minimal_energy_for_that_N.

    Example:
        min_energy_by_N = {8: 120, 10: 260, 12: 410}

    Parameters
    ----------
    min_energy_by_N : dict[int, float]
        Dictionary mapping board size N to minimal energy.
    save_path : str or None
        If provided, saves the figure; otherwise displays it.
    """

    plt.style.use("default")

    # Sort by N
    Ns = np.array(sorted(min_energy_by_N.keys()))
    mins = np.array([min_energy_by_N[N] for N in Ns], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))

    #ax.plot(Ns, mins, marker="4", linewidth=1,linestyle ='dotted', color='firebrick', alpha = 0.2)
    ax.plot(
    Ns, mins,
    linewidth=1,
    linestyle='dotted',
    color='firebrick',
    alpha=0.2,
    )
    ax.plot(
    Ns, mins,
    marker="H",
    linestyle='none',
    color='firebrick',
    alpha=1.0,
    )
    ax.set_xlabel("Cube $N^3$")
    ax.set_ylabel("Minimal energy reached")
    #ax.set_title("Minimal energy vs board size N")
    #ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def get_best_energy(results: list[dict], use_final: bool = True) -> float:
    """
    Return the best (lowest) energy from a list of run histories.

    If use_final is True:
        use E_final = energy[-1] for each run.
    Otherwise:
        use the best energy along the trajectory: min_t energy[t] for each run.
    """
    best = None

    for r in results:
        if not isinstance(r, dict) or "energy" not in r or "error" in r:
            continue

        energies = np.asarray(r["energy"], dtype=float)
        value = energies[-1] if use_final else energies.min()

        if best is None or value < best:
            best = value

    if best is None:
        raise ValueError("No valid runs found in results.")
    return best

