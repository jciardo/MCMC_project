import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib.text import TextPath


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
    plt.style.use("seaborn-v0_8-darkgrid")
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

    title = (
        f"Simulated Annealing Results for {N}-Stacks {state_type if state_type!='stack'else ""} Problem, with {len(results)} simulations and '{mode_init}' Initialization"
        if mode_init != "noisy_latin_square"
        else f"Simulated Annealing Results for {N}-Stacks {state_type if state_type!='stack'else ''} Problem, with {len(results)} simulations and {mode_init} Initialization (p={noisy_p})"
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
            ax.plot(mean_energy, color="red", linewidth=2, label="Mean Energy")

        #?ax.set_title("Energy Evolution Across Simulations") #? Not for the report
        ax.set_xlabel("Steps")
        ax.set_ylabel("Energy")
        ax.grid(True, linestyle="--", alpha=0.7)
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
