import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(N: int, mode_init: str, state_type: str, results: list) -> None:
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

    # 🏆 Identify the Global Winner and Loser Indices
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

    # 🥇 Highlight the BEST RUN (Global Winner)
    winner_energy = results[winner_index]["energy"]
    ax_energy.plot(
        winner_energy,
        color="green",
        linewidth=0.5,
        linestyle="-",
        label=f"BEST Run (Seed {results[winner_index].get('seed', 'N/A')} - Final E={final_energies[winner_index]:.2f})",
    )

    # 💀 Highlight the WORST RUN (Global Loser)
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

    # --- Subplot 2: Attacked Queens (Conflicts) ---
    ax_queens = axes[1]

    # Plot ALL individual runs (faint and transparent)
    for res in results:
        ax_queens.plot(
            res["attacked_queens"], color="tab:blue", alpha=0.1, linewidth=0.8
        )

    # Plot the Mean Run
    if has_aggregate:
        ax_queens.plot(mean_queens, color="navy", linewidth=2, label="Mean Conflicts")

    # 🥇 Highlight the BEST RUN (Conflict Trajectory)
    winner_queens = results[winner_index]["attacked_queens"]
    ax_queens.plot(
        winner_queens,
        color="green",
        linewidth=0.5,
        linestyle="-",
        label=f"BEST Run Trajectory, final numbers: {winner_queens[-1]:.2f}",
    )

    # 💀 Highlight the WORST RUN (Conflict Trajectory)
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

    fig.suptitle(
        f"Simulated Annealing Results for {N}-Stacks {state_type if state_type!='stack'else ""} Problem, with {len(results)} simulations and '{mode_init}' Initialization",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()
