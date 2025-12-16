from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from mcmc.chain import MCMCChain
from mcmc.proposals import SingleStackMove, SingleConstraintStackMove, BlockShuffleMove
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm


class DummyPbar:
    """A dummy progress bar that does nothing."""

    def __init__(self, total=None, desc=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, n=1):
        pass

    def set_postfix(self, ordered_dict=None, refresh=True):
        pass

    def close(self):
        pass


class AnnealingSchedule(ABC):
    """
    Abstract base class for annealing schedules.
    """

    @abstractmethod
    def get_temperature(self) -> float:
        """Return the current temperature."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Advance the schedule by one step."""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Return True if the schedule is finished."""
        pass


@dataclass
class GeometricSchedule(AnnealingSchedule):
    """
    T(k+1) = alpha * T(k)
    """

    T_initial: float
    alpha: float
    max_steps: int

    _current_step: int = field(init=False, default=0)
    _current_T: float = field(init=False)

    def __post_init__(self):
        if not (0 < self.alpha < 1):
            raise ValueError("Alpha must be in (0, 1) for cooling.")
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T

    def step(self) -> None:
        if self._current_step < self.max_steps:
            self._current_T *= self.alpha
            self._current_step += 1

    def is_finished(self) -> bool:
        return self._current_step >= self.max_steps  # or self._current_T <= 1e-9


@dataclass
class LinearSchedule(AnnealingSchedule):
    """
    T(k+1) = T(k) - decrement
    """

    T_initial: float
    decrement: float
    min_temp: float = 0.001

    _current_T: float = field(init=False)

    def __post_init__(self):
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T

    def step(self) -> None:
        self._current_T -= self.decrement

    def is_finished(self) -> bool:
        return self._current_T <= self.min_temp


@dataclass
class AdaptiveSchedule(AnnealingSchedule):
    T_initial: float
    alpha: float
    min_temp: float = 0.001
    max_steps: int = 1000000

    # Parameters for stagnation detection
    stagnation_limit: int = 8000
    reheat_ratio: float = 0.3

    # Internal State
    _current_T: float = field(init=False)
    _stagnation_counter: int = field(init=False, default=0)
    _improving_streak: bool = field(init=False, default=False)
    _max_steps: int = field(init=False)
    _current_step: int = field(init=False, default=0)

    _local_best_energy: float = field(init=False, default=float("inf"))

    def __post_init__(self):
        self._current_T = self.T_initial
        self._max_steps = self.max_steps
        self._current_step = 0
        self._local_best_energy = float("inf")

    def get_temperature(self) -> float:
        return self._current_T

    def update_metrics(self, current_energy: float, global_best_energy: float) -> None:

        if current_energy < self._local_best_energy:
            self._local_best_energy = current_energy
            self._stagnation_counter = 0
            self._improving_streak = True
        else:
            self._improving_streak = False
            self._stagnation_counter += 1

    def step(self) -> None:
        self._current_step += 1

        # Cruising
        if self._improving_streak:
            return

        # Reheating
        if self._stagnation_counter > self.stagnation_limit:
            # Reset Temperature
            self._current_T = self.T_initial * self.reheat_ratio

            self._stagnation_counter = 0
            self._local_best_energy = float("inf")
            return

        # Normal Cooling
        if self._current_T > self.min_temp:
            self._current_T *= self.alpha

    def is_finished(self) -> bool:
        return self._current_step >= self._max_steps


def run_simulated_annealing(
    mcmc_chain,
    schedule: "AnnealingSchedule",
    rng: np.random.Generator,
    verbose_every: int = 1000,
    detailed_stats: bool = False,
    is_watched: bool = False,
    re_heat: bool = False,
) -> dict:
    """
    Run the simulated annealing process using the provided MCMC chain and cooling schedule.
    Stores the temperature, energy, number of attacked queens, and queen positions at each step.
    Returns a dict with the full trajectory.
    """
    iteration = 0
    history = {
        "temperature": [],
        "energy": [],
        "attacked_queens": [],
        "positions": [],
        "best_energy": None,
        "best_positions": None,
    }
    best_energy = mcmc_chain.energy_model.current_energy
    best_positions = list(mcmc_chain.state.iter_queens())

    if is_watched:
        pbar_context = tqdm(total=schedule.max_steps, desc="Simulated Annealing")
    else:
        pbar_context = DummyPbar()

    # The loop continues as long as the cooling schedule dictates (T > T_min or steps < max_steps)
    with pbar_context as pbar:
        while not schedule.is_finished():
            # 1. Retrieve the current temperature T
            T = schedule.get_temperature()

            # 2. Perform one MCMC step (State transition with Metropolis acceptance)
            mcmc_chain.step(rng, T)

            # 3. Update the temperature for the next iteration
            if re_heat == True:
                schedule.update_metrics(
                    mcmc_chain.energy_model.current_energy, best_energy
                )
            schedule.step()

            # Collect stats at every step
            current_energy = mcmc_chain.energy_model.current_energy

            # positions = full configuration at this step (INTERNAL 1-based coords)
            positions = list(mcmc_chain.state.iter_queens())
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions #! added

            attacked = mcmc_chain.energy_model.count_attacked_queens(mcmc_chain.state)
            positions = list(mcmc_chain.state.iter_queens())

            history["temperature"].append(T)
            history["energy"].append(current_energy)
            history["attacked_queens"].append(attacked)
            history["positions"].append(positions)

            # 4. Monitoring and Logging (every 1000 steps)
            if iteration % verbose_every == 0:
                if not detailed_stats:
                    print(
                        f"Iter {iteration}, "
                        f"T={T:.4f}, "
                        f"Energy={current_energy:.4f}, "
                        f"Attacked Queens={attacked}"
                    )
                else:
                    attacked_stats = mcmc_chain.energy_model.attacked_stats(
                        mcmc_chain.state
                    )
                    print(
                        f"Iter {iteration}, "
                        f"T={T:.4f}, "
                        f"Energy={current_energy:.4f}, "
                        f'Attacked Queens={attacked_stats["attacked_queens"]}, '
                        f'mean attack={attacked_stats["mean_attacks"]:.2f}, '
                        f'most attacked={attacked_stats["max_attacks"]:.2f}'
                    )

                # Early stopping condition if the perfect solution (E=0) is found
                if current_energy == 0:
                    print(f"Solution found at iteration {iteration}!")
                    pbar.close()
                    return history

            iteration += 1
            pbar.update(1)

    print("Simulated Annealing complete.")
    history["best_energy"] = best_energy
    history["best_positions"] = best_positions
    pbar.close()
    return history


def calibrate_initial_temperature(
    mcmc_chain: "MCMCChain",
    target_acceptance_rate: float = 0.8,
    n_samples: int = 5000,
    rng: np.random.Generator = np.random.default_rng(),
) -> float:
    """
    Calibrates the initial temperature T0 for Simulated Annealing (SA).
    """
    energy_increases: list[float] = []

    # Start with a copy of the current state
    current_state = mcmc_chain.state.copy()
    energy_model = mcmc_chain.energy_model
    proposal = mcmc_chain.proposal

    # --- 1. Sampling Phase (Pure Random Walk) ---
    for _ in range(n_samples):

        # 1a. Propose a move
        move, delta_E = proposal.propose(current_state, energy_model, rng)

        # 1b. Store degradation
        if delta_E > 0:
            energy_increases.append(delta_E)

        # 2. Apply the proposed move (Must handle ALL move types)

        if isinstance(move, SingleStackMove):
            energy_model.apply_move(
                state=current_state,
                i=move.i,
                j=move.j,
                k_new=move.k_new,
                delta_E=delta_E,
            )

        elif isinstance(move, SingleConstraintStackMove):
            energy_model.apply_move(
                state=current_state,
                i1=move.i1,
                i2=move.i2,
                j=move.j,
                k1=move.k1,
                k2=move.k2,
                delta_E=delta_E,
            )

        elif isinstance(move, BlockShuffleMove):
            for (i, j), k_new in zip(move.indices, move.new_heights):
                if hasattr(current_state, "stacks"):
                    current_state.stacks[i - 1][j - 1] = k_new
                elif hasattr(current_state, "set_height"):
                    current_state.set_height(i, j, k_new)
            # Re-sync energy model
            energy_model.initialize(current_state)

    if not energy_increases:
        print("Warning: No energy-increasing moves found. Defaulting T0 to 10.0.")
        return 10.0

    mean_energy_increase = np.mean(energy_increases)
    T0 = -mean_energy_increase / np.log(target_acceptance_rate)

    print(f"Average energy degradation (mean Delta E > 0): {mean_energy_increase:.2f}")
    return T0
