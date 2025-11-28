#! Simulated annealing on top of the chain

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from mcmc.chain import MCMCChain
import numpy as np


@dataclass
class AnnealingSchedule:
    """
    Container for annealing schedule
    """

    def __init__(
        self,
        temperatures: List[float],
        no_annealing: bool = False,
        num_steps_per_temp: int = 10000,
    ):
        self.temperatures = temperatures
        self.no_annealing = no_annealing
        self.num_steps_per_temp = num_steps_per_temp

    def run(self, mcmc_chain: MCMCChain, rng: np.random.Generator) -> None:
        """
        Run the annealing schedule on the given MCMC chain
        """

        if self.no_annealing:
            # ? Run at constant temperature (first temperature)
            T = self.temperatures[0]
            for step in range(self.num_steps_per_temp):
                mcmc_chain.step(rng, T)
                if step % 1000 == 0:
                    print(
                        f"Step {step} at T={T}, Energy={mcmc_chain.energy_model.current_energy}"
                    )
            print("Annealing complete.")
            # Show the final position of all

        else:
            # ? Run through the temperature schedule
            for T in self.temperatures:
                for step in range(self.num_steps_per_temp):
                    mcmc_chain.step(rng, T)
                    if step % 1000 == 0:
                        print(
                            f"Step {step} at T={T}, Energy={mcmc_chain.energy_model.current_energy}"
                        )
