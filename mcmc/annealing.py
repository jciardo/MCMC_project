from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from mcmc.chain import MCMCChain  
from mcmc.proposals import SingleStackMove, SingleConstraintStackMove
from abc import ABC, abstractmethod
import numpy as np




class AnnealingSchedule(ABC):
    """
    Classe abstraite définissant l'interface pour tout calendrier de refroidissement.
    """
    
    @abstractmethod
    def get_temperature(self) -> float:
        """Retourne la température actuelle."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Met à jour la température pour l'itération suivante."""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Indique si le processus de recuit doit s'arrêter."""
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
            raise ValueError("Alpha doit être entre 0 et 1.")
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T

    def step(self) -> None:
        if self._current_step < self.max_steps:
            self._current_T *= self.alpha
            self._current_step += 1

    def is_finished(self) -> bool:
        return self._current_step >= self.max_steps
    


@dataclass
class LinearSchedule(AnnealingSchedule):
    """
    T(k+1) = T(k) - decremenet
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
    


def run_simulated_annealing(
    mcmc_chain,  
    schedule: 'AnnealingSchedule',  # The abstract cooling schedule object
    rng: np.random.Generator
) -> None:
    """
    Run the simulated annealing process using the provided MCMC chain and cooling schedule.
    """
    iteration = 0
    
    # The loop continues as long as the cooling schedule dictates (T > T_min or steps < max_steps)
    while not schedule.is_finished():
        # 1. Retrieve the current temperature T
        T = schedule.get_temperature()
        
        # 2. Perform one MCMC step (State transition with Metropolis acceptance)
        mcmc_chain.step(rng, T) 
        
        # 3. Update the temperature for the next iteration
        schedule.step()
        
        # 4. Monitoring and Logging (every 1000 steps)
        if iteration % 1000 == 0:
            current_energy = mcmc_chain.energy_model.current_energy
            print(f"Iter {iteration}: T={T:.4f}, Energy={current_energy:.4f}")
            
            # Early stopping condition if the perfect solution (E=0) is found
            if current_energy == 0:
                print(f"Solution found at iteration {iteration}!")
                break
                
        iteration += 1
        
    print("Simulated Annealing complete.")



def calibrate_initial_temperature(
    mcmc_chain: 'MCMCChain', 
    target_acceptance_rate: float = 0.8, 
    n_samples: int = 1000,
    rng: np.random.Generator = np.random.default_rng()
) -> float:
    """
    Calibrates the initial temperature T0 for Simulated Annealing (SA).
    """
    energy_increases: List[float] = []
    
    # Start with a copy of the current state to avoid modifying the chain state
    current_state = mcmc_chain.state.copy()
    energy_model = mcmc_chain.energy_model
    proposal = mcmc_chain.proposal

    # --- 1. Sampling Phase (Pure Random Walk) ---
    for _ in range(n_samples):
        
        # 1a. Propose a move and get the energy difference (Delta E)
        move, delta_E = proposal.propose(current_state, energy_model, rng)
       
        # 1b. Store only moves that cause energy degradation (Delta E > 0)
        if delta_E > 0:
            energy_increases.append(delta_E)
            
        # 2. Apply the proposed move to create the next state for the random walk. 

        energy_model.apply_move(
            current_state,
            *(move.i, move.j, move.k_new) if isinstance(move, SingleStackMove) 
            else (move.i1, move.i2, move.j, move.k1, move.k2),
            delta_E
        )
                   
        # 3. Recalculate energy_model's line_counts for the next proposal. 
        energy_model.initialize(current_state) # Re-initialize the model with the new state

    # --- 2. T0 Calculation ---
    if not energy_increases: # Check if the list is empty
        print("Warning: No energy-increasing moves found during sampling. Defaulting T0 to 10.0.")
        return 10.0
        
    mean_energy_increase = np.mean(energy_increases)
    
    # 4. Calculate T0 using the inverse Metropolis formula
    T0 = -mean_energy_increase / np.log(target_acceptance_rate)
    
    print(f"Average energy degradation (mean Delta E > 0): {mean_energy_increase:.2f}")
    print(f"Initial temperature T0 calculated for {target_acceptance_rate*100}% acceptance: {T0:.2f}")
    
    return T0