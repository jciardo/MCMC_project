from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Assure-toi que les imports correspondent à tes noms de fichiers
from state_space.states import StackState, ConstraintStackState
from energy.energy_model import EnergyModel
from mcmc.proposals import (
    Proposal, 
    SingleStackMove, 
    SingleConstraintStackMove, 
    BlockShuffleMove  # <--- Important : on importe le nouveau type
)

@dataclass
class MCMCChain:
    """
    MCMC chain container
    """

    state: StackState | ConstraintStackState
    energy_model: EnergyModel
    proposal: Proposal

    def acceptance_probability(self, delta_E: int, T: float) -> float:
        """
        Compute acceptance probability for a given energy change at temperature T
        """
        if delta_E <= 0:
            return 1.0
        else:
            return np.exp(-delta_E / T)

    def step(self, rng: np.random.Generator, T: float) -> None:
        """
        Perform a single MCMC step at temperature T
        """

        #! Propose a move
        move, delta_E = self.proposal.propose(self.state, self.energy_model, rng)
        
        #! Compute acceptance probability
        acc_prob = self.acceptance_probability(delta_E, T)

        #! Accept or reject
        if rng.uniform(0, 1) < acc_prob:
            
            # ? CAS 1: Mouvement Simple (1 Reine)
            if isinstance(move, SingleStackMove):
                self.energy_model.apply_move(
                    self.state, move.i, move.j, move.k_new, delta_E
                )

            # ? CAS 2: Mouvement Swap (2 Reines)
            elif isinstance(move, SingleConstraintStackMove):
                # ? Swap heights in constrained stacks
                self.energy_model.apply_move(
                    self.state,
                    i1=move.i1,
                    i2=move.i2,
                    j=move.j,
                    k1=move.k1,
                    k2=move.k2,
                    delta_E=delta_E,
                )
            # ? CAS 3: Mouvement Shuffle (Zone Locale) --- NOUVEAU ---
            elif isinstance(move, BlockShuffleMove):
                # On applique manuellement tous les changements du bloc
                for (i, j), k_new in zip(move.indices, move.new_heights):
                    # Adaptation selon ta structure (stacks ou set_height)
                    if hasattr(self.state, 'stacks'):
                        self.state.stacks[i-1][j-1] = k_new
                    elif hasattr(self.state, 'set_height'):
                        self.state.set_height(i, j, k_new)
                
                # CRITIQUE : On force le modèle d'énergie à se resynchroniser
                # car apply_move n'est pas conçu pour mettre à jour N reines d'un coup.
                self.energy_model.initialize(self.state)