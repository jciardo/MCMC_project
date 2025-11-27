from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from state_space.states import StackState
from energy.energy_model import EnergyModel
#! Proposal (moves)

@dataclass
class SingleStackMove:
    """
    Container for move description
    """
    i: int #! [1, ..., N]
    j: int #! [1, ..., N]
    k_old: int #! [1, ..., N] (current height)
    k_new: int #! [1, ..., N] (proposed new height)

class Proposal(ABC):
    """
    Abstract base class for move generators
    """

    @abstractmethod
    def propose(
        self,
        state: StackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove, int]:
        """
        Propose a move starting from the given state

        Returns:
        (move, delta_E)
        - move : a description of the local change
        - delta_E: energy(state_after_move) - energy(state)
        """
        pass

class SingleStackRandomHeightProposal(Proposal):
    """
    Baseline proposal:
    - choose a random stack (i,j)
    - choose a random new height k_new != current k_old
    - compute delta_E via energy_model
    """

    def __init__(self, N: int):
        self.N = N

    def propose(
        self,
        state: StackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove, int]:

        #! pick a random stack (i,j)
        i = rng.integers(1, self.N + 1)
        j = rng.integers(1, self.N + 1)

        #! get current height
        k_old = state.get_height(i, j)

        #! sample a new height != k_old
        #* simplest: sample from [1...N] (we assume N small)
        while True:
            k_new = rng.integers(1, self.N + 1)
            if k_new != k_old:
                break

        #! compute delta_E using energy model
        delta_E = energy_model.delta_energy_single_move(state, i, j, k_new)

        move = SingleStackMove(i=i, j=j, k_old=k_old, k_new=k_new)
        return move, delta_E
