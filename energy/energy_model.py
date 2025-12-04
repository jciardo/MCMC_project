from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List

from state_space.states import StackState, ConstraintStackState
from state_space.geometry import Board, LineIndex


def nchoose2(n: int) -> int:
    return (n * (n - 1)) // 2


@dataclass
class EnergyModel:
    geometry: Board
    line_index: LineIndex

    def __post_init__(self):

        #! Nb of lines
        L = len(self.line_index.lines)

        #! Count the nb of queens on each line
        self.line_counts = np.zeros(L, dtype=int)

        #! Tot energy
        self.current_energy = 0

    def initialize(self, state: StackState | ConstraintStackState) -> None:
        """
        Compute line_counts and current_energy from scratch for this state
        """
        # ? zero counts
        self.line_counts[:] = 0

        # ? count queens on each line
        for i, j, k in state.iter_queens():

            cell_id = self.geometry.coord_to_id(i, j, k)
            for line_id in self.line_index.cell_to_lines[cell_id]:

                self.line_counts[line_id] += 1

        # ? computes energy
        energy = 0
        for c in self.line_counts:
            if c > 1:
                energy += nchoose2(c)

        self.current_energy = energy

    def get_energy(self) -> int:
        return int(self.current_energy)

    def _line_delta_energy(self, line_id, in_old, in_new):
        c = self.line_counts[line_id]

        if in_old and not in_new:
            dc = -1
        elif in_new and not in_old:
            dc = +1
        else:
            dc = 0

        return nchoose2(c + dc) - nchoose2(c)

    def _delta_energy_generic(
        self, affected_cells_old: list[int], affected_cells_new: list[int]
    ) -> int:
        lid = self.line_index
        old_lines = [lid.cell_to_lines[cell] for cell in affected_cells_old]
        new_lines = [lid.cell_to_lines[cell] for cell in affected_cells_new]
        old_set = set([line_id for lines in old_lines for line_id in lines])
        new_set = set([line_id for lines in new_lines for line_id in lines])
        delta_E = 0
        for line_id in old_set.union(new_set):
            in_old = line_id in old_set
            in_new = line_id in new_set
            delta_E += self._line_delta_energy(line_id, in_old, in_new)
        return delta_E

    def delta_energy(
        self,
        state: StackState | ConstraintStackState,
        i: int = None,
        j: int = None,
        k_new: int = None,
        i1: int = None,
        i2: int = None,
        k1: int = None,
        k2: int = None,
    ) -> int:
        """Energy change for a proposed move."""
        if isinstance(state, StackState):
            old_k = state.get_height(i, j)
            k_new_val = k_new if k_new is not None else old_k
            if k_new_val == old_k:
                return 0
            board = self.geometry
            cell_old = board.coord_to_id(i, j, old_k)
            cell_new = board.coord_to_id(i, j, k_new_val)
            return self._delta_energy_generic([cell_old], [cell_new])
        elif isinstance(state, ConstraintStackState):
            k1_val = k1 if k1 is not None else state.get_height(i1, j)
            k2_val = k2 if k2 is not None else state.get_height(i2, j)
            if k1_val == k2_val:
                return 0
            board = self.geometry
            cell_1_old = board.coord_to_id(i1, j, k1_val)
            cell_2_old = board.coord_to_id(i2, j, k2_val)
            cell_1_new = board.coord_to_id(i1, j, k2_val)
            cell_2_new = board.coord_to_id(i2, j, k1_val)
            return self._delta_energy_generic(
                [cell_1_old, cell_2_old], [cell_1_new, cell_2_new]
            )

    def _apply_move_generic(
        self,
        affected_cells_old: list[int],
        affected_cells_new: list[int],
        delta_E: int = None,
    ) -> None:
        lid = self.line_index
        old_lines = [lid.cell_to_lines[cell] for cell in affected_cells_old]
        new_lines = [lid.cell_to_lines[cell] for cell in affected_cells_new]
        old_set = set([line_id for lines in old_lines for line_id in lines])
        new_set = set([line_id for lines in new_lines for line_id in lines])
        if delta_E is None:
            delta_E = 0
            for line_id in old_set.union(new_set):
                in_old = line_id in old_set
                in_new = line_id in new_set
                delta_E += self._line_delta_energy(line_id, in_old, in_new)
        for line_id in old_set.union(new_set):
            in_old = line_id in old_set
            in_new = line_id in new_set
            if in_old and not in_new:
                self.line_counts[line_id] -= 1
            elif in_new and not in_old:
                self.line_counts[line_id] += 1
        self.current_energy += delta_E

    def apply_move(
        self,
        state: StackState | ConstraintStackState,
        i: int = None,
        j: int = None,
        k_new: int = None,
        i1: int = None,
        i2: int = None,
        k1: int = None,
        k2: int = None,
        delta_E: int = None,
    ) -> None:
        """
        Apply the move proposed, updating line counts and current energy
        """
        if isinstance(state, StackState):
            old_k = state.get_height(i, j)
            k_new_val = k_new if k_new is not None else old_k
            if k_new_val == old_k:
                return
            board = self.geometry
            cell_old = board.coord_to_id(i, j, old_k)
            cell_new = board.coord_to_id(i, j, k_new_val)
            self._apply_move_generic([cell_old], [cell_new], delta_E)
            state.set_height(i, j, k_new_val)
        elif isinstance(state, ConstraintStackState):
            k1_val = k1 if k1 is not None else state.get_height(i1, j)
            k2_val = k2 if k2 is not None else state.get_height(i2, j)
            if k1_val == k2_val:
                return
            board = self.geometry
            cell_1_old = board.coord_to_id(i1, j, k1_val)
            cell_2_old = board.coord_to_id(i2, j, k2_val)
            cell_1_new = board.coord_to_id(i1, j, k2_val)
            cell_2_new = board.coord_to_id(i2, j, k1_val)
            self._apply_move_generic(
                [cell_1_old, cell_2_old], [cell_1_new, cell_2_new], delta_E
            )
            state.set_height(i1, j, k2_val)
            state.set_height(i2, j, k1_val)

    def count_attacked_queens(self, state: StackState) -> int:
        """
        Returns the number of queens that lie on at least one line with >= 2 queens
        """
        attacked = 0
        board = self.geometry
        l_id = self.line_index

        for (i, j, k) in state.iter_queens():
            cell_id = board.coord_to_id(i, j, k)
            lines = l_id.cell_to_lines[cell_id]

            #! is this queen on any conflicting line?
            if any(self.line_counts[line_id] > 1 for line_id in lines):
                attacked += 1

        return attacked
    
    def attacked_stats(self, state: StackState):
        """
        Compute more precise attack stats:
        - attacked_queens: at least one attacker
        - max_attacks: maximum number of attackers any queen has
        - mean_attacks: mean number of attackers among attacked queens
        - most_attacked_queen: coord of the queen or None

        Returns : dict
        """
        board = self.geometry
        l_id = self.line_index
        N = board.N

        attacked_queens = 0
        total_attacks = 0
        max_attacks = 0
        most_attacked_queen = None

        #! queens per layer
        layer_counts = np.zeros(N, dtype=int)

        for (i, j, k) in state.iter_queens():
            
            #? record occupancy
            layer_counts[k-1] += 1   # adjust here if you use 1..N

            cell_id = board.coord_to_id(i, j, k)
            lines = l_id.cell_to_lines[cell_id]

            #? count how many other queens attack this queen
            attacks_here = 0
            for line_id in lines:
                n = self.line_counts[line_id]
                if n > 1:
                    attacks_here += (n - 1)

            if attacks_here > 0:
                attacked_queens += 1
                total_attacks += attacks_here
                if attacks_here > max_attacks:
                    max_attacks = attacks_here
                    most_attacked_queen = (i, j, k)

        mean_attacks = (total_attacks / attacked_queens if attacked_queens > 0 else 0.0)

        return {
            "attacked_queens": attacked_queens,
            "max_attacks": max_attacks,
            "mean_attacks": mean_attacks,
            "most_attacked_queen": most_attacked_queen,
            "layer_counts": layer_counts, # index 0→layer1, 1→layer2,
        }
