import numpy as np
from itertools import product
import numpy as np
import argparse
from tqdm import tqdm
from state_space.states import StackState, ConstraintStackState
from state_space.geometry import Board, LineIndex
from energy.energy_model import EnergyModel
from mcmc.proposals import (
    SingleStackMove,
    SingleConstraintStackMove,
    SingleStackRandomHeightProposal,
)
from mcmc.chain import MCMCChain
from mcmc.annealing import (
    AnnealingSchedule,
    LinearSchedule,
    run_simulated_annealing,
    GeometricSchedule,
    calibrate_initial_temperature
)
from mcmc.proposals import SingleConstraintStackSwapProposal

from itertools import product

'''def brute_force_min_energy(
    N: int,
    Board_cls,
    LineIndex_cls,
    EnergyModel_cls,
    StackState_cls,
    include_vertical: bool = False,
):
    """
    Exhaustively enumerate all StackStates and find the exact minimum energy.

    Returns
    -------
    best_energy : int
    best_states : list[np.ndarray]
        Each element is an N x N array of heights in {1, ..., N},
        representing one optimal configuration.
    """
    # Geometry & line index as in main()
    geometry = Board_cls(N=N)
    line_index = LineIndex_cls(geometry=geometry, include_vertical=include_vertical)
    energy_model = EnergyModel_cls(geometry=geometry, line_index=line_index)

    best_energy = None
    best_states = []

    # There are N^(N^2) states
    total_states = N ** (N * N)
    print(f"[BRUTE FORCE N={N}] exploring {total_states:,} states")

    # Main enumeration loop with tqdm
    for assignment in tqdm(
        product(range(N), repeat=N * N),
        total=total_states,
        unit="state",
        ncols=80
    ):
    
        # Build a StackState for this assignment
        state = StackState_cls.noisy_latin_square(N=N, p =0.2)   # adjust if your __init__ signature differs
        idx = 0
        for i in range(N):
            for j in range(N):
                k0 = assignment[idx]      # 0 .. N-1
                k = k0 + 1                # convert to 1 .. N
                state.set_height(i, j, k)
                idx += 1

        # Compute energy from scratch
        energy_model.initialize(state)
        E = energy_model.current_energy

        # Update optimum
        if best_energy is None or E < best_energy:
            best_energy = E
            # Store a simple N x N array of heights for later inspection
            mat = np.array([
                [assignment[i * N + j] + 1 for j in range(N)]
                for i in range(N)
            ])
            best_states = [mat]
        elif E == best_energy:
            mat = np.array([
                [assignment[i * N + j] + 1 for j in range(N)]
                for i in range(N)
            ])
            best_states.append(mat)

    print(f"[BRUTE FORCE] Exact minimum energy for N={N}: {best_energy}")
    print(f"[BRUTE FORCE] Number of optimal states: {len(best_states)}")

    return best_energy, best_states'''

import numpy as np
from tqdm import tqdm

def exact_min_energy_solver(
    N: int,
    Board_cls,
    LineIndex_cls,
    StackState_cls,
    include_vertical: bool = False,
):
    """
    Exact solver using DFS + branch-and-bound

    Returns:
        best_energy (int)
        best_states (list of N x N numpy arrays of heights)
    """

    # Geometry and line index
    geometry = Board_cls(N=N)
    line_index = LineIndex_cls(geometry=geometry, include_vertical=include_vertical)

    #  mirror the EnergyModel combinatorics:
    #   energy = sum_l C(line_counts[l], 2)
    # and maintain it incrementally
    num_lines = len(line_index.lines)  #! check if LineIndex exposes num_lines in a different way
    line_counts = np.zeros(num_lines, dtype=int)

    # Precompute for each (i,j,k) the list of line_ids it belongs to
    cell_lines = {}
    for i in range(N):
        for j in range(N):
            for k in range(1, N + 1):
                cell_id = geometry.coord_to_id(i, j, k)
                lines = line_index.cell_to_lines[cell_id]
                cell_lines[(i, j, k)] = lines

    # State object to mutate
    state = StackState_cls.random(N=N)  # any constructor; we'll overwrite heights

    best_energy = None
    best_states = []

    # tqdm progress bar over "search nodes" (calls to dfs)
    with tqdm(
        desc=f"Exact search N={N}",
        unit="node",
        ncols=80
    ) as pbar:

        def dfs(cell_idx: int, current_energy: int):
            nonlocal best_energy, best_states

            # Count this DFS node in the progress bar
            pbar.update(1)

            # Branch-and-bound: if already worse than best, prune
            if best_energy is not None and current_energy > best_energy:
                return

            # All N^2 cells assigned → complete configuration
            if cell_idx == N * N:
                if best_energy is None or current_energy < best_energy:
                    best_energy = current_energy
                    best_states = [state.heights.copy()]
                elif current_energy == best_energy:
                    best_states.append(state.heights.copy())
                return

            i = cell_idx // N
            j = cell_idx % N

            # Try all heights k = 1..N for this cell
            for k in range(1, N + 1):
                lines = cell_lines[(i, j, k)]

                # Incremental energy for placing queen at (i,j,k)
                delta_E = 0
                for line_id in lines:
                    c = line_counts[line_id]
                    delta_E += c   # C(c+1,2) - C(c,2) = c

                new_energy = current_energy + delta_E

                # Further branch-and-bound
                if best_energy is not None and new_energy > best_energy:
                    continue

                # Commit choice
                state.set_height(i, j, k)
                for line_id in lines:
                    line_counts[line_id] += 1

                # Recurse
                dfs(cell_idx + 1, new_energy)

                # Backtrack
                for line_id in lines:
                    line_counts[line_id] -= 1
                # No need to undo state.set_height; it'll be overwritten

        # Kick off recursion
        dfs(cell_idx=0, current_energy=0)

    return best_energy, best_states

'''def debug_bruteforce_N3():
    N = 3
    best_E, best_states = brute_force_min_energy(
        N=N,
        Board_cls=Board,
        LineIndex_cls=LineIndex,
        EnergyModel_cls=EnergyModel,
        StackState_cls=StackState,
        include_vertical=False,  # same as in main()
    )

    print("Example optimal configuration (first one):")
    if best_states:
        print(best_states[0])

def debug_bruteforce_N4():
    N = 4
    best_E, best_states = brute_force_min_energy(
        N=N,
        Board_cls=Board,
        LineIndex_cls=LineIndex,
        EnergyModel_cls=EnergyModel,
        StackState_cls=StackState,
        include_vertical=False,  # same as in main()
    )

    print("Example optimal configuration (first one):")
    if best_states:
        print(best_states[0])'''

def debug_exact_solver_N3():
    N = 3
    best_E, best_states = exact_min_energy_solver(
        N=N,
        Board_cls=Board,
        LineIndex_cls=LineIndex,
        StackState_cls=StackState,
        include_vertical=False,
    )

    print(f"[EXACT SOLVER] Exact minimum energy for N={N}: {best_E}")
    print(f"[EXACT SOLVER] Number of optimal states: {len(best_states)}")
    if best_states:
        print("Example optimal configuration:")
        print(best_states[0])
        print(len(best_states))

def debug_exact_solver_N4():
    N = 4
    best_E, best_states = exact_min_energy_solver(
        N=N,
        Board_cls=Board,
        LineIndex_cls=LineIndex,
        StackState_cls=StackState,
        include_vertical=False,
    )

    print(f" Exact minimum energy for N={N}: {best_E}")
    print(f" Number of optimal states: {len(best_states)}")
    if best_states:
        print("Example optimal configuration:")
        print(best_states[0])
        print(len(best_states))

import numpy as np

import numpy as np

def check_energy_for_matrix(
    mat: np.ndarray,
    Board_cls,
    LineIndex_cls,
    EnergyModel_cls,
    StackState_cls,
    include_vertical: bool = False,
):
    N = mat.shape[0]
    assert mat.shape == (N, N), "Matrix must be N x N"

    print("Matrix passed in:")
    print(mat)

    # Rebuild geometry and model exactly like you do in MCMC
    geometry = Board_cls(N=N)
    line_index = LineIndex_cls(geometry=geometry, include_vertical=include_vertical)
    energy_model = EnergyModel_cls(geometry=geometry, line_index=line_index)

    # Build a *fresh* StackState and overwrite heights
    state = StackState_cls.random(N=N)  # any constructor; heights will be overwritten

    for i in range(N):
        for j in range(N):
            k = int(mat[i, j])
            #state.set_height(i, j, k)
            state = state.set_height(i, j, k)

    # Show what the state actually holds
    try:
        arr = state.as_array()
    except TypeError:
        arr = state.as_array  # if it's a property, not a method

    print("State as_array after setting heights:")
    print(arr)

    # Compute energy with your EnergyModel
    energy_model.initialize(state)
    E = energy_model.current_energy
    print("Energy from EnergyModel:", E)
    return E



def energy_via_line_counts(
    mat: np.ndarray,
    Board_cls,
    LineIndex_cls,
    include_vertical: bool = False,
):
    N = mat.shape[0]
    geometry = Board_cls(N=N)
    line_index = LineIndex_cls(geometry=geometry, include_vertical=include_vertical)

    num_lines = len(line_index.lines)  # adapt if needed
    line_counts = np.zeros(num_lines, dtype=int)

    # Count queens per line
    for i in range(N):
        for j in range(N):
            k = int(mat[i, j])
            cell_id = geometry.coord_to_id(i, j, k)
            for line_id in line_index.cell_to_lines[cell_id]:
                line_counts[line_id] += 1

    # Compute energy = sum C(n,2)
    E = 0
    for c in line_counts:
        if c >= 2:
            E += c * (c - 1) // 2
    return E

        

if __name__ == "__main__":
    #debug_exact_solver_N4()
    conf = np.array([
    [2, 2, 1, 3],
    [4, 4, 4, 1],
    [2, 1, 1, 3],
    [1, 4, 3, 1],
    ], dtype=int)

    E_lines = energy_via_line_counts(conf, Board, LineIndex, include_vertical=False)
    E_model = check_energy_for_matrix(conf, Board, LineIndex, EnergyModel, StackState, include_vertical=False)

    print("Energy via line_counts: ", E_lines)
    print("Energy via EnergyModel: ", E_model)



"""answers so far :
Exact search N=3: 6440node [00:00, 163879.76node/s]
Exact minimum energy for N=3: 26
Number of optimal states: 16
Example optimal configuration:
[[1 3 1]
 [1 3 2]
 [1 3 1]]
16


Exact search N=4: 49538023node [06:33, 125802.80node/s]
Exact minimum energy for N=4: 46
Number of optimal states: 96
Example optimal configuration:
[[2 2 1 3]
 [4 4 4 1]
 [2 1 1 3]
 [1 4 3 1]]
"""



