# 3D $N^2$-Queens Solver via MCMC

This repository implements a solver for the 3D $N^2$-queens problem (on an $N \times N \times N$ grid) using Markov Chain Monte Carlo (MCMC) methods and Simulated Annealing.

## 1. Code Structure

The project is organized in a modular fashion:

*   **`main.py`**: The main entry point. Handles argument parsing, simulation initialization, parallel execution, and results plotting.
*   **`energy/`**: Contains the energy model.
    *   `energy_model.py`: Defines the cost function (Hamiltonian) which counts the number of alignments (attacks) between queens.
*   **`mcmc/`**: Core of the MCMC algorithm.
    *   `proposals.py`: Defines possible moves to explore the state space (Shuffle, Swap, Random Height, etc.).
    *   `chain.py`: Manages the Markov Chain logic and Metropolis-Hastings acceptance/rejection criteria.
    *   `annealing.py`: Implements cooling schedules (Linear, Geometric, Adaptive) for simulated annealing.
*   **`state_space/`**: Problem representation.
    *   `geometry.py`: Defines the 3D grid and line indices.
    *   `states.py`: Defines data structures for representing the board state and initialization methods. 
*   **`utils/`**: Utility functions.
    *   `plot_utils.py`: Tools for visualizing energy evolution and the final 3D configuration.

## 2. Repo Function

The goal is to find configurations of $N^2$ queens in an $N \times N \times N$ cube such that no queens attack each other. This is equivalent to minimizing an energy function.

The solver uses **Simulated Annealing** to escape local minima. It starts at a high temperature (frequently accepting random moves that may increase energy) and progressively cools the system to "freeze" it into an optimal configuration (zero energy).

The code supports:
*   Various initialization modes (random, noisy latin square, etc.).
*   Strong constraints (e.g., exactly one queen per vertical column).
*   Running multiple simulations in parallel to leverage multi-core processors.

## 3. How to Run

The main script is `main.py`. It is used via the command line.

### Installation

Ensure required dependencies are installed (typically `numpy`, `matplotlib`, `tqdm`).
You can install them using pip:

```bash
pip install -r requirements.txt
```

### Common Options

Here are the most useful arguments to control the simulation:

*   **Problem Size**: `--N 8` (grid size).
*   **Duration**: `--steps 100000` (number of iterations).
*   **State Type**:
    *   `--state_type stack`: Relaxed constraints (default).
    *   `--state_type constraint`: Strict constraints (Latin Square per column).
*   **Initialization**: `--mode_init random_latin_square` (or `noisy_latin_square`, `layer_balanced_random`).
*   **Geometric Annealing** (Recommended for performance):
    *   Requires specifying `--T_initial`, `--alpha`, and `--max_steps`.
    *   Example: `--T_initial 100 --alpha 0.99 --max_steps 100000`.
*   **Parallelism**:
    *   `--n_simulations 10`: Runs 10 simulations with different seeds.
    *   `--max_workers 4`: Uses 4 CPU cores in parallel.

### Full Example

To launch a serious search for a solution for N=11, in parallel on 4 cores:

```bash
python main.py \
  --N 11 \
  --state_type stack \
  --mode_init noisy_latin_square \
  --T_initial 100 \
  --alpha 0.9999 \
  --base_seed 42 \
  --max_steps 100000 \
  --verbose_every 5000 \
  --n_simulations 4 \
  --max_workers 4
```

### Visualization

At the end of the execution, the script will automatically generate plots showing the energy evolution and the final configuration of the queens.
