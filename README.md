# 3D $N^2$-Queens Solver via MCMC

This repository implements a solver for the 3D $N^2$-queens problem (on an $N \times N \times N$ grid) using Markov Chain Monte Carlo (MCMC) methods and Simulated Annealing.

## 1. Code Structure

The project is organized as follows:

*   **`main.py`**: Entry point for running simulations. Parses command-line arguments and orchestrates the MCMC process.
*   **`state_space/`**: Defines the state space and board representation.
    *   `geometry.py`: Defines the 3D grid geometry and precomputes all the attack lines for queens on the board.
    *   `states.py`: Defines the different state representation `StackState` and `ConstraintState`. Also includes their initialization methods (random, noisy latin square, etc.) and methods to apply moves.
*   **`energy/`**: Contains the energy model.
    *   `energy_model.py`: Defines the energy function. It counts the number of conflicts (attacking queens) in the 3D grid. We also have the logic to compute the moves from proposals at each step. There is also a function to compute the number of attacked queens.
*   **`mcmc/`**: Core of the MCMC algorithm.
    *   `proposals.py`: Defines various proposal mechanisms to suggest new states, including single stack height changes, constraint-respecting swaps, and block shuffles. 
    *   `chain.py`: Defines the MCMC chain structure, including state transitions and acceptance criteria.
    *   `annealing.py`: Implements cooling schedules (Linear, Geometric, Adaptive) for simulated annealing. It also contains the main MCMC loop that iteratively proposes moves, evaluates them, and updates the state.
*   **`utils/`**: Utility functions.
    *   `plot_utils.py`: Tools for visualizing energy evolution and the final 3D configuration.

## 2. Repo Function

The goal is to find configurations of $N^2$ queens in an $N \times N \times N$ cube such that no queens attack each other. This is equivalent to minimizing an energy function.

The solver uses **Simulated Annealing** to escape local minima. It starts at a high temperature (frequently accepting random moves that may increase energy) and progressively cools the system to "freeze" it into an optimal configuration (zero energy).

The code supports:
*   Various initialization modes (random, noisy latin square, etc.).
*   Strong constraints (e.g., exactly one queen per vertical column).
*   Running multiple simulations in parallel to leverage multi-core processors.

## 3. Parameters review

To model the different approaches, we had to define several parameters and hyperparameters:

- **$N$**: the dimension of the problem.
- **$max\_steps$**: the maximum number of steps allowed for the MCMC algorithm. The algorithm stops earlier if a zero-energy state is reached.
- **$T_0$**: the initial temperature used for simulated annealing.
- **$\alpha$**: the annealing coefficient.
- **`state`**: the choice between `StackState` and `ConstraintStackState`, corresponding respectively to the Matrix and Column-Restricted state spaces.
- **`mode_init`**: the initialization mode for the state. We implemented the following options:
  - **Random**: a fully random initialization without any constraint.
  - **Random Latin Square**: for any given column $(i,*)$, we ensure that all heights $k$ are different, enforcing the constraint that all queens in $(i,*)$ have distinct heights.
  - **Noisy Latin Square**: this method is motivated by the fact that the previous initialization may be too constrained. We first generate a Random Latin Square initialization, then perturb each position $(i,j)$ with probability $p$ by replacing its height with a random value in $\{1, \ldots, N\}$.
  - **Layer-Balanced Random**: heights in $\{1, \ldots, N\}$ are assigned to each $(i,j)$ with a bias toward using each height approximately $N$ times overall, without enforcing any row- or column-wise permutation constraints.

  While all these initializations are possible for `StackState`, we only allowed Random Latin Square for `ConstraintStackState` because of the nature of this state.

- **`noisy_p`**: the parameter controlling the perturbation probability $p$ used in the Noisy Latin Square initialization.
- **`re_heat`**: a parameter enabling reheating, following the strategy described in Section II.C.
- **`n_simulations`**: for running simulations in parallel (if possible).


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

## Competition rules :

N=$10$

Send 2 files : Team_name1.txt
               Team_name2.txt

### Contest 1 :
Total energy ; nb of conflicts + 4 for each occupied black square (i.e x + y + z =1(mod2))

### Contest 2 :
Total energy = nb of conflicts + for each occupied square  (x,y,z)  energy = |x+y+2z|
