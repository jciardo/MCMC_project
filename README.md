# Solving the  3D $N^2$-queens problems using the Metropolis–Hastings algorithm

## Running the Simulation

The solver is called from the CLI using the main script. At minimum, specify the board size `N` and the number of MCMC steps.

### Basic Usage

```bash
python main.py --N 8 --steps 1000000
```

Runs an annealing session on an 8x8x8 board with 1 million MCMC updates.

### Random Seed

```bash
python main.py --N 8 --steps 200000 --base_seed 123
```

Ensures reproducible sampling.

### State Type

Choose the type of state representation:
- `stack` (default): unconstrained stacks
- `constraint`: constrained stacks (Latin square per column)

```bash
python main.py --N 8 --state_type stack
python main.py --N 8 --state_type constraint
```

### Initialization Mode

Choose the initialization mode for the state:
- `noisy_latin_square`
- `layer_balanced_random`
- `random_latin_square`

```bash
python main.py --N 8 --mode_init noisy_latin_square
```

### Geometric (Exponential) Annealing

Specify a geometric cooling schedule:

```bash
python main.py --N 12 --T_initial 200 --alpha 0.995 --max_steps 300000
```

Where:

- `T_initial` is the starting temperature  
- `alpha` is the multiplicative cooling factor per step  
- `max_steps` limits the run length when using geometric cooling

### Verbosity and Diagnostics

Print progress every `verbose_every` steps:

```bash
python main.py --verbose_every 5000
```

Enable extended attack count statistics:

```bash
python main.py --stats True
```

### Parallel Simulations

Run several simulations in parallel (default: 10):

```bash
python main.py --n_simulations 5 --max_workers 2
```

### Optimal call for N = 11

```bash
python main.py \
  --N 11 \
  --base_seed 49 \
  --alpha 0.9999 \
  --T_initial 100 \
  --max_steps 100000 \
  --state_type stack \
  --mode_init noisy_latin_square \
  --noisy_p 0.2 \
  --n_simulations 4 \
  --stats True \ 
```

### Arguments Summary

- `--N`: Size of the board (N x N)
- `--steps`: Number of MCMC steps
- `--base_seed`: Random number generator seed (used for parallel runs)
- `--state_type`: 'stack' or 'constraint'
- `--mode_init`: Initialization mode ('noisy_latin_square', 'layer_balanced_random', 'random_latin_square')
- `--T_initial`: Initial temperature for geometric annealing
- `--alpha`: Cooling factor for geometric annealing
- `--max_steps`: Max steps for geometric annealing
- `--verbose_every`: Print diagnostics every n steps
- `--stats`: Print more detailed stats (True/False)
- `--n_simulations`: Number of parallel simulations
- `--max_workers`: Max number of parallel workers
