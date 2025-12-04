# Solving the  3D $N^2$-queens problems using the Metropolis–Hastings algorithm

## Running the Simulation

The solver is called from the cli using the main script. At minimum, specify the board size `N` and the number of MCMC steps.

### Basic Usage

```bash
python main.py --N 8 --steps 1000000
```

Runs an annealing session on an 8x8x8 board with 1 million MCMC updates.

### Random Seed

```bash
python main.py --N 8 --steps 200000 --seed 123
```

Ensures reproducible sampling.

### Discrete Annealing Schedule

Provide a comma-separated list of temperatures:

```bash
python main.py --N 10 --steps 500000 --temperatures 100,50,20,10,5
```

The chain cycles through these temperatures in sequence.

### Geometric (Exponential) Annealing

Instead of discrete temperatures, specify a geometric cooling schedule:

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

### Full Example

```bash
python main.py \
  --N 11 \
  --alpha 0.9999 \
  --T_initial 100 \
  --max_steps 100000 \
  --stats True
```
