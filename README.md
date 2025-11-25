# 3D N-Queens Problem Solver using MCMC

This project solves the 3D N-Queens problem using Markov Chain Monte Carlo (MCMC) with simulated annealing.

## Problem Description

On a 3D N × N × N chessboard, we want to place N² queens such that no two queens attack each other.

A queen placed at position (i, j, k) attacks:
- Any position sharing exactly two common coordinates (positions along the three axes through the queen)
- Any position along a 2D diagonal (in the xy, xz, or yz planes)
- Any position along a 3D diagonal of the chessboard

## Algorithm

The solver uses the **Metropolis-Hastings MCMC algorithm** combined with **simulated annealing**:

1. Start with a random placement of N² queens on the N³ board positions
2. At each iteration, propose moving a random queen to a random empty position
3. Accept or reject the move based on the Metropolis criterion:
   - Always accept if the move reduces conflicts
   - Accept with probability exp(-Δ/T) if the move increases conflicts (where Δ is the change in conflicts and T is the temperature)
4. Gradually cool the temperature to converge toward an optimal solution
5. Stop when a conflict-free configuration is found or maximum iterations reached

## Usage

```bash
# Basic usage (3×3×3 board with 9 queens)
python queens_3d_mcmc.py

# Specify board size
python queens_3d_mcmc.py -n 4

# With verbose output
python queens_3d_mcmc.py -n 3 -v

# Set random seed for reproducibility
python queens_3d_mcmc.py -n 3 -s 42

# Custom parameters
python queens_3d_mcmc.py -n 3 -i 2000000 -t 20.0 -c 0.999999
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --size` | Size of the N×N×N chessboard | 3 |
| `-i, --iterations` | Maximum MCMC iterations | 1000000 |
| `-t, --temperature` | Initial temperature for simulated annealing | 10.0 |
| `-c, --cooling` | Cooling rate (temperature multiplier per iteration) | 0.99999 |
| `-s, --seed` | Random seed for reproducibility | None |
| `-v, --verbose` | Print progress information | False |

## Example Output

```
Solving 3D 3×3×3 N-Queens problem with 9 queens
Using MCMC with simulated annealing
Max iterations: 1000000, Initial temp: 10.0, Cooling: 0.99999

Result after 12345 iterations:
SUCCESS! Found a valid configuration with no conflicts.

3D 3×3×3 chessboard with 9 queens:
==================================================

Layer k=0:
  0 1 2
0 Q . .
1 . . Q
2 . Q .

Layer k=1:
  0 1 2
0 . Q .
1 . . .
2 Q . Q

Layer k=2:
  0 1 2
0 . . Q
1 Q . .
2 . Q .

==================================================
Queen positions: [(0, 0, 0), (0, 1, 1), ...]

Solution validated successfully!
```

## Important Notes

- **Not all N values may have valid solutions.** The 3D N-Queens problem is highly constrained. For small values of N (especially N=2), a conflict-free configuration may not exist mathematically.
- For larger N, finding a solution may require more iterations or tuning of the temperature and cooling parameters.
- The algorithm uses a stochastic approach, so different random seeds may produce different results.

## Requirements

- Python 3.6 or higher
- No external dependencies (uses only standard library)

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT License