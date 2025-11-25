#!/usr/bin/env python3
"""
3D N-Queens Problem Solver using Markov Chain Monte Carlo (MCMC)

This module solves the problem of placing N² queens on a 3D N×N×N chessboard
such that no two queens attack each other.

A queen at position (i, j, k) attacks:
- Any position sharing exactly two coordinates (positions along the 3 axes)
- Any position along 2D diagonals (in xy, xz, yz planes)
- Any position along 3D diagonals of the chessboard

The solver uses the Metropolis-Hastings MCMC algorithm to find valid configurations.
"""

import random
import argparse
import sys
import math
from typing import List, Tuple, Set


def get_all_attacked_positions(pos: Tuple[int, int, int], n: int) -> Set[Tuple[int, int, int]]:
    """
    Get all positions attacked by a queen at the given position.
    
    Args:
        pos: (i, j, k) position of the queen
        n: Size of the chessboard
        
    Returns:
        Set of all attacked positions (excluding the queen's own position)
    """
    i, j, k = pos
    attacked = set()
    
    # Positions sharing exactly two coordinates (3 lines through the queen)
    # Line along x-axis (j and k fixed)
    for x in range(n):
        if x != i:
            attacked.add((x, j, k))
    
    # Line along y-axis (i and k fixed)
    for y in range(n):
        if y != j:
            attacked.add((i, y, k))
    
    # Line along z-axis (i and j fixed)
    for z in range(n):
        if z != k:
            attacked.add((i, j, z))
    
    # 2D diagonals in the xy-plane (k fixed)
    for d in range(1, n):
        if i + d < n and j + d < n:
            attacked.add((i + d, j + d, k))
        if i - d >= 0 and j - d >= 0:
            attacked.add((i - d, j - d, k))
        if i + d < n and j - d >= 0:
            attacked.add((i + d, j - d, k))
        if i - d >= 0 and j + d < n:
            attacked.add((i - d, j + d, k))
    
    # 2D diagonals in the xz-plane (j fixed)
    for d in range(1, n):
        if i + d < n and k + d < n:
            attacked.add((i + d, j, k + d))
        if i - d >= 0 and k - d >= 0:
            attacked.add((i - d, j, k - d))
        if i + d < n and k - d >= 0:
            attacked.add((i + d, j, k - d))
        if i - d >= 0 and k + d < n:
            attacked.add((i - d, j, k + d))
    
    # 2D diagonals in the yz-plane (i fixed)
    for d in range(1, n):
        if j + d < n and k + d < n:
            attacked.add((i, j + d, k + d))
        if j - d >= 0 and k - d >= 0:
            attacked.add((i, j - d, k - d))
        if j + d < n and k - d >= 0:
            attacked.add((i, j + d, k - d))
        if j - d >= 0 and k + d < n:
            attacked.add((i, j - d, k + d))
    
    # 3D diagonals (all 4 space diagonals through the queen)
    for d in range(1, n):
        # (+, +, +) and (-, -, -)
        if i + d < n and j + d < n and k + d < n:
            attacked.add((i + d, j + d, k + d))
        if i - d >= 0 and j - d >= 0 and k - d >= 0:
            attacked.add((i - d, j - d, k - d))
        
        # (+, +, -) and (-, -, +)
        if i + d < n and j + d < n and k - d >= 0:
            attacked.add((i + d, j + d, k - d))
        if i - d >= 0 and j - d >= 0 and k + d < n:
            attacked.add((i - d, j - d, k + d))
        
        # (+, -, +) and (-, +, -)
        if i + d < n and j - d >= 0 and k + d < n:
            attacked.add((i + d, j - d, k + d))
        if i - d >= 0 and j + d < n and k - d >= 0:
            attacked.add((i - d, j + d, k - d))
        
        # (+, -, -) and (-, +, +)
        if i + d < n and j - d >= 0 and k - d >= 0:
            attacked.add((i + d, j - d, k - d))
        if i - d >= 0 and j + d < n and k + d < n:
            attacked.add((i - d, j + d, k + d))
    
    return attacked


def count_conflicts(queens: List[Tuple[int, int, int]], n: int) -> int:
    """
    Count the total number of conflicts (pairs of attacking queens).
    
    Args:
        queens: List of queen positions
        n: Size of the chessboard
        
    Returns:
        Number of conflicting pairs
    """
    conflicts = 0
    queen_set = set(queens)
    
    for queen in queens:
        attacked = get_all_attacked_positions(queen, n)
        conflicts += len(attacked & queen_set)
    
    # Each conflict is counted twice (once for each queen in the pair)
    return conflicts // 2


def generate_initial_configuration(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate a random initial configuration of N² queens.
    
    Args:
        n: Size of the chessboard
        
    Returns:
        List of N² random queen positions
    """
    num_queens = n * n
    all_positions = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    return random.sample(all_positions, num_queens)


def mcmc_solve(n: int, max_iterations: int = 1000000, temperature: float = 1.0,
               cooling_rate: float = 0.9999, verbose: bool = False) -> Tuple[List[Tuple[int, int, int]], int, int]:
    """
    Solve the 3D N-Queens problem using MCMC with simulated annealing.
    
    Args:
        n: Size of the chessboard
        max_iterations: Maximum number of MCMC iterations
        temperature: Initial temperature for simulated annealing
        cooling_rate: Rate at which temperature decreases
        verbose: Print progress information
        
    Returns:
        Tuple of (final queen configuration, number of conflicts, iterations used)
    """
    num_queens = n * n
    
    # Generate initial configuration
    queens = generate_initial_configuration(n)
    queen_set = set(queens)
    
    current_conflicts = count_conflicts(queens, n)
    
    if verbose:
        print(f"Initial configuration: {current_conflicts} conflicts")
    
    # Get all board positions
    all_positions = set((i, j, k) for i in range(n) for j in range(n) for k in range(n))
    
    best_queens = list(queens)
    best_conflicts = current_conflicts
    
    temp = temperature
    
    for iteration in range(max_iterations):
        if current_conflicts == 0:
            if verbose:
                print(f"Solution found at iteration {iteration}!")
            return queens, 0, iteration
        
        # Propose a move: pick a random queen and move it to an empty position
        queen_idx = random.randint(0, num_queens - 1)
        old_pos = queens[queen_idx]
        
        # Find empty positions
        empty_positions = list(all_positions - queen_set)
        if not empty_positions:
            continue
            
        new_pos = random.choice(empty_positions)
        
        # Calculate the change in conflicts
        # Remove old queen temporarily
        queen_set.remove(old_pos)
        queens[queen_idx] = new_pos
        queen_set.add(new_pos)
        
        new_conflicts = count_conflicts(queens, n)
        delta = new_conflicts - current_conflicts
        
        # Metropolis-Hastings acceptance criterion with simulated annealing
        if delta <= 0:
            # Accept move (improves or maintains solution)
            current_conflicts = new_conflicts
        elif temp > 0 and random.random() < math.exp(-delta / temp):
            # Accept worse move with probability
            current_conflicts = new_conflicts
        else:
            # Reject move - revert
            queen_set.remove(new_pos)
            queens[queen_idx] = old_pos
            queen_set.add(old_pos)
        
        # Track best solution
        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_queens = list(queens)
        
        # Cool down
        temp *= cooling_rate
        
        if verbose and iteration % 10000 == 0:
            print(f"Iteration {iteration}: conflicts = {current_conflicts}, best = {best_conflicts}, temp = {temp:.6f}")
    
    # Return the best found solution
    return best_queens, best_conflicts, max_iterations


def validate_solution(queens: List[Tuple[int, int, int]], n: int) -> bool:
    """
    Validate that a configuration is a valid solution.
    
    Args:
        queens: List of queen positions
        n: Size of the chessboard
        
    Returns:
        True if the configuration is valid (no conflicts)
    """
    if len(queens) != n * n:
        return False
    
    if len(set(queens)) != len(queens):
        return False  # Duplicate positions
    
    return count_conflicts(queens, n) == 0


def print_solution(queens: List[Tuple[int, int, int]], n: int) -> None:
    """
    Print the solution in a readable format.
    
    Args:
        queens: List of queen positions
        n: Size of the chessboard
    """
    queen_set = set(queens)
    
    print(f"\n3D {n}×{n}×{n} chessboard with {n*n} queens:")
    print("=" * 50)
    
    for k in range(n):
        print(f"\nLayer k={k}:")
        print("  " + " ".join(str(j) for j in range(n)))
        for i in range(n):
            row = f"{i} "
            for j in range(n):
                if (i, j, k) in queen_set:
                    row += "Q "
                else:
                    row += ". "
            print(row)
    
    print("\n" + "=" * 50)
    print(f"Queen positions: {sorted(queens)}")


def main():
    """Main entry point for the 3D N-Queens MCMC solver."""
    parser = argparse.ArgumentParser(
        description="Solve the 3D N-Queens problem using MCMC (Markov Chain Monte Carlo)"
    )
    parser.add_argument(
        "-n", "--size",
        type=int,
        default=3,
        help="Size of the N×N×N chessboard (default: 3)"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=1000000,
        help="Maximum number of MCMC iterations (default: 1000000)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=10.0,
        help="Initial temperature for simulated annealing (default: 10.0)"
    )
    parser.add_argument(
        "-c", "--cooling",
        type=float,
        default=0.99999,
        help="Cooling rate (default: 0.99999)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    if args.size < 1:
        print("Error: Board size must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    if args.seed is not None:
        random.seed(args.seed)
    
    print(f"Solving 3D {args.size}×{args.size}×{args.size} N-Queens problem with {args.size**2} queens")
    print(f"Using MCMC with simulated annealing")
    print(f"Max iterations: {args.iterations}, Initial temp: {args.temperature}, Cooling: {args.cooling}")
    
    queens, conflicts, iterations = mcmc_solve(
        n=args.size,
        max_iterations=args.iterations,
        temperature=args.temperature,
        cooling_rate=args.cooling,
        verbose=args.verbose
    )
    
    print(f"\nResult after {iterations} iterations:")
    
    if conflicts == 0:
        print("SUCCESS! Found a valid configuration with no conflicts.")
        print_solution(queens, args.size)
        
        # Double-check the solution
        if validate_solution(queens, args.size):
            print("\nSolution validated successfully!")
        else:
            print("\nWARNING: Solution validation failed!")
            sys.exit(1)
    else:
        print(f"Could not find a perfect solution. Best found has {conflicts} conflicts.")
        print_solution(queens, args.size)
        print("\nTry increasing the number of iterations or adjusting temperature parameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
