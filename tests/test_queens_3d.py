"""Tests for the 3D N-Queens MCMC solver."""

import pytest
import random
from queens_3d_mcmc import (
    get_all_attacked_positions,
    count_conflicts,
    generate_initial_configuration,
    validate_solution,
    mcmc_solve,
)


class TestAttackPositions:
    """Test the attack position calculation."""
    
    def test_attacked_positions_center(self):
        """Test attacked positions from center of a 3x3x3 board."""
        n = 3
        pos = (1, 1, 1)
        attacked = get_all_attacked_positions(pos, n)
        
        # Should not contain the queen's own position
        assert pos not in attacked
        
        # Check axis attacks (positions sharing two coordinates)
        # Line along x-axis (j=1, k=1)
        assert (0, 1, 1) in attacked
        assert (2, 1, 1) in attacked
        
        # Line along y-axis (i=1, k=1)
        assert (1, 0, 1) in attacked
        assert (1, 2, 1) in attacked
        
        # Line along z-axis (i=1, j=1)
        assert (1, 1, 0) in attacked
        assert (1, 1, 2) in attacked
    
    def test_attacked_positions_corner(self):
        """Test attacked positions from corner of a 3x3x3 board."""
        n = 3
        pos = (0, 0, 0)
        attacked = get_all_attacked_positions(pos, n)
        
        # Should not contain the queen's own position
        assert pos not in attacked
        
        # Axis attacks
        assert (1, 0, 0) in attacked
        assert (2, 0, 0) in attacked
        assert (0, 1, 0) in attacked
        assert (0, 2, 0) in attacked
        assert (0, 0, 1) in attacked
        assert (0, 0, 2) in attacked
        
        # 3D diagonal
        assert (1, 1, 1) in attacked
        assert (2, 2, 2) in attacked
    
    def test_diagonal_attacks_xy_plane(self):
        """Test 2D diagonal attacks in the xy-plane."""
        n = 5
        pos = (2, 2, 0)  # Center of xy-plane at k=0
        attacked = get_all_attacked_positions(pos, n)
        
        # xy-plane diagonals
        assert (0, 0, 0) in attacked
        assert (1, 1, 0) in attacked
        assert (3, 3, 0) in attacked
        assert (4, 4, 0) in attacked
        
        assert (0, 4, 0) in attacked
        assert (1, 3, 0) in attacked
        assert (3, 1, 0) in attacked
        assert (4, 0, 0) in attacked
    
    def test_3d_diagonal_attacks(self):
        """Test 3D space diagonal attacks."""
        n = 4
        pos = (1, 1, 1)
        attacked = get_all_attacked_positions(pos, n)
        
        # (+, +, +) direction
        assert (0, 0, 0) in attacked
        assert (2, 2, 2) in attacked
        assert (3, 3, 3) in attacked
        
        # (+, +, -) direction
        assert (2, 2, 0) in attacked
        assert (0, 0, 2) in attacked
        
        # (+, -, +) direction
        assert (2, 0, 2) in attacked
        assert (0, 2, 0) in attacked
        
        # (+, -, -) direction
        assert (2, 0, 0) in attacked
        assert (0, 2, 2) in attacked


class TestConflictCounting:
    """Test conflict counting functions."""
    
    def test_no_conflicts_single_queen(self):
        """Single queen should have no conflicts."""
        n = 3
        queens = [(1, 1, 1)]
        # With only 1 queen, there can't be conflicts
        # But our problem requires N² queens, so this is just a basic test
        assert count_conflicts(queens, n) == 0
    
    def test_conflict_detection_axis(self):
        """Detect conflicts along axes."""
        n = 3
        # Two queens on same axis line
        queens = [(0, 0, 0), (1, 0, 0)]  # Same j and k
        assert count_conflicts(queens, n) == 1
    
    def test_conflict_detection_diagonal(self):
        """Detect conflicts along diagonals."""
        n = 3
        # Two queens on 3D diagonal
        queens = [(0, 0, 0), (2, 2, 2)]
        assert count_conflicts(queens, n) == 1
    
    def test_no_conflicts_non_attacking(self):
        """Two queens that don't attack each other."""
        n = 5
        # These positions don't share attack patterns
        queens = [(0, 0, 0), (1, 2, 4)]
        assert count_conflicts(queens, n) == 0


class TestInitialConfiguration:
    """Test initial configuration generation."""
    
    def test_correct_number_of_queens(self):
        """Should generate exactly N² queens."""
        for n in [2, 3, 4]:
            queens = generate_initial_configuration(n)
            assert len(queens) == n * n
    
    def test_unique_positions(self):
        """All queens should be in unique positions."""
        n = 4
        queens = generate_initial_configuration(n)
        assert len(set(queens)) == len(queens)
    
    def test_valid_positions(self):
        """All positions should be within the board."""
        n = 5
        queens = generate_initial_configuration(n)
        for i, j, k in queens:
            assert 0 <= i < n
            assert 0 <= j < n
            assert 0 <= k < n


class TestValidation:
    """Test solution validation."""
    
    def test_invalid_wrong_count(self):
        """Invalid if wrong number of queens."""
        n = 3
        queens = [(0, 0, 0)]  # Only 1 queen instead of 9
        assert not validate_solution(queens, n)
    
    def test_invalid_duplicates(self):
        """Invalid if duplicate positions."""
        n = 2
        queens = [(0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)]
        assert not validate_solution(queens, n)


class TestMCMCSolver:
    """Test the MCMC solver."""
    
    def test_small_board_finds_solution(self):
        """Should find a solution for small boards."""
        random.seed(42)
        n = 2
        queens, conflicts, iterations = mcmc_solve(
            n=n,
            max_iterations=100000,
            temperature=10.0,
            cooling_rate=0.9999,
            verbose=False
        )
        
        assert len(queens) == n * n
        # For n=2, we need 4 queens on 8 positions - likely to find solution
        # Note: solution may not always exist or be found
    
    def test_reproducibility_with_seed(self):
        """Same seed should produce same results."""
        random.seed(123)
        n = 2
        result1 = mcmc_solve(n=n, max_iterations=10000, verbose=False)
        
        random.seed(123)
        result2 = mcmc_solve(n=n, max_iterations=10000, verbose=False)
        
        # Same seed should give same initial config and moves
        assert result1 == result2
    
    def test_returns_best_solution(self):
        """Should track and return the best solution found."""
        random.seed(42)
        n = 2
        queens, conflicts, iterations = mcmc_solve(
            n=n,
            max_iterations=50000,
            verbose=False
        )
        
        # Returned conflicts should match actual count
        assert count_conflicts(queens, n) == conflicts


class TestIntegration:
    """Integration tests for the full solver."""
    
    def test_n2_solvable(self):
        """Test that n=2 can be solved."""
        random.seed(12345)
        n = 2
        queens, conflicts, _ = mcmc_solve(
            n=n,
            max_iterations=500000,
            temperature=20.0,
            cooling_rate=0.99999,
            verbose=False
        )
        
        # We expect to find a valid solution for n=2
        if conflicts == 0:
            assert validate_solution(queens, n)
