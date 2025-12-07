from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import numpy as np


Coord3D = Tuple[int, int, int]  # ? (i, j, k)


class State(ABC):
    """
    Abstract interface for a 3D board configuration
    """

    def __init__(self, N: int):
        self.N = N

    @abstractmethod
    def copy(self) -> "State":
        """
        Deep copy of the state
        """
        pass

    @abstractmethod
    def iter_queens(self) -> Iterable[Coord3D]:
        """
        Iterate over all queen positions (i, j, k)
        """
        pass

    @abstractmethod
    def as_array(self):
        """
        Return an array-like representation (for logging / plotting)
        """
        pass


@dataclass
class StackState(State):
    """
    One queen per "vertical stack" (i,j,*)
    Internally stored as heights[i, j] = k in {1,...,N}
    """

    heights: np.ndarray  #! shape (N, N), values in [1..N]

    def __init__(self, heights: np.ndarray):

        N = heights.shape[0]
        super().__init__(N)

        # ? Heights as a (N x N) matrix
        assert heights.shape == (N, N)
        self.heights = heights

    @classmethod
    def random(cls, N: int, rng=np.random.default_rng()):
        """
        Uniformly random state
        """
        heights = rng.integers(1, N + 1, size=(N, N))

        return cls(heights)

    @classmethod
    def random_latin_square(cls, N: int, rng=np.random.default_rng()):
        """
        Implement a random state respecting constraints :
        for one column (i, *), all heights k are different
        """
        print("Generating random Latin square for unconstrained stacks")
        base = np.arange(1, N + 1)
        heights = np.array([np.roll(base, i) for i in range(N)])

        rng.shuffle(heights, axis=0)
        rng.shuffle(heights, axis=1)

        return cls(heights)

    @classmethod
    def noisy_latin_square(cls, N: int, p: float = 1, rng=np.random.default_rng()):
        """
        Noisy Latin square initialization ;
        i) Build Latin
        ii) Perturb each cell (i,j) wit proba p (replace height by random {1, ..., N})

        Parameters :
        N : board size
        p : probability (0 : pure Latin -> 1 : random init.)

        Returns : cls instance initialized w heights
        """
        print(f"Generating noisy Latin square with p={p:.2f} for unconstrained stacks")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        #! i) Latin square (std implementation)
        base = np.arange(1, N + 1)
        heights = np.array([np.roll(base, i) for i in range(N)])

        rng.shuffle(heights, axis=0)
        rng.shuffle(heights, axis=1)

        #! ii) add noise
        if p > 0.0:
            mask = rng.random(size=(N, N)) < p
            random_heights = rng.integers(1, N + 1, size=(N, N))
            heights[mask] = random_heights[mask]

        return cls(heights)

    @classmethod
    def layer_balanced_random(cls, N: int, rng=np.random.default_rng()):
        """
        Layer-balanced random initialization :
        - Assign heights in {1, ..., N} to each (i,j),
        with a bias toward using each height about N times overall

        (no enforcment of row- or column-wise permutations)
        """
        print("Generating layer-balanced random state for unconstrained stacks")

        heights = np.empty((N, N), dtype=int)

        #! counts[h-1] = how many times height h has been used
        counts = np.zeros(N, dtype=int)
        target = N  # * target usage per height

        for i in range(N):
            for j in range(N):
                #! Favor underused heights
                deficits = target - counts
                weights = np.clip(deficits, 0, None)
                probs = weights / weights.sum()
                idx = rng.choice(N, p=probs)
                h = idx + 1

                heights[i, j] = h
                counts[idx] += 1

        return cls(heights)

    @classmethod
    def init_state(
        cls,
        N: int,
        rng=np.random.default_rng(),
        mode: str = "noisy_latin_square",
        p: float = 0.2,
    ):
        """
        Initialize state based on the specified mode.
        Parameters:
        - N: Size of the board.
        - rng: Random number generator.
        - mode: Initialization mode ('noisy_latin_square', 'layer_balanced_random', 'random_latin_square', 'random').
        Returns:
        - An instance of StackState initialized according to the specified mode.
        """
        if mode == "noisy_latin_square":
            return cls.noisy_latin_square(N=N, rng=rng, p=p)
        elif mode == "layer_balanced_random":
            return cls.layer_balanced_random(N=N, rng=rng)
        elif mode == "random_latin_square":
            return cls.random_latin_square(N=N, rng=rng)
        elif mode == "random":
            return cls.random(N=N, rng=rng)
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

    def copy(self):
        """
        Deep copy
        """
        return StackState(self.heights.copy())

    def iter_queens(self) -> Iterable[Coord3D]:
        """
        Provides the full set of queen coordinates [do better]
        Interface for the energy model to counts conflicts, initialize line counts, ...
        """
        for i in range(self.N):
            for j in range(self.N):
                k = self.heights[i, j]
                yield (i + 1, j + 1, int(k))

    def as_array(self) -> np.ndarray:
        return self.heights

    #! Helpers for proposal

    def get_height(self, i: int, j: int) -> int:
        """
        Get queen's height for stack (i,j)
        """
        return int(self.heights[i - 1, j - 1])

    def set_height(self, i: int, j: int, k: int) -> None:
        """
        Set the queen's height at stack (i,j) to k
        """
        assert 1 <= k <= self.N
        self.heights[i - 1, j - 1] = k


@dataclass
class ConstraintStackState(State):
    """
    One queen per "vertical stack" (i,j,*)
    Internally stored as heights[i, j] = k in {1,...,N}
    """

    heights: np.ndarray  #! shape (N, N), values in [1..N]

    def __init__(self, heights: np.ndarray):

        N = heights.shape[0]
        super().__init__(N)
        assert heights.shape == (N, N), f"Heights shape must be ({N}, {N})"
        # Vérifie que chaque colonne est une permutation de 1..N
        for j in range(N):
            if set(heights[:, j]) != set(range(1, N + 1)):
                raise ValueError(f"Column {j} is not a permutation of 1..N")
        self.heights = heights

    @classmethod
    def random_latin_square(cls, N: int, rng=np.random.default_rng()):
        """
        Generate a random Latin square for constrained stacks:
        Each column is a permutation of 1..N
        """
        print("Generating random Latin square for constrained stacks")
        heights = np.zeros((N, N), dtype=int)
        for j in range(N):
            heights[:, j] = rng.permutation(np.arange(1, N + 1))
        return cls(heights)

    @classmethod
    def init_state(
        cls, N: int, rng=np.random.default_rng(), mode: str = "random_latin_square"
    ):
        """
        Initialize state based on the specified mode.
        Parameters:
        - N: Size of the board.
        - rng: Random number generator.
        - mode: Initialization mode ('random_latin_square').
        Returns:
        - An instance of ConstraintStackState initialized according to the specified mode.
        """
        if mode == "random_latin_square":
            return cls.random_latin_square(N=N, rng=rng)
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

    def copy(self):
        """
        Deep copy
        """
        return ConstraintStackState(self.heights.copy())

    def iter_queens(self) -> Iterable[Coord3D]:
        """
        Provides the full set of queen coordinates [do better]
        Interface for the energy model to counts conflicts, initialize line counts, ...
        """
        for i in range(self.N):
            for j in range(self.N):
                k = self.heights[i, j]
                yield (i + 1, j + 1, int(k))

    def as_array(self) -> np.ndarray:
        return self.heights

    #! Helpers for proposal

    def get_height(self, i: int, j: int) -> int:
        """
        Get queen's height for stack (i,j)
        """
        return int(self.heights[i - 1, j - 1])

    def set_height(self, i: int, j: int, k: int) -> None:
        """
        Set the queen's height at stack (i,j) to k
        """
        assert 1 <= k <= self.N, f"Height k={k} out of bounds"
        self.heights[i - 1, j - 1] = k
