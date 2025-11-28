#! Simulated annealing on top of the chain

import math
import random


# Pourquoi pas tester plusieurs fonction de refroidissement ? !!

class AnnealingSchedule:
    """
    Implements the Geometric (Exponential) cooling schedule for Simulated Annealing (SA).
    Cooling Law: T(k+1) = alpha * T(k)
    """
    
    def __init__(self, T_initial: float, alpha: float, max_steps: int):
        """
        Initializes the annealing schedule parameters.
        """
        if not (0 < alpha < 1):
            raise ValueError("The cooling factor 'alpha' must be strictly between 0 and 1.")
        if T_initial <= 0:
            raise ValueError("The initial temperature must be positive.")

        self.T_initial = T_initial
        self.alpha = alpha
        self.max_steps = max_steps
        self.current_step = 0
        self.current_T = T_initial

    def next_temperature(self) -> float:
        """
        Calculates and returns the temperature for the next step (k+1).
        """

        if self.current_step >= self.max_steps:
            # If the maximum number of steps is reached, the system is considered "frozen."
            return 0.0

        if self.current_step > 0:
            # Apply the geometric cooling law from the second step onwards (k=1, 2, ...).
            self.current_T *= self.alpha
        
        self.current_step += 1
        return self.current_T

    def get_current_T(self) -> float:
        """
        Returns the temperature of the current step (T(k)).
        """
        return self.current_T
