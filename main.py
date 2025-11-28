# ! Main script to run MCMC annealing
import numpy as np
import argparse
from state_space.states import StackState, ConstraintStackState
from state_space.geometry import Board, LineIndex
from energy.energy_model import EnergyModel
from mcmc.proposals import (
    SingleStackMove,
    SingleConstraintStackMove,
    SingleStackRandomHeightProposal,
)
from mcmc.chain import MCMCChain
from mcmc.annealing import AnnealingSchedule
from mcmc.proposals import SingleConstraintStackSwapProposal


def main(N: int, number_of_steps: int, rng_seed: int) -> None:
    # ? Initialize random number generator
    rng = np.random.default_rng(rng_seed)

    # ? Initialize state (example: empty stacks)
    initial_heights = np.zeros((N, N), dtype=int)
    state = StackState(heights=initial_heights)

    # ? Initialize energy model (example: dummy geometry and line index)
    geometry = Board(N=N)  # ? Initialize Board object
    line_index = LineIndex(
        geometry=geometry, include_vertical=False
    )  # ? Initialize LineIndex object
    energy_model = EnergyModel(geometry=geometry, line_index=line_index)
    energy_model.initialize(state)

    # ? Initialize proposal mechanism
    proposal = SingleStackRandomHeightProposal(
        N
    )  # ? Replace with actual proposal class

    # ? Initialize MCMC chain
    mcmc_chain = MCMCChain(state=state, energy_model=energy_model, proposal=proposal)

    # ? Define annealing schedule
    # temperatures = np.linspace(10.0, 0.1, num=100).tolist()  # ? Example schedule
    temperatures = [100]
    annealing_schedule = AnnealingSchedule(
        temperatures=temperatures,
        no_annealing=True,
        num_steps_per_temp=number_of_steps,
    )
    # ? Run annealing
    annealing_schedule.run(mcmc_chain, rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC Annealing for Stack States")
    parser.add_argument("--N", type=int, default=8, help="Size of the board (N x N)")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MCMC steps")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random number generator seed"
    )
    args = parser.parse_args()

    main(N=args.N, number_of_steps=args.steps, rng_seed=args.seed)
