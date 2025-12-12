# ! Main script to run MCMC annealing
import os
import numpy as np
import argparse
import concurrent.futures
import time
import matplotlib.pyplot as plt
from utils.plot_utils import plot_results, _aggregate_time_series, plot_sa_vs_constant
from state_space.states import StackState, ConstraintStackState
from state_space.geometry import Board, LineIndex
from energy.energy_model import EnergyModel
from mcmc.proposals import (
    SingleStackMove,
    SingleConstraintStackMove,
    SingleStackRandomHeightProposal,
)
from mcmc.chain import MCMCChain
from mcmc.annealing import (
    AnnealingSchedule,
    LinearSchedule,
    run_simulated_annealing,
    GeometricSchedule,
    ConstantSchedule,
    calibrate_initial_temperature,
)
from mcmc.proposals import SingleConstraintStackSwapProposal


def main(
    N: int,
    number_of_steps: int,
    rng_seed: int,
    state: StackState | ConstraintStackState = StackState,
    mode_init: str = "noisy_latin_square",
    T_initial: float = None,
    alpha: float = None,
    max_steps: int = None,
    verbose_every: int = 1000,
    detailed_stats: bool = False,
    is_watched: bool = False,
    noisy_p: float = 0.2,
    schedule_type: str = "geometric",
) -> None:
    # ? Initialize random number generator
    rng = np.random.default_rng(rng_seed)

    # ? Initialize state (example: empty stacks)
    """ Ici il pourrait être intéressant d'ajouter une logique pour initialiser des états différents respectant certaines contraintes """
    if mode_init == "noisy_latin_square":
        state = state.init_state(N=N, rng=rng, mode=mode_init, p=noisy_p)
    else:
        state = state.init_state(N=N, rng=rng, mode=mode_init)

    # ? Initialize energy model (example: dummy geometry and line index)
    geometry = Board(N=N)  # ? Initialize Board object
    line_index = LineIndex(
        geometry=geometry, include_vertical=False
    )  # ? Initialize LineIndex object
    energy_model = EnergyModel(geometry=geometry, line_index=line_index)
    energy_model.initialize(state)

    # ? Initialize proposal mechanism
    proposal = (
        SingleStackRandomHeightProposal(N)
        if isinstance(state, StackState)
        else SingleConstraintStackSwapProposal(N)
    )
    # ? Initialize MCMC chain
    mcmc_chain = MCMCChain(state=state, energy_model=energy_model, proposal=proposal)
    mcmc_chain_calibration = MCMCChain(
        state=state, energy_model=energy_model, proposal=proposal
    )
    # T_initial = calibrate_initial_temperature(
    #     mcmc_chain_calibration, target_acceptance_rate=0.85, n_samples=5000, rng=rng
    # )
    print(f"Calibrated initial temperature: T0 = {T_initial:.4f}")

    if T_initial is None or max_steps is None:
        print("You must provide T_initial and max_steps")
        return

    if schedule_type == "geometric":
        if alpha is None:
            print("You must provide alpha for geometric annealing!")
            return
        annealing_schedule = GeometricSchedule(
            T_initial=T_initial,
            alpha=alpha,
            max_steps=max_steps,
        )
    elif schedule_type == "constant":
        annealing_schedule = ConstantSchedule(
            T=T_initial,
            max_steps=max_steps,
        )

    # ? Run annealing
    try:
        history = run_simulated_annealing(
            mcmc_chain,
            annealing_schedule,
            rng,
            verbose_every,
            detailed_stats,
            is_watched,
        )
    except Exception as e:
        print(f"Simulated Annealing failed with error: {e}")
        return {"error": str(e)}
    return history


def run_single_simulation(args_dict: dict) -> dict:
    """
    Wrapper to run a single simulation with given arguments.
    """
    seed = args_dict["rng_seed"]
    try:
        # We can override verbose_every here if needed to not flood the output
        args_dict["verbose_every"] = (
            10000000
            if args_dict.get("is_watched", False)
            else args_dict["verbose_every"]
        )
        print(f"--> Start simulation Seed {seed} on process {os.getpid()}")

        result = main(**args_dict)
        result["seed"] = seed
        print(
            f"<-- End simulation Seed {seed}. Energie: {result['energy'][-1]:.4f}, Queen attacked : {result['attacked_queens'][-1]:.4f} on process {os.getpid()} in {len(result['energy'])} steps"
        )
        return result
    except AssertionError as e:
        print(f"!! Simulation Seed {seed} failed with error: {e}")
        return {"seed": seed, "error": str(e)}
    

def experiment_sa_vs_constant(
    N: int,
    n_simulations: int,
    base_seed: int,
    max_workers: int | None,
    T_initial: float,
    alpha: float,
    max_steps: int,
    mode_init: str,
    state_type: str,
    noisy_p: float = 0.2,
):
    common_kwargs = dict(
        number_of_steps=max_steps,
        T_initial=T_initial,
        alpha=alpha,
        max_steps=max_steps,
        mode_init=mode_init,
        state=StackState if state_type == "stack" else ConstraintStackState,
        verbose_every=1000000,
        detailed_stats=False,
        noisy_p=noisy_p,
    )

    # With simulated annealing
    sa_results = run_batch(
        N=N,
        schedule_type="geometric",
        n_simulations=n_simulations,
        base_seed=base_seed,
        max_workers=max_workers,
        **common_kwargs,
    )

    # Without simulated annealing (constant T)
    const_results = run_batch(
        N=N,
        schedule_type="constant",
        n_simulations=n_simulations,
        base_seed=base_seed + 10,  # different seeds if you like
        max_workers=max_workers,
        **common_kwargs,
    )

    # Now either:
    #  - call a modified plot_results that can take two groups, or
    #  - write a small dedicated plotting function:
    plot_sa_vs_constant(
    N=N,
    sa_results=sa_results,
    const_results=const_results,
    mode_init=mode_init,
    state_type=state_type,
    noisy_p=noisy_p,
    save_path="plots/energy_sa_vs_constant_N12.png",  
    )

def run_batch(
    N: int,
    schedule_type: str,
    n_simulations: int,
    base_seed: int,
    max_workers: int | None,
    **kwargs,
    ) -> list[dict]:
    """
    Run n_simulations in parallel for a given schedule_type and return histories.
    kwargs are forwarded to main().
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    simulations_configs = []
    for i in range(n_simulations):
        should_be_watched = (i % max_workers == 0)
        cfg = {
            "N": N,
            "rng_seed": base_seed + i,
            "schedule_type": schedule_type,
            **kwargs,
            "is_watched": should_be_watched,
        }
                # noisy_p etc can be in kwargs
        simulations_configs.append(cfg)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_simulation, cfg): cfg
            for cfg in simulations_configs
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel MCMC Annealing for N-Stacks")
    parser.add_argument("--N", type=int, default=8, help="Size of the board (N x N)")
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Number of MCMC steps"
    )
    parser.add_argument(
        "--base_seed", type=int, default=42, help="Random number generator seed"
    )
    parser.add_argument(
        "--state_type",
        type=str,
        default="stack",
        help="Type of state: 'stack' or 'constraint'",
    )
    parser.add_argument(
        "--mode_init",
        type=str,
        default="random_latin_square",
        help="Initialization mode: 'noisy_latin_square' or 'layer_balanced_random' or 'random_latin_square'",
    )

    parser.add_argument(
        "--T_initial",
        type=float,
        default=None,
        help="Initial temperature for geometric annealing",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Cooling factor for geometric annealing",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max steps for geometric annealing"
    )
    parser.add_argument(
        "--verbose_every",
        type=int,
        default=1000,
        help="Print energy/diagnostics every n steps",
    )
    parser.add_argument(
        "--stats",
        type=bool,
        default=False,
        help="Print more detailed stats for detecting pathological cases",
    )
    parser.add_argument(
        "--n_simulations", type=int, default=10, help="Number of parallel simulations"
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Max number of parallel workers"
    )
    parser.add_argument(
        "--noisy_p",
        type=float,
        default=0.2,
        help="Probability parameter for noisy_latin_square initialization (only used if mode_init is 'noisy_latin_square')",
    )
    parser.add_argument(
    "--schedule_type",
    type=str,
    default="geometric",
    choices=["geometric", "constant"],
    help="Use 'geometric' for simulated annealing or 'constant' for fixed-T baseline.",
    )

    parser.add_argument(
    "--experiment",
    type=str,
    default="single",
    choices=["single", "sa_vs_constant"],
    help="Run a single annealing run (default) or the SA vs Constant-T experiment.",
    )

    args = parser.parse_args()

    '''max_workers = (
        args.max_workers if args.max_workers else os.cpu_count()
    )  # ? Use all available CPUs if not specified

    simulations_configs = []

    for i in range(args.n_simulations):
        # Is watch is observed for each "batch" of simulations. For example if n_simulations=10 and max_workers=2, 5 batches will be run with is_watched=True
        should_be_watched = (
            i % max_workers == 0
        )  # Here if we don't want to have tqdm we can set it to False
        config = {
            "N": args.N,
            "number_of_steps": args.steps,
            "rng_seed": args.base_seed + i,
            "T_initial": args.T_initial,
            "mode_init": args.mode_init,
            "state": StackState if args.state_type == "stack" else ConstraintStackState,
            "alpha": args.alpha,
            "max_steps": args.max_steps,
            "verbose_every": args.verbose_every,
            "detailed_stats": args.stats,
            "is_watched": should_be_watched,
            "schedule_type": args.schedule_type,
        }
        # Adding noisy_p only if needed
        if args.mode_init == "noisy_latin_square":
            config["noisy_p"] = args.noisy_p
        simulations_configs.append(config)

    # ? Run simulations in parallel
    results = []
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_simulation, cfg): cfg
            for cfg in simulations_configs
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)

    duration = time.time() - start_time

    print(f"All simulations completed in {duration:.2f} seconds.")

    # ? Plot results
    plot_results(
        N=args.N,
        mode_init=args.mode_init,
        state_type=args.state_type,
        results=results,
        noisy_p=args.noisy_p,
        plot_cube=False,
    )'''

    max_workers = args.max_workers if args.max_workers else os.cpu_count()

    # ------------------------------------------------------------
    # CASE 1: STANDARD OPERATION (your current workflow)
    # ------------------------------------------------------------
    if args.experiment == "single":
        # Build configs for *one* schedule type (whatever user provided)
        simulations_configs = []

        for i in range(args.n_simulations):
            should_be_watched = (i % max_workers == 0)
            config = {
                "N": args.N,
                "number_of_steps": args.steps,
                "rng_seed": args.base_seed + i,
                "T_initial": args.T_initial,
                "mode_init": args.mode_init,
                "state": StackState if args.state_type == "stack" else ConstraintStackState,
                "alpha": args.alpha,
                "max_steps": args.max_steps,
                "verbose_every": args.verbose_every,
                "detailed_stats": args.stats,
                "is_watched": should_be_watched,
                "schedule_type": "geometric",   # or use a CLI arg if desired
            }
            if args.mode_init == "noisy_latin_square":
                config["noisy_p"] = args.noisy_p
            simulations_configs.append(config)

        # Parallel execution
        results = []
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_simulation, cfg): cfg for cfg in simulations_configs}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        duration = time.time() - start_time
        print(f"All simulations completed in {duration:.2f} seconds.")

        # Plot using your normal function
        plot_results(
            N=args.N,
            mode_init=args.mode_init,
            state_type=args.state_type,
            results=results,
            noisy_p=args.noisy_p,
            plot_cube=False,
        )

    # ------------------------------------------------------------
    # CASE 2: SA VS CONSTANT-T EXPERIMENT
    # ------------------------------------------------------------
    elif args.experiment == "sa_vs_constant":

        print("\n=== Running SA vs Constant-T experiment ===\n")

        # Common kwargs forwarded to "run_batch"
        common_kwargs = dict(
            number_of_steps=args.max_steps,        # or args.steps
            T_initial=args.T_initial,
            alpha=args.alpha,
            max_steps=args.max_steps,
            mode_init=args.mode_init,
            state=StackState if args.state_type == "stack" else ConstraintStackState,
            verbose_every=args.verbose_every,
            detailed_stats=args.stats,
            noisy_p=args.noisy_p,
        )

        # Run SA batch
        sa_results = run_batch(
            N=args.N,
            schedule_type="geometric",
            n_simulations=args.n_simulations,
            base_seed=args.base_seed,
            max_workers=max_workers,
            **common_kwargs,
        )

        # Run Constant-T batch
        const_results = run_batch(
            N=args.N,
            schedule_type="constant",
            n_simulations=args.n_simulations,
            base_seed=args.base_seed + 10000,
            max_workers=max_workers,
            **common_kwargs,
        )

        # Produce comparison plot
        plot_sa_vs_constant(
            N=args.N,
            sa_results=sa_results,
            const_results=const_results,
            mode_init=args.mode_init,
            state_type=args.state_type,
            noisy_p=args.noisy_p,
            save_path=f"plots/sa_vs_const_N{args.N}.png",
        )


