import numpy as np
import argparse
import os

from games.rps_games import (
    RockPaperScissors,
    BiasedRockPaperScissors,
    AsymmetricRockPaperScissors,
    RockPaperScissorsWithNoise
)
from algorithms.fictitious_play import FictitiousPlay, SmoothFictitiousPlay
from algorithms.exp3 import EXP3
from algorithms.hedge import Hedge
from visualizations.simplex_plot import plot_two_player_simplex


def run_rps_experiments(game_name='all', algorithm_name='all', num_iterations=300):
    """
    Run experiments on Rock Paper Scissors games with learning algorithms.
    """
    # Define games (only 3-action games)
    games = {
        'RPS': RockPaperScissors(),
        'BiasedRPS': BiasedRockPaperScissors(rock_bonus=0.2),
        'AsymmetricRPS': AsymmetricRockPaperScissors(p2_scale=0.7),
        'NoisyRPS': RockPaperScissorsWithNoise(noise_level=0.1)
    }

    # Learning rate configuration for EXP3 and Hedge
    eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}

    # Define algorithms
    algorithms = {
        'FictitiousPlay': lambda game, num_iter: FictitiousPlay(game, num_iter),
        'SmoothFP_T0.1': lambda game, num_iter: SmoothFictitiousPlay(game, num_iter, temperature=0.1),
        'SmoothFP_T0.5': lambda game, num_iter: SmoothFictitiousPlay(game, num_iter, temperature=0.5),
        'EXP3': lambda game, num_iter: EXP3(game, num_iter, eta_config),
        'Hedge': lambda game, num_iter: Hedge(game, num_iter, eta_config)
    }

    # Single initial condition for Fictitious Play - action counts
    # Format: [p1_action_counts, p2_action_counts]
    fp_initial_condition = [np.array([5.0, 1.0, 1.0]), np.array([1.0, 5.0, 1.0])]

    # Single initial condition for EXP3/Hedge - log-space scores
    # Format: [p1_log_scores, p2_log_scores]
    exp_initial_condition = [np.array([1.5, -1.5, 0.0]), np.array([-1.0, 1.0, 0.0])]

    # Determine which games to run
    games_to_run = {}
    if game_name == 'all':
        games_to_run.update(games)
    elif game_name in games:
        games_to_run[game_name] = games[game_name]
    else:
        print(f"Unknown game: {game_name}")
        return

    # Determine which algorithms to run
    algorithms_to_run = algorithms if algorithm_name == 'all' else {algorithm_name: algorithms[algorithm_name]}

    # Run experiments
    for gname, game in games_to_run.items():
        print(f'\nRunning experiments for {gname}...')

        action_labels = ['Rock', 'Paper', 'Scissors']

        # Collect trajectories for all algorithms
        player1_trajectories = []
        player2_trajectories = []
        algorithm_names = []

        for aname, alg_factory in algorithms_to_run.items():
            print(f'  Running {aname}...')

            # Choose appropriate initial conditions based on algorithm
            if aname in ['EXP3', 'Hedge']:
                init_scores = exp_initial_condition
            else:  # Fictitious Play variants
                init_scores = fp_initial_condition

            alg = alg_factory(game, num_iterations)
            alg.run(initial_scores=init_scores)

            # Extract player strategies
            # Convert to numpy array if needed
            strategies = np.array(alg.strategies)
            # strategies has shape (num_iterations, num_players, num_actions)
            p1_trajectory = strategies[:, 0, :]
            p2_trajectory = strategies[:, 1, :]

            player1_trajectories.append(p1_trajectory)
            player2_trajectories.append(p2_trajectory)
            algorithm_names.append(aname)

        # Create combined visualization
        print(f'Creating visualization for {gname}...')
        plot_two_player_simplex(
            player1_trajectories,
            player2_trajectories,
            gname,
            'AllAlgorithms' if len(algorithms_to_run) > 1 else list(algorithms_to_run.keys())[0],
            action_labels_p1=action_labels,
            action_labels_p2=action_labels,
            save_dir='visualizations',
            algorithm_names=algorithm_names
        )

    print('\nAll experiments completed!')
    print('Visualizations saved in visualizations/ directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Rock Paper Scissors experiments with learning algorithms'
    )
    parser.add_argument(
        '--game',
        type=str,
        default='all',
        choices=['all', 'RPS', 'BiasedRPS', 'AsymmetricRPS', 'NoisyRPS'],
        help='Game to run experiments on'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='all',
        choices=['all', 'FictitiousPlay', 'SmoothFP_T0.1', 'SmoothFP_T0.5', 'EXP3', 'Hedge'],
        help='Algorithm to use'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=1000,
        help='Number of iterations to run'
    )

    args = parser.parse_args()

    run_rps_experiments(
        game_name=args.game,
        algorithm_name=args.algorithm,
        num_iterations=args.num_iterations
    )
