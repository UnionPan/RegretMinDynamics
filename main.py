
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import argparse
import os

from games.implemented_games import (
    PureCoordination,
    CoordinationWithSpectator,
    MatchingPenniesWithTwist,
    MatchingPenniesWithOutsideOption
)
from algorithms.exp3 import EXP3
from algorithms.pga import ProjectedGradientAscent
from algorithms.opt_ftrl import OptimisticFTRL
from algorithms.regret_matching import RegretMatching
from algorithms.regret_matching_softmax import RegretMatchingSoftmax
from algorithms.hedge import Hedge
from algorithms.b_ftrl_exp3 import BFTL_EXP3

def plot_strategy_evolution(trajectories, game_name, algorithm_name, equilibria=None):
    # Ensure visualizations directory exists
    os.makedirs('visualizations', exist_ok=True)

    # Pre-convert trajectories to numpy arrays once
    trajectories = [np.array(strategies) for strategies in trajectories]

    # Pre-compute cube geometry (only once)
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    # Default to empty list if no equilibria provided
    if equilibria is None:
        equilibria = []

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set view angle and static properties once
    ax.view_init(10, 22)  # Eye-level view: elevation=10 (lower), azimuth=22
    ax.grid(False)

    # Create GIF
    filenames = []
    num_steps = len(trajectories[0])

    for t in range(0, num_steps, 10):  # every 10 steps
        ax.cla()

        # Set plot properties
        ax.grid(False)
        ax.set_xlabel('Player 1 - P(Action 0)')
        ax.set_ylabel('Player 2 - P(Action 0)')
        ax.set_zlabel('Player 3 - P(Action 0)')
        ax.set_title(f'{game_name} - {algorithm_name} - Strategy Evolution')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.view_init(10, 22)  # Eye-level view: elevation=10 (lower), azimuth=22

        # Draw cube edges once per frame
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, color='gray', linewidth=1, alpha=0.3)

        # Plot Nash Equilibria as gold stars
        for idx, eq in enumerate(equilibria):
            x, y, z, label = eq
            # Only add to legend for the first equilibrium point
            if idx == 0:
                ax.scatter([x], [y], [z], color='gold', marker='*', s=300,
                          edgecolors='black', linewidths=1.5, alpha=0.9, zorder=100,
                          label='Nash Equilibrium')
            else:
                ax.scatter([x], [y], [z], color='gold', marker='*', s=300,
                          edgecolors='black', linewidths=1.5, alpha=0.9, zorder=100)

        # Add legend if equilibria exist
        if equilibria:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Plot trajectories with vectorized color computation
        for strategies in trajectories:
            x = strategies[:t+1, 0, 0]
            y = strategies[:t+1, 1, 0]
            z = strategies[:t+1, 2, 0]

            if len(x) > 1:
                # Compute all colors at once using vectorization
                num_segments = len(x) - 1
                intensities = np.arange(num_segments) / max(1, t)

                # Use Line3DCollection from mpl_toolkits for much faster rendering
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                segments = [[[x[i], y[i], z[i]], [x[i+1], y[i+1], z[i+1]]]
                           for i in range(num_segments)]
                colors = np.column_stack([
                    0.2 + 0.6 * (1 - intensities),
                    0.4 + 0.4 * (1 - intensities),
                    np.full(num_segments, 0.9),
                    np.full(num_segments, 0.8)  # alpha
                ])

                lc = Line3DCollection(segments, colors=colors, linewidths=2)
                ax.add_collection(lc)

        filename = f'visualizations/frame_{t}.png'
        plt.savefig(filename, dpi=80, bbox_inches='tight')  # Reduced DPI for speed
        filenames.append(filename)

    # Write GIF with optimized settings
    with imageio.get_writer(f'visualizations/{game_name}_{algorithm_name}.gif',
                           mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up frames
    for filename in filenames:
        os.remove(filename)

    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='all', help='Game to run')
    parser.add_argument('--algorithm', type=str, default='all', help='Algorithm to run')
    args = parser.parse_args()

    num_iterations = 1000
    eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}
    # With full-feedback (no importance sampling), we can use normal learning rates
    hedge_eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}
    opt_ftrl_eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}
    delta_config = {'initial_delta': 0.1, 'decay_rate': -0.15} # For B-FTRL EXP3
    bftrl_eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}

    games = {
        'MatchingPenniesWithOutsideOption': MatchingPenniesWithOutsideOption(),
        'CoordinationWithSpectator': CoordinationWithSpectator(),
        'MatchingPenniesWithTwist': MatchingPenniesWithTwist(),
        'PureCoordination': PureCoordination()
    }

    algorithms = {
        'OptFTRL': OptimisticFTRL,
        'Hedge': Hedge,
        'EXP3': EXP3,
        'PGA': ProjectedGradientAscent,
        'RegretMatching': RegretMatching,
        'RegretMatchingSoftmax': RegretMatchingSoftmax,
        'BFTL_EXP3': BFTL_EXP3,
    }

    # Game-specific initial scores
    # Each entry represents [player1_scores, player2_scores, player3_scores]
    # Scores are in log-space for softmax-based algorithms (like OptFTRL, EXP3, Hedge)
    # and direct strategy space for projection-based algorithms (like PGA)

    # Initial scores for PureCoordination
    initial_scores_pure_coordination = [
        [np.array([0.3, -0.3]), np.array([-0.2, 0.2]), np.array([0.25, -0.25])],
        [np.array([-0.25, 0.25]), np.array([0.3, -0.3]), np.array([-0.3, 0.3])],
        [np.array([0.2, -0.15]), np.array([-0.3, 0.3]), np.array([0.15, -0.2])],
        [np.array([1.0, -0.5]), np.array([-0.5, 1.0]), np.array([0.8, -0.8])],
        [np.array([2.5, -2.5]), np.array([2.5, -2.5]), np.array([2.5, -2.5])],
        [np.array([-2.5, 2.5]), np.array([-2.5, 2.5]), np.array([-2.5, 2.5])],
    ]

    # Initial scores for CoordinationWithSpectator
    # Player 3 is a spectator (always gets 0 payoff), so we vary P1 and P2 more
    initial_scores_coordination_spectator = [
        # P1 and P2 start with opposite biases, P3 varies across spectrum
        [np.array([1.5, -1.5]), np.array([-1.5, 1.5]), np.array([2.0, -2.0])],
        [np.array([1.5, -1.5]), np.array([-1.5, 1.5]), np.array([-2.0, 2.0])],
        [np.array([1.5, -1.5]), np.array([-1.5, 1.5]), np.array([0.0, 0.0])],

        # P1 and P2 start with same biases, P3 varies
        [np.array([1.5, -1.5]), np.array([1.5, -1.5]), np.array([1.5, -1.5])],
        [np.array([-1.5, 1.5]), np.array([-1.5, 1.5]), np.array([-1.5, 1.5])],

        # P1 and P2 near uniform, P3 varies strongly
        [np.array([0.3, -0.3]), np.array([-0.3, 0.3]), np.array([2.5, -2.5])],
        [np.array([0.3, -0.3]), np.array([-0.3, 0.3]), np.array([-2.5, 2.5])],

        # Strong corners for P1 and P2
        [np.array([2.5, -2.5]), np.array([2.5, -2.5]), np.array([1.0, -1.0])],
        [np.array([-2.5, 2.5]), np.array([-2.5, 2.5]), np.array([-1.0, 1.0])],

        # Mixed scenarios
        [np.array([2.0, -2.0]), np.array([-2.0, 2.0]), np.array([1.5, -1.5])],
        [np.array([-2.0, 2.0]), np.array([2.0, -2.0]), np.array([-1.5, 1.5])],
    ]

    # Initial scores for MatchingPenniesWithTwist
    # NOTE: P1 has a dominant strategy (action 1 always gives 0.1, action 0 gives 0)
    # To create richer trajectories, we need diverse starting points with smaller magnitudes
    initial_scores_matching_pennies_twist = [
        # Near center - slow convergence
        [np.array([0.1, -0.1]), np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        [np.array([0.0, 0.0]), np.array([0.2, -0.2]), np.array([-0.2, 0.2])],
        [np.array([-0.15, 0.15]), np.array([0.15, -0.15]), np.array([0.1, -0.1])],

        # Moderate diversity - P2 and P3 start at different corners
        [np.array([0.5, -0.5]), np.array([1.2, -1.2]), np.array([-1.2, 1.2])],
        [np.array([0.5, -0.5]), np.array([-1.2, 1.2]), np.array([1.2, -1.2])],
        [np.array([0.3, -0.3]), np.array([0.8, -0.8]), np.array([-0.8, 0.8])],

        # P1 starts favoring action 0 (will need to switch)
        [np.array([0.8, -0.8]), np.array([1.0, -1.0]), np.array([0.5, -0.5])],
        [np.array([1.0, -1.0]), np.array([-1.0, 1.0]), np.array([1.0, -1.0])],

        # Mixed P2 and P3 positions
        [np.array([0.2, -0.2]), np.array([1.5, -1.5]), np.array([0.3, -0.3])],
        [np.array([0.4, -0.4]), np.array([-1.5, 1.5]), np.array([-0.3, 0.3])],
        [np.array([0.6, -0.6]), np.array([0.5, -0.5]), np.array([1.5, -1.5])],
    ]

    # Initial scores for MatchingPenniesWithOutsideOption
    # NOTE: BFTL_EXP3 is sensitive to extreme initial values - keep magnitudes moderate
    initial_scores_matching_pennies_outside = [
        # Near center - allows gradual exploration
        [np.array([0.4, -0.4]), np.array([-0.4, 0.4]), np.array([0.3, -0.3])],
        [np.array([-0.3, 0.3]), np.array([0.5, -0.5]), np.array([-0.4, 0.4])],
        [np.array([0.2, -0.2]), np.array([0.3, -0.3]), np.array([-0.2, 0.2])],

        # Moderate biases - not too extreme
        [np.array([1.0, -1.0]), np.array([-1.0, 1.0]), np.array([0.8, -0.8])],
        [np.array([-1.0, 1.0]), np.array([1.0, -1.0]), np.array([-0.8, 0.8])],
        [np.array([0.8, -0.8]), np.array([0.8, -0.8]), np.array([0.6, -0.6])],

        # Corners but not extreme (reduced from ±1.8 to ±1.2)
        [np.array([1.2, -1.2]), np.array([1.2, -1.2]), np.array([1.0, -1.0])],
        [np.array([-1.2, 1.2]), np.array([-1.2, 1.2]), np.array([-1.0, 1.0])],

        # Mixed configurations
        [np.array([0.6, -0.6]), np.array([-0.8, 0.8]), np.array([0.5, -0.5])],
        [np.array([-0.6, 0.6]), np.array([0.8, -0.8]), np.array([-0.5, 0.5])],
    ]

    # PGA-specific initial scores (used for all games with PGA)
    pga_initial_scores_config = [
        [np.array([0.1, -0.1]), np.array([-0.05, 0.05]), np.array([0.05, -0.05])],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, -0.5])],
        [np.array([2.0, -2.0]), np.array([2.0, -2.0]), np.array([2.0, -2.0])],
        [np.array([-2.0, 2.0]), np.array([-2.0, 2.0]), np.array([-2.0, 2.0])],
        [np.array([1.5, -0.5]), np.array([-0.5, 1.5]), np.array([0.8, -0.8])],
    ]

    # Map games to their initial scores
    game_initial_scores = {
        'PureCoordination': initial_scores_pure_coordination,
        'CoordinationWithSpectator': initial_scores_coordination_spectator,
        'MatchingPenniesWithTwist': initial_scores_matching_pennies_twist,
        'MatchingPenniesWithOutsideOption': initial_scores_matching_pennies_outside,
    }

    # Nash Equilibrium points for each game
    # Format: list of (x, y, z) coordinates where x=P1's prob of action 0, etc.
    game_equilibria = {
        'PureCoordination': [
            (1.0, 1.0, 1.0, 'NE1: All action 0'),
            (1.0, 0.0, 0.0, 'NE2: P1=0, P2=1, P3=1'),
            (0.0, 1.0, 0.0, 'NE3: P1=1, P2=0, P3=1'),
            (0.0, 0.0, 1.0, 'NE4: P1=1, P2=1, P3=0'),
        ],
        'CoordinationWithSpectator': [
            # Two equilibrium edges (sample 3 points per edge)
            (1.0, 1.0, 0.0, 'NE: P1=0, P2=0, P3=0'),
            (1.0, 1.0, 0.5, 'NE: P1=0, P2=0, P3=0.5'),
            (1.0, 1.0, 1.0, 'NE: P1=0, P2=0, P3=1'),
            (0.0, 0.0, 0.0, 'NE: P1=1, P2=1, P3=0'),
            (0.0, 0.0, 0.5, 'NE: P1=1, P2=1, P3=0.5'),
            (0.0, 0.0, 1.0, 'NE: P1=1, P2=1, P3=1'),
        ],
        'MatchingPenniesWithTwist': [
            (0.0, 0.5, 0.5, 'NE: P1=1 (dom), P2=P3=0.5'),
        ],
        'MatchingPenniesWithOutsideOption': [
            (0.5, 0.5, 0.5, 'NE: Mixed (approx)'),
            (0.0, 0.0, 1.0, 'PO: Pareto optimal'),
        ],
    }

    games_to_run = games.keys() if args.game == 'all' else [args.game]
    algorithms_to_run = algorithms.keys() if args.algorithm == 'all' else [args.algorithm]

    for game_name in games_to_run:
        game = games[game_name]
        for alg_name, alg_class in algorithms.items():
            if alg_name not in algorithms_to_run: # Added this line to filter algorithms
                continue
            print(f'Running simulation for {game_name} with {alg_name}...')

            # Select initial scores based on algorithm and game
            if alg_name == 'PGA':
                current_initial_scores = pga_initial_scores_config
            else:
                current_initial_scores = game_initial_scores[game_name]

            trajectories = []
            for initial_scores in current_initial_scores:
                if alg_name == 'RegretMatching':
                    alg = alg_class(game, num_iterations)
                elif alg_name == 'RegretMatchingSoftmax':
                    alg = alg_class(game, num_iterations, eta_config)
                elif alg_name == 'BFTL_EXP3':
                    alg = alg_class(game, num_iterations, bftrl_eta_config, delta_config)
                elif alg_name == 'Hedge':
                    alg = alg_class(game, num_iterations, hedge_eta_config)
                elif alg_name == 'OptFTRL':
                    alg = alg_class(game, num_iterations, opt_ftrl_eta_config)
                else:
                    alg = alg_class(game, num_iterations, eta_config)
                alg.run(initial_scores=initial_scores)
                trajectories.append(alg.strategies)

            # Get equilibria for this game
            equilibria = game_equilibria.get(game_name, [])
            plot_strategy_evolution(trajectories, game_name, alg_name, equilibria=equilibria)
