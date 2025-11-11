import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
import inspect

# Import game classes
from games.implemented_games import PureCoordination, CoordinationWithSpectator, MatchingPenniesWithTwist, MatchingPenniesWithOutsideOption
from games.rps_games import RockPaperScissors, BiasedRockPaperScissors, AsymmetricRockPaperScissors, RockPaperScissorsWithNoise

# Import algorithm classes
from algorithms.b_ftrl_exp3 import BFTL_EXP3
from algorithms.exp3 import EXP3
from algorithms.fictitious_play import FictitiousPlay, SmoothFictitiousPlay
from algorithms.hedge import Hedge
from algorithms.opt_ftrl import OptimisticFTRL
from algorithms.pga import ProjectedGradientAscent
from algorithms.regret_matching import RegretMatching
from algorithms.regret_matching_softmax import RegretMatchingSoftmax

st.title("Regret Minimization Dynamics")

# --- Mode Selection ---
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose a mode", ["Single Simulation", "Algorithm Comparison"])

# --- Game and Algorithm Selection ---
st.sidebar.header("Simulation Settings")

game_options = {
    "Rock Paper Scissors": RockPaperScissors,
    "Biased Rock Paper Scissors": BiasedRockPaperScissors,
    "Asymmetric Rock Paper Scissors": AsymmetricRockPaperScissors,
    "Noisy Rock Paper Scissors": RockPaperScissorsWithNoise,
    "Pure Coordination": PureCoordination,
    "Coordination with Spectator": CoordinationWithSpectator,
    "Matching Pennies with Twist": MatchingPenniesWithTwist,
    "Matching Pennies with Outside Option": MatchingPenniesWithOutsideOption,
}

game_equilibria = {
    'Pure Coordination': [
        (1.0, 1.0, 1.0, 'NE1'), (1.0, 0.0, 0.0, 'NE2'),
        (0.0, 1.0, 0.0, 'NE3'), (0.0, 0.0, 1.0, 'NE4'),
    ],
    'Coordination with Spectator': [
        (1.0, 1.0, 0.0, 'NE'), (1.0, 1.0, 0.5, 'NE'), (1.0, 1.0, 1.0, 'NE'),
        (0.0, 0.0, 0.0, 'NE'), (0.0, 0.0, 0.5, 'NE'), (0.0, 0.0, 1.0, 'NE'),
    ],
    'Matching Pennies with Twist': [(0.0, 0.5, 0.5, 'NE')],
    'Matching Pennies with Outside Option': [(0.5, 0.5, 0.5, 'NE')],
    'Rock Paper Scissors': [(1/3, 1/3, 1.0/3)], # For player 1
}


algorithm_options = {
    "BFTL-EXP3": BFTL_EXP3,
    "EXP3": EXP3,
    "Fictitious Play": FictitiousPlay,
    "Smooth Fictitious Play": SmoothFictitiousPlay,
    "Hedge": Hedge,
    "Optimistic FTRL": OptimisticFTRL,
    "Projected Gradient Ascent": ProjectedGradientAscent,
    "Regret Matching": RegretMatching,
    "Regret Matching Softmax": RegretMatchingSoftmax,
}

def is_rps_game(name):
    return "Rock Paper Scissors" in name

def barycentric_to_cartesian(strategies):
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    return np.dot(strategies, v)

def create_rps_animation(trajectories, num_iterations, game_name):
    # ... (implementation from previous turn)
    pass

def create_3d_cube_animation(trajectories, game_name, algorithm_name, equilibria=None):
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
    gif_path = f'visualizations/{game_name}_{algorithm_name}.gif'
    with imageio.get_writer(gif_path,
                           mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up frames
    for filename in filenames:
        os.remove(filename)

    plt.close(fig)
    return gif_path

def create_comparison_animation(trajectories_dict, game_name, num_iterations):
    os.makedirs('visualizations', exist_ok=True)
    filenames = []
    
    for t in range(0, num_iterations, 10):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
        
        # Player 1
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f"Player 1 Strategy (Iteration {t})")
        simplex_points = barycentric_to_cartesian(np.eye(3))
        ax1.add_patch(plt.Polygon(simplex_points, edgecolor='black', fill=False))
        _ = ax1.text(simplex_points[0, 0] - 0.1, simplex_points[0, 1], "Rock")
        _ = ax1.text(simplex_points[1, 0] + 0.05, simplex_points[1, 1], "Paper")
        _ = ax1.text(simplex_points[2, 0] - 0.05, simplex_points[2, 1] + 0.05, "Scissors")

        # Player 2
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f"Player 2 Strategy (Iteration {t})")
        ax2.add_patch(plt.Polygon(simplex_points, edgecolor='black', fill=False))
        _ = ax2.text(simplex_points[0, 0] - 0.1, simplex_points[0, 1], "Rock")
        _ = ax2.text(simplex_points[1, 0] + 0.05, simplex_points[1, 1], "Paper")
        _ = ax2.text(simplex_points[2, 0] - 0.05, simplex_points[2, 1] + 0.05, "Scissors")

        for algo_name, strategies in trajectories_dict.items():
            coords1 = barycentric_to_cartesian(strategies[:t+1, 0, :])
            coords2 = barycentric_to_cartesian(strategies[:t+1, 1, :])
            ax1.plot(coords1[:, 0], coords1[:, 1], linewidth=2, alpha=0.7, label=algo_name)
            ax2.plot(coords2[:, 0], coords2[:, 1], linewidth=2, alpha=0.7, label=algo_name)

        ax1.legend()
        ax2.legend()
        
        filename = f'visualizations/frame_{t}.png'
        plt.savefig(filename, dpi=80, bbox_inches='tight')
        filenames.append(filename)
        plt.close(fig)

    gif_path = f'visualizations/{game_name}_comparison.gif'
    with imageio.get_writer(gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)
        
    return gif_path


if mode == "Single Simulation":
    selected_game_name = st.sidebar.selectbox("Choose a game", list(game_options.keys()))
    selected_algo_name = st.sidebar.selectbox("Choose an algorithm", list(algorithm_options.keys()))
    num_iterations = st.sidebar.number_input("Number of Iterations", 100, 100000, 1000)
    num_initializations = st.sidebar.slider("Number of Initializations", 1, 10, 3)
    st.sidebar.header("Display Options")
    show_play_info = st.sidebar.checkbox("Show Iterative Play Information")
    show_strategy_plot = st.sidebar.checkbox("Show Strategy Evolution Plot")

    if st.sidebar.button("Run Simulation"):
        game_class = game_options[selected_game_name]
        game = game_class()
        
        algo_class = algorithm_options[selected_algo_name]
        
        initial_scores_map = {
            "Pure Coordination": [np.array([0.3, -0.3]), np.array([-0.2, 0.2]), np.array([0.25, -0.25])],
            "Coordination with Spectator": [np.array([1.5, -1.5]), np.array([-1.5, 1.5]), np.array([2.0, -2.0])],
            "Matching Pennies with Twist": [np.array([0.1, -0.1]), np.array([0.0, 0.0]), np.array([0.0, 0.0])],
            "Matching Pennies with Outside Option": [np.array([0.4, -0.4]), np.array([-0.4, 0.4]), np.array([0.3, -0.3])]
        }
        
        initial_scores_list = [initial_scores_map.get(selected_game_name, [np.array([0.1, -0.1]), np.array([0.1, -0.1]), np.array([0.1, -0.1])]) for _ in range(num_initializations)]

        trajectories = []
        for initial_scores in initial_scores_list:
            eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}
            
            if selected_algo_name == "BFTL-EXP3":
                delta_config = {'initial_delta': 0.1, 'decay_rate': -0.15}
                algorithm = algo_class(game, num_iterations, eta_config, delta_config)
            else:
                sig = inspect.signature(algo_class)
                if 'eta_config' in sig.parameters:
                    algorithm = algo_class(game, num_iterations, eta_config)
                else:
                    algorithm = algo_class(game, num_iterations)

            algorithm.run(initial_scores=initial_scores)
            trajectories.append(np.array(algorithm.strategies))

        equilibria = game_equilibria.get(selected_game_name, [])
        
        if not is_rps_game(selected_game_name):
            st.write("Generating 3D animation...")
            gif_path = create_3d_cube_animation(trajectories, selected_game_name, selected_algo_name, equilibria=equilibria)
            st.image(gif_path)
        else:
            st.write("RPS games are visualized in Algorithm Comparison mode.")

else: # Algorithm Comparison
    rps_games = {name: game for name, game in game_options.items() if is_rps_game(name)}
    selected_game_name = st.sidebar.selectbox("Choose an RPS game", list(rps_games.keys()))
    
    comparison_algos = {
        'FictitiousPlay': FictitiousPlay,
        'SmoothFP_T0.1': lambda game, num_iter: SmoothFictitiousPlay(game, num_iter, temperature=0.1),
        'EXP3': EXP3,
        'Hedge': Hedge
    }
    selected_algos = st.sidebar.multiselect("Choose algorithms to compare", list(comparison_algos.keys()))
    num_iterations = st.sidebar.number_input("Number of Iterations", 100, 100000, 1000)

    if st.sidebar.button("Run Comparison"):
        if not selected_algos:
            st.warning("Please select at least one algorithm to compare.")
        else:
            game_class = rps_games[selected_game_name]
            game = game_class()
            
            trajectories_dict = {}
            for algo_name in selected_algos:
                algo_factory = comparison_algos[algo_name]
                
                eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}
                if algo_name in ['EXP3', 'Hedge']:
                    algorithm = algo_factory(game, num_iterations, eta_config)
                    initial_scores = [np.array([1.5, -1.5, 0.0]), np.array([-1.0, 1.0, 0.0])]
                else: # Fictitious Play variants
                    algorithm = algo_factory(game, num_iterations)
                    initial_scores = [np.array([5.0, 1.0, 1.0]), np.array([1.0, 5.0, 1.0])]

                algorithm.run(initial_scores=initial_scores)
                trajectories_dict[algo_name] = np.array(algorithm.strategies)

            st.write("Generating comparison animation...")
            gif_path = create_comparison_animation(trajectories_dict, selected_game_name, num_iterations)
            st.image(gif_path)

if 'mode' not in st.session_state:
    st.session_state.mode = "Single Simulation"

if st.session_state.mode == "Single Simulation":
    if 'selected_game_name' not in st.session_state:
        st.session_state.selected_game_name = list(game_options.keys())[0]
    # ... and so on for all the widgets
else:
    # ...
    pass