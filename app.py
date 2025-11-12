import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
import inspect
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
st.sidebar.header("Game Category")
mode = st.sidebar.radio("Choose a category", ["Box Games", "RPS Games"])

# --- Game Options by Category ---
box_game_options = {
    "Pure Coordination": PureCoordination,
    "Coordination with Spectator": CoordinationWithSpectator,
    "Matching Pennies with Twist": MatchingPenniesWithTwist,
    "Matching Pennies with Outside Option": MatchingPenniesWithOutsideOption,
}

rps_game_options = {
    "Rock Paper Scissors": RockPaperScissors,
    "Biased Rock Paper Scissors": BiasedRockPaperScissors,
    "Asymmetric Rock Paper Scissors": AsymmetricRockPaperScissors,
    "Noisy Rock Paper Scissors": RockPaperScissorsWithNoise,
}

# Select appropriate games based on mode
if mode == "Box Games":
    game_options = box_game_options
else:
    game_options = rps_game_options

# --- Game and Algorithm Selection ---
st.sidebar.header("Simulation Settings")

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


# Algorithm options by category
box_algorithm_options = {
    "BFTL-EXP3": BFTL_EXP3,
    "EXP3": EXP3,
    "Hedge": Hedge,
    "Optimistic FTRL": OptimisticFTRL,
    "Projected Gradient Ascent": ProjectedGradientAscent,
    "Regret Matching": RegretMatching,
    "Regret Matching Softmax": RegretMatchingSoftmax,
}

rps_algorithm_options = {
    'Fictitious Play': FictitiousPlay,
    'Smooth FP (T=0.1)': lambda game, num_iter: SmoothFictitiousPlay(game, num_iter, temperature=0.1),
    'EXP3': EXP3,
    'Hedge': Hedge
}

# Select appropriate algorithms based on mode
if mode == "Box Games":
    algorithm_options = box_algorithm_options
else:
    algorithm_options = rps_algorithm_options

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

def create_3d_cube_animation_plotly(trajectories, game_name, algorithm_name, equilibria=None):
    """
    Create an interactive 3D animation using Plotly (no file generation needed).
    Returns a Plotly figure object that can be displayed with st.plotly_chart().
    """
    # Pre-convert trajectories to numpy arrays
    trajectories = [np.array(strategies) for strategies in trajectories]

    # Default to empty list if no equilibria provided
    if equilibria is None:
        equilibria = []

    # Color palette for different trajectories
    trajectory_colors = [
        'rgb(31, 119, 180)',   # blue
        'rgb(255, 127, 14)',   # orange
        'rgb(44, 160, 44)',    # green
        'rgb(214, 39, 40)',    # red
        'rgb(148, 103, 189)',  # purple
        'rgb(140, 86, 75)',    # brown
        'rgb(227, 119, 194)',  # pink
        'rgb(127, 127, 127)',  # gray
        'rgb(188, 189, 34)',   # olive
        'rgb(23, 190, 207)',   # cyan
    ]

    # Cube vertices and edges
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    # Create figure
    fig = go.Figure()

    # Add cube edges
    for edge in edges:
        points = vertices[edge]
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines',
            line=dict(color='gray', width=2),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add Nash Equilibria as gold stars
    if equilibria:
        eq_x, eq_y, eq_z, eq_labels = [], [], [], []
        for eq in equilibria:
            x, y, z, label = eq
            eq_x.append(x)
            eq_y.append(y)
            eq_z.append(z)
            eq_labels.append(label)

        fig.add_trace(go.Scatter3d(
            x=eq_x, y=eq_y, z=eq_z,
            mode='markers',
            marker=dict(
                size=12,
                color='gold',
                symbol='diamond',
                line=dict(color='black', width=2)
            ),
            name='Nash Equilibrium',
            text=eq_labels,
            hovertemplate='<b>%{text}</b><br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

    # Prepare animation frames
    num_steps = len(trajectories[0])
    frames = []

    # Sample frames (every 10 steps)
    frame_indices = list(range(0, num_steps, 10))
    if frame_indices[-1] != num_steps - 1:
        frame_indices.append(num_steps - 1)

    # Create frames for animation
    for t in frame_indices:
        frame_data = []

        # Add cube edges to each frame
        for edge in edges:
            points = vertices[edge]
            frame_data.append(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='lines',
                line=dict(color='gray', width=2),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add equilibria to each frame
        if equilibria:
            eq_x, eq_y, eq_z, eq_labels = [], [], [], []
            for eq in equilibria:
                x, y, z, label = eq
                eq_x.append(x)
                eq_y.append(y)
                eq_z.append(z)
                eq_labels.append(label)

            frame_data.append(go.Scatter3d(
                x=eq_x, y=eq_y, z=eq_z,
                mode='markers',
                marker=dict(
                    size=12,
                    color='gold',
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                name='Nash Equilibrium',
                text=eq_labels,
                hovertemplate='<b>%{text}</b><br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))

        # Add trajectories up to time t
        for traj_idx, strategies in enumerate(trajectories):
            x = strategies[:t+1, 0, 0]
            y = strategies[:t+1, 1, 0]
            z = strategies[:t+1, 2, 0]

            # Get color for this trajectory
            traj_color = trajectory_colors[traj_idx % len(trajectory_colors)]

            if len(x) > 1:
                # Draw the trajectory line
                frame_data.append(go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='lines',
                    line=dict(color=traj_color, width=4),
                    name=f'Trajectory {traj_idx+1}' if len(trajectories) > 1 else 'Trajectory',
                    showlegend=(t == frame_indices[0]),
                    legendgroup=f'traj{traj_idx}',
                    hovertemplate=f'Trajectory {traj_idx+1}<br>P1: %{{x:.3f}}<br>P2: %{{y:.3f}}<br>P3: %{{z:.3f}}<extra></extra>'
                ))

                # Add current position marker
                frame_data.append(go.Scatter3d(
                    x=[x[-1]],
                    y=[y[-1]],
                    z=[z[-1]],
                    mode='markers',
                    marker=dict(size=8, color=traj_color, symbol='circle',
                               line=dict(color='white', width=1)),
                    showlegend=False,
                    legendgroup=f'traj{traj_idx}',
                    hovertemplate=f'Trajectory {traj_idx+1}<br>t={t}<br>P1: %{{x:.3f}}<br>P2: %{{y:.3f}}<br>P3: %{{z:.3f}}<extra></extra>'
                ))
            else:
                # Just starting point
                frame_data.append(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(size=8, color=traj_color, symbol='circle',
                               line=dict(color='white', width=1)),
                    name=f'Trajectory {traj_idx+1}' if len(trajectories) > 1 else 'Trajectory',
                    showlegend=(t == frame_indices[0]),
                    legendgroup=f'traj{traj_idx}',
                    hovertemplate=f'Trajectory {traj_idx+1}<br>t={t}<br>P1: %{{x:.3f}}<br>P2: %{{y:.3f}}<br>P3: %{{z:.3f}}<extra></extra>'
                ))

        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Add initial trajectories (t=0) - must match frame structure
    for traj_idx, strategies in enumerate(trajectories):
        x = strategies[0:1, 0, 0]
        y = strategies[0:1, 1, 0]
        z = strategies[0:1, 2, 0]

        traj_color = trajectory_colors[traj_idx % len(trajectory_colors)]

        # Add empty line trace (will be populated by frames)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=traj_color, width=4),
            name=f'Trajectory {traj_idx+1}' if len(trajectories) > 1 else 'Trajectory',
            legendgroup=f'traj{traj_idx}',
            hovertemplate=f'Trajectory {traj_idx+1}<br>P1: %{{x:.3f}}<br>P2: %{{y:.3f}}<br>P3: %{{z:.3f}}<extra></extra>'
        ))

        # Add initial marker
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color=traj_color, symbol='circle',
                       line=dict(color='white', width=1)),
            showlegend=False,
            legendgroup=f'traj{traj_idx}',
            hovertemplate=f'Trajectory {traj_idx+1}<br>t=0<br>P1: %{{x:.3f}}<br>P2: %{{y:.3f}}<br>P3: %{{z:.3f}}<extra></extra>'
        ))

    # Add frames to figure
    fig.frames = frames

    # Update layout with animation settings
    fig.update_layout(
        title=f'{game_name} - {algorithm_name} - Strategy Evolution',
        scene=dict(
            xaxis=dict(title='Player 1 - P(Action 0)', range=[0, 1]),
            yaxis=dict(title='Player 2 - P(Action 0)', range=[0, 1]),
            zaxis=dict(title='Player 3 - P(Action 0)', range=[0, 1]),
            camera=dict(
                eye=dict(x=1.8, y=0.7, z=0.35)  # Matches main.py: elevation=10, azimuth=22
            )
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 50, 'easing': 'linear'}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0,
            'xanchor': 'left',
            'yanchor': 'bottom'
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'label': str(t),
                    'method': 'animate',
                    'args': [[str(t)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for t in frame_indices
            ],
            'x': 0.1,
            'len': 0.9,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'pad': {'b': 10, 't': 50},
            'currentvalue': {
                'visible': True,
                'prefix': 'Iteration: ',
                'xanchor': 'right',
                'font': {'size': 16}
            }
        }],
        height=700,
        hovermode='closest'
    )

    return fig

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

def create_comparison_animation_plotly(trajectories_dict, game_name, num_iterations):
    """
    Create interactive side-by-side simplex plots for RPS game comparison using Plotly.
    Returns a Plotly figure object that can be displayed with st.plotly_chart().
    """
    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Player 1 Strategy", "Player 2 Strategy"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.12
    )

    # Simplex triangle vertices in 2D
    simplex_points = barycentric_to_cartesian(np.eye(3))

    # Add simplex triangles to both subplots
    for col in [1, 2]:
        # Triangle filled area (white)
        fig.add_trace(go.Scatter(
            x=np.append(simplex_points[:, 0], simplex_points[0, 0]),
            y=np.append(simplex_points[:, 1], simplex_points[0, 1]),
            mode='lines',
            fill='toself',
            fillcolor='white',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=col)

        # Labels
        labels = ["Rock", "Paper", "Scissors"]
        offsets = [(-0.1, 0), (0.05, 0), (-0.05, 0.05)]
        for i, (label, offset) in enumerate(zip(labels, offsets)):
            fig.add_annotation(
                x=simplex_points[i, 0] + offset[0],
                y=simplex_points[i, 1] + offset[1],
                text=label,
                showarrow=False,
                font=dict(size=12),
                row=1, col=col
            )

    # Prepare animation frames
    frame_indices = list(range(0, num_iterations, 10))
    if frame_indices[-1] != num_iterations - 1:
        frame_indices.append(num_iterations - 1)

    # Color palette for different algorithms
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(trajectories_dict.keys())}

    frames = []

    # Create frames for animation
    for t in frame_indices:
        frame_data = []

        # Add simplex triangles to each frame (both subplots)
        for col_idx in range(2):
            # Triangle filled area (white)
            frame_data.append(go.Scatter(
                x=np.append(simplex_points[:, 0], simplex_points[0, 0]),
                y=np.append(simplex_points[:, 1], simplex_points[0, 1]),
                mode='lines',
                fill='toself',
                fillcolor='white',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip',
                xaxis='x' if col_idx == 0 else 'x2',
                yaxis='y' if col_idx == 0 else 'y2'
            ))

        # Add trajectories for each algorithm
        for algo_name, strategies in trajectories_dict.items():
            # Player 1 trajectory
            coords1 = barycentric_to_cartesian(strategies[:t+1, 0, :])
            frame_data.append(go.Scatter(
                x=coords1[:, 0],
                y=coords1[:, 1],
                mode='lines+markers',
                line=dict(color=algo_colors[algo_name], width=2),
                marker=dict(size=4),
                name=algo_name,
                showlegend=True,
                legendgroup=algo_name,
                hovertemplate=f'{algo_name}<br>R: %{{customdata[0]:.3f}}<br>P: %{{customdata[1]:.3f}}<br>S: %{{customdata[2]:.3f}}<extra></extra>',
                customdata=strategies[:t+1, 0, :],
                xaxis='x',
                yaxis='y'
            ))

            # Player 2 trajectory
            coords2 = barycentric_to_cartesian(strategies[:t+1, 1, :])
            frame_data.append(go.Scatter(
                x=coords2[:, 0],
                y=coords2[:, 1],
                mode='lines+markers',
                line=dict(color=algo_colors[algo_name], width=2),
                marker=dict(size=4),
                name=algo_name,
                showlegend=False,
                legendgroup=algo_name,
                hovertemplate=f'{algo_name}<br>R: %{{customdata[0]:.3f}}<br>P: %{{customdata[1]:.3f}}<br>S: %{{customdata[2]:.3f}}<extra></extra>',
                customdata=strategies[:t+1, 1, :],
                xaxis='x2',
                yaxis='y2'
            ))

        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Add initial trajectories (t=0) to the figure
    for algo_name, strategies in trajectories_dict.items():
        # Player 1 initial point
        coords1 = barycentric_to_cartesian(strategies[0:1, 0, :])
        fig.add_trace(go.Scatter(
            x=coords1[:, 0],
            y=coords1[:, 1],
            mode='lines+markers',
            line=dict(color=algo_colors[algo_name], width=2),
            marker=dict(size=4),
            name=algo_name,
            legendgroup=algo_name,
            hovertemplate=f'{algo_name}<br>R: %{{customdata[0]:.3f}}<br>P: %{{customdata[1]:.3f}}<br>S: %{{customdata[2]:.3f}}<extra></extra>',
            customdata=strategies[0:1, 0, :]
        ), row=1, col=1)

        # Player 2 initial point
        coords2 = barycentric_to_cartesian(strategies[0:1, 1, :])
        fig.add_trace(go.Scatter(
            x=coords2[:, 0],
            y=coords2[:, 1],
            mode='lines+markers',
            line=dict(color=algo_colors[algo_name], width=2),
            marker=dict(size=4),
            name=algo_name,
            showlegend=False,
            legendgroup=algo_name,
            hovertemplate=f'{algo_name}<br>R: %{{customdata[0]:.3f}}<br>P: %{{customdata[1]:.3f}}<br>S: %{{customdata[2]:.3f}}<extra></extra>',
            customdata=strategies[0:1, 1, :]
        ), row=1, col=2)

    # Add frames to figure
    fig.frames = frames

    # Update layout
    fig.update_layout(
        title=f'{game_name} - Algorithm Comparison',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.5,
            'y': 1.15,
            'xanchor': 'center',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'label': str(t),
                    'method': 'animate',
                    'args': [[str(t)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for t in frame_indices
            ],
            'x': 0.5,
            'len': 0.8,
            'xanchor': 'center',
            'y': -0.05,
            'yanchor': 'top',
            'pad': {'b': 10, 't': 50},
            'currentvalue': {
                'visible': True,
                'prefix': 'Iteration: ',
                'xanchor': 'center',
                'font': {'size': 16}
            }
        }],
        height=600,
        hovermode='closest',
        margin=dict(t=100, b=100)
    )

    # Update axes to be equal aspect ratio and hide grid
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)

    return fig


if mode == "Box Games":
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
        algorithms_run = []  # Store algorithm instances for additional info

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
            algorithms_run.append(algorithm)

        equilibria = game_equilibria.get(selected_game_name, [])

        # Generate 3D cube visualization for Box Games
        st.write("Generating interactive 3D animation...")
        fig = create_3d_cube_animation_plotly(trajectories, selected_game_name, selected_algo_name, equilibria=equilibria)
        st.plotly_chart(fig, use_container_width=True)

        # Show additional information if requested
        if show_play_info:
            st.subheader("Strategy Information")
            for i, (traj, algo) in enumerate(zip(trajectories, algorithms_run)):
                with st.expander(f"Trajectory {i+1} - Final Strategies", expanded=(i==0)):
                    final_strategies = traj[-1]
                    st.write(f"**Player 1:** Action 0: {final_strategies[0, 0]:.4f}, Action 1: {final_strategies[0, 1]:.4f}")
                    st.write(f"**Player 2:** Action 0: {final_strategies[1, 0]:.4f}, Action 1: {final_strategies[1, 1]:.4f}")
                    st.write(f"**Player 3:** Action 0: {final_strategies[2, 0]:.4f}, Action 1: {final_strategies[2, 1]:.4f}")

                    # Show distance to nearest equilibrium if available
                    if equilibria:
                        min_dist = float('inf')
                        closest_eq = None
                        for eq in equilibria:
                            eq_point = np.array([eq[0], eq[1], eq[2]])
                            strategy_point = np.array([final_strategies[0, 0], final_strategies[1, 0], final_strategies[2, 0]])
                            dist = np.linalg.norm(eq_point - strategy_point)
                            if dist < min_dist:
                                min_dist = dist
                                closest_eq = eq[3] if len(eq) > 3 else "NE"
                        st.write(f"**Distance to closest equilibrium ({closest_eq}):** {min_dist:.4f}")

        if show_strategy_plot:
            st.subheader("Strategy Evolution Over Time")

            # Create line plots for each player showing probability evolution
            fig_evolution = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Player 1", "Player 2", "Player 3")
            )

            colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
                     'rgb(214, 39, 40)', 'rgb(148, 103, 189)']

            for traj_idx, traj in enumerate(trajectories):
                color = colors[traj_idx % len(colors)]
                iterations = np.arange(len(traj))

                # Player 1
                fig_evolution.add_trace(
                    go.Scatter(x=iterations, y=traj[:, 0, 0],
                             mode='lines', name=f'Traj {traj_idx+1}',
                             line=dict(color=color, width=2),
                             legendgroup=f'traj{traj_idx}',
                             showlegend=True),
                    row=1, col=1
                )

                # Player 2
                fig_evolution.add_trace(
                    go.Scatter(x=iterations, y=traj[:, 1, 0],
                             mode='lines', name=f'Traj {traj_idx+1}',
                             line=dict(color=color, width=2),
                             legendgroup=f'traj{traj_idx}',
                             showlegend=False),
                    row=1, col=2
                )

                # Player 3
                fig_evolution.add_trace(
                    go.Scatter(x=iterations, y=traj[:, 2, 0],
                             mode='lines', name=f'Traj {traj_idx+1}',
                             line=dict(color=color, width=2),
                             legendgroup=f'traj{traj_idx}',
                             showlegend=False),
                    row=1, col=3
                )

            # Update axes labels
            fig_evolution.update_xaxes(title_text="Iteration", row=1, col=1)
            fig_evolution.update_xaxes(title_text="Iteration", row=1, col=2)
            fig_evolution.update_xaxes(title_text="Iteration", row=1, col=3)
            fig_evolution.update_yaxes(title_text="P(Action 0)", range=[0, 1], row=1, col=1)
            fig_evolution.update_yaxes(title_text="P(Action 0)", range=[0, 1], row=1, col=2)
            fig_evolution.update_yaxes(title_text="P(Action 0)", range=[0, 1], row=1, col=3)

            fig_evolution.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_evolution, use_container_width=True)

else:  # RPS Games
    selected_game_name = st.sidebar.selectbox("Choose a game", list(game_options.keys()))
    selected_algos = st.sidebar.multiselect("Choose algorithms to compare", list(algorithm_options.keys()))
    num_iterations = st.sidebar.number_input("Number of Iterations", 100, 100000, 1000)

    if st.sidebar.button("Run Comparison"):
        if not selected_algos:
            st.warning("Please select at least one algorithm to compare.")
        else:
            game_class = game_options[selected_game_name]
            game = game_class()

            trajectories_dict = {}
            for algo_name in selected_algos:
                algo_factory = algorithm_options[algo_name]

                eta_config = {'initial_eta': 0.2, 'decay_rate': -0.5}
                if algo_name in ['EXP3', 'Hedge']:
                    algorithm = algo_factory(game, num_iterations, eta_config)
                    initial_scores = [np.array([1.5, -1.5, 0.0]), np.array([-1.0, 1.0, 0.0])]
                else:  # Fictitious Play variants
                    algorithm = algo_factory(game, num_iterations)
                    initial_scores = [np.array([5.0, 1.0, 1.0]), np.array([1.0, 5.0, 1.0])]

                algorithm.run(initial_scores=initial_scores)
                trajectories_dict[algo_name] = np.array(algorithm.strategies)

            st.write("Generating interactive comparison animation...")
            fig = create_comparison_animation_plotly(trajectories_dict, selected_game_name, num_iterations)
            st.plotly_chart(fig, use_container_width=True)

if 'mode' not in st.session_state:
    st.session_state.mode = "Box Games"