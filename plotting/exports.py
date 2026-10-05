"""Bounded GIF rendering with full-history context and no shared frame files."""

import io
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

if TYPE_CHECKING:
    from research.simulation import ExperimentResult


MAX_FRAMES = 60
MAX_PATH_POINTS = 1000
COLORS = ("#2367a6", "#cc6b40", "#39836e", "#8b67a5", "#bf9639", "#567488", "#bd5f84")


def _indices(iterations: int, limit: int) -> np.ndarray:
    if iterations < 1:
        raise ValueError("Animation requires at least one iteration.")
    return np.unique(np.linspace(0, iterations - 1, min(iterations, limit), dtype=int))


def frame_indices(iterations: int) -> np.ndarray:
    """Select at most 60 frames, including the first and last iterations."""
    return _indices(iterations, MAX_FRAMES)


def path_indices(iterations: int) -> np.ndarray:
    """Select at most 1,000 path points, including every animated timestamp."""
    frames = frame_indices(iterations)
    background = _indices(iterations, MAX_PATH_POINTS - len(frames))
    return np.union1d(background, frames)


def _cube_scene(figure, result, spec, colors, indices):
    axis = figure.add_subplot(111, projection="3d")
    axis.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    axis.set_xlabel(f"Player 1 - P({spec.action_labels[0]})")
    axis.set_ylabel(f"Player 2 - P({spec.action_labels[0]})")
    axis.set_zlabel(f"Player 3 - P({spec.action_labels[0]})")
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=35)
    for equilibrium in spec.equilibria:
        axis.scatter(*equilibrium[:, 0], marker="*", s=80, color="#bf9639", edgecolor="#775a17")
    animated = []
    for run in result.runs:
        points = run.strategies[indices, :, 0]
        color = colors[run.algorithm]
        axis.plot(*points.T, color=color, alpha=0.16, linewidth=1)
        line, = axis.plot([], [], [], color=color, linewidth=1.5)
        marker, = axis.plot([], [], [], marker="o", color=color, markersize=5)
        animated.append((line, marker, points))
    return animated


def _polygon_vertices(actions):
    if actions == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]])
    angles = np.pi / 2 - np.arange(actions) * 2 * np.pi / actions
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _simplex_scene(figure, result, spec, colors, indices):
    actions = len(spec.action_labels)
    vertices = _polygon_vertices(actions)
    animated = []
    for player in range(result.runs[0].strategies.shape[1]):
        axis = figure.add_subplot(1, 2, player + 1)
        axis.set(xlim=(-1.4, 1.4), ylim=(-1.3, 1.35), title=f"Player {player + 1}")
        axis.set_aspect("equal")
        axis.axis("off")
        boundary = np.vstack((vertices, vertices[0])) if actions > 2 else vertices
        axis.plot(*boundary.T, color="#a9b4bf", linewidth=1.3)
        for label, point in zip(spec.action_labels, vertices):
            axis.text(*(point * 1.18), label, ha="center", va="center", fontsize=10, color="#34485d")
        for equilibrium in spec.equilibria:
            point = equilibrium[player] @ vertices
            axis.scatter(*point, marker="*", s=100, color="#bf9639", edgecolor="#775a17", zorder=4)
        for run in result.runs:
            points = run.strategies[indices, player] @ vertices
            color = colors[run.algorithm]
            axis.plot(*points.T, color=color, alpha=0.16, linewidth=1)
            line, = axis.plot([], [], color=color, linewidth=1.5)
            marker, = axis.plot([], [], marker="o", color=color, markersize=5)
            animated.append((line, marker, points))
    return animated


def _update_artists(animated, indices, iteration, is_cube):
    visible = indices <= iteration
    for line, marker, points in animated:
        trajectory = points[visible]
        if is_cube:
            line.set_data_3d(*trajectory.T)
            marker.set_data_3d(*trajectory[-1:, :].T)
        else:
            line.set_data(*trajectory.T)
            marker.set_data(*trajectory[-1:, :].T)


def render_gif(result: "ExperimentResult") -> bytes:
    """Render decision-policy paths using bounded in-memory raster frames.

    For four or more actions, the polygon projection is explicitly labeled as
    non-injective. Research histories are retained separately in result archives.
    """
    from research.catalog import GAMES

    spec = GAMES[result.config.game_name]
    iterations, players, actions = result.runs[0].strategies.shape
    indices = path_indices(iterations)
    frames = frame_indices(iterations)
    colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(result.config.algorithm_names)}
    is_cube = players == 3 and actions == 2
    figure = Figure(figsize=(10, 5.4), dpi=85, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    try:
        animated = (
            _cube_scene(figure, result, spec, colors, indices) if is_cube
            else _simplex_scene(figure, result, spec, colors, indices)
        )
        figure.subplots_adjust(top=0.79, bottom=0.13, left=0.06, right=0.92)
        title = figure.suptitle(spec.name, y=0.97, fontsize=15, color="#24374c")
        handles = []
        for name, color in colors.items():
            handles.append(Line2D([0], [0], color=color, label=name, linewidth=2))
        if spec.equilibria:
            handles.append(Line2D([0], [0], linestyle="none", marker="*", color="#bf9639", label="Certified equilibrium example", markersize=8))
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=min(4, len(handles)), frameon=False, fontsize=8)
        projection = " n-gon projection is not one-to-one." if actions >= 4 else ""
        figure.text(0.5, 0.035, f"Decision policies; faint lines show sampled full-history context.{projection}", ha="center", fontsize=8, color="#52657a")
        images = []
        for iteration in frames:
            _update_artists(animated, indices, iteration, is_cube)
            title.set_text(f"{spec.name} | iteration {iteration + 1:,} / {iterations:,}")
            canvas.draw()
            raster = np.asarray(canvas.buffer_rgba())[:, :, :3]
            images.append(Image.fromarray(raster).quantize(colors=128))
        output = io.BytesIO()
        images[0].save(output, format="GIF", save_all=True, append_images=images[1:], duration=140, loop=0, optimize=False)
        return output.getvalue()
    finally:
        figure.clear()
