"""Pure research figures with bounded display samples and marker-only animation.

Research histories retain every iteration.
The static path shows the complete run at evenly spaced display samples.
Animation follows a cursor over that path and never rebuilds history prefixes.
"""

from collections import Counter
from zlib import adler32

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_PATH_POINTS = 1000
MAX_FIGURE_PATH_POINTS = 12000
MAX_DIAGNOSTIC_POINTS = 1000
MAX_FRAMES = 80
INK = "#172235"
MUTED = "#66758c"
GRID = "#e6ebf2"
PLAYER_COLORS = ("#3866e9", "#c78322", "#218c63")
PALETTE = (
    "#3866e9",
    "#178b8b",
    "#c78a25",
    "#7f5ac4",
    "#ca5d6d",
    "#4389ad",
    "#6b7840",
    "#ad694b",
    "#708092",
)
ALGORITHMS = (
    "Hedge",
    "EXP3",
    "Fictitious Play",
    "Smooth Fictitious Play",
    "BFTL-EXP3",
    "Optimistic FTRL",
    "Projected Gradient Ascent",
    "Regret Matching",
    "Regret Matching Softmax",
)
DASHES = ("solid", "dash", "dot", "dashdot", "longdash")


def algorithm_color(algorithm):
    """Return a deterministic color independent of selected algorithms."""
    index = (
        ALGORITHMS.index(algorithm)
        if algorithm in ALGORITHMS
        else adler32(algorithm.encode()) % len(PALETTE)
    )
    return PALETTE[index]


def probability_vertices(actions):
    """Return action vertices for a line, equilateral simplex, or n-gon."""
    if actions < 2:
        raise ValueError("A policy visualization requires at least two actions.")
    if actions == 2:
        return np.array([[1.0, 0.0], [0.0, 0.0]])
    angles = np.pi / 2 + np.arange(actions) * 2 * np.pi / actions
    return np.column_stack((np.cos(angles), np.sin(angles)))


def project_strategies(probabilities):
    """Project probabilities by their weighted action vertices."""
    probabilities = np.asarray(probabilities)
    return probabilities @ probability_vertices(probabilities.shape[-1])


def projection_note(game_spec):
    """Explain what information the game family's trajectory preserves."""
    actions = len(game_spec.action_labels)
    if game_spec.category == "Box Games" and actions == 2:
        return f"Each cube axis is one player's probability of {game_spec.action_labels[0]} (action 0). The cube preserves the joint policy profile."
    if actions == 2:
        return f"The line shows P({game_spec.action_labels[0]}); the other action has the remaining probability."
    if actions == 3:
        return "The triangle is a one-to-one representation of each player's three-action policy. Each vertex is a pure action."
    return "This n-gon is a lossy projection: different policies can occupy the same point. Overlapping paths do not establish convergence; inspect full action probabilities and the Nash gap."


def _indices(length, limit):
    return np.linspace(0, length - 1, min(length, limit), dtype=int)


def _metadata(role, run=None, player=None):
    meta = {"role": role}
    if run is not None:
        meta = {**meta, "algorithm": run.algorithm, "run": run.run}
    return {**meta, "player": player} if player is not None else meta


def _name(run):
    return f"{run.algorithm} · Run {run.run + 1}"


def _style(run):
    return {
        "color": algorithm_color(run.algorithm),
        "width": 2,
        "dash": DASHES[run.run % len(DASHES)],
    }


def _base_layout(figure, title, height=490):
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 15}, "x": 0.01},
        font={"family": "Arial, sans-serif", "size": 12, "color": INK},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=height,
        margin={"l": 30, "r": 30, "t": 68, "b": 75},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.13,
            "x": 0,
            "font": {"size": 11},
        },
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
        uirevision=title,
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    for player, annotation in enumerate(figure.layout.annotations):
        annotation.font = {"size": 14, "color": PLAYER_COLORS[player % len(PLAYER_COLORS)]}
    return figure


def _policies(run, view, indices):
    if view == "Current policy":
        return run.strategies[indices]
    if view in ("Empirical frequencies", "Empirical policy"):
        if run.empirical_strategies is None:
            raise ValueError(f"{run.algorithm} does not record empirical frequencies.")
        return run.empirical_strategies[indices]
    if view == "Time-average policy":
        # Reduce disjoint blocks first so stored prefix means scale with display samples.
        ends = indices + 1
        starts = np.concatenate(([0], ends[:-1]))
        totals = np.add.reduceat(run.strategies, starts, axis=0)
        return np.cumsum(totals, axis=0) / ends[:, None, None]
    raise ValueError(f"Unknown policy view: {view}")


def _hover(labels, cube=False, name="%{fullData.name}"):
    if cube:
        probabilities = "<br>".join(
            f"Player {i + 1} · P({labels[0]}): %{{customdata[{i + 1}]:.3f}}" for i in range(3)
        )
    else:
        probabilities = "<br>".join(
            f"{label}: %{{customdata[{i + 1}]:.3f}}" for i, label in enumerate(labels)
        )
    return f"Iteration %{{customdata[0]:,.0f}}<br>{probabilities}<extra>{name}</extra>"


def _geometry(figure, labels, players):
    vertices = probability_vertices(len(labels))
    closed = np.concatenate((vertices, vertices[:1]))
    for player in range(players):
        col = player + 1
        color = PLAYER_COLORS[player % len(PLAYER_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=closed[:, 0],
                y=closed[:, 1],
                mode="lines",
                line={"color": color, "width": 1.5},
                fill="toself" if len(labels) > 2 else None,
                fillcolor="#f6f8fb",
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
                meta=_metadata("boundary"),
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                text=labels,
                mode="markers+text",
                marker={"size": 4, "color": color},
                textposition="top center",
                textfont={"color": color, "size": 11},
                showlegend=False,
                hoverinfo="skip",
                cliponaxis=False,
                meta=_metadata("action_labels"),
            ),
            row=1,
            col=col,
        )
        if len(labels) == 2:
            figure.update_xaxes(
                title_text=f"P({labels[0]})",
                title_font={"color": color, "size": 11},
                tickfont={"color": color, "size": 10},
                range=[-0.07, 1.07],
                tickvals=[0, 0.5, 1],
                row=1,
                col=col,
            )
            figure.update_yaxes(range=[-0.2, 0.2], visible=False, row=1, col=col)
        else:
            axis_name = "x" if col == 1 else f"x{col}"
            figure.update_xaxes(range=[-1.22, 1.22], visible=False, row=1, col=col)
            figure.update_yaxes(
                range=[-1.2, 1.25],
                visible=False,
                scaleanchor=axis_name,
                scaleratio=1,
                row=1,
                col=col,
            )


def _equilibria(figure, spec, players, cube):
    labels = getattr(spec, "equilibrium_labels", ())
    for index, profile in enumerate(getattr(spec, "equilibria", ())):
        profile = np.asarray(profile)
        label = labels[index] if index < len(labels) else f"Verified equilibrium {index + 1}"
        common = {
            "mode": "markers",
            "name": "Equilibrium references",
            "marker": {"color": "#172235", "size": 8, "symbol": "diamond"},
            "legendgroup": "equilibria",
            "meta": _metadata("equilibrium"),
        }
        if cube:
            x, y, z = profile[:, 0]
            figure.add_trace(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    showlegend=index == 0,
                    hovertemplate=f"{label}<br>P1: %{{x:.3f}}<br>P2: %{{y:.3f}}<br>P3: %{{z:.3f}}<extra></extra>",
                    **common,
                )
            )
        else:
            points = project_strategies(profile)
            for player, (x, y) in enumerate(points):
                figure.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        showlegend=index == 0 and player == 0,
                        customdata=[profile[player]],
                        hovertemplate="<br>".join(
                            f"{action}: %{{customdata[{i}]:.3f}}"
                            for i, action in enumerate(spec.action_labels)
                        )
                        + f"<extra>{label}</extra>",
                        **common,
                    ),
                    row=1,
                    col=player + 1,
                )


def _cube_axes(figure, action):
    def axis(player):
        return {
            "title": {
                "text": f"P{player}: P({action})",
                "font": {"size": 11, "color": PLAYER_COLORS[player - 1]},
            },
            "tickfont": {"size": 10, "color": PLAYER_COLORS[player - 1]},
            "range": [0, 1],
            "tickvals": [0, 0.5, 1],
            "backgroundcolor": "#f6f8fb",
            "gridcolor": GRID,
        }

    figure.update_layout(
        scene={
            "xaxis": axis(1),
            "yaxis": axis(2),
            "zaxis": axis(3),
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.15, "y": 1.15, "z": 1.0}},
        }
    )


def _trace(
    run,
    points,
    policies,
    indices,
    labels,
    role,
    player=None,
    cube=False,
    showlegend=False,
):
    current = role == "current"
    probabilities = policies[:, :, 0] if cube else policies[:, player]
    common = {
        "x": points[:, 0],
        "y": points[:, 1],
        "name": run.algorithm,
        "legendgroup": run.algorithm,
        "mode": "markers" if current else "lines",
        "line": _style(run),
        "marker": {
            "size": 8 if current else 3,
            "color": algorithm_color(run.algorithm),
            "line": {"width": 1, "color": "white"},
        },
        "opacity": 1 if current else max(0.35, 0.75 - 0.08 * run.run),
        "showlegend": showlegend,
        "customdata": np.column_stack((indices + 1, probabilities)),
        "hovertemplate": _hover(labels, cube, _name(run)),
        "meta": _metadata(role, run, player),
    }
    return go.Scatter3d(z=points[:, 2], **common) if cube else go.Scatter(**common)


def _add_run_paths(figure, run, view, labels, players, cube, point_limit, animate, showlegend):
    length = len(run.strategies)
    indices = _indices(length, point_limit)
    policies = _policies(run, view, indices)
    frame_indices = _indices(length, MAX_FRAMES) if animate else np.array([length - 1])
    frame_policies = _policies(run, view, frame_indices)
    moving = []
    for player in (None,) if cube else range(players):
        points = policies[:, :, 0] if cube else project_strategies(policies[:, player])
        cursor = frame_policies[:, :, 0] if cube else project_strategies(frame_policies[:, player])
        history = _trace(
            run,
            points,
            policies,
            indices,
            labels,
            "history",
            player,
            cube,
            showlegend and player in (None, 0),
        )
        marker = _trace(
            run,
            cursor[:1],
            frame_policies[:1],
            frame_indices[:1],
            labels,
            "current",
            player,
            cube,
        )
        position = {} if cube else {"row": 1, "col": player + 1}
        figure.add_trace(history, **position)
        marker_index = len(figure.data)
        figure.add_trace(marker, **position)
        moving.append((marker_index, run, player, cursor, frame_policies, frame_indices))
    return moving


def _iteration_annotation(iteration):
    return {
        "text": f"Iteration {iteration + 1:,}",
        "x": 0.98,
        "y": -0.025,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "right",
        "yanchor": "top",
        "showarrow": False,
        "font": {"size": 11, "color": MUTED},
    }


def _animation(figure, moving, labels, cube):
    frames = []
    player_headers = [annotation.to_plotly_json() for annotation in figure.layout.annotations]
    for frame_index, iteration in enumerate(moving[0][-1]):
        data = [
            _trace(
                run,
                points[frame_index : frame_index + 1],
                policies[frame_index : frame_index + 1],
                indices[frame_index : frame_index + 1],
                labels,
                "current",
                player,
                cube,
            )
            for _, run, player, points, policies, indices in moving
        ]
        frames.append(
            go.Frame(
                name=str(iteration + 1),
                data=data,
                traces=[item[0] for item in moving],
                layout={"annotations": player_headers + [_iteration_annotation(iteration)]},
            )
        )
    figure.frames = frames
    labeled_steps = set(_indices(len(frames), 5))
    settings = {
        "frame": {"duration": 70, "redraw": cube},
        "transition": {"duration": 0},
        "mode": "immediate",
    }
    figure.update_layout(
        height=580,
        margin={"b": 190},
        annotations=player_headers + [_iteration_annotation(moving[0][-1][0])],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.025,
                "yanchor": "top",
                "font": {"size": 11},
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, settings]},
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0,
                "len": 1,
                "y": -0.22,
                "yanchor": "top",
                "pad": {"t": 0, "b": 0},
                "font": {"size": 10},
                "ticklen": 3,
                "minorticklen": 0,
                "currentvalue": {"visible": False},
                "steps": [
                    {
                        "label": frame.name if index in labeled_steps else "",
                        "value": frame.name,
                        "method": "animate",
                        "args": [
                            [frame.name],
                            {**settings, "frame": {"duration": 0, "redraw": cube}},
                        ],
                    }
                    for index, frame in enumerate(frames)
                ],
            }
        ],
        legend={"y": -0.44, "yanchor": "top"},
    )


def trajectory_figure(result, game_spec, view="Current policy", animate=True):
    """Plot joint binary cube profiles or a separate policy space per player."""
    if not result.runs:
        raise ValueError("A trajectory requires at least one run.")
    players, actions = result.runs[0].strategies.shape[1:]
    cube = game_spec.category == "Box Games" and players == 3 and actions == 2
    figure = (
        go.Figure()
        if cube
        else make_subplots(
            rows=1,
            cols=players,
            subplot_titles=[f"Player {i + 1}" for i in range(players)],
        )
    )
    _base_layout(
        figure,
        view,
    )
    if cube:
        _cube_axes(figure, game_spec.action_labels[0])
    else:
        _geometry(figure, game_spec.action_labels, players)
    _equilibria(figure, game_spec, players, cube)
    path_count = len(result.runs) * (1 if cube else players)
    point_limit = min(MAX_PATH_POINTS, max(2, MAX_FIGURE_PATH_POINTS // path_count))
    moving = []
    seen = set()
    for run in result.runs:
        moving.extend(
            _add_run_paths(
                figure,
                run,
                view,
                game_spec.action_labels,
                players,
                cube,
                point_limit,
                animate,
                run.algorithm not in seen,
            )
        )
        seen = seen | {run.algorithm}
    if animate:
        _animation(figure, moving, game_spec.action_labels, cube)
    return figure


METRICS = {
    "nash_gap": ("Nash gap", "Sum of unilateral expected utility improvements"),
    "average_regret": (
        "Average expected external regret",
        "Best fixed action versus the observed opponents' policy sequence",
    ),
    "payoffs": (
        "Expected utility",
        "Expected utility under the current policy profile",
    ),
}


def diagnostics_figure(result, metric="nash_gap"):
    """Show sampled research metrics, preserving independent runs and players."""
    if metric not in METRICS:
        raise ValueError(f"Unknown diagnostic metric: {metric}")
    title, _ = METRICS[metric]
    players = 1 if metric == "nash_gap" else result.runs[0].strategies.shape[1]
    figure = make_subplots(
        rows=1,
        cols=players,
        subplot_titles=[] if players == 1 else [f"Player {i + 1}" for i in range(players)],
    )
    _base_layout(figure, title, height=400)
    seen = set()
    for run in result.runs:
        values = getattr(run, metric)
        indices = _indices(len(values), MAX_DIAGNOSTIC_POINTS)
        for player in range(players):
            y = values[indices] if metric == "nash_gap" else values[indices, player]
            figure.add_trace(
                go.Scatter(
                    x=indices + 1,
                    y=y,
                    mode="lines",
                    name=run.algorithm,
                    line=_style(run),
                    opacity=max(0.45, 1 - 0.08 * run.run),
                    showlegend=player == 0 and run.algorithm not in seen,
                    legendgroup=run.algorithm,
                    meta=_metadata("diagnostic", run, player),
                    hovertemplate=f"Iteration %{{x:,.0f}}<br>{title}: %{{y:.4f}}<extra>{_name(run)}</extra>",
                ),
                row=1,
                col=player + 1,
            )
        seen = seen | {run.algorithm}
    figure.update_xaxes(title_text="Iteration", tickformat=",d")
    figure.update_yaxes(title_text=title)
    if metric in ("nash_gap", "average_regret"):
        figure.add_hline(y=0, line={"color": "#a7b4c8", "width": 1, "dash": "dot"})
    return figure


def strategy_figure(result, game_spec):
    """Show full final current-policy probabilities with independent-run ranges."""
    players = result.runs[0].strategies.shape[1]
    figure = make_subplots(
        rows=1, cols=players, subplot_titles=[f"Player {i + 1}" for i in range(players)]
    )
    _base_layout(figure, "Final current policies · repeat mean and range", height=360)
    algorithms = tuple(dict.fromkeys(run.algorithm for run in result.runs))
    counts = Counter(run.algorithm for run in result.runs)
    for algorithm in algorithms:
        finals = np.stack([run.strategies[-1] for run in result.runs if run.algorithm == algorithm])
        for player in range(players):
            probabilities = finals[:, player]
            mean = probabilities.mean(axis=0)
            lower, upper = probabilities.min(axis=0), probabilities.max(axis=0)
            figure.add_trace(
                go.Bar(
                    x=game_spec.action_labels,
                    y=mean,
                    name=algorithm,
                    legendgroup=algorithm,
                    showlegend=player == 0,
                    marker={"color": algorithm_color(algorithm)},
                    offsetgroup=algorithm,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": upper - mean,
                        "arrayminus": mean - lower,
                        "thickness": 1,
                        "width": 3,
                    },
                    customdata=np.column_stack((lower, upper)),
                    hovertemplate=f"%{{x}}<br>Mean probability: %{{y:.3f}}<br>Run range: %{{customdata[0]:.3f}} to %{{customdata[1]:.3f}}<extra>{algorithm} · {counts[algorithm]} runs</extra>",
                    meta={
                        "role": "final_policy",
                        "algorithm": algorithm,
                        "player": player,
                    },
                ),
                row=1,
                col=player + 1,
            )
    figure.update_layout(barmode="group", bargap=0.25)
    figure.update_yaxes(title_text="Probability", range=[0, 1.05], tickvals=[0, 0.5, 1])
    return figure
