"""Geometry and payload contracts for the research figures."""

from types import SimpleNamespace

import numpy as np
import pytest

from ui.plots import (
    algorithm_color,
    diagnostics_figure,
    probability_vertices,
    project_strategies,
    projection_note,
    strategy_figure,
    trajectory_figure,
)


def experiment(steps=12, players=2, actions=3, algorithms=("Hedge",), repeats=1):
    first = np.eye(actions)[0]
    last = np.eye(actions)[-1]
    weight = np.linspace(0, 1, steps)[:, None, None]
    policies = np.broadcast_to((1 - weight) * first + weight * last, (steps, players, actions))
    runs = tuple(
        SimpleNamespace(
            algorithm=algorithm,
            run=repeat,
            seed=repeat,
            strategies=policies,
            empirical_strategies=None,
            payoffs=np.broadcast_to(np.arange(players), (steps, players)),
            average_regret=np.broadcast_to(np.linspace(1, 0, steps)[:, None], (steps, players)),
            nash_gap=np.linspace(1, 0, steps),
        )
        for algorithm in algorithms
        for repeat in range(repeats)
    )
    return SimpleNamespace(runs=runs, config=SimpleNamespace(iterations=steps))


def spec(actions=3, players=2, category="RPS Games", equilibria=()):
    return SimpleNamespace(
        name="Example",
        category=category,
        action_labels=tuple(f"Action {i}" for i in range(actions)),
        equilibria=equilibria,
    )


def role_traces(figure, role):
    return [trace for trace in figure.data if trace.meta and trace.meta.get("role") == role]


def test_triangle_vertices_project_pure_actions_and_uniform_policy_exactly():
    vertices = probability_vertices(3)
    np.testing.assert_allclose(project_strategies(np.eye(3)), vertices)
    np.testing.assert_allclose(
        project_strategies(np.full((1, 3), 1 / 3)),
        vertices.mean(axis=0)[None],
        atol=1e-15,
    )
    distances = np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1)
    np.testing.assert_allclose(distances, np.repeat(distances[0], 3))


def test_two_actions_are_a_probability_line_with_clear_action_labels():
    figure = trajectory_figure(experiment(actions=2), spec(actions=2), animate=False)
    for trace in role_traces(figure, "history"):
        assert set(trace.y) == {0}
        assert trace.x[0] == 1
        assert trace.x[-1] == 0
    assert "P(Action 0)" in figure.layout.xaxis.title.text
    assert all("Action 1" in trace.text for trace in role_traces(figure, "action_labels"))
    assert not figure.frames


def test_triangle_boundary_has_each_action_label_at_its_vertex():
    figure = trajectory_figure(experiment(), spec(), animate=False)
    labels = role_traces(figure, "action_labels")
    assert len(labels) == 2
    vertices = probability_vertices(3)
    for trace in labels:
        assert tuple(trace.text) == spec().action_labels
        np.testing.assert_allclose(np.column_stack((trace.x, trace.y)), vertices)


def test_equilibrium_markers_use_the_profile_for_each_player():
    profile = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    game_spec = spec(equilibria=(profile,))
    game_spec.equilibrium_labels = ("Reference",)
    figure = trajectory_figure(experiment(), game_spec, animate=False)
    markers = role_traces(figure, "equilibrium")
    assert len(markers) == 2
    expected = project_strategies(profile)
    for player, trace in enumerate(markers):
        np.testing.assert_allclose([trace.x[0], trace.y[0]], expected[player])
        np.testing.assert_array_equal(trace.customdata[0], profile[player])
        assert trace.name == "Equilibrium references"
        assert "Reference" in trace.hovertemplate
    assert markers[0].xaxis != markers[1].xaxis


def test_ngon_projection_explicitly_discloses_lost_information():
    square = np.array([[0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5]])
    np.testing.assert_allclose(
        project_strategies(square)[0], project_strategies(square)[1], atol=1e-15
    )
    note = projection_note(spec(actions=4))
    assert "different policies" in note.lower()
    assert "projection" in note.lower()
    assert "one-to-one" in projection_note(spec()).lower()


def test_cube_uses_probability_of_action_zero_for_history_and_equilibria():
    equilibrium = np.array([[1, 0], [0, 1], [1, 0]])
    figure = trajectory_figure(
        experiment(players=3, actions=2),
        spec(actions=2, players=3, category="Box Games", equilibria=(equilibrium,)),
        animate=False,
    )
    history = role_traces(figure, "history")[0]
    np.testing.assert_allclose([history.x[0], history.y[0], history.z[0]], [1, 1, 1])
    equilibrium_trace = role_traces(figure, "equilibrium")[0]
    np.testing.assert_allclose(
        [equilibrium_trace.x[0], equilibrium_trace.y[0], equilibrium_trace.z[0]],
        [1, 0, 1],
    )
    for axis in (
        figure.layout.scene.xaxis,
        figure.layout.scene.yaxis,
        figure.layout.scene.zaxis,
    ):
        assert "P(Action 0)" in axis.title.text


def test_cube_axes_keep_action_index_explicit_with_named_actions():
    game_spec = spec(actions=2, category="Box Games")
    game_spec.action_labels = ("Dove", "Hawk")
    figure = trajectory_figure(experiment(players=3, actions=2), game_spec, animate=False)
    assert figure.layout.scene.xaxis.title.text == "P1: P(Dove)"
    assert "Dove (action 0)" in projection_note(game_spec)


def test_cube_axis_titles_are_compact_without_duplicating_generic_action_labels():
    figure = trajectory_figure(
        experiment(players=3, actions=2), spec(actions=2, category="Box Games")
    )
    for player, axis in enumerate(
        (figure.layout.scene.xaxis, figure.layout.scene.yaxis, figure.layout.scene.zaxis)
    ):
        assert axis.title.text == f"P{player + 1}: P(Action 0)"
        assert "<br>" not in axis.title.text
        assert axis.title.font.size <= 11
    assert figure.layout.scene.camera.eye.x < 1.6


@pytest.mark.parametrize("category,players,actions", [("Box Games", 3, 2), ("RPS Games", 2, 3)])
def test_equilibrium_legend_is_one_group_with_individual_reference_hover_labels(
    category, players, actions
):
    profiles = tuple(np.full((players, actions), 1 / actions) for _ in range(5))
    game_spec = spec(actions=actions, category=category, equilibria=profiles)
    game_spec.equilibrium_labels = tuple(f"Reference {i + 1}" for i in range(5))
    figure = trajectory_figure(experiment(players=players, actions=actions), game_spec)
    equilibria = role_traces(figure, "equilibrium")
    entries = [trace for trace in equilibria if trace.showlegend]
    assert len(entries) == 1
    assert entries[0].name == "Equilibrium references"
    assert all(trace.legendgroup == "equilibria" for trace in equilibria)
    assert all(
        any(label in trace.hovertemplate for trace in equilibria)
        for label in game_spec.equilibrium_labels
    )


def test_animation_controls_and_legend_have_separate_vertical_rows():
    figure = trajectory_figure(
        experiment(1000, players=3, actions=2), spec(actions=2, category="Box Games")
    )
    buttons = figure.layout.updatemenus[0]
    slider = figure.layout.sliders[0]
    legend = figure.layout.legend
    assert buttons.y > slider.y > legend.y
    assert buttons.y - slider.y >= 0.15
    assert slider.y - legend.y >= 0.15
    assert buttons.yanchor == slider.yanchor == legend.yanchor == "top"
    assert figure.layout.margin.b >= 180
    assert figure.layout.height - figure.layout.margin.t - figure.layout.margin.b >= 300
    assert len(slider.steps) == len(figure.frames) == 80
    assert slider.currentvalue.visible is False
    assert sum(bool(step.label) for step in slider.steps) <= 5
    assert slider.steps[-1].label == "1000"
    assert figure.layout.annotations[-1].text == "Iteration 1"
    assert figure.frames[-1].layout.annotations[-1].text == "Iteration 1,000"


def test_player_colors_distinguish_cube_axes_and_policy_spaces_without_changing_algorithm_colors():
    colors = ("#3866e9", "#c78322", "#218c63")
    cube = trajectory_figure(
        experiment(players=3, actions=2), spec(actions=2, category="Box Games")
    )
    for axis, color in zip(
        (cube.layout.scene.xaxis, cube.layout.scene.yaxis, cube.layout.scene.zaxis), colors
    ):
        assert axis.title.font.color == color
        assert axis.tickfont.color == color
    simplex = trajectory_figure(experiment(), spec())
    for annotation, color in zip(simplex.layout.annotations[:2], colors):
        assert annotation.font.color == color
    for trace, color in zip(role_traces(simplex, "action_labels"), colors):
        assert trace.textfont.color == color
    for frame in simplex.frames:
        assert [annotation.text for annotation in frame.layout.annotations[:2]] == [
            "Player 1",
            "Player 2",
        ]
        assert [annotation.font.color for annotation in frame.layout.annotations[:2]] == list(
            colors[:2]
        )
    for trace in role_traces(simplex, "history"):
        assert trace.line.color == algorithm_color("Hedge")


def test_plot_titles_are_short_and_do_not_repeat_explanations():
    result = experiment()
    for figure in (
        trajectory_figure(result, spec()),
        diagnostics_figure(result),
        strategy_figure(result, spec()),
    ):
        assert "<br>" not in figure.layout.title.text
        assert "<sup>" not in figure.layout.title.text


def test_large_comparison_legend_has_one_entry_per_algorithm():
    result = experiment(algorithms=("Hedge", "EXP3"), repeats=5)
    for figure in (trajectory_figure(result, spec()), diagnostics_figure(result)):
        entries = [trace for trace in figure.data if trace.showlegend]
        assert {trace.name for trace in entries} == {"Hedge", "EXP3"}
        assert len(entries) == 2
        hedge = [
            trace for trace in figure.data if trace.meta and trace.meta.get("algorithm") == "Hedge"
        ]
        assert all(trace.legendgroup == "Hedge" for trace in hedge)


@pytest.mark.parametrize("players,actions,category", [(2, 3, "RPS Games"), (3, 2, "Box Games")])
def test_animation_only_updates_one_point_in_explicit_current_marker_traces(
    players, actions, category
):
    result = experiment(steps=1000, players=players, actions=actions, repeats=2)
    figure = trajectory_figure(result, spec(actions=actions, category=category))
    assert len(figure.frames) <= 80
    marker_indices = [
        i
        for i, trace in enumerate(figure.data)
        if trace.meta and trace.meta.get("role") == "current"
    ]
    assert marker_indices
    for frame in figure.frames:
        assert tuple(frame.traces) == tuple(marker_indices)
        assert all(len(trace.x) == len(trace.y) == 1 for trace in frame.data)
        if category == "Box Games":
            assert all(len(trace.z) == 1 for trace in frame.data)
    for mapped_index, frame_trace in zip(figure.frames[0].traces, figure.frames[0].data):
        trace = figure.data[mapped_index]
        np.testing.assert_allclose(frame_trace.x, trace.x)
        np.testing.assert_allclose(frame_trace.y, trace.y)
    final = figure.frames[-1]
    assert final.name == "1000"


def test_payload_is_bounded_for_a_hundredfold_increase_in_iterations():
    small = trajectory_figure(experiment(1000, repeats=3), spec())
    large = trajectory_figure(experiment(100000, repeats=3), spec())
    assert len(large.to_json()) < 1.35 * len(small.to_json())
    assert all(len(trace.x) <= 1000 for trace in role_traces(large, "history"))
    assert len(large.frames) <= 80


def test_empirical_view_uses_its_own_history_and_rejects_missing_history():
    result = experiment()
    with pytest.raises(ValueError, match="empirical"):
        trajectory_figure(result, spec(), view="Empirical frequencies")
    empirical = np.broadcast_to([0, 1, 0], result.runs[0].strategies.shape)
    result.runs[0].empirical_strategies = empirical
    figure = trajectory_figure(result, spec(), view="Empirical frequencies", animate=False)
    vertex = probability_vertices(3)[1]
    for trace in role_traces(figure, "history"):
        np.testing.assert_allclose(trace.x, vertex[0])
        np.testing.assert_allclose(trace.y, vertex[1])


def test_diagnostics_keep_player_series_separate_and_sample_true_iteration_numbers():
    result = experiment(100000, algorithms=("Hedge", "EXP3"), repeats=2)
    figure = diagnostics_figure(result, metric="payoffs")
    traces = role_traces(figure, "diagnostic")
    assert len(traces) == 8
    for trace in traces:
        assert len(trace.x) <= 1000
        assert trace.x[0] == 1
        assert trace.x[-1] == 100000
        assert np.unique(trace.y).size == 1
        assert trace.line.color == algorithm_color(trace.meta["algorithm"])
    assert figure.layout.yaxis.title.text == "Expected utility"
    assert figure.layout.yaxis2.title.text == "Expected utility"


def test_nash_gap_has_a_zero_reference_and_does_not_mean_player_payoff():
    figure = diagnostics_figure(experiment(), metric="nash_gap")
    assert figure.layout.yaxis.title.text == "Nash gap"
    assert any(shape.y0 == shape.y1 == 0 for shape in figure.layout.shapes)
    with pytest.raises(ValueError, match="metric"):
        diagnostics_figure(experiment(), metric="made-up metric")


def test_final_policy_bars_display_full_probabilities_and_repeat_range():
    result = experiment(actions=4, repeats=2)
    result.runs[1].strategies = np.broadcast_to([0.5, 0.5, 0, 0], (12, 2, 4))
    figure = strategy_figure(result, spec(actions=4))
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert len(bars) == 2
    for trace in bars:
        assert tuple(trace.x) == spec(actions=4).action_labels
        np.testing.assert_allclose(trace.y, [0.25, 0.25, 0, 0.5])
        np.testing.assert_allclose(trace.error_y.array, [0.25, 0.25, 0, 0.5])
        assert trace.marker.color == algorithm_color("Hedge")
    assert "mean" in figure.layout.title.text.lower()
    assert "range" in figure.layout.title.text.lower()


def test_colors_remain_consistent_when_algorithm_selection_changes():
    alone = diagnostics_figure(experiment(algorithms=("EXP3",)))
    comparison = diagnostics_figure(experiment(algorithms=("Hedge", "EXP3")))
    a = role_traces(alone, "diagnostic")[0]
    b = next(
        trace
        for trace in role_traces(comparison, "diagnostic")
        if trace.meta["algorithm"] == "EXP3"
    )
    assert a.line.color == b.line.color
    assert a.name == "EXP3"
    assert "EXP3 · Run 1" in a.hovertemplate


def test_plotting_does_not_modify_research_histories():
    result = experiment()
    original = result.runs[0].strategies.copy()
    result.runs[0].strategies.flags.writeable = False
    trajectory_figure(result, spec())
    strategy_figure(result, spec())
    diagnostics_figure(result)
    np.testing.assert_array_equal(result.runs[0].strategies, original)


def test_time_average_view_is_the_prefix_mean_of_current_policies():
    result = experiment(steps=3)
    figure = trajectory_figure(result, spec(), view="Time-average policy", animate=False)
    expected = project_strategies(
        np.cumsum(result.runs[0].strategies[:, 0], axis=0) / np.arange(1, 4)[:, None]
    )
    for trace in role_traces(figure, "history"):
        np.testing.assert_allclose(np.column_stack((trace.x, trace.y)), expected)


@pytest.mark.parametrize(
    "players,actions,category,algorithms,repeats",
    [
        (
            3,
            2,
            "Box Games",
            (
                "BFTL-EXP3",
                "EXP3",
                "Hedge",
                "Optimistic FTRL",
                "Projected Gradient Ascent",
                "Regret Matching",
                "Regret Matching Softmax",
            ),
            5,
        ),
        (2, 6, "RPS Games", ("Fictitious Play", "Smooth Fictitious Play", "Hedge", "EXP3"), 5),
    ],
)
def test_worst_public_comparison_trajectory_payload_stays_under_five_mib(
    players, actions, category, algorithms, repeats
):
    result = experiment(10000, players, actions, algorithms, repeats)
    figure = trajectory_figure(result, spec(actions=actions, category=category))
    assert len(figure.to_json().encode()) <= 5 * 1024 * 1024
    assert all(len(trace.x) <= 1000 for trace in role_traces(figure, "history"))


def test_time_average_view_remains_exact_under_sparse_display_sampling():
    result = experiment(100001)
    probabilities = result.runs[0].strategies.copy()
    probabilities[1000:2000] = [0, 1, 0]
    result.runs[0].strategies = probabilities
    expected = np.cumsum(probabilities, axis=0) / np.arange(1, 100002)[:, None, None]
    figure = trajectory_figure(result, spec(), view="Time-average policy")
    for trace in role_traces(figure, "history"):
        indices = np.asarray(trace.customdata[:, 0], dtype=int) - 1
        player = trace.meta["player"]
        np.testing.assert_allclose(
            np.column_stack((trace.x, trace.y)),
            project_strategies(expected[indices, player]),
            atol=1e-12,
        )
    for frame in (
        figure.frames[0],
        figure.frames[len(figure.frames) // 2],
        figure.frames[-1],
    ):
        index = int(frame.name) - 1
        for trace in frame.data:
            player = trace.meta["player"]
            np.testing.assert_allclose(
                np.column_stack((trace.x, trace.y)),
                project_strategies(expected[index : index + 1, player]),
                atol=1e-12,
            )
