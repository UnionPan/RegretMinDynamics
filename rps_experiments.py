"""Compatible RPS entry point using the shared seeded research engine."""

from main import run_cli, run_selected_experiments


def run_rps_experiments(
    game_name="all", algorithm_name="all", num_iterations=300, **options
):
    """Run RPS-family experiments, retaining the historical callable signature."""
    return run_selected_experiments(
        game_name, algorithm_name, category="RPS Games", iterations=num_iterations, **options
    )


if __name__ == "__main__":
    raise SystemExit(run_cli(category="RPS Games"))
