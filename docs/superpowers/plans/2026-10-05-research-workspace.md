# Research Workspace Implementation Plan

> **For agentic workers:** Use task-by-task execution with explicit ownership and review.

**Goal:** Build a reproducible, mathematically transparent simulation workspace with efficient visualizations and a concise React interface on the existing GCP deployment.

**Architecture:** Keep the NumPy learning algorithms and Python research workflow.
Separate game/algorithm metadata, seeded experiment execution, diagnostics, plotting, exports, the HTTP adapter, and the React interface.
Compute full histories for research and downsample only the visualization layer.

**Tech Stack:** Python, NumPy, FastAPI, React, TypeScript, Vite, Three.js, SVG, Plotly, Matplotlib, pytest, Vitest, Cloud Run.

## Scope and deployment decision

The user approved staying on GCP, then explicitly requested React after reviewing the Streamlit design.
The React revision prioritizes concise wording, optional explanations, consistent player colors, and a spacious chart.
September 2026 monitoring shows 2,021 seconds of billable instance time, 4,749 requests, and 0.193 GB sent.
The current service uses one CPU and 512 MiB with no configured minimum instances.
React can be hosted statically on Netlify, while the Python API still needs server compute.
The September project billing table shows a cost of $0.17.
Browser-side Python would move compute to visitors and requires a separate feasibility evaluation.
No cloud deletion, production deployment, or credential changes are part of the local implementation.

## File boundaries and shared contracts

- `research/catalog.py`: one registry for games, descriptions, action labels, valid equilibrium profiles, and algorithm metadata.
- `research/simulation.py`: validated experiment configuration, seeded independent runs, immutable results, and initialization conversion.
- `research/diagnostics.py`: expected utilities, unilateral deviation gains, Nash gap, and explicitly defined expected external regret.
- `algorithms/*.py`: local RNG injection, input copying, histories of current policies and empirical frequencies where applicable, and safe reuse.
- `games/*.py`: consistent payoff-table access and seeded noise.
- `ui/plots.py`: pure bounded Plotly figures with stable animation trace mapping.
- `frontend/src/`: React setup, persistent results, optional explanations, accessible chart tabs, and downloads.
- `frontend/src/TrajectoryCanvas.tsx` and trajectory modules: event-driven Three.js cube, SVG policies, and sampled-path playback.
- `frontend/src/styles.css`: restrained visual design and responsive layout.
- `api/`: validated experiments, bounded temporary results, figure/archive requests, and static frontend serving.
- `app.py`: ASGI entry point.
- `research/exports.py`: portable result archive and metadata.
- `plotting/exports.py`: bounded CLI GIF rendering without untracked source imports.
- `main.py` and `rps_experiments.py`: compatible CLI entry points using the shared engine.
- `tests/`: unit, API integration, CLI, rendering lifecycle, and payload regression coverage.
- `requirements.txt`, `requirements-dev.txt`, and `pyproject.toml`: reproducible dependency and check configuration.
- `README.md`: current usage, metric definitions, deployment decision, and limitations.

Registry names use the existing human-readable game and algorithm names.
Categories are `Box Games` and `RPS Games`.
The UI and both CLIs use the same catalog and runner.

The public engine contracts are:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    game_name: str
    algorithm_names: tuple[str, ...]
    iterations: int = 1000
    runs: int = 3
    seed: int = 42
    learning_rate: float = 0.2
    decay: float = -0.5
    temperature: float = 0.1
    exploration: float = 0.1
    initialization: str = "random"

@dataclass(frozen=True)
class RunResult:
    algorithm: str
    run: int
    seed: int
    strategies: np.ndarray
    empirical_strategies: np.ndarray | None
    payoffs: np.ndarray
    average_regret: np.ndarray
    nash_gap: np.ndarray

@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    runs: tuple[RunResult, ...]
    runtime_seconds: float

def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...
```

`strategies` always means the distribution actually used to sample actions, including Fictitious Play's decision policy and BFTL's exploration mixture.
Fictitious Play's historical empirical frequencies remain separately available.
Arrays in returned results are copied and made read-only.
Diagnostics use the base payoff table and describe expected utilities, including for noisy games.
Expected external regret compares a fixed action with the sequence of opponents' mixed policies and the learner's expected utilities.
It is distinct from realized bandit regret and from a convergence guarantee.
Nash gap is the sum of nonnegative unilateral expected utility improvements.

## Task 1: Reproducible engine and scientific diagnostics

- [x] Reproduce the existing shared-initialization defect and incorrect equilibrium annotations before changes.
- [x] Add failing tests for input isolation, seed reproducibility, probability validity, independent repeats, metric definitions, and equilibrium certificates.
- [x] Add local RNG arguments to algorithms and noisy games, preserving existing positional constructors and CLI compatibility.
- [x] Preserve all existing mathematical update rules in this scope and correct misleading descriptions.
- [x] Record actual decision policies separately from empirical frequencies.
- [x] Implement validated catalog, runner, and vectorized diagnostics.
- [x] Enumerate pure equilibria from payoff tables and certify displayed mixed profiles numerically.
- [x] Expose feedback assumptions honestly and avoid unsupported regret/convergence claims.
- [x] Run `python -m pytest tests/test_simulation.py tests/test_diagnostics.py tests/test_catalog.py -q`.

Expected first run: failures expose missing contracts and reproduced defects.
Expected final run: all tests pass with deterministic seeded results and valid probabilities.

## Task 2: Bounded, consistent visualizations

- [x] Add failing payload-scaling and animation trace-mapping tests.
- [x] Build reusable dark-text-on-light-background plots with consistent player and algorithm colors.
- [x] Bound background paths, animation frames, and moving trails independently of simulation length.
- [x] Use at most 80 animation frames and 1,000 static points per path, with moving-marker frames and an acceptance limit of 5 MiB serialized per public comparison figure.
- [x] Preserve full histories in engine results and exports.
- [x] Handle two-action games as a line and explain that n-gon projections for four or more actions are not one-to-one.
- [x] Make 3D axis labels explicitly refer to probability of action zero.
- [x] Run `python -m pytest tests/test_plots.py -q` and compare 1,000 versus 100,000 iteration payloads.

Expected final behavior: plot payload growth is bounded, initial and final frames map to the same traces, and both game families are readable.

## Task 3: Concise React workspace and HTTP adapter

- [x] Reproduce disappearing Streamlit results and initial UI density/axis overlap.
- [x] Obtain explicit user approval to replace Streamlit with React.
- [x] Define a warm, quiet interface with a setup rail, spacious chart, and details drawer.
- [x] Build React/Vite with lazy partial chart bundles and explicit simulation requests.
- [x] Preserve results across setup and display changes; handle failures and cancellations clearly.
- [x] Put explanations, methods, reference equilibria, and payoff exploration behind toggles.
- [x] Color player axes, headings, and payoff values consistently; group equilibrium legends.
- [x] Expose a validated FastAPI adapter with bounded temporary storage, body sizes, request rates, and concurrency.
- [x] Verify API experiments, figures, export consistency, expiry, and resource guards.
- [x] Verify React critical flows, asynchronous chart cleanup, and at least 80% frontend coverage.
- [x] Inspect real browser UI at desktop and narrower widths.

## Task 4: Working CLIs and portable research exports

- [x] Reproduce the missing `visualizations.simplex_plot` import through the documented CLI.
- [x] Add failing subprocess tests for both entry points.
- [x] Use shared engine configuration with existing CLI aliases retained.
- [x] Add seed, iterations, repeats, output directory, and optional GIF controls.
- [x] Export full arrays plus configuration, seed, version metadata, and summaries without pickle dependencies.
- [x] Render GIFs in memory or unique temporary directories to avoid shared frame-file collisions.
- [x] Verify CLI exports round-trip and seeded CLI/UI configurations agree.

## Task 5: Dedicated trajectory rendering

The user approved improving rendering after seeing the React workspace.
Trajectories use an antialiased Three.js cube and crisp SVG player panels.
Diagnostics and final policies retain the lazy-loaded basic Plotly bundle.
Playback interpolates linearly between sampled points and identifies approximate positions.
Metrics and exports retain recorded policies.
Rendering stays idle between interactions, with bounded pixel ratio, no shadows, and deterministic cleanup.

- [x] Decode numeric arrays and preserve action, player, repeat, and iteration metadata.
- [x] Fit the cube with small colored labels and grouped equilibrium references.
- [x] Verify replay, legend toggles, probability mapping, and lifecycle cleanup.
- [x] Inspect both game families and narrow layouts in the browser.

## Task 6: Deployment hygiene, review, and completion checks

- [x] Pin supported runtime dependencies and add development test/lint dependencies.
- [x] Update the obsolete Python container base and add a focused Docker ignore file.
- [x] Retain existing GCP workflow; add verification before builds without changing cloud resources.
- [x] Update README instructions and explain metric and visualization limitations.
- [x] Run full tests with at least 80% coverage of changed runtime modules.
- [x] Run lint and compile checks, inspect all diffs, and perform independent code review.
- [x] Inspect Box and RPS results, rotation, replay, display persistence, and responsive layouts in the browser.
- [x] Verify full-resolution downloads through API roundtrips and frontend interaction tests.
- [x] Report actual verification results and any environment-specific limitations.

Do not modify generated changelogs or commit/push without a separate request.

## Final verification

### Particle trails and stroke refinement

- [x] Reproduce thick dashed-path clutter in the browser.
- [x] Test continuous path styling, screen-sized outlined markers, time-clipped tails, and static restoration before implementation.
- [x] Share clipped sample semantics in `frontend/src/trajectoryData.ts` and reuse GPU buffers in `frontend/src/trajectoryLine.ts`.
- [x] Integrate short fading tails into `frontend/src/trajectoryCube.ts` and six lightweight opacity bands into `frontend/src/TrajectoryCanvas.tsx`.
- [x] Verify resource disposal, absence of future samples, probability fidelity, production build, and independent review.

Run `npm run test:coverage --prefix frontend`, `npm run build --prefix frontend`, and `git diff --check` after the final changes.
Inspect real 3D and SVG playback in Safari and Chrome.

- Python: 202 tests passed with warnings treated as errors and 96.30% branch-inclusive coverage.
- Frontend: 90 tests passed; statements 96.66%, branches 86.43%, functions 95.86%, lines 98.96%.
- Production frontend build, strict TypeScript, Ruff, compilation, and diff whitespace checks passed.
- JavaScript gzip sizes: initial 75.6 KB, trajectory controls 5.1 KB, Three.js cube 144.9 KB, basic Plotly 380.2 KB.
- Independent rendering checks verified 42 real figures and 10,368 sampled policies across all 21 games.
- Browser inspection covered cube rotation and replay, SVG triangles, short fading tails, completed path restoration, and responsive setup/player-panel layouts.
- Known-vulnerability audits found none in the resolved runtime requirements and npm dependencies.
- Docker is unavailable in this environment, so the container image was not built locally.
- The user requested a commit and push after successful verification.
- GitHub verification, Docker build, and Cloud Run deployment succeeded for `e4e8787`.
- A live experiment rendered and replayed successfully without browser console errors.
- The deployed health smoke check exposed Cloud Run's interception of `/healthz`.
  The endpoint moved to `/health`, with API regression tests first reproducing the missing route.
