# Regret Lab / RegretMinDynamics

A reproducible research workspace for learning dynamics in finite games.
The React interface keeps the chart prominent and puts explanations, methods, and payoff details in an optional Info drawer.
Python and NumPy run simulations on the server; Cloud Run remains the hosting target.

## Run the workspace

Use Python 3.12 and Node.js 24.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
uvicorn app:app --host 127.0.0.1 --port 8512
```

Open `http://127.0.0.1:8512`.
The health endpoint is `/health`, which avoids Cloud Run's reserved URL paths.
For frontend development, run the API on port 8512 and `npm run dev --prefix frontend` in a second terminal.
Vite proxies API requests to the Python server.

Choose a game, learning rules, and iteration count, then click Run experiment.
Advanced settings contain repeats, seed, initialization, and learning parameters.
Setup changes preserve the completed experiment until the next successful run.
Player 1 is blue, Player 2 is orange, and Player 3 is green across axes, subplot headers, and payoff details.
Trajectory lines use algorithm colors, fine continuous strokes, and lighter additional repeats.
Replay is optional and disabled initially.

Simulations run only on explicit requests.
The public API caps work at 200,000 total steps: iterations multiplied by repeats and algorithms.
Paths use at most 1,000 points each and an aggregate point budget.
Trajectories use an antialiased Three.js cube and native SVG player panels.
The cube uses a perspective view, softly shaded back faces, small colored axes, and label collision handling.
Rendering stays idle between interactions; pixel ratio is capped at 1.5 and there are no shadows or post-processing effects.
Replay moves outlined particles linearly between sampled points and labels interpolated positions as approximate.
Particles leave short fading tails showing the most recent 12% of iterations; future paths stay hidden.
Pausing preserves thin paths up to the particles; completed playback preserves the full trajectories.
Stopping replay restores the complete sampled trajectories.
Playing, pausing, and scrubbing use the existing browser data without server requests.
Legend buttons hide algorithms or equilibrium references to clarify comparisons.
The cube renderer and basic Plotly diagnostic/final-policy bundle load only when needed.
The initial JavaScript bundle has a 100 KB gzip budget; optional chart bundles have separate budgets.

## Interpret the results

| Quantity | Meaning |
| --- | --- |
| Current policy | The distribution used to sample actions, including BFTL exploration and Fictitious Play's response rule. |
| Time-average policy | Arithmetic mean of the sampling policies through each iteration. |
| Empirical frequencies | Fictitious Play's action counts normalized with initial pseudocounts; available for FP methods only. |
| Expected payoff | Utility under the current independent policy profile and the expected payoff table. |
| Average expected external regret | Best fixed action's cumulative expected utility against the opponents' policy sequence, minus the learner's cumulative expected utility, divided by elapsed iterations. |
| Nash gap | Sum of nonnegative unilateral expected utility improvements at the current profile. |

Expected external regret can be negative and does not measure realized bandit regret.
The summary averages final expected regret across players and runs, and final Nash gaps across runs.
Low external regret and low current-profile Nash gap are different criteria.
Method descriptions identify actual feedback assumptions; names do not establish theoretical guarantees for these implementations.
Existing mathematical update recurrences are preserved.

Noisy-game diagnostics use the base expected payoff table without drawing extra noise.
FP methods query fresh noisy payoffs when estimating responses in noisy games, preserving their existing behavior.
Each run owns its random generators and does not modify NumPy's global random state.
Repeat initialization seeds are shared across algorithms, with explicit conversion into scores, probabilities, or FP beliefs.
FP pseudocounts initialize beliefs, so identical initial decision policies across methods are not assumed.

The binary three-player cube shows each player's probability of action zero.
Three-action triangles preserve each player's complete policy.
Polygons for four or more actions are lossy projections; different policies can occupy the same location.
Use final policy probabilities and diagnostics for those games.
Displayed equilibrium references are certified against unilateral deviations and can omit other mixed equilibria or continua.

## Command-line research

Both CLIs use the same seeded engine as the web application.
Existing aliases remain supported.

```bash
python main.py --game PureCoordination --algorithm OptFTRL --iterations 1000 --runs 3 --seed 42
python rps_experiments.py --game RPS --algorithm FictitiousPlay --num_iterations 1000 --seed 42
python rps_experiments.py --game RPS --algorithm Hedge --iterations 100000 --seed 42 --no-gif
```

Use `--output-dir` to choose a destination; the default is `outputs/`.
ZIP archives contain configuration, run seeds, versions, complete arrays, and per-run CSV summaries.
Load the NPZ with `allow_pickle=False`.
GIFs use at most 60 frames and 1,000 path points; `--no-gif` saves research data directly.
The CLI has no public API work limit.
Use `--game all --algorithm all` to run the four original games in that entry point with compatible algorithms.
Catalog names also work when quoted.

## Project layout

| Path | Responsibility |
| --- | --- |
| `frontend/` | React, TypeScript, Vite, lightweight styling, and UI tests. |
| `api/` | Validated HTTP adapter, resource limits, temporary results, and static frontend serving. |
| `app.py` | ASGI entry point. |
| `research/catalog.py` | Game and method metadata, action labels, certified equilibrium references. |
| `research/simulation.py` | Seeded experiments and immutable result arrays. |
| `research/diagnostics.py` | Expected utilities, external regret, and Nash gaps. |
| `algorithms/`, `games/` | Learning rules, owned random streams, finite payoff tables, and noisy feedback. |
| `ui/plots.py` | Bounded figure geometry, diagnostics, and optional server animation. |
| `frontend/src/trajectory*` | Numeric decoding, sampled playback, Three.js cube, and SVG player panels. |
| `research/exports.py`, `plotting/exports.py` | Portable archives and bounded GIFs. |
| `main.py`, `rps_experiments.py` | Compatible research CLIs. |

## Validation and deployment

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest --cov --cov-report=term-missing
npm run test:coverage --prefix frontend
npm run build --prefix frontend
```

Python and frontend coverage gates are 80%.
CI verifies both sides before building the production image.
Pull requests run checks; pushes to `main` deploy after verification.
The Docker image builds the React assets and serves them with the Python API as one application.
It excludes authentication artifacts and runs as an unprivileged user.

Results are temporary: at most four experiments, 96 MiB of arrays, and a 20-minute lifetime.
Older results can be evicted sooner, and a server restart clears them.
Download research archives to retain a completed experiment.
The deployment recipe uses one worker and one Cloud Run instance because result storage is process-local.
A future deployment with multiple instances needs shared result storage.
The API limits concurrent simulations, rendering, and downloads, and enforces request and input-size budgets.

## Hosting decision

Keep the frontend and Python compute together on GCP.
React can be hosted statically on Netlify, but the Python simulation API still needs server compute.
The September 2026 GCP billing cost table shows $0.17 for `box-regret-ae01b8`.
Monitoring recorded approximately 34 billable instance-minutes, 4,749 requests, and 0.19 GB sent.
The measured service used one CPU and 512 MiB with no minimum instances.

Cloud Run's free allowances and registry pricing explain the small bill: [Cloud Run pricing](https://cloud.google.com/run/pricing), [Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing).
Netlify Free currently includes 300 credits; Personal starts at $9 per month: [Netlify pricing](https://www.netlify.com/pricing/).
Those plans provide no compute-cost advantage for the Python engine in this project.
Local implementation work does not change or redeploy the running GCP service.
