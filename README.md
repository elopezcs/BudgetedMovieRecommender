# Budgeted Interactive Movie Recommender Using Reinforcement Learning

Offline RL training, evaluation, and demo stack for a budgeted interactive movie recommendation session.

This project builds and trains policies in Python, saves reusable model artifacts, serves
them through a lightweight FastAPI demo bridge, and presents the results in a React frontend.

## What This Project Includes

- Custom `Gymnasium` environment for short interactive recommendation sessions.
- User simulator with four behavioral user types.
- RL algorithms:
  - tabular Q-learning
  - DQN (Stable Baselines3)
  - PPO (Stable Baselines3)
- Baseline policies:
  - `always_recommend`
  - `always_ask`
  - `ask_once_then_recommend`
  - `random_policy`
- Evaluation scripts and side-by-side comparison outputs.
- FastAPI demo bridge and React presentation frontend for live walkthroughs.
- Tests for environment, reward logic, simulator, Q-learning update, demo API behavior, and artifact round-trip.

## Project Structure

```text
frontend/
docs/
src/
  env/
  agents/
  training/
  evaluation/
  demo_api/
  inference/
  utils/
models/
results/
configs/
tests/
```

## Environment Definition

`src/env/movie_recommender_env.py` defines `BudgetedMovieRecommenderEnv`.

- **Observation (12-dim normalized vector)**
  - genre belief (5)
  - uncertainty
  - interaction history summary (asked/recommended/repeat ratios)
  - remaining question budget ratio
  - engagement
  - step ratio
- **Actions (11 discrete)**
  - question actions:
    - familiar, exploratory, serious, light, fast-paced, calm
  - recommendation actions:
    - action, comedy, drama, sci-fi, documentary
- **Termination**
  - abandonment or max-step truncation
- **Question Budget Modes**
  - `soft`: questions beyond the budget are still allowed but incur `over_budget_penalty`
  - `hard`: once the budget is exhausted, extra question actions are blocked and treated as over-budget attempts
- **Reward**
  - positive: recommendation acceptance, engagement continuation
  - negative: skipping, abandonment, question friction, over-budget questioning, repetition

`src/env/user_simulator.py` contains four user profiles:

- `action_focused`
- `balanced_viewer`
- `novelty_seeking`
- `question_sensitive`

Each profile has hidden genre preferences, question tolerance, repetition sensitivity,
and abandonment dynamics.

## Quick Start (Recommended)

If you want the demo to launch reliably on Windows, use Python 3.11 and create a fresh virtualenv.

From repository root (`c:\AIML\RL\MovieRecommender2`):

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

The default environment config in `configs/default.yaml` includes a question-budget enforcement mode:

```yaml
environment:
  question_budget: 4
  question_budget_mode: soft
```

- `soft` preserves the current behavior: policies may ask beyond the budget, but those steps receive `over_budget_penalty`
- `hard` turns the budget into a true cap: once the allowed questions are used, additional question actions do not execute a real user-question transition

When comparing `soft` and `hard` settings, retrain and re-evaluate models under each mode instead of reusing artifacts trained under a different environment dynamic.

Start backend API:

```bash
python -m uvicorn src.demo_api.app:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal, start frontend:

```bash
cd frontend
npm install
npm run dev
```

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://localhost:5173`
- API docs: `http://127.0.0.1:8000/docs`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For best compatibility in this project, prefer Python 3.11 (`py -3.11`).

## Train Models

Use the default config or pass your own with `--config`.

```bash
python -m src.training.train_q_learning --config configs/default.yaml
python -m src.training.train_dqn --config configs/default.yaml
python -m src.training.train_ppo --config configs/default.yaml
python -m src.training.train_all --config configs/default.yaml
```

Changing `environment.question_budget_mode` changes the environment dynamics seen during training. If you switch between `soft` and `hard`, treat those as separate experiments and retrain the policies.

Optional reproducibility override:

```bash
python -m src.training.train_q_learning --config configs/default.yaml --seed 123
```

## Evaluate Models

Evaluate latest run for each algorithm:

```bash
python -m src.evaluation.evaluate_q_learning --config configs/default.yaml
python -m src.evaluation.evaluate_dqn --config configs/default.yaml
python -m src.evaluation.evaluate_ppo --config configs/default.yaml
```

Evaluation and the Live Demo backend both read the same config-driven environment behavior, so keep the config aligned with the experiment you want to demonstrate.

Evaluate a specific model directory:

```bash
python -m src.evaluation.evaluate_dqn --config configs/default.yaml --model-dir models/dqn/dqn_seed42_steps6000
```

## Compare RL Algorithms and Ask-Once Baseline

```bash
python -m src.evaluation.compare_algorithms --config configs/default.yaml
```

This writes comparison artifacts using the latest `q_learning`, `dqn`, and `ppo` evaluation summaries plus the `ask_once_then_recommend` baseline:

- `results/comparisons/<tag>/comparison_summary.csv`
- `results/comparisons/<tag>/comparison_summary.json`
- `results/comparisons/<tag>/comparison_reward_chart.png`

## Artifacts and Output Formats

### Q-Learning

- Model:
  - `models/q_learning/<run_id>/q_table.npy`
  - `models/q_learning/<run_id>/metadata.json`
- Results:
  - `results/q_learning/<run_id>/summary.json`
  - `results/q_learning/<run_id>/evaluation_episodes.csv`
  - `results/q_learning/<run_id>/training_curve.csv`
  - `results/q_learning/<run_id>/config_snapshot.json`

### DQN / PPO

- Model:
  - `models/<algo>/<run_id>/model.zip` (SB3 artifact)
  - `models/<algo>/<run_id>/metadata.json`
- Results:
  - `results/<algo>/<run_id>/summary.json`
  - `results/<algo>/<run_id>/evaluation_episodes.csv`
  - `results/<algo>/<run_id>/config_snapshot.json`

## Metrics Produced

Each summary file includes:

- average cumulative reward
- recommendation acceptance rate
- abandonment rate
- average questions asked
- average session length

## Reproducibility

- Seed in `configs/default.yaml` controls training/evaluation randomness.
- Environment and algorithm seeding are set in training scripts.
- Run names are deterministic (`{algo}_seed{seed}_steps{n}`).
- Config snapshot is stored with each run.

## Tests

```bash
pytest -q
```

## Frontend Integration Notes

The shipped demo uses a concrete frontend/backend split:

- `frontend/` renders the Overview, Live Demo, Algorithm Comparison, and Technical Credibility pages.
- `src/demo_api/app.py` serves the live demo session endpoints and results endpoints consumed by the frontend.
- `src/inference/inference_adapter.py` loads trained RL artifacts and returns actions for `q_learning`, `dqn`, and `ppo`.
- `results/` comparison and summary artifacts feed the dashboard KPIs and charts.

## Inference Adapter (JSON I/O)

A minimal adapter is available at `src/inference/inference_adapter.py`.

It expects an input JSON payload:

```json
{
  "observation": [0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.1, 0.2, 0.0, 0.75, 0.8, 0.1]
}
```

Run it (reads from STDIN by default):

```bash
echo "{\"observation\":[0.2,0.2,0.2,0.2,0.2,0.6,0.1,0.2,0.0,0.75,0.8,0.1]}" | python -m src.inference.inference_adapter --algo q_learning
```

Or with files:

```bash
python -m src.inference.inference_adapter --algo dqn --input-json request.json --output-json response.json
```

Response format:

```json
{
  "algorithm": "q_learning",
  "model_dir": "models/q_learning/q_learning_seed42_steps700",
  "action_id": 6,
  "action_name": "rec_action"
}
```

## Frontend Demo Application

A presentation-ready React application now lives in `frontend/`. It provides:

- product-pitch overview with KPI cards
- live step-by-step recommender demo with budget and belief updates
- algorithm comparison charts and metric table
- technical credibility view describing RL architecture and artifacts
- demo policy selection for `q_learning`, `dqn`, and `ppo`
- session mode selection for `auto` and `interactive`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` and proxies `/api/*` calls to `http://127.0.0.1:8000`.

In the Live Demo page:

- `auto` mode advances the environment with the built-in user simulator.
- `interactive` mode pauses on each pending action and waits for manual feedback before continuing.
- `Play Session` / autoplay is only available in `auto` mode.

### Run the Lightweight Demo API Bridge

From repository root:

```bash
python -m uvicorn src.demo_api.app:app --reload --host 0.0.0.0 --port 8000
```

Note: prefer `python -m uvicorn ...` over `uvicorn ...` to ensure the command runs from the active virtual environment.

This bridge reuses existing backend components:

- session stepping via `src/env/movie_recommender_env.py`
- interactive/manual response flow from `src/demo_api/app.py`
- RL policy inference from `src/inference/inference_adapter.py`
- metric artifacts from `results/`

Primary demo endpoints include:

- `GET /api/demo/options`
- `POST /api/demo/session/start`
- `POST /api/demo/session/next`
- `POST /api/demo/session/respond`
- `POST /api/demo/session/reset`

### Presentation Flow (Recommended)

1. Open **Overview** to frame the problem and KPI outcomes.
2. Run **Live Demo**, choose one of `q_learning`, `dqn`, or `ppo`, then show both `auto` playback and `interactive` manual-response mode.
3. Use **Algorithm Comparison** to show reward/acceptance/abandonment tradeoffs.
4. Close on **Technical Credibility** to connect visuals to the real offline RL stack.

### Missing Artifacts Behavior

If comparison or summary files are missing, the app shows friendly guidance instead of crashing. Run:

```bash
python -m src.evaluation.compare_algorithms --config configs/default.yaml
```

to refresh comparison outputs.

## Troubleshooting Launch Issues

- `No module named 'src'`:
  - Run modules from repository root with `python -m ...` (do not run `python app.py` from root).
- `Program 'uvicorn.exe' failed to run: An Application Control ...`:
  - Use `python -m uvicorn ...` instead of `uvicorn ...`.
- NumPy import errors such as `No module named 'numpy._core._multiarray_umath'`:
  - Recreate `.venv` using Python 3.11 and reinstall dependencies with `--no-cache-dir`.
- DQN/PPO marked as skipped in `train_all`:
  - This indicates SB3 dependency import is blocked by the local environment policy; Q-learning still runs.

