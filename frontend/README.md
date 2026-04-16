# Frontend Demo (React + Vite)

This app is the presentation layer for the Budgeted Interactive Movie Recommender.

## What It Shows

- **Overview page**: product framing and KPI highlights
- **Live Demo page**: step-by-step ask-vs-recommend session playback
- **Algorithm Comparison page**: charts and table from `results/comparisons/...`
- **Technical Credibility page**: concise architecture and artifact evidence

## Prerequisites

- Node.js 20+
- Python environment with repository requirements installed
- Existing model and result artifacts under `models/` and `results/`

## Install and Run

From repository root:

```bash
cd frontend
npm install
```

Start the API bridge (in one terminal, from repo root):

```bash
python -m uvicorn src.demo_api.app:app --reload --port 8000
```

Start the frontend (second terminal):

```bash
cd frontend
npm run dev
```

Open: `http://localhost:5173`

## Data and Inference Sources

- **Overview / comparison metrics** come from latest files under:
  - `results/comparisons/*/comparison_summary.json`
  - fallback: latest `results/q_learning/*/summary.json`, `results/dqn/*/summary.json`, `results/ppo/*/summary.json`
- **Live RL actions** (`q_learning`, `dqn`, `ppo`) come from:
  - `src/inference/inference_adapter.py` via the API bridge
- **Baseline actions** come from:
  - `src/agents/baselines.py`
- **Session state transitions** are computed by:
  - `src/env/movie_recommender_env.py`

## Demo Controls (Live Demo Page)

- Select user profile, policy, question budget, and optional seed
- Click **Start**
- Use **Next Step** for narrated walkthrough
- Use **Play Session** for automatic playback
- Use **Reset** to replay the same scenario
- Expand **Advanced Details** for technical Q&A

## Build

```bash
cd frontend
npm run build
```
# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
