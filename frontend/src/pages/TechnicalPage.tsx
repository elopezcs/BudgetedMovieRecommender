import { Card } from '../components/Card'
import { PageTitle } from '../components/PageTitle'

export function TechnicalPage() {
  return (
    <div className="space-y-6">
      <PageTitle
        eyebrow="Architecture"
        title="Technical Credibility"
        description="This frontend is a product-grade presentation layer over the existing offline RL pipeline. It does not replace training logic; it exposes it."
      />

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <h3 className="text-lg font-semibold text-white">Environment</h3>
          <p className="mt-2 text-sm text-slate-300">
            The backend uses a custom Gymnasium environment with 12-dimensional state, 11 discrete actions, and budgeted
            question constraints that penalize over-questioning and abandonment.
          </p>
        </Card>
        <Card>
          <h3 className="text-lg font-semibold text-white">User Simulator</h3>
          <p className="mt-2 text-sm text-slate-300">
            Four behavioral personas with hidden preferences and stochastic responses drive realistic interaction dynamics
            for ask-vs-recommend decisions.
          </p>
        </Card>
        <Card>
          <h3 className="text-lg font-semibold text-white">Learning Algorithms</h3>
          <p className="mt-2 text-sm text-slate-300">
            Q-learning (tabular), DQN, and PPO are trained offline and evaluated with consistent metrics including reward,
            acceptance, abandonment, and question usage.
          </p>
        </Card>
        <Card>
          <h3 className="text-lg font-semibold text-white">Saved Artifacts</h3>
          <p className="mt-2 text-sm text-slate-300">
            The app reads existing outputs under <code>models/</code> and <code>results/</code>, including summary JSON,
            evaluation CSVs, comparison exports, and model metadata.
          </p>
        </Card>
      </div>

      <Card>
        <h3 className="text-lg font-semibold text-white">Inference Adapter</h3>
        <p className="mt-2 text-sm text-slate-300">
          Real RL action selection for q_learning, dqn, and ppo is routed through the existing Python adapter at
          <code>src/inference/inference_adapter.py</code>. A minimal API wrapper calls this adapter and keeps the frontend
          clean and framework-agnostic.
        </p>
      </Card>
    </div>
  )
}
