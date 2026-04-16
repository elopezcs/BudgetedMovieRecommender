import { useEffect, useMemo, useState } from 'react'
import { Card } from '../components/Card'
import { FeedbackState } from '../components/FeedbackState'
import { PageTitle } from '../components/PageTitle'
import { ComparisonTable } from '../features/comparison/ComparisonTable'
import { MetricRadarPanel } from '../features/comparison/MetricRadarPanel'
import { RewardComparisonChart } from '../features/comparison/RewardComparisonChart'
import { fetchLatestComparison } from '../services/resultsService'
import type { ComparisonMetric, ComparisonResponse } from '../types/results'

type SortKey = keyof Pick<
  ComparisonMetric,
  'average_cumulative_reward' | 'acceptance_rate' | 'abandonment_rate' | 'average_questions_asked' | 'average_session_length'
>

export function ComparisonPage() {
  const [data, setData] = useState<ComparisonResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<SortKey>('average_cumulative_reward')

  useEffect(() => {
    fetchLatestComparison()
      .then(setData)
      .catch((err: Error) => setError(err.message))
  }, [])

  const sortedMetrics = useMemo(() => {
    return [...(data?.metrics ?? [])].sort((a, b) => {
      const direction = sortBy === 'abandonment_rate' ? 1 : -1
      return direction * (b[sortBy] - a[sortBy])
    })
  }, [data?.metrics, sortBy])

  return (
    <div className="space-y-6">
      <PageTitle
        eyebrow="Model Benchmarking"
        title="Algorithm Comparison Dashboard"
        description="Compare reinforcement-learning policies and baseline strategies using the latest evaluation artifacts in the repository."
      />

      {error ? <FeedbackState title="Comparison data unavailable" message={error} /> : null}

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h3 className="text-lg font-semibold text-white">Average Reward by Algorithm</h3>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Sort table by
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SortKey)}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            >
              <option value="average_cumulative_reward">Reward</option>
              <option value="acceptance_rate">Acceptance Rate</option>
              <option value="abandonment_rate">Abandonment Rate</option>
              <option value="average_questions_asked">Questions Asked</option>
              <option value="average_session_length">Session Length</option>
            </select>
          </label>
        </div>
        {sortedMetrics.length ? (
          <div className="mt-4">
            <RewardComparisonChart metrics={sortedMetrics} />
          </div>
        ) : (
          <div className="mt-4">
            <FeedbackState
              title="No metrics to compare"
              message="Generate comparison artifacts under results/comparisons to populate this page."
            />
          </div>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <h3 className="text-lg font-semibold text-white">Acceptance vs Abandonment</h3>
          <p className="mt-2 text-sm text-slate-400">
            Visual signal of conversion quality and drop-off risk across top policies.
          </p>
          <div className="mt-4">{sortedMetrics.length ? <MetricRadarPanel metrics={sortedMetrics} /> : null}</div>
        </Card>

        <Card className="lg:col-span-3">
          <h3 className="text-lg font-semibold text-white">Metric Table</h3>
          <p className="mt-2 text-sm text-slate-400">All core evaluation metrics from comparison_summary outputs.</p>
          <div className="mt-4">
            <ComparisonTable metrics={sortedMetrics} />
          </div>
        </Card>
      </div>
    </div>
  )
}
