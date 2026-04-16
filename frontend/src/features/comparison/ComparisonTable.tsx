import type { ComparisonMetric } from '../../types/results'
import { toFixed, toPercent, titleCase } from '../../utils/format'

interface ComparisonTableProps {
  metrics: ComparisonMetric[]
}

export function ComparisonTable({ metrics }: ComparisonTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-slate-700 text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th className="px-3 py-3">Algorithm</th>
            <th className="px-3 py-3">Avg Reward</th>
            <th className="px-3 py-3">Acceptance</th>
            <th className="px-3 py-3">Abandonment</th>
            <th className="px-3 py-3">Avg Questions</th>
            <th className="px-3 py-3">Session Length</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric.algorithm} className="border-b border-slate-800/80 text-slate-200">
              <td className="px-3 py-3 font-medium">{titleCase(metric.algorithm)}</td>
              <td className="px-3 py-3">{toFixed(metric.average_cumulative_reward, 3)}</td>
              <td className="px-3 py-3">{toPercent(metric.acceptance_rate)}</td>
              <td className="px-3 py-3">{toPercent(metric.abandonment_rate)}</td>
              <td className="px-3 py-3">{toFixed(metric.average_questions_asked, 2)}</td>
              <td className="px-3 py-3">{toFixed(metric.average_session_length, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
