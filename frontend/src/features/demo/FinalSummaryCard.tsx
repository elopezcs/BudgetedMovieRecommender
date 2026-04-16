import type { SessionSummary } from '../../types/demo'
import { toFixed } from '../../utils/format'

interface FinalSummaryCardProps {
  summary: SessionSummary
}

export function FinalSummaryCard({ summary }: FinalSummaryCardProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <Metric label="Total Reward" value={toFixed(summary.total_reward, 2)} />
      <Metric label="Accepted Recommendations" value={String(summary.accepted_recommendations)} />
      <Metric label="Skipped Recommendations" value={String(summary.skipped_recommendations)} />
      <Metric label="Questions Used" value={String(summary.questions_used)} />
      <Metric label="Session Length" value={String(summary.session_length)} />
      <Metric label="Abandoned" value={summary.abandoned ? 'Yes' : 'No'} />
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4">
      <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
    </div>
  )
}
