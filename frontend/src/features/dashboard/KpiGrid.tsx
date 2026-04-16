import { KpiCard } from '../../components/KpiCard'

interface KpiGridProps {
  bestReward: string
  topPolicy: string
  bestAcceptanceRate: string
  avgQuestionsAsked: string
  bestAlgorithm: string
}

export function KpiGrid({
  bestReward,
  topPolicy,
  bestAcceptanceRate,
  avgQuestionsAsked,
  bestAlgorithm,
}: KpiGridProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <KpiCard label="Best Reward" value={bestReward} subtext={`Top policy: ${topPolicy}`} />
      <KpiCard label="Best Acceptance Rate" value={bestAcceptanceRate} />
      <KpiCard label="Average Questions Asked" value={avgQuestionsAsked} />
      <KpiCard label="Best Performing Algorithm" value={bestAlgorithm} />
    </div>
  )
}
