import { Card } from './Card'

interface KpiCardProps {
  label: string
  value: string
  subtext?: string
}

export function KpiCard({ label, value, subtext }: KpiCardProps) {
  return (
    <Card className="min-h-32">
      <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-3 text-3xl font-semibold text-white">{value}</p>
      {subtext ? <p className="mt-2 text-sm text-slate-400">{subtext}</p> : null}
    </Card>
  )
}
