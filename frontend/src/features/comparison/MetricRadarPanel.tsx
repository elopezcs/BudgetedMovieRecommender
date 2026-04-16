import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, Tooltip } from 'recharts'
import type { ComparisonMetric } from '../../types/results'
import { titleCase } from '../../utils/format'

interface MetricRadarPanelProps {
  metrics: ComparisonMetric[]
}

export function MetricRadarPanel({ metrics }: MetricRadarPanelProps) {
  const top = metrics.slice(0, 5)

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <RadarChart data={top}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="algorithm" tickFormatter={titleCase} stroke="#cbd5e1" />
          <Tooltip
            contentStyle={{ backgroundColor: '#020617', border: '1px solid #334155', borderRadius: 12 }}
            labelFormatter={(value) => titleCase(String(value))}
          />
          <Radar name="Acceptance" dataKey="acceptance_rate" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.3} />
          <Radar name="Abandonment" dataKey="abandonment_rate" stroke="#f97316" fill="#f97316" fillOpacity={0.2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
