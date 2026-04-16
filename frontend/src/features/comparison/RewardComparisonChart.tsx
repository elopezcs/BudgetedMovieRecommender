import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ComparisonMetric } from '../../types/results'
import { titleCase } from '../../utils/format'

interface RewardComparisonChartProps {
  metrics: ComparisonMetric[]
}

export function RewardComparisonChart({ metrics }: RewardComparisonChartProps) {
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <BarChart data={metrics}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="algorithm" tickFormatter={titleCase} stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip
            contentStyle={{ backgroundColor: '#020617', border: '1px solid #334155', borderRadius: 12 }}
            labelFormatter={(value) => titleCase(String(value))}
          />
          <Bar dataKey="average_cumulative_reward" fill="#34d399" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
