import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { SessionStep } from '../../types/demo'

interface BeliefChartProps {
  step: SessionStep | null
}

export function BeliefChart({ step }: BeliefChartProps) {
  const data = step ? Object.entries(step.belief).map(([genre, score]) => ({ genre, score })) : []

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="genre" stroke="#94a3b8" />
          <YAxis domain={[0, 1]} stroke="#94a3b8" />
          <Tooltip contentStyle={{ backgroundColor: '#020617', border: '1px solid #334155', borderRadius: 12 }} />
          <Bar dataKey="score" fill="#22d3ee" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
