interface BudgetBarProps {
  used: number
  total: number
}

export function BudgetBar({ used, total }: BudgetBarProps) {
  const ratio = total === 0 ? 0 : Math.min(100, Math.round((used / total) * 100))
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-300">
        <span>Question Budget</span>
        <span>
          {used} / {total}
        </span>
      </div>
      <div className="h-3 rounded-full bg-slate-800">
        <div className="h-full rounded-full bg-emerald-400 transition-all" style={{ width: `${ratio}%` }} />
      </div>
    </div>
  )
}
