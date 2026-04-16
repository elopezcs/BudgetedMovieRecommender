import type { DemoOptionsResponse, DemoStartRequest } from '../../types/demo'
import { titleCase } from '../../utils/format'

interface ScenarioControlsProps {
  options: DemoOptionsResponse | null
  value: DemoStartRequest
  loading: boolean
  onChange: (next: DemoStartRequest) => void
  onStart: () => void
  onNext: () => void
  onReset: () => void
  canStep: boolean
  hasSession: boolean
}

export function ScenarioControls({
  options,
  value,
  loading,
  onChange,
  onStart,
  onNext,
  onReset,
  canStep,
  hasSession,
}: ScenarioControlsProps) {
  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <label className="flex flex-col gap-2 text-sm text-slate-300">
        User Profile
        <select
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
          value={value.user_profile}
          onChange={(event) => onChange({ ...value, user_profile: event.target.value })}
        >
          {(options?.user_profiles ?? []).map((profile) => (
            <option key={profile} value={profile}>
              {titleCase(profile)}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-2 text-sm text-slate-300">
        Policy
        <select
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
          value={value.policy}
          onChange={(event) => onChange({ ...value, policy: event.target.value as DemoStartRequest['policy'] })}
        >
          {(options?.policies ?? []).map((policy) => (
            <option key={policy} value={policy}>
              {titleCase(policy)}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-2 text-sm text-slate-300">
        Question Budget
        <input
          type="number"
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
          value={value.question_budget}
          min={options?.budget_range.min ?? 1}
          max={options?.budget_range.max ?? 10}
          onChange={(event) => onChange({ ...value, question_budget: Number(event.target.value) })}
        />
      </label>
      <label className="flex flex-col gap-2 text-sm text-slate-300">
        Seed (Optional)
        <input
          type="number"
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
          value={value.seed ?? ''}
          placeholder="Default"
          onChange={(event) =>
            onChange({
              ...value,
              seed: event.target.value === '' ? null : Number(event.target.value),
            })
          }
        />
      </label>
      <div className="flex items-end gap-2">
        <button
          type="button"
          className="rounded-lg bg-emerald-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:opacity-50"
          disabled={loading}
          onClick={onStart}
        >
          Start
        </button>
        <button
          type="button"
          className="rounded-lg bg-blue-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-blue-300 disabled:opacity-50"
          disabled={!hasSession || !canStep || loading}
          onClick={onNext}
        >
          Next Step
        </button>
        <button
          type="button"
          className="rounded-lg border border-slate-600 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-slate-500 disabled:opacity-50"
          disabled={!hasSession || loading}
          onClick={onReset}
        >
          Reset
        </button>
      </div>
    </div>
  )
}
