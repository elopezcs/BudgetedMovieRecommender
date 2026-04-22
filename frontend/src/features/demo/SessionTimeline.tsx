import type { SessionStep } from '../../types/demo'
import { toFixed, titleCase } from '../../utils/format'

interface SessionTimelineProps {
  timeline: SessionStep[]
}

export function SessionTimeline({ timeline }: SessionTimelineProps) {
  return (
    <div className="space-y-3">
      {timeline.map((step) => (
        <article key={step.step_index} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-white">Step {step.step_index + 1}</h4>
            <span
              className={`rounded-full px-2 py-1 text-xs ${
                step.action_type === 'ask' ? 'bg-purple-400/20 text-purple-300' : 'bg-cyan-400/20 text-cyan-300'
              }`}
            >
              {step.action_type === 'ask' ? 'Ask' : 'Recommend'}
            </span>
          </div>
          <div className="mt-3 grid gap-2 text-sm text-slate-300 md:grid-cols-2">
            <p>
              Action: <span className="text-white">{titleCase(step.action_name)}</span>
            </p>
            <p>
              Reward: <span className="text-white">{toFixed(step.reward, 2)}</span>
            </p>
            <p>
              User Response:{' '}
              <span className="text-white">{step.user_response_label}</span>
            </p>
            <p>
              Engagement: <span className="text-white">{toFixed(step.engagement, 2)}</span>
            </p>
            <p>
              Question Budget: <span className="text-white">{step.questions_used} / {step.question_budget}</span>
            </p>
            <p>
              Budget Remaining: <span className="text-white">{step.question_budget_remaining}</span>
            </p>
            {step.fallback_applied ? (
              <p className="md:col-span-2 text-amber-300">
                {step.fallback_reason}
                {/* step.original_attempted_action_name ? (
                  <span className="text-slate-300"> Original policy choice: {titleCase(step.original_attempted_action_name)}.</span>
                ) : null */}
              </p>
            ) : null}
            {step.manual_response ? (
              <p className="md:col-span-2">
                Manual Response: <span className="text-white">{step.manual_response}</span>
              </p>
            ) : null}
            {step.question_text ? (
              <p className="md:col-span-2">
                Question Prompt: <span className="text-white">{step.question_text}</span>
              </p>
            ) : null}
            {step.recommendation_genre ? (
              <p className="md:col-span-2">
                Recommendation Genre: <span className="text-white">{titleCase(step.recommendation_genre)}</span>
              </p>
            ) : null}
          </div>
        </article>
      ))}
    </div>
  )
}
