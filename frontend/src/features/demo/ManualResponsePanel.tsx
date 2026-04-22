import { useState } from 'react'
import type {
  ContinuationChoice,
  ManualResponseRequest,
  PendingAction,
  QuestionFeedback,
  RecommendationFeedback,
} from '../../types/demo'
import { titleCase } from '../../utils/format'

const genreOptions = ['action', 'comedy', 'drama', 'scifi', 'documentary'] as const

interface ManualResponsePanelProps {
  pendingAction: PendingAction
  loading: boolean
  onSubmit: (payload: Omit<ManualResponseRequest, 'session_id'>) => void
}

export function ManualResponsePanel({ pendingAction, loading, onSubmit }: ManualResponsePanelProps) {
  const [continuation, setContinuation] = useState<ContinuationChoice>('continue')
  const [questionFeedback, setQuestionFeedback] = useState<QuestionFeedback>('helpful')
  const [hintedGenre, setHintedGenre] = useState<string>(genreOptions[0])
  const [recommendationFeedback, setRecommendationFeedback] = useState<RecommendationFeedback>('accepted_strong')

  function handleSubmit() {
    if (pendingAction.action_type === 'ask') {
      onSubmit({
        continuation,
        question_feedback: questionFeedback,
        hinted_genre: hintedGenre,
      })
      return
    }

    onSubmit({
      continuation,
      recommendation_feedback: recommendationFeedback,
    })
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-white">Interactive Response</h3>
        <p className="mt-2 text-sm text-slate-400">
          The policy has selected its next action. Choose how the human audience wants the user to respond before the
          step is committed to the session.
        </p>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-sm text-slate-300">
        <p>
          Proposed Action: <span className="text-white">{titleCase(pendingAction.action_name)}</span>
        </p>
        {pendingAction.fallback_applied ? (
          <p className="mt-2 text-amber-300">
            {pendingAction.fallback_reason}
            {/* pendingAction.original_attempted_action_name ? (
              <span className="text-slate-300"> Original policy choice: {titleCase(pendingAction.original_attempted_action_name)}.</span>
            ) : null */}
          </p>
        ) : null}
        {pendingAction.question_text ? (
          <p className="mt-2">
            Question Prompt: <span className="text-white">{pendingAction.question_text}</span>
          </p>
        ) : null}
        {pendingAction.recommendation_genre ? (
          <p className="mt-2">
            Recommendation Genre: <span className="text-white">{titleCase(pendingAction.recommendation_genre)}</span>
          </p>
        ) : null}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {pendingAction.action_type === 'ask' ? (
          <>
          <label className="flex flex-col gap-2 text-sm text-slate-300">
              Hinted Genre
              <select
                className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
                value={hintedGenre}
                onChange={(event) => setHintedGenre(event.target.value)}
              >
                {genreOptions.map((genre) => (
                  <option key={genre} value={genre}>
                    {titleCase(genre)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-2 text-sm text-slate-300">
              Question Quality
              <select
                className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
                value={questionFeedback}
                onChange={(event) => setQuestionFeedback(event.target.value as QuestionFeedback)}
              >
                <option value="helpful">Helpful</option>
                <option value="neutral">Neutral</option>
                <option value="annoying">Annoying</option>
              </select>
            </label>
            
          </>
        ) : (
          <label className="flex flex-col gap-2 text-sm text-slate-300 lg:col-span-2">
            Recommendation Outcome
            <select
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
              value={recommendationFeedback}
              onChange={(event) => setRecommendationFeedback(event.target.value as RecommendationFeedback)}
            >
              <option value="accepted_strong">Accepted Strongly</option>
              <option value="accepted_weak">Accepted Weakly</option>
              <option value="skipped">Skipped</option>
              <option value="rejected_annoyed">Rejected and Annoyed</option>
            </select>
          </label>
        )}

        <label className="flex flex-col gap-2 text-sm text-slate-300">
          Session Continuation
          <select
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
            value={continuation}
            onChange={(event) => setContinuation(event.target.value as ContinuationChoice)}
          >
            <option value="continue">Continue</option>
            <option value="abandon">Abandon</option>
          </select>
        </label>
      </div>

      <button
        type="button"
        className="rounded-lg bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-200 disabled:opacity-50"
        disabled={loading}
        onClick={handleSubmit}
      >
        Apply Response
      </button>
    </div>
  )
}
