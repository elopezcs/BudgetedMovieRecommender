interface FeedbackStateProps {
  title: string
  message: string
}

export function FeedbackState({ title, message }: FeedbackStateProps) {
  return (
    <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/60 px-6 py-10 text-center">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <p className="mx-auto mt-2 max-w-2xl text-sm text-slate-400">{message}</p>
    </div>
  )
}
