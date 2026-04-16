interface PageTitleProps {
  eyebrow: string
  title: string
  description: string
}

export function PageTitle({ eyebrow, title, description }: PageTitleProps) {
  return (
    <header className="mb-8">
      <p className="text-xs uppercase tracking-[0.25em] text-emerald-300">{eyebrow}</p>
      <h2 className="mt-3 text-4xl font-semibold text-white">{title}</h2>
      <p className="mt-3 max-w-3xl text-base text-slate-300">{description}</p>
    </header>
  )
}
