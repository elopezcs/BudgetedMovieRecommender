import type { PropsWithChildren } from 'react'

interface CardProps extends PropsWithChildren {
  className?: string
}

export function Card({ children, className = '' }: CardProps) {
  return <section className={`rounded-2xl border border-slate-800 bg-slate-900/70 p-5 ${className}`}>{children}</section>
}
