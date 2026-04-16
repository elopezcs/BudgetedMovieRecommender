import type { PropsWithChildren } from 'react'
import { NavLink } from 'react-router-dom'
import { BrainCircuit } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/demo', label: 'Live Demo' },
  { to: '/comparison', label: 'Algorithm Comparison' },
  { to: '/technical', label: 'Technical Credibility' },
]

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-emerald-500/20 p-2 text-emerald-300">
              <BrainCircuit className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Budgeted RL Recommender</p>
              <h1 className="text-sm font-semibold text-slate-100">Interactive Movie Intelligence</h1>
            </div>
          </div>
          <nav className="flex gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded-md px-3 py-2 text-sm transition ${
                    isActive ? 'bg-emerald-500 text-slate-950' : 'text-slate-300 hover:text-white'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl px-6 py-8">{children}</main>
    </div>
  )
}
