import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './AppShell'
import { ComparisonPage } from '../pages/ComparisonPage'
import { LiveDemoPage } from '../pages/LiveDemoPage'
import { OverviewPage } from '../pages/OverviewPage'
import { TechnicalPage } from '../pages/TechnicalPage'

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/demo" element={<LiveDemoPage />} />
        <Route path="/comparison" element={<ComparisonPage />} />
        <Route path="/technical" element={<TechnicalPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  )
}
