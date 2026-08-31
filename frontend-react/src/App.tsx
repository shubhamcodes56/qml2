import { useState } from 'react'
import PatientInput from './components/PatientInput'
import ClassicalMLView from './components/ClassicalMLView'
import QMLEntanglementView from './components/QMLEntanglementView'
import AdvisoryPanel from './components/AdvisoryPanel'
import XrayPanel from './components/XrayPanel'
import TimelineView from './components/TimelineView'

const API = 'http://localhost:8000'
type TabId = 'timeline' | 'analysis'

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('timeline')
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [xrayResult, setXrayResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  // Keep existing static functions for the "Static Analysis" tab if needed
  const handleAnalyze = async (vitals: Record<string, number>) => {
    setLoading(true)
    try {
      setXrayResult(null)
      const res = await fetch(`${API}/analyze-patient`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vitals }),
      })
      setAnalysisResult(await res.json())
    } catch (err) { console.error(err) } finally { setLoading(false) }
  }

  const handleStaticXrayUpload = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    try {
      setAnalysisResult(null)
      const res = await fetch(`${API}/upload-xray`, { method: 'POST', body: formData })
      setXrayResult(await res.json())
    } catch (err) { console.error(err) }
  }

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>
      {/* Clinical Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-white text-xl" style={{ background: 'var(--accent-blue)' }}>
            +
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              Clinical Decision Support <span style={{ color: 'var(--accent-blue)', fontWeight: 400 }}>| QML Edition</span>
            </h1>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Department of Pulmonology & Critical Care</p>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <nav className="flex px-6 border-b" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        {[
          { id: 'timeline' as TabId, label: 'Live ICU Monitoring' },
          { id: 'analysis' as TabId, label: 'Static Patient Analysis' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className="px-5 py-4 text-sm font-semibold transition-all border-b-2"
            style={{
              color: activeTab === tab.id ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottomColor: activeTab === tab.id ? 'var(--accent-blue)' : 'transparent',
            }}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === 'timeline' && (
        <div className="p-6">
          <TimelineView apiBase={API} />
        </div>
      )}

      {activeTab === 'analysis' && (
        <div className="p-6">
           <div className="grid gap-6 mb-6" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <PatientInput onAnalyze={handleAnalyze} loading={loading} />
            <XrayPanel onUpload={handleStaticXrayUpload} result={xrayResult} />
          </div>
          {/* Results */}
          {(analysisResult || xrayResult) && (
            <>
              <div className="grid gap-6 mb-6" style={{ gridTemplateColumns: analysisResult?.classical_ml ? '1fr 1fr' : '1fr' }}>
                {analysisResult?.classical_ml && (
                  <ClassicalMLView data={analysisResult.classical_ml} />
                )}
                
                {(xrayResult?.qml_with_xray || analysisResult?.quantum_ml) && (
                  <QMLEntanglementView data={xrayResult?.qml_with_xray || analysisResult?.quantum_ml} />
                )}
              </div>
              <AdvisoryPanel
                alerts={analysisResult?.advisory || []}
                classicalRisk={analysisResult?.classical_ml?.risk_score || 0}
                qmlRisk={xrayResult?.qml_with_xray?.risk_score || analysisResult?.quantum_ml?.risk_score || 0}
                xrayFindings={xrayResult?.advisory || []}
              />
            </>
          )}
        </div>
      )}
    </div>
  )
}
