import { useState } from 'react'

const PRESET_TIER1 = {
  heart_rate: 78.5, spo2: 97.3, resp_rate: 16.2, temperature: 36.85,
  wbc_count: 7.2, esr: 11.5, crp: 2.5, lymphocyte_pct: 29.8,
  hemoglobin: 13.7, albumin: 3.9,
  xray_opacity: 0.08, xray_cavity: 0.05, xray_nodule: 0.07, xray_pleural: 0.06,
  ada_level: 16.5, mantoux_mm: 5.8,
}

const FEATURE_GROUPS = [
  {
    label: 'Vital Signs',
    icon: '💓',
    features: [
      { key: 'heart_rate', label: 'Heart Rate', unit: 'bpm', range: '60-100' },
      { key: 'spo2', label: 'SpO2', unit: '%', range: '95-100' },
      { key: 'resp_rate', label: 'Respiratory Rate', unit: 'br/min', range: '12-20' },
      { key: 'temperature', label: 'Temperature', unit: '°C', range: '36.1-37.5' },
    ],
  },
  {
    label: 'Blood Work',
    icon: '🩸',
    features: [
      { key: 'wbc_count', label: 'WBC Count', unit: 'x10³/µL', range: '4-11' },
      { key: 'esr', label: 'ESR', unit: 'mm/hr', range: '0-20' },
      { key: 'crp', label: 'CRP', unit: 'mg/dL', range: '0-10' },
      { key: 'lymphocyte_pct', label: 'Lymphocyte %', unit: '%', range: '20-40' },
      { key: 'hemoglobin', label: 'Hemoglobin', unit: 'g/dL', range: '12-17' },
      { key: 'albumin', label: 'Albumin', unit: 'g/dL', range: '3.5-5.0' },
    ],
  },
  {
    label: 'X-Ray Markers',
    icon: '🫁',
    features: [
      { key: 'xray_opacity', label: 'Opacity Score', unit: '', range: '0.0-1.0' },
      { key: 'xray_cavity', label: 'Cavity Prob.', unit: '', range: '0.0-1.0' },
      { key: 'xray_nodule', label: 'Nodule Density', unit: '', range: '0.0-1.0' },
      { key: 'xray_pleural', label: 'Pleural Thickening', unit: '', range: '0.0-1.0' },
    ],
  },
  {
    label: 'TB-Specific Tests',
    icon: '🧬',
    features: [
      { key: 'ada_level', label: 'ADA Level', unit: 'U/L', range: '0-40' },
      { key: 'mantoux_mm', label: 'Mantoux Test', unit: 'mm', range: '0-30' },
    ],
  },
]

interface Props {
  onAnalyze: (vitals: Record<string, number>) => void
  loading: boolean
}

export default function PatientInput({ onAnalyze, loading }: Props) {
  const [vitals, setVitals] = useState<Record<string, number>>(PRESET_TIER1)

  const handleChange = (key: string, value: string) => {
    setVitals((prev) => ({ ...prev, [key]: parseFloat(value) || 0 }))
  }

  const loadPreset = () => setVitals({ ...PRESET_TIER1 })

  return (
    <div className="rounded-xl p-5 border" style={{ background: '#111827', borderColor: '#2d3555' }}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-bold" style={{ color: '#f1f5f9' }}>
          Patient Vitals (16 Features)
        </h2>
        <button
          onClick={loadPreset}
          className="px-3 py-1 rounded text-xs font-semibold transition-all hover:opacity-80"
          style={{ background: '#1e293b', color: '#f59e0b', border: '1px solid #f59e0b' }}
        >
          Load Tier-1 (Hidden TB)
        </button>
      </div>

      {FEATURE_GROUPS.map((group) => (
        <div key={group.label} className="mb-4">
          <p className="text-xs font-semibold mb-2 flex items-center gap-1.5" style={{ color: '#94a3b8' }}>
            <span>{group.icon}</span> {group.label}
          </p>
          <div className="grid grid-cols-2 gap-2">
            {group.features.map((f) => (
              <div
                key={f.key}
                className="flex items-center gap-2 rounded-lg px-3 py-2"
                style={{ background: '#1a1f35', border: '1px solid #2d3555' }}
              >
                <div className="flex-1 min-w-0">
                  <label className="block text-xs truncate" style={{ color: '#94a3b8' }}>{f.label}</label>
                  <input
                    type="number"
                    step="any"
                    value={vitals[f.key] ?? ''}
                    onChange={(e) => handleChange(f.key, e.target.value)}
                    className="w-full bg-transparent text-sm font-semibold outline-none mt-0.5"
                    style={{ color: '#f1f5f9' }}
                  />
                </div>
                <span className="text-xs whitespace-nowrap" style={{ color: '#64748b' }}>
                  {f.unit} <br />
                  <span style={{ color: '#10b981', fontSize: '0.65rem' }}>{f.range}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}

      <button
        onClick={() => onAnalyze(vitals)}
        disabled={loading}
        className="w-full py-3 rounded-lg font-bold text-sm transition-all"
        style={{
          background: loading ? '#334155' : 'linear-gradient(135deg, #8b5cf6, #6366f1)',
          color: '#fff',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'SIMULATING QUANTUM STATE...' : 'RUN ANALYSIS (ML + QML)'}
      </button>
    </div>
  )
}
