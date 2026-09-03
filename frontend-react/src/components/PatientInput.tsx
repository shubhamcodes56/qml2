import { useState } from 'react'

const PRESET_TIER1 = {
  heart_rate: 78.5, spo2: 97.3, resp_rate: 16.2, temperature: 36.85,
  wbc_count: 7.2, esr: 11.5, crp: 2.5, lymphocyte_pct: 29.8,
  hemoglobin: 13.7, albumin: 3.9, platelet_count: 255.0, blood_sugar: 102.0,
  xray_opacity: 0.08, xray_cavity: 0.05, xray_nodule: 0.07, xray_pleural: 0.06,
  ada_level: 16.5, mantoux_mm: 5.8, sputum_afb: 0.02, genexpert_ct: 34.5,
  bmi: 21.5, treatment_days: 0,
}

const PRESET_SEVERE_TB = {
  heart_rate: 105.0, spo2: 88.0, resp_rate: 24.0, temperature: 39.5,
  wbc_count: 18.0, esr: 85.0, crp: 45.0, lymphocyte_pct: 10.0,
  hemoglobin: 9.5, albumin: 2.5, platelet_count: 450.0, blood_sugar: 110.0,
  xray_opacity: 0.85, xray_cavity: 0.60, xray_nodule: 0.75, xray_pleural: 0.40,
  ada_level: 85.0, mantoux_mm: 22.0, sputum_afb: 0.90, genexpert_ct: 15.0,
  bmi: 16.0, treatment_days: 0,
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
      { key: 'platelet_count', label: 'Platelet Count', unit: 'x10³/µL', range: '150-400' },
      { key: 'blood_sugar', label: 'Blood Sugar', unit: 'mg/dL', range: '70-140' },
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
      { key: 'sputum_afb', label: 'Sputum AFB', unit: '', range: '0.0-1.0' },
      { key: 'genexpert_ct', label: 'GeneXpert Ct', unit: '', range: '10-45' },
    ],
  },
  {
    label: 'Patient Profile',
    icon: '📋',
    features: [
      { key: 'bmi', label: 'BMI', unit: 'kg/m²', range: '18-25' },
      { key: 'treatment_days', label: 'Treatment Days', unit: 'days', range: '0-365' },
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

  return (
    <div className="rounded-xl p-5 border" style={{ background: '#111827', borderColor: '#2d3555' }}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-bold" style={{ color: '#f1f5f9' }}>
          Patient Vitals (22 Features)
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setVitals({ ...PRESET_TIER1 })}
            className="px-3 py-1 rounded text-xs font-semibold transition-all hover:opacity-80"
            style={{ background: '#1e293b', color: '#f59e0b', border: '1px solid #f59e0b' }}
          >
            Tier-1 (Hidden TB)
          </button>
          <button
            onClick={() => setVitals({ ...PRESET_SEVERE_TB })}
            className="px-3 py-1 rounded text-xs font-semibold transition-all hover:opacity-80"
            style={{ background: '#1e293b', color: '#ef4444', border: '1px solid #ef4444' }}
          >
            Severe TB
          </button>
        </div>
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
        {loading ? 'COMPUTING 256-DIM QUANTUM STATE...' : 'RUN ANALYSIS (ML + QML 8-Qubit)'}
      </button>
    </div>
  )
}
