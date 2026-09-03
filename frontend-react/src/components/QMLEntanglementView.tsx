import React from 'react'

interface Props {
  data: {
    risk_score: number
    severity: string
    von_neumann_entropy: number
    circuit_angles: number[]
    density_matrix_heatmap: number[][]
    probability_landscape: number[][]
    quantum_attention_map?: Array<{
      pair: string
      strength: number
      normalized: number
    }>
    xray_quantum_focus?: Record<string, {
      angle: number
      focus_score: number
      raw_value: number
    }>
    per_qubit_analysis?: Array<{
      label: string
      pauliz: number
      prob_excited: number
      status: string
    }>
  }
}

const ANGLE_LABELS = [
  'HR', 'SpO2', 'RR', 'Temp', 'WBC', 'ESR', 'CRP', 'Lymph',
  'Hb', 'Alb', 'Plat', 'Gluc', 'X-Opa', 'X-Cav', 'X-Nod', 'X-Ple',
  'ADA', 'Mantx', 'AFB', 'GeneX', 'BMI', 'TrtD'
]

export default function QMLEntanglementView({ data }: Props) {
  const risk = data.risk_score
  const entropy = data.von_neumann_entropy

  const severityColor =
    data.severity === 'CRITICAL' ? '#ef4444' :
    data.severity === 'WARNING' ? '#f59e0b' :
    data.severity === 'INCONCLUSIVE' ? '#eab308' :
    data.severity === 'LOW' ? '#38bdf8' : '#10b981'

  const matrix = data.density_matrix_heatmap || []

  return (
    <div className="rounded-xl p-5 border" style={{ background: '#111827', borderColor: '#8b5cf633', borderWidth: 2 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-base font-bold" style={{ color: '#8b5cf6' }}>
          QML Deep Entanglement View
        </h2>
        <span className="text-xs px-2 py-0.5 rounded" style={{ background: '#8b5cf622', color: '#8b5cf6' }}>
          8-Qubit (256-Dim) Variational Circuit
        </span>
      </div>
      <p className="text-xs mb-4" style={{ color: '#64748b' }}>
        All 22 features <strong>entangled</strong> — cross-correlations reveal hidden patterns
      </p>

      {/* Risk + Entropy Row */}
      <div className="grid grid-cols-2 gap-3 mb-5">
        <div className="rounded-lg p-3 text-center" style={{ background: '#1a1f35', border: `1px solid ${severityColor}44` }}>
          <p className="text-xs mb-1" style={{ color: '#94a3b8' }}>QML Risk Score</p>
          <p className="text-2xl font-bold" style={{ color: severityColor }}>{risk}%</p>
          <p className="text-xs font-semibold mt-1" style={{ color: severityColor }}>{data.severity}</p>
        </div>

        <div className="rounded-lg p-3 text-center" style={{ background: '#1a1f35', border: '1px solid #2d3555' }}>
          <p className="text-xs mb-1" style={{ color: '#94a3b8' }}>Von Neumann Entropy</p>
          <p className="text-2xl font-bold" style={{ color: entropy > 1.5 ? '#f59e0b' : '#10b981' }}>
            {entropy.toFixed(2)}
          </p>
          <p className="text-xs mt-1" style={{ color: '#64748b' }}>bits (System Coherence)</p>
        </div>
      </div>

      {/* X-Ray Quantum Focus */}
      {data.xray_quantum_focus && (
        <div className="mb-5">
          <p className="text-xs font-semibold mb-2" style={{ color: '#ef4444' }}>
            X-Ray Quantum Attention Map
          </p>
          <div className="grid grid-cols-4 gap-2">
            {Object.entries(data.xray_quantum_focus).map(([name, val]) => (
              <div key={name} className="text-center rounded-lg p-2" style={{ background: '#1a1f35', border: '1px solid #2d3555' }}>
                <div style={{ height: '6px', background: '#0f172a', borderRadius: '3px', overflow: 'hidden', marginBottom: '4px' }}>
                  <div style={{
                    width: `${val.focus_score}%`,
                    height: '100%',
                    background: val.focus_score > 60 ? '#ef4444' : val.focus_score > 30 ? '#f59e0b' : '#10b981',
                    transition: 'width 0.5s ease'
                  }} />
                </div>
                <span className="block text-xs font-bold uppercase" style={{ color: '#94a3b8' }}>{name}</span>
                <span className="block text-xs" style={{ color: '#f1f5f9' }}>{val.focus_score.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-Qubit PauliZ Analysis */}
      {data.per_qubit_analysis && (
        <div className="mb-5">
          <p className="text-xs font-semibold mb-2" style={{ color: '#38bdf8' }}>
            Per-Qubit PauliZ Measurement
          </p>
          <div className="grid grid-cols-4 gap-2">
            {data.per_qubit_analysis.map((q, i) => (
              <div key={i} className="rounded p-2 text-center" style={{
                background: '#1a1f35',
                border: `1px solid ${q.status === 'anomaly' ? '#ef4444' : q.status === 'watch' ? '#f59e0b' : '#10b981'}`
              }}>
                <span className="block text-xs font-bold" style={{ color: '#38bdf8' }}>{q.label}</span>
                <span className="block font-mono text-xs mt-1" style={{ color: '#fff' }}>Z={q.pauliz > 0 ? '+' : ''}{q.pauliz.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Circuit Angles — Entanglement Heatmap */}
      <div className="mb-5">
        <p className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>
          Quantum Circuit Angles (22 Features)
        </p>
        <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
          {data.circuit_angles.map((angle, i) => {
            const intensity = Math.min(1, Math.abs(angle) / Math.PI)
            const hue = angle > 0 ? 270 : 160
            return (
              <div
                key={i}
                className="rounded p-1 text-center"
                title={`${ANGLE_LABELS[i]}: ${angle.toFixed(3)} rad`}
                style={{
                  background: `hsla(${hue}, 70%, 50%, ${0.15 + intensity * 0.6})`,
                  border: `1px solid hsla(${hue}, 70%, 50%, ${0.3 + intensity * 0.5})`,
                }}
              >
                <span className="block text-[10px] font-bold" style={{ color: '#f1f5f9' }}>
                  {ANGLE_LABELS[i] || `F${i}`}
                </span>
                <span className="block text-[10px]" style={{ color: '#94a3b8' }}>
                  {angle.toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* ZZ Entanglement Top Interactions */}
      {data.quantum_attention_map && (
        <div className="mb-5">
          <p className="text-xs font-semibold mb-2" style={{ color: '#8b5cf6' }}>
            Top ZZ-Entanglement Interactions
          </p>
          <div className="space-y-2">
            {data.quantum_attention_map.slice(0, 4).map((zz, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="text-[10px] font-mono text-gray-400 w-32 truncate">{zz.pair}</span>
                <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div style={{ width: `${Math.min(zz.normalized * 100, 100)}%`, background: 'linear-gradient(90deg, #38bdf8, #8b5cf6)', height: '100%' }} />
                </div>
                <span className="text-[10px] font-mono text-white w-10 text-right">{zz.strength.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Density Matrix Visualization */}
      <div className="mb-4">
        <p className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>
          Density Matrix (32x32) — off-diagonal = cross-variable entanglement
        </p>
        <div
          className="rounded-lg overflow-hidden"
          style={{ background: '#0a0e1a', border: '1px solid #2d3555', padding: 2 }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${matrix.length || 1}, 1fr)`,
              gap: 0,
              aspectRatio: '1',
            }}
          >
            {matrix.flatMap((row, ri) =>
              row.map((val, ci) => {
                const norm = Math.min(1, val * 5)
                const isOffDiag = ri !== ci
                return (
                  <div
                    key={`${ri}-${ci}`}
                    style={{
                      background: isOffDiag
                        ? `rgba(139, 92, 246, ${norm})`
                        : `rgba(56, 189, 248, ${norm})`,
                      width: '100%',
                      aspectRatio: '1',
                    }}
                  />
                )
              })
            )}
          </div>
        </div>
      </div>

      {/* Key Insight */}
      <div className="rounded-lg p-3 text-xs" style={{ background: '#8b5cf611', color: '#c4b5fd', border: '1px solid #8b5cf633' }}>
        <strong>Quantum Advantage:</strong> Unlike Classical ML, QML encodes ALL 22 features into entangled qubits.
        The ZZ-Feature Map creates hidden correlations that are mathematically invisible to independent-variable analysis.
      </div>
    </div>
  )
}
