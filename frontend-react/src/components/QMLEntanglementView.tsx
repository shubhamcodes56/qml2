import {
  ResponsiveContainer,
  Treemap,
  Tooltip,
} from 'recharts'

interface Props {
  data: {
    risk_score: number
    severity: string
    von_neumann_entropy: number
    mutual_information: {
      entropy_vitals: number
      entropy_blood_xray: number
      mutual_information: number
    }
    circuit_angles: number[]
    density_matrix_heatmap: number[][]
    probability_landscape: number[][]
  }
}

const ANGLE_LABELS = [
  'HR', 'SpO2', 'RR', 'Temp', 'WBC', 'ESR', 'CRP', 'Lymph',
  'Hb', 'Alb', 'X-Opa', 'X-Cav', 'X-Nod', 'X-Ple', 'ADA', 'Mantx'
]

export default function QMLEntanglementView({ data }: Props) {
  const risk = data.risk_score
  const entropy = data.von_neumann_entropy
  const mi = data.mutual_information

  const severityColor =
    data.severity === 'CRITICAL' ? '#ef4444' :
    data.severity === 'WARNING' ? '#f59e0b' :
    data.severity === 'LOW' ? '#38bdf8' : '#10b981'

  // Build entanglement data for treemap visualization
  const entanglementData = data.circuit_angles.map((angle, i) => ({
    name: ANGLE_LABELS[i] || `F${i}`,
    size: Math.abs(angle) * 100,
    value: angle,
  }))

  // Group by subsystem for treemap
  const treemapData = [
    {
      name: 'Vitals Subsystem (Q0-Q3)',
      children: entanglementData.slice(0, 8).map(d => ({ ...d, size: Math.max(d.size, 5) })),
    },
    {
      name: 'Blood/X-ray Subsystem (Q4-Q7)',
      children: entanglementData.slice(8, 16).map(d => ({ ...d, size: Math.max(d.size, 5) })),
    },
  ]

  // Density Matrix as a grid of colored cells
  const matrix = data.density_matrix_heatmap || []

  return (
    <div className="rounded-xl p-5 border" style={{ background: '#111827', borderColor: '#8b5cf633', borderWidth: 2 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-base font-bold" style={{ color: '#8b5cf6' }}>
          QML Entanglement View
        </h2>
        <span className="text-xs px-2 py-0.5 rounded" style={{ background: '#8b5cf622', color: '#8b5cf6' }}>
          8-Qubit Variational Circuit
        </span>
      </div>
      <p className="text-xs mb-4" style={{ color: '#64748b' }}>
        All 16 features <strong>entangled</strong> — cross-correlations reveal hidden patterns
      </p>

      {/* Risk + Entropy Row */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        {/* QML Risk */}
        <div className="rounded-lg p-3 text-center" style={{ background: '#1a1f35', border: `1px solid ${severityColor}44` }}>
          <p className="text-xs mb-1" style={{ color: '#94a3b8' }}>QML Risk Score</p>
          <p className="text-2xl font-bold" style={{ color: severityColor }}>{risk}%</p>
          <p className="text-xs font-semibold mt-1" style={{ color: severityColor }}>{data.severity}</p>
        </div>

        {/* Von Neumann Entropy */}
        <div className="rounded-lg p-3 text-center" style={{ background: '#1a1f35', border: '1px solid #2d3555' }}>
          <p className="text-xs mb-1" style={{ color: '#94a3b8' }}>Entanglement Entropy</p>
          <p className="text-2xl font-bold" style={{ color: entropy > 1.5 ? '#f59e0b' : '#10b981' }}>
            {entropy.toFixed(2)}
          </p>
          <p className="text-xs mt-1" style={{ color: '#64748b' }}>bits (S von Neumann)</p>
        </div>

        {/* Mutual Information */}
        <div className="rounded-lg p-3 text-center" style={{ background: '#1a1f35', border: '1px solid #2d3555' }}>
          <p className="text-xs mb-1" style={{ color: '#94a3b8' }}>Mutual Information</p>
          <p className="text-lg font-bold" style={{ color: '#38bdf8' }}>{mi.mutual_information.toFixed(2)}</p>
          <p className="text-xs mt-1" style={{ color: '#64748b' }}>
            Vitals: {mi.entropy_vitals.toFixed(2)} | Blood: {mi.entropy_blood_xray.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Circuit Angles — Entanglement Heatmap */}
      <div className="mb-4">
        <p className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>
          Quantum Circuit Angles (RY encoding) — color intensity = correlation strength
        </p>
        <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(8, 1fr)' }}>
          {data.circuit_angles.map((angle, i) => {
            const intensity = Math.min(1, Math.abs(angle) / Math.PI)
            const hue = angle > 0 ? 270 : 160 // purple for positive, green for negative
            return (
              <div
                key={i}
                className="rounded p-2 text-center"
                title={`${ANGLE_LABELS[i]}: ${angle.toFixed(3)} rad`}
                style={{
                  background: `hsla(${hue}, 70%, 50%, ${0.15 + intensity * 0.6})`,
                  border: `1px solid hsla(${hue}, 70%, 50%, ${0.3 + intensity * 0.5})`,
                }}
              >
                <span className="block text-xs font-bold" style={{ color: '#f1f5f9' }}>
                  {ANGLE_LABELS[i]}
                </span>
                <span className="block text-xs" style={{ color: '#94a3b8' }}>
                  {angle.toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Density Matrix Visualization */}
      <div className="mb-4">
        <p className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>
          Density Matrix (32x32 downsampled) — off-diagonal = cross-variable entanglement
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
        <p className="text-xs mt-1" style={{ color: '#64748b' }}>
          <span style={{ color: '#38bdf8' }}>■</span> Diagonal (self) &nbsp;
          <span style={{ color: '#8b5cf6' }}>■</span> Off-diagonal (entanglement)
        </p>
      </div>

      {/* Key Insight */}
      <div className="rounded-lg p-3 text-xs" style={{ background: '#8b5cf611', color: '#c4b5fd', border: '1px solid #8b5cf633' }}>
        <strong>Quantum Advantage:</strong> Unlike Classical ML, QML encodes ALL 16 features into entangled qubits.
        The off-diagonal density matrix elements reveal hidden correlations (HR↔Temp sync, SpO2↔WBC coupling)
        that are mathematically invisible to independent-variable analysis.
      </div>
    </div>
  )
}
