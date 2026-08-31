import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface Props {
  data: {
    risk_score: number
    diagnosis: string
    features: Array<{
      name: string
      value: number
      importance: number
      risk_contribution: number
      status: string
    }>
  }
}

export default function ClassicalMLView({ data }: Props) {
  const riskPct = (data.risk_score * 100).toFixed(1)
  const isHealthy = data.risk_score < 0.5

  // Sort features by importance for the bar chart
  const chartData = [...data.features]
    .sort((a, b) => b.importance - a.importance)
    .map((f) => ({
      name: f.name.replace('_', ' ').replace('xray ', 'x-').substring(0, 12),
      fullName: f.name,
      importance: +(f.importance * 100).toFixed(1),
      value: f.value,
    }))

  return (
    <div className="rounded-xl p-5 border" style={{ background: '#111827', borderColor: '#2d3555' }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-base font-bold" style={{ color: '#94a3b8' }}>
          Classical ML View
        </h2>
        <span className="text-xs px-2 py-0.5 rounded" style={{ background: '#1e293b', color: '#94a3b8' }}>
          Random Forest
        </span>
      </div>
      <p className="text-xs mb-4" style={{ color: '#64748b' }}>
        Each variable analyzed <strong>independently</strong> — no cross-correlations
      </p>

      {/* Risk Score */}
      <div className="flex items-center gap-4 mb-5 p-4 rounded-lg" style={{ background: '#1a1f35', border: '1px solid #2d3555' }}>
        <div
          className="w-16 h-16 rounded-full flex items-center justify-center text-lg font-bold"
          style={{
            background: isHealthy
              ? 'radial-gradient(circle, rgba(16,185,129,0.2), transparent)'
              : 'radial-gradient(circle, rgba(239,68,68,0.2), transparent)',
            border: `2px solid ${isHealthy ? '#10b981' : '#ef4444'}`,
            color: isHealthy ? '#10b981' : '#ef4444',
          }}
        >
          {riskPct}%
        </div>
        <div>
          <p className="font-bold text-sm" style={{ color: isHealthy ? '#10b981' : '#ef4444' }}>
            {data.diagnosis}
          </p>
          <p className="text-xs" style={{ color: '#94a3b8' }}>
            {isHealthy
              ? 'All individual vitals within safe thresholds. No alert triggered.'
              : 'Threshold exceeded on one or more vitals.'}
          </p>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="mb-3">
        <p className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>
          Feature Importance (%) — how much each variable contributed individually
        </p>
        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3555" />
              <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis
                dataKey="name"
                type="category"
                stroke="#64748b"
                tick={{ fontSize: 9, fill: '#94a3b8' }}
                width={80}
              />
              <Tooltip
                contentStyle={{ background: '#1a1f35', border: '1px solid #2d3555', borderRadius: 8, color: '#f1f5f9' }}
                formatter={(value: number, _: string, entry: any) =>
                  [`${value}%`, `Importance (${entry.payload.fullName})`]
                }
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={i < 3 ? '#94a3b8' : '#475569'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-lg p-3 text-xs" style={{ background: '#1a1f35', color: '#94a3b8', border: '1px solid #f59e0b22' }}>
        <strong style={{ color: '#f59e0b' }}>Limitation:</strong> Classical ML treats each feature as an independent variable.
        It cannot detect hidden correlations between HR-Temp synchronization, SpO2-WBC coupling, or multi-system entanglement patterns.
      </div>
    </div>
  )
}
