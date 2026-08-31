import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Area, AreaChart,
} from 'recharts'

interface Props {
  apiBase: string
}

export default function TimelineView({ apiBase }: Props) {
  const [history, setHistory] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [driftTriggered, setDriftTriggered] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const fetchLiveHistory = async () => {
      try {
        const res = await fetch(`${apiBase}/live-history`)
        const data = await res.json()
        if (data.history) {
          setHistory(data.history)
          setLoading(false)
        }
      } catch (err) {
        console.error("Live fetch error:", err)
      }
    }

    // Poll every 1 second
    fetchLiveHistory()
    const interval = setInterval(fetchLiveHistory, 1000)
    return () => clearInterval(interval)
  }, [apiBase])

  const triggerDrift = async () => {
    try {
      await fetch(`${apiBase}/trigger-drift`, { method: 'POST' })
      setDriftTriggered(true)
    } catch (e) { console.error(e) }
  }

  const resetSimulation = async () => {
    try {
      await fetch(`${apiBase}/reset`, { method: 'POST' })
      setDriftTriggered(false)
      setHistory([])
    } catch (e) { console.error(e) }
  }

  const handleXrayUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    try {
      await fetch(`${apiBase}/upload-xray-live`, {
        method: 'POST',
        body: formData,
      })
    } catch (err) {
      console.error('X-ray upload error:', err)
    }
  }

  if (loading || history.length === 0) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 rounded-full mb-4 mx-auto" style={{ borderColor: 'var(--border)', borderTopColor: 'var(--accent-blue)' }}></div>
          <p style={{ color: 'var(--text-secondary)' }}>Connecting to Live ICU Monitor...</p>
        </div>
      </div>
    )
  }

  // Format data for Recharts
  const chartData = history.map((tick) => ({
    time: tick.time_index,
    timeLabel: `t=${tick.time_index}s`,
    ...tick.classical_vitals,
    scaled_temp: tick.classical_vitals.temperature * 2, // scale temp to fit near HR (74 ~ 37*2)
    ...tick.qml_correlations,
    ...tick.risk,
    annotations: tick.annotations,
    is_drift: tick.is_drift
  }))

  const tooltipStyle = {
    contentStyle: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-primary)', fontSize: 12 },
  }

  // Collect annotations to plot as ReferenceLines
  const annotationLines: any[] = []
  chartData.forEach(d => {
    if (d.annotations && d.annotations.length > 0) {
      d.annotations.forEach((ann: any) => {
        annotationLines.push({
          time: d.time,
          label: ann.label,
          type: ann.type
        })
      })
    }
  })

  // Find first drift point if it exists
  const driftStartPoint = chartData.find(d => d.is_drift)

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center justify-between mb-6 p-4 rounded-xl border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div>
          <h2 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
            Live Stream: Classical ML vs Quantum ML
          </h2>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Real-time monitoring. Data refreshes every second.
          </p>
        </div>
        
        <div className="flex gap-3">
          {/* X-Ray Upload */}
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleXrayUpload} className="hidden" />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all hover:opacity-80"
            style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
          >
            <span className="text-lg">🫁</span> Inject X-Ray Data
          </button>

          {/* Trigger Drift */}
          <button
            onClick={triggerDrift}
            disabled={driftTriggered}
            className="px-4 py-2 rounded-lg text-xs font-semibold transition-all hover:opacity-80"
            style={{ 
              background: driftTriggered ? 'var(--bg-secondary)' : 'var(--accent-red)', 
              color: driftTriggered ? 'var(--text-muted)' : '#fff',
              border: 'none'
            }}
          >
            {driftTriggered ? 'Micro-Drift Active' : 'Trigger Micro-Drift (Infection)'}
          </button>

          {/* Reset */}
          <button
            onClick={resetSimulation}
            className="px-4 py-2 rounded-lg text-xs font-semibold transition-all hover:opacity-80"
            style={{ background: 'transparent', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Reset
          </button>
        </div>
      </div>

      {/* Chart 1: Classical Vitals */}
      <div className="rounded-xl p-5 mb-6 border shadow-sm" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <h3 className="text-sm font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
          Standard Vitals Monitor (Classical Thresholds)
        </h3>
        <p className="text-xs mb-4" style={{ color: 'var(--text-secondary)' }}>
          Notice how during a micro-drift, all absolute values remain strictly within safe normal clinical boundaries.
        </p>

        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ left: 10, right: 10, top: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} hide />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} domain={[70, 100]} />
              <Tooltip {...tooltipStyle} formatter={(val: number, name: string) => {
                if (name === 'scaled_temp') return [(val / 2).toFixed(2) + '°C', 'Temperature']
                return [val.toFixed(1), name === 'heart_rate' ? 'Heart Rate' : 'SpO2']
              }} />
              <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-secondary)' }} />
              
              {/* Drift Line */}
              {driftStartPoint && (
                <ReferenceLine x={driftStartPoint.time} stroke="var(--accent-red)" strokeDasharray="4 4" label={{ value: 'Infection Start', fill: 'var(--accent-red)', fontSize: 10, position: 'top' }} />
              )}
              
              {/* Annotations */}
              {annotationLines.map((ann, i) => (
                <ReferenceLine key={i} x={ann.time} stroke="var(--accent-blue)" strokeDasharray="2 2" label={{ value: ann.label, fill: 'var(--accent-blue)', fontSize: 10, position: 'insideTopLeft' }} />
              ))}

              <Line type="monotone" dataKey="heart_rate" stroke="var(--accent-blue)" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="scaled_temp" stroke="var(--accent-yellow)" strokeWidth={2} dot={false} isAnimationActive={false} name="Temperature" />
              <Line type="monotone" dataKey="spo2" stroke="var(--accent-green)" strokeWidth={1.5} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: QML Entanglement View */}
      <div className="rounded-xl p-5 mb-6 border shadow-sm" style={{ background: 'var(--bg-card)', borderColor: 'var(--accent-purple)', borderWidth: 2 }}>
        <h3 className="text-sm font-bold mb-1" style={{ color: 'var(--accent-purple)' }}>
          Quantum Entanglement Correlations (QML)
        </h3>
        <p className="text-xs mb-4" style={{ color: 'var(--text-secondary)' }}>
          QML measures how subsystems sync together. Infection breaks the HR-Temp and SpO2-WBC synchronization before absolute values fail.
        </p>

        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ left: 10, right: 10, top: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} hide />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} domain={[-1.5, 1.5]} />
              <Tooltip {...tooltipStyle} formatter={(val: number, name: string) => [val.toFixed(3), name]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              
              {driftStartPoint && (
                <ReferenceLine x={driftStartPoint.time} stroke="var(--accent-red)" strokeDasharray="4 4" />
              )}

              {annotationLines.map((ann, i) => (
                <ReferenceLine key={i} x={ann.time} stroke="var(--accent-purple)" strokeDasharray="2 2" label={{ value: ann.label, fill: 'var(--accent-purple)', fontSize: 10, position: 'insideTopLeft' }} />
              ))}

              <Area type="monotone" dataKey="hr_temp_sync" stroke="var(--accent-purple)" fill="var(--accent-purple)" fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
              <Area type="monotone" dataKey="spo2_wbc_coupling" stroke="var(--accent-teal)" fill="var(--accent-teal)" fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: Risk Score Comparison */}
      <div className="rounded-xl p-5 border shadow-sm" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <h3 className="text-sm font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
          Risk Probability: Classical RF vs 8-Qubit QML
        </h3>
        <p className="text-xs mb-4" style={{ color: 'var(--text-secondary)' }}>
          Notice the gap: QML flags risk early based on broken correlations, while Classical ML waits for standard thresholds.
        </p>

        <div style={{ height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ left: 10, right: 10, top: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} hide />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} domain={[0, 100]} />
              <Tooltip {...tooltipStyle} formatter={(val: number, name: string) => [`${val.toFixed(1)}%`, name]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              
              {driftStartPoint && (
                <ReferenceLine x={driftStartPoint.time} stroke="var(--accent-red)" strokeDasharray="4 4" />
              )}
              
              {annotationLines.map((ann, i) => (
                <ReferenceLine key={i} x={ann.time} stroke="var(--accent-blue)" strokeDasharray="2 2" label={{ value: ann.label, fill: 'var(--accent-blue)', fontSize: 10, position: 'insideTopLeft' }} />
              ))}

              <Area type="monotone" dataKey="classical" stroke="var(--text-muted)" fill="var(--text-muted)" fillOpacity={0.1} strokeWidth={2} name="Classical ML Risk" isAnimationActive={false} />
              <Area type="monotone" dataKey="qml" stroke="var(--accent-red)" fill="var(--accent-red)" fillOpacity={0.1} strokeWidth={2} name="QML Risk" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      
    </div>
  )
}
