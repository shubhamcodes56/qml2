import { useRef, useState } from 'react'

interface Props {
  onUpload: (file: File) => void
  result: any
}

export default function XrayPanel({ onUpload, result }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [selectedZone, setSelectedZone] = useState<any | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setFileName(file.name)

    // Preview
    const reader = new FileReader()
    reader.onload = (ev) => setPreview(ev.target?.result as string)
    reader.readAsDataURL(file)

    // Upload
    setSelectedZone(null)
    onUpload(file)
  }

  const xray = result?.xray_analysis
  const findings = xray?.findings || []

  return (
    <div className="rounded-xl p-5 border shadow-sm" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>
        🫁 X-Ray Upload & Deep Analysis
      </h2>

      {/* Dropzone */}
      <div
        onClick={() => fileInputRef.current?.click()}
        className="rounded-lg p-6 text-center cursor-pointer transition-all hover:opacity-80 mb-4"
        style={{
          background: 'var(--bg-secondary)',
          border: '2px dashed var(--border)',
          ...(preview ? {} : {}),
        }}
      >
        {preview ? (
          <div>
            <img
              src={preview}
              alt="X-ray preview"
              className="max-h-48 mx-auto rounded-lg mb-2"
              style={{ objectFit: 'contain' }}
            />
            <p className="text-xs" style={{ color: '#94a3b8' }}>{fileName}</p>
          </div>
        ) : (
          <div>
            <p className="text-3xl mb-2">📤</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
              Click to upload Chest X-Ray
            </p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              PNG, JPG, DICOM • Max 10MB
            </p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Results */}
      {xray && (
        <div>
          {/* Feature extraction results */}
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
            QML Feature Extraction ({xray.analysis_method})
          </p>
          <div className="grid grid-cols-2 gap-2 mb-4">
            {[
              { label: 'Opacity', value: xray.opacity_score, key: 'opacity' },
              { label: 'Cavity', value: xray.cavity_probability, key: 'cavity' },
              { label: 'Nodules', value: xray.nodule_density, key: 'nodule' },
              { label: 'Pleural', value: xray.pleural_thickening, key: 'pleural' },
            ].map((f) => (
              <div
                key={f.key}
                className="rounded-lg p-3"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
              >
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{f.label}</p>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-2 rounded-full" style={{ background: 'var(--border)' }}>
                    <div
                      className="h-2 rounded-full transition-all"
                      style={{
                        width: `${Math.min(100, f.value * 100)}%`,
                        background: f.value > 0.1 ? 'var(--accent-yellow)' : 'var(--accent-green)',
                      }}
                    />
                  </div>
                  <span className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
                    {(f.value * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          {xray.heatmap_overlay_b64 && (
            <div className="mb-4">
              <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
                QML Heatmap Overlay
              </p>
              <img
                src={`data:image/png;base64,${xray.heatmap_overlay_b64}`}
                alt="QML Heatmap"
                className="w-full rounded-lg"
                style={{ maxHeight: 200, objectFit: 'contain' }}
              />
            </div>
          )}

          {/* Gemini AI Report */}
          {result?.gemini_report && (
            <div className="mb-4 p-4 rounded-xl shadow-sm" style={{ background: '#f8fafc', border: '1px solid #e2e8f0' }}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">✨</span>
                <p className="text-sm font-bold text-slate-800">
                  AI Pulmonologist Diagnostic Report
                </p>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed italic">
                "{result.gemini_report}"
              </p>
            </div>
          )}

          {/* Findings */}
          {findings.length > 0 && (
            <div>
              <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
                X-Ray Findings
              </p>
              {findings.map((f: any, i: number) => (
                <div
                  key={i}
                  className="rounded-lg p-3 mb-2 text-xs"
                  style={{
                    background: f.severity === 'warning' ? '#fef3c7' : 'var(--bg-secondary)',
                    border: `1px solid ${f.severity === 'warning' ? 'var(--accent-yellow)' : 'var(--border)'}`,
                    color: 'var(--text-primary)',
                  }}
                >
                  <strong style={{ color: f.severity === 'warning' ? 'var(--accent-yellow)' : 'var(--text-secondary)' }}>
                    {f.feature}:
                  </strong>{' '}
                  {f.message}
                </div>
              ))}
            </div>
          )}

          {/* Zone Analysis */}
          {xray.zones && (
            <div className="mt-3">
              <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
                9-Zone Lung Field Analysis
              </p>
              <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                {xray.zones.map((z: any) => {
                  const intensity = z.mean_intensity
                  const isHot = intensity > 0.55
                  return (
                    <div
                      key={z.zone_id}
                      onClick={() => setSelectedZone(z)}
                      className="rounded p-2 text-center text-xs cursor-pointer transition-all hover:scale-105"
                      style={{
                        background: isHot ? '#fee2e2' : 'var(--bg-secondary)',
                        border: `1px solid ${isHot ? 'var(--accent-red)' : 'var(--border)'}`,
                        color: isHot ? 'var(--accent-red)' : 'var(--text-secondary)',
                        boxShadow: selectedZone?.zone_id === z.zone_id ? '0 0 0 2px var(--accent-blue)' : 'none'
                      }}
                    >
                      Z{z.zone_id}: {(intensity * 100).toFixed(0)}%
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Zoomed Zone View */}
          {selectedZone && selectedZone.zone_image_b64 && (
            <div className="mt-4 p-4 rounded-xl border shadow-sm" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <div className="flex justify-between items-start mb-3">
                <h4 className="font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
                  🔬 Zone {selectedZone.zone_id} — Actual QML Deep Analysis
                </h4>
                <button 
                  onClick={() => setSelectedZone(null)}
                  className="text-xs font-bold px-2 py-1 rounded" 
                  style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}
                >
                  ✕
                </button>
              </div>
              
              <div className="flex gap-4">
                <img 
                  src={`data:image/png;base64,${selectedZone.zone_image_b64}`} 
                  alt={`Zone ${selectedZone.zone_id}`}
                  className="rounded-lg border shadow-sm"
                  style={{ width: '120px', height: '120px', objectFit: 'cover', borderColor: selectedZone.qml_zone_risk > 50 ? 'var(--accent-red)' : 'var(--border)' }}
                />
                
                <div className="flex-1 space-y-2">
                  {/* OpenCV Extracted Features */}
                  <p className="text-[10px] font-bold uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
                    DenseNet-121 Trained Features → Fed to QML Circuit
                  </p>
                  <div className="grid grid-cols-4 gap-1 text-[10px]">
                    {selectedZone.cv_features && Object.entries(selectedZone.cv_features).map(([key, val]: [string, any]) => (
                      <div key={key} className="p-1.5 rounded border text-center" style={{ background: 'var(--bg-secondary)' }}>
                        <span className="block text-gray-500 font-bold capitalize">{key}</span>
                        <strong style={{ color: val > 0.3 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
                          {(val * 100).toFixed(1)}%
                        </strong>
                      </div>
                    ))}
                  </div>
                  
                  {/* Actual QML Circuit Result */}
                  <div className="p-3 rounded-lg border" style={{ 
                    background: selectedZone.qml_zone_risk > 50 ? '#fef2f2' : 'var(--bg-secondary)', 
                    borderColor: selectedZone.qml_zone_risk > 50 ? 'var(--accent-red)' : 'var(--border)' 
                  }}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] font-bold uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
                        8-Qubit QML Circuit Output
                      </span>
                      {selectedZone.qml_details?.severity && (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ 
                          background: selectedZone.qml_details.severity === 'NORMAL' ? '#dcfce7' : 
                                     selectedZone.qml_details.severity === 'WARNING' ? '#fef3c7' : 
                                     selectedZone.qml_details.severity === 'CRITICAL' ? '#fee2e2' : '#f1f5f9',
                          color: selectedZone.qml_details.severity === 'NORMAL' ? '#166534' :
                                 selectedZone.qml_details.severity === 'WARNING' ? '#92400e' :
                                 selectedZone.qml_details.severity === 'CRITICAL' ? '#991b1b' : '#475569'
                        }}>
                          {selectedZone.qml_details.severity}
                        </span>
                      )}
                    </div>
                    <div className="flex items-end gap-3">
                      <span className="text-2xl font-bold" style={{ color: selectedZone.qml_zone_risk > 50 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                        {selectedZone.qml_zone_risk}%
                      </span>
                      {selectedZone.qml_details?.von_neumann_entropy != null && (
                        <span className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>
                          Entropy: {selectedZone.qml_details.von_neumann_entropy} bits
                        </span>
                      )}
                      {selectedZone.qml_details?.confidence != null && (
                        <span className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>
                          Conf: {selectedZone.qml_details.confidence}%
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Per-Qubit Readings */}
                  {selectedZone.qml_details?.per_qubit && selectedZone.qml_details.per_qubit.length > 0 && (
                    <div className="grid grid-cols-4 gap-1 text-[9px]">
                      {selectedZone.qml_details.per_qubit.map((q: any, i: number) => (
                        <div key={i} className="p-1 rounded border text-center" style={{ 
                          background: q.status === 'anomaly' ? '#fee2e2' : q.status === 'watch' ? '#fef3c7' : '#f0fdf4',
                          borderColor: q.status === 'anomaly' ? '#fca5a5' : q.status === 'watch' ? '#fcd34d' : '#bbf7d0'
                        }}>
                          <span className="block font-bold">{q.label}</span>
                          <span>Z={q.pauliz?.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
