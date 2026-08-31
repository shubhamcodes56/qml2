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
            <div className="mt-4 p-4 rounded-xl border" style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}>
              <div className="flex justify-between items-start mb-2">
                <h4 className="font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
                  Zone {selectedZone.zone_id} Deep Analysis
                </h4>
                <button 
                  onClick={() => setSelectedZone(null)}
                  className="text-xs font-bold" 
                  style={{ color: 'var(--text-muted)' }}
                >
                  ✕
                </button>
              </div>
              
              <div className="flex gap-4">
                <img 
                  src={`data:image/png;base64,${selectedZone.zone_image_b64}`} 
                  alt={`Zone ${selectedZone.zone_id}`}
                  className="rounded-lg border shadow-sm"
                  style={{ width: '120px', height: '120px', objectFit: 'cover', borderColor: selectedZone.mean_intensity > 0.55 ? 'var(--accent-red)' : 'var(--border)' }}
                />
                
                <div className="flex-1 space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 rounded bg-white border">
                      <span className="block text-gray-500">Mean Opacity</span>
                      <strong style={{ color: selectedZone.mean_intensity > 0.55 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
                        {(selectedZone.mean_intensity * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div className="p-2 rounded bg-white border">
                      <span className="block text-gray-500">Max Intensity</span>
                      <strong>{(selectedZone.max_intensity * 100).toFixed(1)}%</strong>
                    </div>
                  </div>
                  
                  <p className="text-xs p-2 rounded" style={{ background: selectedZone.mean_intensity > 0.55 ? '#fee2e2' : '#f1f5f9', color: selectedZone.mean_intensity > 0.55 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                    {selectedZone.mean_intensity > 0.55 
                      ? '⚠️ High opacity detected in this quadrant. Classical thresholds often miss early density changes here, but the quantum model flagged it based on localized mean intensity vs global background.'
                      : '✅ Zone appears clear. Opacity within normal baseline limits.'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Zoomed Zone View */}
          {selectedZone && selectedZone.zone_image_b64 && (
            <div className="mt-4 p-4 rounded-xl border shadow-sm" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <div className="flex justify-between items-start mb-3">
                <h4 className="font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
                  Zone {selectedZone.zone_id} Deep Analysis
                </h4>
                <button 
                  onClick={() => setSelectedZone(null)}
                  className="text-xs font-bold" 
                  style={{ color: 'var(--text-muted)' }}
                >
                  ✕
                </button>
              </div>
              
              <div className="flex gap-4">
                <img 
                  src={`data:image/png;base64,${selectedZone.zone_image_b64}`} 
                  alt={`Zone ${selectedZone.zone_id}`}
                  className="rounded-lg border shadow-sm"
                  style={{ width: '120px', height: '120px', objectFit: 'cover', borderColor: selectedZone.mean_intensity > 0.55 ? 'var(--accent-red)' : 'var(--border)' }}
                />
                
                <div className="flex-1 space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 rounded bg-white border">
                      <span className="block text-gray-500">Mean Opacity</span>
                      <strong style={{ color: selectedZone.mean_intensity > 0.55 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
                        {(selectedZone.mean_intensity * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div className="p-2 rounded bg-white border">
                      <span className="block text-gray-500">Max Intensity</span>
                      <strong>{(selectedZone.max_intensity * 100).toFixed(1)}%</strong>
                    </div>
                  </div>
                  
                  <p className="text-xs p-2 rounded leading-relaxed" style={{ background: selectedZone.mean_intensity > 0.55 ? '#fee2e2' : 'var(--bg-secondary)', color: selectedZone.mean_intensity > 0.55 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                    {selectedZone.mean_intensity > 0.55 
                      ? '⚠️ High opacity detected in this quadrant. Classical thresholds often miss early density changes here, but the quantum model flagged it based on localized mean intensity vs global background.'
                      : '✅ Zone appears clear. Opacity within normal baseline limits.'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
