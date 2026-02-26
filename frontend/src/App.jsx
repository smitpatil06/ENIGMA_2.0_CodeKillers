import React, { useState } from 'react';
import { Upload, Activity, AlertTriangle, CheckCircle, Info, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// ── Custom tooltip for charts ──────────────────────────────────────────────
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-sm shadow-xl">
        <p className="text-white font-semibold">{payload[0].name}</p>
        <p style={{ color: payload[0].payload.color || payload[0].fill }}>
          {typeof payload[0].value === 'number' && payload[0].value > 1000
            ? payload[0].value.toLocaleString() + ' px'
            : payload[0].value + '%'}
        </p>
      </div>
    );
  }
  return null;
};

// ── Stat card ──────────────────────────────────────────────────────────────
const StatCard = ({ label, value, color, description }) => (
  <div className="flex flex-col gap-1">
    <div className="flex items-center gap-2">
      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
      <span className="text-xs text-gray-400 uppercase tracking-widest">{label}</span>
    </div>
    <p className="text-3xl font-black tabular-nums" style={{ color }}>{value}<span className="text-lg font-normal text-gray-500">%</span></p>
    {description && <p className="text-xs text-gray-500">{description}</p>}
  </div>
);

// ── Upload zone ────────────────────────────────────────────────────────────
const UploadZone = ({ label, accent, file, onChange, hint, accept = 'image/*' }) => (
  <label className="group relative flex flex-col gap-2 cursor-pointer">
    <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: accent }}>{label}</span>
    <div
      className="relative border-2 border-dashed rounded-xl p-4 transition-all duration-200 flex items-center gap-3 hover:bg-gray-700/40"
      style={{ borderColor: file ? accent : '#374151' }}
    >
      <div
        className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors"
        style={{ backgroundColor: file ? accent + '22' : '#1f2937' }}
      >
        <Upload size={18} style={{ color: file ? accent : '#6b7280' }} />
      </div>
      <div className="min-w-0">
        {file
          ? <p className="text-sm text-white font-medium truncate">{file.name}</p>
          : <p className="text-sm text-gray-500">{hint}</p>}
      </div>
      {file && (
        <div className="ml-auto w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: accent }} />
      )}
    </div>
    <input type="file" accept={accept} className="hidden" onChange={e => onChange(e.target.files[0])} />
  </label>
);

// ── Legend pill ────────────────────────────────────────────────────────────
const LegendPill = ({ color, label, value }) => (
  <div className="flex items-center gap-2">
    <div className="w-4 h-4 rounded-sm flex-shrink-0" style={{ backgroundColor: color }} />
    <span className="text-xs text-gray-400">{label}</span>
    {value !== undefined && (
      <span className="ml-auto text-xs font-bold tabular-nums text-white">{value}%</span>
    )}
  </div>
);

export default function App() {
  const [rgbFile, setRgbFile]     = useState(null);
  const [nirFile, setNirFile]     = useState(null);
  const [tifFile, setTifFile]     = useState(null);
  const [resultImg, setResultImg] = useState(null);
  const [previewImg, setPreviewImg] = useState(null);
  const [metrics, setMetrics]     = useState(null);
  const [modelType, setModelType] = useState('');
  const [selectedModel, setSelectedModel] = useState('unet'); // Default to U-Net
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  const canAnalyze = selectedModel === 'unet'
    ? Boolean(rgbFile && nirFile)
    : Boolean(tifFile);

  const handleAnalyze = async () => {
    if (!canAnalyze) {
      setError(selectedModel === 'unet'
        ? 'Upload both RGB and NIR images first.'
        : 'Upload a hyperspectral .tif file for this model.');
      return;
    }
    setError(null);
    setLoading(true);
    const form = new FormData();
    if (selectedModel === 'unet') {
      form.append('rgb_file', rgbFile);
      form.append('nir_file', nirFile);
    } else {
      form.append('tif_file', tifFile);
    }
    form.append('model_type', selectedModel);
    try {
      const res  = await fetch('http://localhost:8000/api/analyze', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');
      setResultImg(data.image_base64);
      setPreviewImg(data.preview_base64 || null);
      setMetrics(data.metrics);
      setModelType(data.model_type);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Derived chart data ───────────────────────────────────────────────────
  // Filter slices < 1% — Recharts renders near-zero values as a visible
  // gap/notch in the donut ring even when the value is essentially 0.
  const pieData = metrics ? [
    { name: 'Healthy',  value: metrics.healthy_percent,  color: '#22c55e' },
    { name: 'Moderate', value: metrics.moderate_percent, color: '#eab308' },
    { name: 'Severe',   value: metrics.severe_percent,   color: '#ef4444' },
  ].filter(d => d.value >= 1) : [];

  const barData = metrics ? [
    { name: 'Healthy',  pixels: metrics.severity.healthy,  fill: '#22c55e' },
    { name: 'Moderate', pixels: metrics.severity.moderate, fill: '#eab308' },
    { name: 'Severe',   pixels: metrics.severity.severe,   fill: '#ef4444' },
  ] : [];

  const statusLevel = metrics
    ? metrics.severe_percent > 20    ? 'high-risk'    // genuinely bad — act now
    : metrics.severe_percent > 5     ? 'at-risk'      // some red zones — watch it
    : metrics.moderate_percent > 20  ? 'early-warning' // mostly yellow — preventive
    : 'healthy'
    : null;

  // Actionable recommendation text per level
  const recommendation = {
    'high-risk':      { title: 'Action Recommended',   body: `${metrics?.severe_percent}% of the field shows high-confidence stress markers. Consider targeted irrigation or fungicide application within 48–72 hours to prevent further spread.` },
    'at-risk':        { title: 'Early Intervention Advised', body: `${metrics?.severe_percent}% shows elevated stress. These areas may deteriorate if conditions don't improve — schedule a field check and prepare a treatment plan now.` },
    'early-warning':  { title: 'Pre-Visual Stress Detected', body: `${metrics?.moderate_percent}% of the field shows early stress signatures not yet visible to the naked eye. Proactive irrigation or soil sampling now could prevent crop loss.` },
    'healthy':        { title: 'Looking Good',          body: 'No significant stress signatures detected. Continue regular monitoring — early detection is your biggest advantage.' },
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100" style={{ fontFamily: "'DM Mono', monospace" }}>

      {/* ── Header ── */}
      <header className="border-b border-gray-800 px-8 py-5 flex items-center gap-4">
        <div className="w-9 h-9 rounded-lg bg-green-500/10 border border-green-500/30 flex items-center justify-center">
          <Activity size={18} className="text-green-400" />
        </div>
        <div>
          <h1 className="text-lg font-bold tracking-tight text-white">Orbital Agronomy</h1>
          <p className="text-xs text-gray-500 tracking-widest uppercase">Stress-Vision · Pre-Visual Detection</p>
        </div>
        {modelType && (
          <div className="ml-auto flex items-center gap-2 text-xs text-gray-500 border border-gray-700 rounded-full px-3 py-1">
            <Zap size={11} className="text-yellow-400" />
            {modelType}
          </div>
        )}
      </header>

      {/* Top Section: Controls + Images */}
      <div className="p-8 grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)] gap-6">

        {/* ── Left panel: upload + status ── */}
        <div className="flex flex-col gap-6">

          {/* Upload card */}
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-5">
            <h2 className="text-sm font-bold uppercase tracking-widest text-gray-300">Upload Telemetry</h2>

            {selectedModel === 'unet' ? (
              <>
                <UploadZone
                  label="RGB Imagery"
                  accent="#3b82f6"
                  file={rgbFile}
                  onChange={setRgbFile}
                  hint="Visual spectrum (.jpg / .png)"
                  accept="image/*"
                />
                <UploadZone
                  label="NIR Imagery"
                  accent="#a855f7"
                  file={nirFile}
                  onChange={setNirFile}
                  hint="Infrared channel (.jpg / .png)"
                  accept="image/*"
                />
              </>
            ) : (
              <UploadZone
                label="Hyperspectral TIF"
                accent="#f59e0b"
                file={tifFile}
                onChange={setTifFile}
                hint="Upload .tif / .tiff"
                accept=".tif,.tiff"
              />
            )}

            {/* Model Selector */}
            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold uppercase tracking-widest text-gray-300">Analysis Model</span>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setSelectedModel('unet')}
                  className={`py-2.5 px-3 rounded-lg text-xs font-bold transition-all duration-200 border ${
                    selectedModel === 'unet'
                      ? 'bg-green-500/20 border-green-500 text-green-300'
                      : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600'
                  }`}
                >
                  U-Net
                  <div className="text-[10px] font-normal mt-0.5 opacity-70">CNN 2D</div>
                </button>
                <button
                  onClick={() => setSelectedModel('deep')}
                  className={`py-2.5 px-3 rounded-lg text-xs font-bold transition-all duration-200 border ${
                    selectedModel === 'deep'
                      ? 'bg-blue-500/20 border-blue-500 text-blue-300'
                      : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600'
                  }`}
                >
                  Deep MLP
                  <div className="text-[10px] font-normal mt-0.5 opacity-70">CNN 1D + Scikit</div>
                </button>
                <button
                  onClick={() => setSelectedModel('rf')}
                  className={`py-2.5 px-3 rounded-lg text-xs font-bold transition-all duration-200 border ${
                    selectedModel === 'rf'
                      ? 'bg-purple-500/20 border-purple-500 text-purple-300'
                      : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600'
                  }`}
                >
                  Random Forest
                  <div className="text-[10px] font-normal mt-0.5 opacity-70">Classic ML</div>
                </button>
              </div>
              <p className="text-xs text-gray-600 mt-1">
                {selectedModel === 'unet' && 'Deep segmentation model with test-time augmentation'}
                {selectedModel === 'deep' && 'Multi-layer perceptron with NDWI & PRI features'}
                {selectedModel === 'rf' && 'Ensemble classifier for fast, reliable predictions'}
              </p>
            </div>

            {error && (
              <div className="flex items-start gap-2 text-xs text-red-400 bg-red-900/20 border border-red-800/40 rounded-lg p-3">
                <AlertTriangle size={13} className="flex-shrink-0 mt-0.5" />
                {error}
              </div>
            )}

            <button
              onClick={handleAnalyze}
                disabled={loading || !canAnalyze}
              className="w-full py-3 rounded-xl text-sm font-bold tracking-wide transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{
                background: loading ? '#166534' : 'linear-gradient(135deg, #16a34a, #15803d)',
                color: '#fff',
                boxShadow: loading ? 'none' : '0 0 20px #16a34a55',
              }}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4" strokeDashoffset="10" />
                  </svg>
                  Analysing multiband data…
                </span>
              ) : 'Run Pre-Visual Analysis'}
            </button>
          </div>

          {/* Heatmap legend */}
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-3">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500">Overlay Legend</h2>
            <LegendPill color="#22c55e" label="Healthy  (prob < 0.35)" value={metrics?.healthy_percent} />
            <LegendPill color="#eab308" label="At-Risk  (0.35 – 0.70)" value={metrics?.moderate_percent} />
            <LegendPill color="#ef4444" label="High-Risk (prob > 0.70)" value={metrics?.severe_percent} />
            <p className="text-xs text-gray-600 mt-1 leading-relaxed">
              Yellow zones indicate early stress that can still be treated. Red zones show high-confidence detections that need attention.
            </p>
          </div>

          {/* System status */}
          {metrics && (
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
              <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500">System Status</h2>
              {statusLevel === 'high-risk' && (
                <div className="flex gap-3 p-3 bg-orange-900/30 border border-orange-600/50 rounded-xl">
                  <AlertTriangle size={16} className="text-orange-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-bold text-orange-300 mb-0.5">{recommendation['high-risk'].title}</p>
                    <p className="text-xs text-orange-200/80 leading-relaxed">{recommendation['high-risk'].body}</p>
                  </div>
                </div>
              )}
              {statusLevel === 'at-risk' && (
                <div className="flex gap-3 p-3 bg-yellow-900/30 border border-yellow-600/50 rounded-xl">
                  <AlertTriangle size={16} className="text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-bold text-yellow-300 mb-0.5">{recommendation['at-risk'].title}</p>
                    <p className="text-xs text-yellow-200/80 leading-relaxed">{recommendation['at-risk'].body}</p>
                  </div>
                </div>
              )}
              {statusLevel === 'early-warning' && (
                <div className="flex gap-3 p-3 bg-blue-900/30 border border-blue-600/50 rounded-xl">
                  <Info size={16} className="text-blue-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-bold text-blue-300 mb-0.5">{recommendation['early-warning'].title}</p>
                    <p className="text-xs text-blue-200/80 leading-relaxed">{recommendation['early-warning'].body}</p>
                  </div>
                </div>
              )}
              {statusLevel === 'healthy' && (
                <div className="flex gap-3 p-3 bg-green-900/30 border border-green-600/50 rounded-xl">
                  <CheckCircle size={16} className="text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-bold text-green-300 mb-0.5">{recommendation['healthy'].title}</p>
                    <p className="text-xs text-green-200/80 leading-relaxed">{recommendation['healthy'].body}</p>
                  </div>
                </div>
              )}
              <div className="grid grid-cols-3 gap-3 pt-1">
                <StatCard label="Healthy"  value={metrics.healthy_percent}  color="#22c55e" />
                <StatCard label="Moderate" value={metrics.moderate_percent} color="#eab308" />
                <StatCard label="Severe"   value={metrics.severe_percent}   color="#ef4444" />
              </div>
              <p className="text-xs text-gray-600">
                Mean stress score: <span className="text-gray-400 font-bold">{metrics.mean_stress}</span>
              </p>
            </div>
          )}
        </div>

        {/* ── Image Display Area ── */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-3 min-h-[600px]">
          <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500">Stress Map Output</h2>
          <div className="flex-1 flex items-center justify-center rounded-xl bg-gray-950 overflow-hidden">
            {resultImg ? (
              previewImg ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full p-4">
                  <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-3 flex flex-col gap-2">
                    <p className="text-[11px] uppercase tracking-widest text-gray-500">Raw Preview</p>
                    <div className="aspect-[4/3] w-full rounded-xl bg-gray-950 overflow-hidden">
                      <img src={previewImg} alt="Raw preview" className="w-full h-full object-contain" />
                    </div>
                  </div>
                  <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-3 flex flex-col gap-2">
                    <p className="text-[11px] uppercase tracking-widest text-gray-500">Stress Overlay</p>
                    <div className="aspect-[4/3] w-full rounded-xl bg-gray-950 overflow-hidden">
                      <img src={resultImg} alt="Stress heatmap" className="w-full h-full object-contain" />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full p-4">
                  <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-3 h-full flex flex-col gap-2">
                    <p className="text-[11px] uppercase tracking-widest text-gray-500">Stress Overlay</p>
                    <div className="aspect-[4/3] w-full rounded-xl bg-gray-950 overflow-hidden">
                      <img src={resultImg} alt="Stress heatmap" className="w-full h-full object-contain" />
                    </div>
                  </div>
                </div>
              )
            ) : (
              <div className="text-center text-gray-700 select-none p-8">
                <Activity size={40} className="mx-auto mb-3 opacity-30" />
                <p className="text-sm">Output map will appear here</p>
                <p className="text-xs mt-1 opacity-60">Green · Yellow · Red overlay</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Section: Charts */}
      <div className="px-8 pb-8 grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Pie chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 h-80 flex flex-col">
          <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-2">Health Breakdown</h2>
          {metrics ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  innerRadius={55} outerRadius={80}
                  paddingAngle={pieData.length > 1 ? 3 : 0}
                  minAngle={3}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  formatter={v => <span style={{ color: '#9ca3af', fontSize: 11 }}>{v}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-700 text-xs">Run analysis to populate</div>
          )}
        </div>

        {/* Bar chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 h-80 flex flex-col">
          <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-2">Pixel Severity Count</h2>
          {metrics ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} barSize={28}>
                <XAxis dataKey="name" stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} width={48} tickFormatter={v => v > 1000 ? (v/1000).toFixed(0)+'k' : v} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: '#ffffff08' }} />
                <Bar dataKey="pixels" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-700 text-xs">Run analysis to populate</div>
          )}
        </div>

      </div>
    </div>
  );
}
