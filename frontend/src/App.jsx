import React, { useState } from 'react';
import { Upload, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, Tooltip, ResponsiveContainer } from 'recharts';

function App() {
  const [rgbFile, setRgbFile] = useState(null);
  const [nirFile, setNirFile] = useState(null);
  const [resultImg, setResultImg] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!rgbFile || !nirFile) return alert("Please upload both RGB and NIR images.");
    setLoading(true);
    const formData = new FormData();
    formData.append("rgb_file", rgbFile);
    formData.append("nir_file", nirFile);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResultImg(data.image_base64);
      setMetrics(data.metrics);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const pieData = metrics ? [
    { name: 'Healthy', value: metrics.healthy_percent, color: '#22c55e' },
    { name: 'Stressed', value: metrics.stressed_percent, color: '#ef4444' }
  ] : [];

  const barData = metrics ? [
    { name: 'Healthy', pixels: metrics.severity.healthy, fill: '#22c55e' },
    { name: 'Moderate', pixels: metrics.severity.moderate, fill: '#eab308' },
    { name: 'Severe', pixels: metrics.severity.severe, fill: '#ef4444' }
  ] : [];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold flex items-center gap-3">
          <Activity className="text-green-500" /> Orbital Agronomy: Stress-Vision
        </h1>
        <p className="text-gray-400 mt-2">Pre-Visual Crop Stress Detection</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><Upload /> Upload Telemetry</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">RGB Imagery (Visual)</label>
              <input type="file" onChange={(e) => setRgbFile(e.target.files[0])} className="w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-blue-600 file:text-white cursor-pointer" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">NIR Imagery (Thermal/Infrared)</label>
              <input type="file" onChange={(e) => setNirFile(e.target.files[0])} className="w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-purple-600 file:text-white cursor-pointer" />
            </div>
            <button onClick={handleAnalyze} disabled={loading} className="w-full mt-4 bg-green-600 hover:bg-green-500 text-white font-bold py-3 px-4 rounded-lg disabled:opacity-50">
              {loading ? "Analyzing Multiband Data..." : "Run Pre-Visual Analysis"}
            </button>
          </div>
        </div>

        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 flex flex-col items-center justify-center min-h-[300px]">
          {resultImg ? <img src={resultImg} alt="Output" className="rounded-lg max-h-80 object-contain" /> : <p className="text-gray-500 text-center">Output map will appear here.</p>}
        </div>
      </div>

      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 flex flex-col justify-center">
            <h3 className="text-lg font-semibold mb-4">System Status</h3>
            {metrics.stressed_percent > 15 ? (
              <div className="p-4 bg-red-900/50 border border-red-500 rounded-lg flex gap-3">
                <AlertTriangle className="text-red-500 flex-shrink-0" />
                <p className="text-red-200">Critical: {metrics.stressed_percent}% of crop area at risk.</p>
              </div>
            ) : (
              <div className="p-4 bg-green-900/50 border border-green-500 rounded-lg flex gap-3">
                <CheckCircle className="text-green-500 flex-shrink-0" />
                <p className="text-green-200">Crop healthy. No severe signatures detected.</p>
              </div>
            )}
          </div>
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 h-64">
            <h3 className="text-sm font-semibold text-gray-400 mb-2 text-center">Health Breakdown</h3>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                  {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 h-64">
            <h3 className="text-sm font-semibold text-gray-400 mb-2 text-center">Severity Distribution</h3>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <XAxis dataKey="name" stroke="#9ca3af" fontSize={12} />
                <Tooltip cursor={{fill: '#374151'}} />
                <Bar dataKey="pixels" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
export default App;

