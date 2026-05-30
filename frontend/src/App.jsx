import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Activity,
  Cpu,
  BarChart3,
  Terminal,
  Zap,
  ShieldCheck,
  AlertTriangle,
  RefreshCcw,
  Layers,
  PlaySquare,
  StopCircle,
  PlusCircle,
  UploadCloud,
  FolderOpen,
  AlertCircle,
  CheckCircle,
  Trash2,
  DownloadCloud
} from 'lucide-react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import { motion, AnimatePresence } from 'framer-motion'

// Global placeholder commented out in favor of dynamic component-level state
// const API_BASE = '/api'

// Helper to generate mock structured tabular data for common demo domains
const generateSampleInput = (modelId, features, skew = false) => {
  const inputs = {}

  // 1. Fraud Detector Schema — supports both naming conventions
  if (modelId.toLowerCase().includes('fraud')) {
    const amt = skew ? Number((Math.random() * 1200 + 800).toFixed(2)) : Number((Math.random() * 95 + 5).toFixed(2))
    const dist = skew ? Number((Math.random() * 400 + 100).toFixed(2)) : Number((Math.random() * 14 + 0.1).toFixed(2))
    const intl = skew ? 1 : (Math.random() > 0.85 ? 1 : 0)
    // All known feature name variants
    inputs['amount'] = amt
    inputs['transaction_amount'] = amt
    inputs['distance'] = dist
    inputs['transaction_distance'] = dist
    inputs['is_international'] = intl
  }
  // 2. Customer Churn Schema — supports both naming conventions
  else if (modelId.toLowerCase().includes('churn')) {
    const tenure = skew ? Math.floor(Math.random() * 3 + 1) : Math.floor(Math.random() * 58 + 3)
    const charges = skew ? Number((Math.random() * 130 + 120).toFixed(2)) : Number((Math.random() * 70 + 20).toFixed(2))
    const calls = skew ? Math.floor(Math.random() * 6 + 6) : Math.floor(Math.random() * 3)
    // All known feature name variants
    inputs['tenure'] = tenure
    inputs['monthly_charges'] = charges
    inputs['support_calls'] = calls
    inputs['customer_calls'] = calls
    inputs['num_support_calls'] = calls
  }
  // 3. House Price Schema — supports both naming conventions
  else if (modelId.toLowerCase().includes('house') || modelId.toLowerCase().includes('price')) {
    const sqft = skew ? Number((Math.random() * 8000 + 4500).toFixed(0)) : Number((Math.random() * 2200 + 800).toFixed(0))
    const beds = skew ? Math.floor(Math.random() * 5 + 6) : Math.floor(Math.random() * 3 + 1)
    const baths = skew ? Number((Math.random() * 3 + 5).toFixed(1)) : Number((Math.random() * 2 + 1).toFixed(1))
    const age = skew ? Number((Math.random() * 30 + 90).toFixed(0)) : Number((Math.random() * 45 + 2).toFixed(0))
    // All known feature name variants
    inputs['sqft'] = sqft
    inputs['square_feet'] = sqft
    inputs['bedrooms'] = beds
    inputs['bathrooms'] = baths
    inputs['age'] = age
    inputs['house_age'] = age
  }
  // 4. Generic Fallback — generates reasonable numeric values for any feature schema
  else {
    features.forEach(f => {
      inputs[f] = skew ? Number((Math.random() * 50 + 50).toFixed(2)) : Number((Math.random() * 10).toFixed(2))
    })
  }

  // Final pass: map only features registered in the model schema
  // Anything unrecognized gets a random fallback so the form is always fully populated
  const finalPayload = {}
  features.forEach(f => {
    finalPayload[f] = inputs[f] !== undefined ? inputs[f] : Number((Math.random() * 20).toFixed(2))
  })
  return finalPayload
}

function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [backendHost, setBackendHost] = useState(() => {
    const saved = localStorage.getItem('MLOPS_AWS_IP')
    if (saved) return saved.trim()
    if (import.meta.env.VITE_BACKEND_IP) return import.meta.env.VITE_BACKEND_IP.trim()
    const currentHost = window.location.hostname
    if (currentHost === 'localhost' || currentHost === '127.0.0.1') return 'localhost'
    return ''
  })

  const [connectionStatus, setConnectionStatus] = useState('checking')
  const [tempHost, setTempHost] = useState(backendHost === 'localhost' ? '' : backendHost)
  const [showSettings, setShowSettings] = useState(false)
  const [monitorModelId, setMonitorModelId] = useState('')

  const API_BASE = backendHost === 'localhost' || !backendHost
    ? '/api'
    : (backendHost.startsWith('http://') || backendHost.startsWith('https://')
      ? `${backendHost}/api`
      : `http://${backendHost}:8000/api`)

  const baseGrafanaUrl = backendHost === 'localhost' || !backendHost
    ? 'http://localhost:3000/d/enterprise-mlops-dashboard?orgId=1&kiosk'
    : (backendHost.startsWith('http://') || backendHost.startsWith('https://')
      ? `${backendHost.replace(/:\d+$/, '')}:3000/d/enterprise-mlops-dashboard?orgId=1&kiosk`
      : `http://${backendHost}:3000/d/enterprise-mlops-dashboard?orgId=1&kiosk`)

  const grafanaUrl = monitorModelId ? `${baseGrafanaUrl}&var-model_id=${monitorModelId}` : baseGrafanaUrl

  const isMixedContentBlocked = window.location.protocol === 'https:' &&
    (backendHost !== 'localhost' && backendHost !== '' && !backendHost.startsWith('https://'))

  useEffect(() => {
    const checkConnection = async () => {
      setConnectionStatus('checking')
      try {
        const pingUrl = backendHost === 'localhost' || !backendHost
          ? '/api/dashboard/stats'
          : (backendHost.startsWith('http://') || backendHost.startsWith('https://')
            ? `${backendHost}/api/dashboard/stats`
            : `http://${backendHost}:8000/api/dashboard/stats`)
        await axios.get(pingUrl, { timeout: 3000 })
        setConnectionStatus('connected')
      } catch (err) {
        console.warn("Connection verification failed:", err)
        setConnectionStatus('disconnected')
      }
    }
    checkConnection()
  }, [backendHost])
  const [stats, setStats] = useState({
    total_predictions: 0,
    avg_latency_ms: 0,
    active_models_count: 0,
    unresolved_alerts_count: 0,
    models: []
  })

  const [registeredModels, setRegisteredModels] = useState([])
  const [registryVersions, setRegistryVersions] = useState([])
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(false)
  const [predictionHistory, setPredictionHistory] = useState([])
  const [dynamicChartData, setDynamicChartData] = useState([])

  // Interactive Sandbox state
  const [sandboxModelId, setSandboxModelId] = useState('')
  const [sandboxFeatures, setSandboxFeatures] = useState({})
  const [sandboxResult, setSandboxResult] = useState(null)
  const [sandboxLoading, setSandboxLoading] = useState(false)
  const [lastSandboxModelId, setLastSandboxModelId] = useState('')

  // Simulation states
  const [isSimulatingTraffic, setIsSimulatingTraffic] = useState(false)
  const [isSimulatingDrift, setIsSimulatingDrift] = useState(false)

  // Upload Form State
  const [uploadFormData, setUploadFormData] = useState({
    model_id: '',
    model_name: '',
    version: 'v1',
    framework: 'scikit-learn',
    task_type: 'classification',
    features: ''
  })
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadMessage, setUploadMessage] = useState({ type: '', text: '' })

  // 1. Fetch dashboard stats, models list, and alerts
  const fetchAllData = async () => {
    try {
      const statsRes = await axios.get(`${API_BASE}/dashboard/stats`)
      setStats(statsRes.data)

      const modelsRes = await axios.get(`${API_BASE}/models`)
      const models = modelsRes.data
      setRegisteredModels(models)

      const versionRows = []
      const ids = [...new Set(models.map((m) => m.model_id))]
      await Promise.all(
        ids.map(async (modelId) => {
          try {
            const vRes = await axios.get(`${API_BASE}/models/${encodeURIComponent(modelId)}/versions`)
            const { model_name, active_filename, versions } = vRes.data
            versions.forEach((v) => {
              versionRows.push({
                ...v,
                model_id: modelId,
                model_name,
                active_filename,
                is_live_file: v.filename === active_filename,
              })
            })
          } catch (e) {
            console.warn(`Could not load versions for ${modelId}`, e)
          }
        })
      )
      versionRows.sort((a, b) => {
        if (a.model_id !== b.model_id) return a.model_id.localeCompare(b.model_id)
        return (b.registered_at || '').localeCompare(a.registered_at || '')
      })
      setRegistryVersions(versionRows)

      const alertsRes = await axios.get(`${API_BASE}/alerts`)
      setAlerts(alertsRes.data)
    } catch (err) {
      console.error("Failed to gather system operational summaries:", err)
    }
  }

  useEffect(() => {
    fetchAllData()
    const interval = setInterval(fetchAllData, 5000)
    return () => clearInterval(interval)
  }, [backendHost]) // Re-initialize polling only when backend host changes

  // Autofill Sandbox Model dropdown if empty
  useEffect(() => {
    if (registeredModels.length > 0 && !sandboxModelId) {
      const active = registeredModels.filter(m => m.deployment_status === 'ACTIVE')
      if (active.length > 0) {
        setSandboxModelId(active[0].model_id)
      }
    }
  }, [registeredModels, sandboxModelId])

  // Reset sandbox form inputs when selected model changes
  // NOTE: Uses lastSandboxModelId to prevent polling from constantly resetting the inputs
  useEffect(() => {
    if (sandboxModelId && sandboxModelId !== lastSandboxModelId) {
      const target = registeredModels.find(m => m.model_id === sandboxModelId)
      if (target) {
        const initialInputs = {}
        target.features.forEach(f => {
          initialInputs[f] = ''
        })
        setSandboxFeatures(initialInputs)
        setSandboxResult(null)
        setLastSandboxModelId(sandboxModelId)
      }
    }
  }, [sandboxModelId, registeredModels, lastSandboxModelId])

  // 2. In-browser inference and drift simulator
  useEffect(() => {
    let interval;
    if (isSimulatingTraffic) {
      interval = setInterval(async () => {
        const activeModels = registeredModels.filter(m => m.deployment_status === 'ACTIVE')
        if (activeModels.length === 0) return

        // Pick random active model
        const model = activeModels[Math.floor(Math.random() * activeModels.length)]

        // Generate payload (skews transaction amounts/charges if drift injection is enabled!)
        const payloadFeatures = generateSampleInput(model.model_id, model.features, isSimulatingDrift)

        try {
          const res = await axios.post(`${API_BASE}/predict/${encodeURIComponent(model.model_id)}`, { features: payloadFeatures })
          const data = res.data

          const newPrediction = {
            id: data.request_id,
            model_id: data.model_id,
            timestamp: new Date().toLocaleTimeString(),
            features: JSON.stringify(payloadFeatures),
            prediction: data.prediction,
            confidence: data.confidence !== null ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A',
            latency: `${data.latency_ms.toFixed(1)}ms`,
            latencyValue: data.latency_ms,
            drifted: isSimulatingDrift
          }

          setPredictionHistory(prev => [newPrediction, ...prev].slice(0, 15))
          setDynamicChartData(prev => [...prev, { name: prev.length + 1 + '', latency: data.latency_ms }].slice(-15))
        } catch (err) {
          if (err.response?.status !== 429) {
            console.error("Simulation inference pulse crashed:", err)
          }
          // Silently skip 429s during simulation — they resolve within 5s automatically
        }
      }, 2500) // 2.5s interval keeps simulation well under rate limits
    }
    return () => clearInterval(interval)
  }, [isSimulatingTraffic, isSimulatingDrift, registeredModels])

  // 3. Deploy/Activate model version
  const handleDeployModel = async (modelId, version) => {
    setLoading(true)
    try {
      await axios.post(`${API_BASE}/models/${encodeURIComponent(modelId)}/deploy?version=${version}`)
      await fetchAllData()
    } catch (err) {
      console.error("Activation failed:", err)
      alert("Failed to swap deployment active version.")
    } finally {
      setLoading(false)
    }
  }

  // 4. Force Retrain Pipeline
  const handleForceRetrain = async (modelId) => {
    try {
      await axios.post(`${API_BASE}/retrain/${encodeURIComponent(modelId)}`)
      alert(`Automated self-healing retraining triggered for '${modelId}' in background. Swapping active inference shortly!`)
    } catch (err) {
      console.error("Retraining failed to trigger:", err)
    }
  }

  // Delete model from registry
  const handleDeleteModel = async (modelId) => {
    const confirmDelete = window.confirm(
      `Are you sure you want to permanently delete model '${modelId}'?\n\nThis will completely purge all associated database logs, telemetry metrics, and disk binary files.`
    )
    if (!confirmDelete) return

    setLoading(true)
    try {
      await axios.delete(`${API_BASE}/models/${encodeURIComponent(modelId)}`)
      alert(`Successfully deleted model '${modelId}' from registry.`)
      if (monitorModelId === modelId) {
        setMonitorModelId('')
      }
      await fetchAllData()
    } catch (err) {
      console.error("Deletion failed:", err)
      alert(err.response?.data?.detail || "Failed to delete model. Verify server connection.")
    } finally {
      setLoading(false)
    }
  }

  // 5. Interactive sandbox submit
  const handleSandboxPredict = async (e) => {
    e.preventDefault()
    if (!sandboxModelId) return

    // Check if selected model is ACTIVE — only active models can serve predictions
    const selectedModel = registeredModels.find(m => m.model_id === sandboxModelId)
    if (selectedModel && selectedModel.deployment_status !== 'ACTIVE') {
      alert(`Model '${selectedModel.model_name}' is currently INACTIVE.\n\nGo to Model Registry → click "Deploy" to activate it first, then retry sandbox prediction.`)
      return
    }

    // Validate all fields are filled with real numeric values
    const emptyFields = Object.keys(sandboxFeatures).filter(k => sandboxFeatures[k] === '' || sandboxFeatures[k] === null || sandboxFeatures[k] === undefined)
    if (emptyFields.length > 0) {
      alert(`Please fill in all feature fields before running prediction.\n\nMissing: ${emptyFields.join(', ')}\n\nTip: Use "Fill Normal Sample" to auto-populate all fields.`)
      return
    }

    // Cast all inputs to proper numbers (input[type=number] returns strings)
    const processedFeatures = {}
    Object.keys(sandboxFeatures).forEach(k => {
      const val = sandboxFeatures[k]
      const num = parseFloat(val)
      processedFeatures[k] = isNaN(num) ? 0 : num
    })

    setSandboxLoading(true)
    const doPredict = async (isRetry = false) => {
      try {
        const res = await axios.post(
          `${API_BASE}/predict/${encodeURIComponent(sandboxModelId)}`,
          { features: processedFeatures },
          { headers: { 'X-Request-Source': 'sandbox' } }
        )
        setSandboxResult(res.data)
        const data = res.data
        const newPrediction = {
          id: data.request_id,
          model_id: data.model_id,
          timestamp: new Date().toLocaleTimeString(),
          features: JSON.stringify(processedFeatures),
          prediction: data.prediction,
          confidence: data.confidence !== null ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A',
          latency: `${data.latency_ms.toFixed(1)}ms`,
          latencyValue: data.latency_ms,
          drifted: false
        }
        setPredictionHistory(prev => [newPrediction, ...prev].slice(0, 15))
      } catch (err) {
        console.error('Sandbox predict failed:', err)
        if (err.response?.status === 429 && !isRetry) {
          setSandboxResult({ _retrying: true })
          setTimeout(() => doPredict(true), 2000)
          return
        }
        setSandboxResult(null)
        if (err.response?.status === 404) {
          alert(`Model '${sandboxModelId}' is not currently active.\n\nGo to Model Registry and Deploy it first.`)
        } else if (err.response?.status === 429) {
          alert('Server is rate-limited. Please wait 5 seconds and try again.')
        } else {
          alert(err.response?.data?.detail || 'Inference failed. Check all feature values are valid numbers.')
        }
      } finally {
        setSandboxLoading(false)
      }
    }
    doPredict()
  }

  // 6. Handle model file upload
  const handleUploadModelSubmit = async (e) => {
    e.preventDefault()
    setUploadMessage({ type: '', text: '' })

    // Validate model_id: no spaces or special characters
    const modelIdRaw = uploadFormData.model_id.trim()
    if (!modelIdRaw) {
      setUploadMessage({ type: 'error', text: 'Model ID is required.' })
      return
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(modelIdRaw)) {
      setUploadMessage({ type: 'error', text: 'Model ID must only contain letters, numbers, underscores (_) or hyphens (-). No spaces allowed.' })
      return
    }

    if (!uploadFile) {
      setUploadMessage({ type: 'error', text: 'Please select a serialized model (.pkl or .joblib) file.' })
      return
    }

    // Validate file extension on client side
    const fname = uploadFile.name.toLowerCase()
    if (!fname.endsWith('.pkl') && !fname.endsWith('.joblib')) {
      setUploadMessage({ type: 'error', text: `Unsupported file: "${uploadFile.name}". Only .pkl and .joblib files are accepted.` })
      return
    }

    // Convert comma-separated features to JSON list
    const featuresList = uploadFormData.features
      .split(',')
      .map(f => f.trim())
      .filter(f => f !== '')

    if (featuresList.length === 0) {
      setUploadMessage({ type: 'error', text: 'Please declare at least one feature name in the schema (comma-separated).' })
      return
    }

    // Check for duplicate feature names
    const uniqueFeatures = new Set(featuresList)
    if (uniqueFeatures.size !== featuresList.length) {
      setUploadMessage({ type: 'error', text: 'Feature list contains duplicate names. Each feature must be unique.' })
      return
    }

    const versionTag = uploadFormData.version.trim() || 'v1'

    const payload = new FormData()
    payload.append('model_id', modelIdRaw)
    payload.append('model_name', uploadFormData.model_name.trim())
    payload.append('version', versionTag)
    payload.append('framework', uploadFormData.framework.trim() || 'scikit-learn')
    payload.append('task_type', uploadFormData.task_type)
    payload.append('features', JSON.stringify(featuresList))
    payload.append('file', uploadFile)

    setLoading(true)
    setUploadMessage({ type: '', text: '' })
    try {
      await axios.post(`${API_BASE}/models/upload`, payload, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setUploadMessage({ type: 'success', text: `✓ Model "${uploadFormData.model_name || modelIdRaw}" (${versionTag}) registered successfully!` })
      setUploadFormData({
        model_id: '',
        model_name: '',
        version: 'v1',
        framework: 'scikit-learn',
        task_type: 'classification',
        features: ''
      })
      setUploadFile(null)
      document.getElementById('file-upload-input').value = ''
      await fetchAllData()
    } catch (err) {
      console.error('Upload error:', err)
      const detail = err.response?.data?.detail
      if (err.response?.status === 400) {
        setUploadMessage({ type: 'error', text: `Validation error: ${detail}` })
      } else if (err.response?.status === 500) {
        setUploadMessage({ type: 'error', text: `Server error: ${detail || 'Upload failed. Check the backend logs.'}` })
      } else {
        setUploadMessage({ type: 'error', text: detail || 'Upload failed. Ensure the file is a valid scikit-learn .pkl or .joblib model.' })
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="portal-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div style={{ marginBottom: '1rem' }}>
          <h1 className="vibrant-text" style={{ fontSize: '1.45rem', letterSpacing: '-0.04em', lineHeight: '1.2' }}>MLOps Observer</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.70rem' }}>Model Observability Portal v2.0</p>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <NavItem
            id="overview"
            label="Overview"
            icon={<Activity size={18} />}
            active={activeTab === 'overview'}
            onClick={setActiveTab}
          />
          <NavItem
            id="registry"
            label="Model Registry"
            icon={<FolderOpen size={18} />}
            active={activeTab === 'registry'}
            onClick={setActiveTab}
          />
          <NavItem
            id="alerts"
            label="Alerts Center"
            icon={<AlertCircle size={18} />}
            active={activeTab === 'alerts'}
            onClick={setActiveTab}
          />
          <NavItem
            id="monitoring"
            label="Grafana Visualizer"
            icon={<BarChart3 size={18} />}
            active={activeTab === 'monitoring'}
            onClick={setActiveTab}
          />
        </nav>

        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {/* AWS Connection Settings Panel */}
          <div className="glass-card" style={{ padding: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-muted)' }}>AWS BACKEND LINK</span>
              <button
                onClick={() => setShowSettings(!showSettings)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--accent-primary)',
                  fontSize: '0.65rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  padding: 0,
                  textDecoration: 'underline'
                }}
              >
                {showSettings ? 'Close' : 'Configure'}
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span className="status-indicator" style={{
                background: connectionStatus === 'connected' ? 'var(--accent-success)' : connectionStatus === 'checking' ? 'var(--accent-warning)' : 'var(--accent-error)',
                boxShadow: connectionStatus === 'connected' ? '0 0 8px var(--accent-success)' : 'none',
                color: connectionStatus === 'connected' ? 'var(--accent-success)' : connectionStatus === 'checking' ? 'var(--accent-warning)' : 'var(--accent-error)'
              }}></span>
              <span style={{ fontSize: '0.75rem', fontWeight: 700 }}>
                {connectionStatus === 'connected'
                  ? (backendHost === 'localhost' ? 'Local Host (5173)' : 'AWS Connected')
                  : connectionStatus === 'checking' ? 'Verifying Link...' : 'Offline (Disconnected)'}
              </span>
            </div>

            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              Host: {backendHost === 'localhost' ? 'Local System' : (backendHost || 'Not Configured')}
            </span>

            {showSettings && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                  borderTop: '1px solid var(--glass-border)',
                  paddingTop: '0.5rem',
                  marginTop: '0.25rem'
                }}
              >
                <label style={{ fontSize: '0.6rem', fontWeight: 700, color: 'var(--text-muted)' }}>AWS PUBLIC IP OR HOSTNAME</label>
                <input
                  type="text"
                  value={tempHost}
                  onChange={(e) => setTempHost(e.target.value)}
                  placeholder="e.g. 54.210.12.34"
                  style={{
                    padding: '0.35rem 0.5rem',
                    borderRadius: '6px',
                    border: '1px solid var(--glass-border)',
                    background: 'rgba(0,0,0,0.4)',
                    color: 'white',
                    fontSize: '0.75rem'
                  }}
                />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem' }}>
                  <button
                    type="button"
                    onClick={() => {
                      setBackendHost('localhost')
                      setTempHost('')
                      localStorage.removeItem('MLOPS_AWS_IP')
                      setShowSettings(false)
                    }}
                    className="btn-premium"
                    style={{ padding: '0.3rem', fontSize: '0.65rem', justifyContent: 'center' }}
                  >
                    Use Local
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const hostValue = tempHost.trim()
                      if (hostValue) {
                        setBackendHost(hostValue)
                        localStorage.setItem('MLOPS_AWS_IP', hostValue)
                      }
                      setShowSettings(false)
                    }}
                    className="btn-premium"
                    style={{ padding: '0.3rem', fontSize: '0.65rem', justifyContent: 'center' }}
                  >
                    Save & Test
                  </button>
                </div>
              </motion.div>
            )}
          </div>

          <div className="glass-card" style={{ padding: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.35rem' }}>
              <span className="status-indicator" style={{
                background: stats.active_models_count > 0 ? 'var(--accent-success)' : 'var(--accent-warning)',
                color: stats.active_models_count > 0 ? 'var(--accent-success)' : 'var(--accent-warning)'
              }}></span>
              <span style={{ fontSize: '0.65rem', fontWeight: 700 }}>GATEWAY: ACTIVE</span>
            </div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              {stats.active_models_count} Active Models
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="bento-grid"
            >
              {/* Header */}
              <motion.div layout style={{ gridColumn: 'span 12', marginBottom: '0.5rem' }}>
                <h2 style={{ fontSize: '2.1rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Observability Dashboard</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Real-time tabular model performance telemetry, statistical data drift detection, and auto-healing workflows.</p>
              </motion.div>

              {/* Metric Cards */}
              <motion.section layout style={{ gridColumn: 'span 12', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem' }}>
                <MetricCard title="Throughput" value={stats.total_predictions} icon={<Zap size={22} />} color="var(--accent-primary)" subtitle="Total Inference Pulses" />
                <MetricCard title="Avg Latency" value={`${stats.avg_latency_ms.toFixed(1)}ms`} icon={<Cpu size={22} />} color="var(--accent-secondary)" subtitle="Relational Session Average" />
                <MetricCard title="Active Models" value={stats.active_models_count} icon={<Layers size={22} />} color="var(--accent-success)" subtitle="Currently Deployed" />
                <MetricCard title="Unresolved Alerts" value={stats.unresolved_alerts_count} icon={<AlertCircle size={22} />} color={stats.unresolved_alerts_count > 0 ? "var(--accent-error)" : "var(--accent-muted)"} subtitle="Warnings & Drift Alarms" />
              </motion.section>

              {/* Active Model Grid */}
              <motion.div layout style={{ gridColumn: 'span 12' }}>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', fontWeight: 600 }}>Deployed Inference Endpoints</h3>
                {stats.models.length > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.25rem' }}>
                    {stats.models.map(m => (
                      <div className="glass-card" key={m.model_id} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.70rem', fontWeight: 700, background: 'rgba(255,255,255,0.06)', padding: '0.25rem 0.5rem', borderRadius: '0.5rem', color: 'var(--accent-primary)' }}>
                            {m.framework.toUpperCase()}
                          </span>
                          <span style={{ fontSize: '0.65rem', fontWeight: 700, color: m.drift_status === 'LOW' ? 'var(--accent-success)' : m.drift_status === 'WARNING' ? 'var(--accent-warning)' : 'var(--accent-error)' }}>
                            DRIFT: {m.drift_status}
                          </span>
                        </div>
                        <div>
                          <h4 style={{ fontSize: '1.15rem', fontWeight: 700 }}>{m.model_name}</h4>
                          <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>ID: {m.model_id} | Version: {m.version}</p>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', borderTop: '1px solid var(--glass-border)', paddingTop: '0.75rem', marginTop: '0.25rem' }}>
                          <div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>THROUGHPUT</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 700 }}>{m.throughput} reqs</div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>AVG LATENCY</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 700 }}>{m.avg_latency.toFixed(1)}ms</div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>TASK TYPE</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 700, textTransform: 'capitalize' }}>{m.task_type}</div>
                          </div>
                        </div>

                        <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                          <button
                            className="btn-premium"
                            style={{ padding: '0.4rem 0.8rem', fontSize: '0.7rem', width: '50%', justifyContent: 'center' }}
                            onClick={() => handleForceRetrain(m.model_id)}
                          >
                            <RefreshCcw size={12} /> Retrain
                          </button>
                          <button
                            className="btn-premium"
                            style={{
                              padding: '0.4rem 0.8rem',
                              fontSize: '0.7rem',
                              width: '50%',
                              justifyContent: 'center',
                              borderColor: 'rgba(129, 140, 248, 0.4)',
                              color: 'var(--accent-primary)'
                            }}
                            onClick={() => {
                              setMonitorModelId(m.model_id)
                              setActiveTab('monitoring')
                            }}
                          >
                            <BarChart3 size={12} /> Monitor
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="glass-card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No deployed models active. Go to the **Model Registry** tab to upload and deploy models.
                  </div>
                )}
              </motion.div>

              {/* Performance Chart & Controls */}
              <motion.div layout className="glass-card" style={{ gridColumn: 'span 8' }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: 600 }}>Real-time Gateway Latencies</h3>
                <div style={{ height: '220px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={dynamicChartData.length > 0 ? dynamicChartData : chartData}>
                      <defs>
                        <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.35} />
                          <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="name" hide />
                      <YAxis stroke="var(--text-muted)" fontSize={10} />
                      <Tooltip contentStyle={{ background: '#0a0a0c', border: '1px solid var(--glass-border)', borderRadius: '8px', fontSize: '12px' }} />
                      <Area type="monotone" dataKey="latency" stroke="var(--accent-primary)" fillOpacity={1} fill="url(#colorLatency)" strokeWidth={2.5} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Simulation Controls */}
              <motion.div layout className="glass-card" style={{ gridColumn: 'span 4', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Drift Injection Control</h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>In-Browser Gateway Simulator</span>
                  <button
                    className="magic-button"
                    onClick={() => setIsSimulatingTraffic(!isSimulatingTraffic)}
                    style={{ height: '2.6rem' }}
                  >
                    <span className="magic-button-bg"></span>
                    <span className="magic-button-content" style={{ fontSize: '0.75rem', padding: '0.5rem 1rem' }}>
                      {isSimulatingTraffic ? <StopCircle size={14} color="var(--accent-error)" /> : <PlaySquare size={14} color="var(--accent-primary)" />}
                      {isSimulatingTraffic ? 'Halt Simulation' : 'Launch Simulation'}
                    </span>
                  </button>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: 'auto' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Inject Feature Skew (Drift)</span>
                  <button
                    className={`btn-premium ${isSimulatingDrift ? 'active' : ''}`}
                    style={{ justifyContent: 'center', fontSize: '0.8rem', padding: '0.5rem 1rem' }}
                    disabled={!isSimulatingTraffic}
                    onClick={() => setIsSimulatingDrift(!isSimulatingDrift)}
                  >
                    <AlertTriangle size={14} /> {isSimulatingDrift ? 'Injecting Skewed Data...' : 'Inject Out-of-Distribution Drift'}
                  </button>
                  {isSimulatingTraffic && (
                    <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                      {isSimulatingDrift ? '⚠️ Skewing feature limits (e.g. 10x price/charges)' : '🟢 Generating clean baseline transaction logs'}
                    </p>
                  )}
                </div>
              </motion.div>

              {/* Interactive Prediction Sandbox */}
              {registeredModels.length > 0 && (
                <motion.div layout className="glass-card" style={{ gridColumn: 'span 12' }}>
                  <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: 600 }}>Interactive Prediction Sandbox</h3>
                  <div style={{ display: 'flex', gap: '1.5rem' }}>
                    <form onSubmit={handleSandboxPredict} style={{ flex: 1.2, display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                        <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>SELECT MODEL:</label>
                        <select
                          value={sandboxModelId}
                          onChange={(e) => setSandboxModelId(e.target.value)}
                          style={{
                            flex: 1,
                            padding: '0.5rem',
                            borderRadius: '8px',
                            background: 'rgba(0,0,0,0.4)',
                            border: '1px solid var(--glass-border)',
                            color: 'white',
                            fontSize: '0.85rem'
                          }}
                        >
                          <option value="" disabled>-- Choose a model --</option>
                          {registeredModels.map(m => (
                            <option key={m.model_id} value={m.model_id}>
                              {m.model_name} ({m.version}) {m.deployment_status === 'ACTIVE' ? '🟢 Active' : '⚪ Inactive'}
                            </option>
                          ))}
                        </select>
                      </div>

                      {/* Inline warning if selected model is INACTIVE */}
                      {sandboxModelId && (() => {
                        const sel = registeredModels.find(m => m.model_id === sandboxModelId)
                        return sel && sel.deployment_status !== 'ACTIVE' ? (
                          <div style={{
                            display: 'flex', alignItems: 'center', gap: '0.5rem',
                            padding: '0.6rem 0.85rem',
                            borderRadius: '8px',
                            background: 'rgba(251, 191, 36, 0.1)',
                            border: '1px solid rgba(251, 191, 36, 0.35)',
                            color: '#fbbf24',
                            fontSize: '0.78rem',
                            fontWeight: 600
                          }}>
                            <AlertTriangle size={14} />
                            This model is <strong style={{ marginLeft: '0.25rem', marginRight: '0.25rem' }}>INACTIVE</strong> — go to Model Registry to Deploy it before running inference.
                          </div>
                        ) : null
                      })()}

                      {sandboxModelId && Object.keys(sandboxFeatures).length > 0 ? (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.75rem', background: 'rgba(0,0,0,0.15)', padding: '1rem', borderRadius: '1rem', border: '1px solid var(--glass-border)' }}>
                          {Object.keys(sandboxFeatures).map(feature => (
                            <div key={feature} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                              <label style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{feature.replace(/_/g, ' ')}</label>
                              <input
                                type="number"
                                step="any"
                                value={sandboxFeatures[feature]}
                                onChange={(e) => setSandboxFeatures({ ...sandboxFeatures, [feature]: e.target.value })}
                                placeholder="0.0"
                                style={{
                                  padding: '0.4rem 0.6rem',
                                  borderRadius: '6px',
                                  border: '1px solid var(--glass-border)',
                                  background: 'rgba(0,0,0,0.3)',
                                  color: 'white',
                                  fontSize: '0.8rem'
                                }}
                                required
                              />
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.15)', borderRadius: '1rem', border: '1px dashed var(--glass-border)', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                          Select a model above to load its feature schema.
                        </div>
                      )}

                      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button
                            type="button"
                            className="btn-premium"
                            onClick={() => {
                              const target = registeredModels.find(m => m.model_id === sandboxModelId)
                              if (target) setSandboxFeatures(generateSampleInput(sandboxModelId, target.features, false))
                            }}
                            disabled={!sandboxModelId}
                            style={{ fontSize: '0.75rem', padding: '0.4rem 0.8rem' }}
                          >
                            <CheckCircle size={13} /> Fill Normal Sample
                          </button>
                        </div>
                        <button type="submit" className="btn-premium" disabled={sandboxLoading || !sandboxModelId || Object.keys(sandboxFeatures).length === 0} style={{ padding: '0.5rem 1.25rem', fontSize: '0.8rem' }}>
                          {sandboxLoading ? <RefreshCcw className="animate-spin" size={14} /> : <Zap size={14} />}
                          {sandboxLoading ? 'Running Inference...' : 'Run Prediction'}
                        </button>
                      </div>
                    </form>

                    <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      {sandboxResult?._retrying ? (
                        <div className="glass-card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', border: '1px solid rgba(251,191,36,0.3)', background: 'rgba(251,191,36,0.05)' }}>
                          <RefreshCcw size={20} className="animate-spin" style={{ color: '#fbbf24' }} />
                          <p style={{ color: '#fbbf24', fontSize: '0.8rem', fontWeight: 600, textAlign: 'center' }}>
                            Rate limit hit — auto-retrying in 2s...
                          </p>
                          <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', textAlign: 'center' }}>
                            Your request is queued and will complete automatically.
                          </p>
                        </div>
                      ) : sandboxResult && sandboxResult.prediction !== undefined ? (
                        <div className="glass-card" style={{ background: 'rgba(0, 0, 0, 0.35)', border: '1px solid rgba(129, 140, 248, 0.25)', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '0.5rem' }}>
                          <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase' }}>Inference Success ✓</div>

                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>PREDICTION OUTCOME:</span>
                            <span style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent-success)' }}>
                              {sandboxResult.prediction}
                            </span>
                          </div>

                          {sandboxResult.confidence !== null && sandboxResult.confidence !== undefined && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>CONFIDENCE SCORE:</span>
                              <span style={{ fontSize: '1rem', fontWeight: 700 }}>
                                {(sandboxResult.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}

                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--glass-border)', paddingTop: '0.5rem', marginTop: '0.25rem' }}>
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>LATENCY:</span>
                            <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>{sandboxResult.latency_ms.toFixed(1)}ms</span>
                          </div>

                          <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                            Session UUID: {sandboxResult.request_id.slice(0, 8)}...
                          </div>
                        </div>
                      ) : (
                        <div className="glass-card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', borderStyle: 'dashed', borderColor: 'var(--glass-border)', background: 'transparent' }}>
                          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center' }}>
                            Fill in features and click Run Prediction to see results.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Log Stream */}
              <motion.div layout className="glass-card" style={{ gridColumn: 'span 12' }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: 600 }}>Live Prediction Stream</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ textAlign: 'left', color: 'var(--text-muted)', borderBottom: '1px solid var(--glass-border)' }}>
                        <th style={{ paddingBottom: '0.75rem' }}>TIMESTAMP</th>
                        <th>MODEL ID</th>
                        <th>INPUT FEATURES SCHEMA</th>
                        <th>OUTCOME</th>
                        <th>CONFIDENCE</th>
                        <th>LATENCY</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionHistory.length > 0 ? (
                        predictionHistory.map((row) => (
                          <tr key={row.id} className="row-pulse" style={{ borderBottom: '1px solid var(--glass-border)' }}>
                            <td style={{ padding: '0.75rem 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>{row.timestamp}</td>
                            <td style={{ fontWeight: 600 }}>{row.model_id}</td>
                            <td style={{ color: 'var(--text-muted)', maxWidth: '350px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
                              {row.features}
                            </td>
                            <td>
                              <span style={{
                                background: 'rgba(129, 140, 248, 0.1)',
                                color: 'var(--accent-primary)',
                                padding: '0.15rem 0.5rem',
                                borderRadius: '0.5rem',
                                fontSize: '0.75rem',
                                fontWeight: 700
                              }}>
                                {row.prediction}
                              </span>
                            </td>
                            <td style={{ fontWeight: 600 }}>{row.confidence}</td>
                            <td>{row.latency}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan="6" style={{ padding: '1.5rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>
                            Simulation off. Launch the Simulator or use Sandbox above.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'registry' && (
            <motion.div
              key="registry"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="bento-grid"
            >
              {/* Header */}
              <motion.div layout style={{ gridColumn: 'span 12', marginBottom: '0.5rem' }}>
                <h2 style={{ fontSize: '2.1rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Model Registry</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Live registry from SQLite + <code style={{ fontSize: '0.85em' }}>models/</code> on disk.
                  Deployed artifact is set in <code style={{ fontSize: '0.85em' }}>active_model.txt</code>.
                  {' '}
                  <strong style={{ color: 'var(--accent-primary)' }}>{registryVersions.length}</strong> version(s) across{' '}
                  <strong style={{ color: 'var(--accent-primary)' }}>{registeredModels.length}</strong> model(s).
                </p>
              </motion.div>

              {/* Upload Form */}
              <motion.div layout className="glass-card" style={{ gridColumn: 'span 5' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', fontWeight: 600 }}>Register & Upload Model</h3>
                <form onSubmit={handleUploadModelSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>MODEL ID (unique, no spaces)</label>
                    <input
                      type="text"
                      value={uploadFormData.model_id}
                      onChange={(e) => setUploadFormData({ ...uploadFormData, model_id: e.target.value })}
                      placeholder="e.g. credit_fraud_detector"
                      required
                      style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)', color: 'white', fontSize: '0.8rem' }}
                    />
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>MODEL DISPLAY NAME</label>
                    <input
                      type="text"
                      value={uploadFormData.model_name}
                      onChange={(e) => setUploadFormData({ ...uploadFormData, model_name: e.target.value })}
                      placeholder="e.g. Fraud Detector v1"
                      required
                      style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)', color: 'white', fontSize: '0.8rem' }}
                    />
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>VERSION TAG</label>
                      <input
                        type="text"
                        value={uploadFormData.version}
                        onChange={(e) => setUploadFormData({ ...uploadFormData, version: e.target.value })}
                        placeholder="v1"
                        required
                        style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)', color: 'white', fontSize: '0.8rem' }}
                      />
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>TASK TYPE</label>
                      <select
                        value={uploadFormData.task_type}
                        onChange={(e) => setUploadFormData({ ...uploadFormData, task_type: e.target.value })}
                        style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)', color: 'white', fontSize: '0.8rem' }}
                      >
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                      </select>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>FRAMEWORK</label>
                    <input
                      type="text"
                      value={uploadFormData.framework}
                      onChange={(e) => setUploadFormData({ ...uploadFormData, framework: e.target.value })}
                      placeholder="scikit-learn"
                      required
                      style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)', color: 'white', fontSize: '0.8rem' }}
                    />
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>SCHEMA FEATURES (comma-separated list)</label>
                    <input
                      type="text"
                      value={uploadFormData.features}
                      onChange={(e) => setUploadFormData({ ...uploadFormData, features: e.target.value })}
                      placeholder="amount, distance, is_international"
                      required
                      style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)', color: 'white', fontSize: '0.8rem' }}
                    />
                    <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>Declare the exact names expected by your model in order.</span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.25rem' }}>
                    <label style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>MODEL BINARY FILE (.pkl, .joblib)</label>
                    <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px dashed var(--glass-border)', borderRadius: '10px', padding: '1rem', background: 'rgba(0,0,0,0.15)', cursor: 'pointer' }}>
                      <input
                        id="file-upload-input"
                        type="file"
                        accept=".pkl,.joblib"
                        onChange={(e) => setUploadFile(e.target.files[0])}
                        style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
                        required
                      />
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.25rem' }}>
                        <UploadCloud size={24} style={{ color: 'var(--accent-primary)' }} />
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-main)', fontWeight: 600 }}>
                          {uploadFile ? uploadFile.name : 'Select binary file'}
                        </span>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>Drag & drop or browse</span>
                      </div>
                    </div>
                  </div>

                  <button type="submit" className="btn-premium" disabled={loading || registeredModels.some(m => m.model_id === uploadFormData.model_id && m.version === uploadFormData.version)} style={{ justifyContent: 'center', width: '100%', marginTop: '0.5rem', padding: '0.6rem' }}>
                    {loading ? <RefreshCcw className="animate-spin" size={16} /> : <PlusCircle size={16} />}
                    {loading ? 'Uploading...' : 'Upload & Register Model'}
                  </button>

                  {registryVersions.some(m => m.model_id === uploadFormData.model_id && m.version === uploadFormData.version) && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--accent-warning)', textAlign: 'center' }}>
                      Warning: A model with this ID and version already exists in the registry.
                    </div>
                  )}

                  {uploadMessage.text && (
                    <div style={{
                      padding: '0.5rem',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      textAlign: 'center',
                      background: uploadMessage.type === 'success' ? 'rgba(52, 211, 153, 0.1)' : 'rgba(248, 113, 113, 0.1)',
                      color: uploadMessage.type === 'success' ? 'var(--accent-success)' : 'var(--accent-error)',
                      border: `1px solid ${uploadMessage.type === 'success' ? 'rgba(52, 211, 153, 0.2)' : 'rgba(248, 113, 113, 0.2)'}`
                    }}>
                      {uploadMessage.text}
                    </div>
                  )}
                </form>
              </motion.div>

              {/* Present registry — all versions from API */}
              <motion.div layout className="glass-card" style={{ gridColumn: 'span 7' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '1.25rem', fontWeight: 600 }}>Registered Versions (live)</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                    <thead>
                      <tr style={{ textAlign: 'left', color: 'var(--text-muted)', borderBottom: '1px solid var(--glass-border)' }}>
                        <th style={{ paddingBottom: '0.75rem' }}>MODEL</th>
                        <th>VER</th>
                        <th>FILE</th>
                        <th>STATUS</th>
                        <th>ACCURACY</th>
                        <th>TRAINED</th>
                        <th>ACTION</th>
                      </tr>
                    </thead>
                    <tbody>
                      {registryVersions.length > 0 ? (
                        registryVersions.map((v, idx) => {
                          const catalog = registeredModels.find((m) => m.model_id === v.model_id)
                          const showDelete = idx === 0 || registryVersions[idx - 1]?.model_id !== v.model_id
                          return (
                            <tr key={`${v.model_id}-${v.version}`} style={{ borderBottom: '1px solid var(--glass-border)' }}>
                              <td style={{ padding: '0.85rem 0', fontWeight: 600, maxWidth: '140px' }}>
                                {v.model_name}
                                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 500 }}>{v.model_id}</div>
                                {v.is_live_file && (
                                  <div style={{ fontSize: '0.6rem', color: 'var(--accent-primary)', marginTop: '0.2rem' }}>
                                    active_model.txt
                                  </div>
                                )}
                              </td>
                              <td>{v.version}</td>
                              <td style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>{v.filename}</td>
                              <td>
                                <span style={{
                                  background: v.status === 'ACTIVE' ? 'rgba(52, 211, 153, 0.1)' : 'rgba(255,255,255,0.04)',
                                  color: v.status === 'ACTIVE' ? 'var(--accent-success)' : 'var(--text-muted)',
                                  padding: '0.15rem 0.5rem',
                                  borderRadius: '2rem',
                                  fontSize: '0.65rem',
                                  fontWeight: 700
                                }}>
                                  {v.status}
                                </span>
                              </td>
                              <td style={{ fontSize: '0.75rem' }}>
                                {v.accuracy != null ? Number(v.accuracy).toFixed(3) : '—'}
                              </td>
                              <td style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                                {v.trained_at ? new Date(v.trained_at).toLocaleString() : '—'}
                              </td>
                              <td style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', padding: '0.85rem 0', flexWrap: 'wrap' }}>
                                {v.status !== 'ACTIVE' ? (
                                  <button
                                    className="btn-premium"
                                    style={{ padding: '0.2rem 0.5rem', fontSize: '0.65rem' }}
                                    disabled={loading}
                                    onClick={() => handleDeployModel(v.model_id, v.version)}
                                  >
                                    Deploy
                                  </button>
                                ) : (
                                  <span style={{ fontSize: '0.7rem', color: 'var(--accent-success)', fontWeight: 600 }}>Live</span>
                                )}
                                {catalog && (
                                  <button
                                    type="button"
                                    className="btn-premium"
                                    style={{
                                      padding: '0.2rem 0.4rem',
                                      fontSize: '0.65rem',
                                      borderColor: 'rgba(52, 211, 153, 0.3)',
                                      color: 'var(--accent-success)',
                                    }}
                                    disabled={loading}
                                    onClick={() => window.open(`${API_BASE}/models/${encodeURIComponent(v.model_id)}/download`, '_blank')}
                                    title="Download model metadata JSON"
                                  >
                                    <DownloadCloud size={12} />
                                  </button>
                                )}
                                {showDelete && (
                                  <button
                                    type="button"
                                    className="btn-premium"
                                    style={{
                                      padding: '0.2rem 0.4rem',
                                      fontSize: '0.65rem',
                                      borderColor: 'rgba(239, 68, 68, 0.3)',
                                      color: 'var(--accent-error)',
                                    }}
                                    disabled={loading}
                                    onClick={() => handleDeleteModel(v.model_id)}
                                    title="Delete entire model and all versions"
                                  >
                                    <Trash2 size={12} />
                                  </button>
                                )}
                              </td>
                            </tr>
                          )
                        })
                      ) : (
                        <tr>
                          <td colSpan="7" style={{ padding: '2rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>
                            No models registered yet. Upload a <code>.joblib</code> file to get started.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                {registeredModels.length > 0 && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--glass-border)', fontSize: '0.75rem' }}>
                    <div style={{ fontWeight: 700, marginBottom: '0.35rem', color: 'var(--text-muted)' }}>ACTIVE DEPLOYMENTS</div>
                    {registeredModels.filter((m) => m.deployment_status === 'ACTIVE').map((m) => (
                      <div key={m.model_id} style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.25rem' }}>
                        <span style={{ fontWeight: 600 }}>{m.model_id}</span>
                        <span style={{ color: 'var(--text-muted)' }}>→</span>
                        <span style={{ color: 'var(--accent-success)' }}>{m.filename || `model_${m.version}.joblib`}</span>
                        <span style={{ color: 'var(--text-muted)' }}>({m.version})</span>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'alerts' && (
            <motion.div
              key="alerts"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="bento-grid"
            >
              {/* Header */}
              <motion.div layout style={{ gridColumn: 'span 12', marginBottom: '0.5rem' }}>
                <h2 style={{ fontSize: '2.1rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Alerts Center</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Real-time audit log of statistical drift violations, database warnings, and auto-healing events.</p>
              </motion.div>

              {/* Alerts Log */}
              <motion.div layout className="glass-card" style={{ gridColumn: 'span 12' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '1.25rem', fontWeight: 600 }}>Operational Alerts Log</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  {alerts.length > 0 ? (
                    alerts.map(a => (
                      <div
                        key={a.id}
                        style={{
                          display: 'flex',
                          gap: '1rem',
                          padding: '1rem',
                          borderRadius: '1rem',
                          border: `1px solid ${a.severity === 'CRITICAL'
                            ? 'rgba(248, 113, 113, 0.2)'
                            : a.severity === 'WARNING'
                              ? 'rgba(251, 191, 36, 0.2)'
                              : 'rgba(52, 211, 153, 0.2)'
                            }`,
                          background: `linear-gradient(135deg, ${a.severity === 'CRITICAL'
                            ? 'rgba(248, 113, 113, 0.04)'
                            : a.severity === 'WARNING'
                              ? 'rgba(251, 191, 36, 0.04)'
                              : 'rgba(52, 211, 153, 0.04)'
                            }, rgba(0,0,0,0.2))`
                        }}
                      >
                        <div style={{ color: a.severity === 'CRITICAL' ? 'var(--accent-error)' : a.severity === 'WARNING' ? 'var(--accent-warning)' : 'var(--accent-success)' }}>
                          {a.severity === 'CRITICAL' ? <AlertTriangle size={24} /> : a.severity === 'WARNING' ? <AlertCircle size={24} /> : <CheckCircle size={24} />}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                            <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>
                              {a.alert_type.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                              {new Date(a.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <p style={{ fontSize: '0.8rem', color: 'var(--text-main)', marginBottom: '0.4rem' }}>{a.message}</p>
                          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                            <span>MODEL: {a.model_id}</span>
                            <span>•</span>
                            <span>VERSION: {a.version}</span>
                            <span>•</span>
                            <span style={{
                              color: a.resolved ? 'var(--accent-success)' : 'var(--accent-error)',
                              fontWeight: 700
                            }}>
                              {a.resolved ? 'RESOLVED (AUTO-HEALED)' : 'ACTIVE'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div style={{ padding: '3rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No drift or latency warnings registered. System is stable.
                    </div>
                  )}
                </div>
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'monitoring' && (
            <motion.div
              key="monitoring"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ height: '100%' }}
            >
              {isMixedContentBlocked ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  textAlign: 'center',
                  padding: '2rem',
                  background: 'rgba(0,0,0,0.25)',
                  border: '1px dashed var(--glass-border)',
                  borderRadius: '1.25rem'
                }}>
                  <div className="glass-card" style={{ maxWidth: '600px', padding: '2.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', alignItems: 'center' }}>
                    <div style={{
                      background: 'rgba(129, 140, 248, 0.1)',
                      color: 'var(--accent-primary)',
                      padding: '1rem',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <BarChart3 size={36} />
                    </div>
                    <div>
                      <h3 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '0.5rem' }}>Browser Security Policy Active</h3>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.5' }}>
                        To protect your sessions, web browsers restrict loading insecure HTTP frames (Grafana on port 3000) inside HTTPS portals like Vercel.
                      </p>
                    </div>
                    {monitorModelId && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--accent-primary)', fontWeight: 700, margin: '0.25rem 0' }}>
                        🎯 ACTIVE INDIVIDUAL FILTER: {monitorModelId}
                      </div>
                    )}
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--glass-border)', width: '100%' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>TARGET GRAFANA METRICS:</span>
                      <code style={{ fontSize: '0.8rem', color: 'var(--accent-secondary)', fontWeight: 600 }}>{grafanaUrl}</code>
                    </div>
                    <a
                      href={grafanaUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="magic-button"
                      style={{ height: '3rem', width: '100%', textDecoration: 'none' }}
                    >
                      <span className="magic-button-bg"></span>
                      <span className="magic-button-content" style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                        Open Grafana in New Secure Tab ↗
                      </span>
                    </a>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
                  {monitorModelId && (
                    <div className="glass-card" style={{ padding: '0.75rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(129, 140, 248, 0.05)', border: '1px solid rgba(129, 140, 248, 0.2)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <BarChart3 size={18} style={{ color: 'var(--accent-primary)' }} />
                        <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                          Currently Filtering Observability: <span className="vibrant-text" style={{ fontWeight: 800 }}>{monitorModelId}</span>
                        </span>
                      </div>
                      <button
                        className="btn-premium"
                        style={{ padding: '0.3rem 0.75rem', fontSize: '0.7rem' }}
                        onClick={() => setMonitorModelId('')}
                      >
                        Show All Models
                      </button>
                    </div>
                  )}
                  <iframe
                    src={grafanaUrl}
                    className="iframe-view"
                    title="Grafana Dashboard"
                    style={{ flex: 1, border: 'none', borderRadius: '12px' }}
                  />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

function NavItem({ id, label, icon, active, onClick }) {
  return (
    <div className={`nav-item ${active ? 'active' : ''}`} onClick={() => onClick(id)} style={{ fontSize: '0.85rem', padding: '0.8rem 1rem' }}>
      {icon}
      <span>{label}</span>
    </div>
  )
}

function MetricCard({ title, value, icon, color, subtitle }) {
  const [displayValue, setDisplayValue] = useState(value);

  useEffect(() => {
    if (typeof value === 'number') {
      let start = typeof displayValue === 'number' ? displayValue : 0;
      const end = value;
      if (start === end) return;

      const duration = 800;
      const startTime = performance.now();

      const updateValue = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 4);

        setDisplayValue(Math.floor(start + (end - start) * easeProgress));

        if (progress < 1) {
          requestAnimationFrame(updateValue);
        }
      };
      requestAnimationFrame(updateValue);
    } else {
      setDisplayValue(value);
    }
  }, [value]);

  return (
    <motion.div layout className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 700 }}>{title.toUpperCase()}</span>
        <div style={{ color: color }}>{icon}</div>
      </div>
      <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em' }}>{displayValue}</div>
      <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', fontWeight: 500 }}>{subtitle}</div>
    </motion.div>
  )
}

const chartData = [
  { name: '1', latency: 45 },
  { name: '2', latency: 55 },
  { name: '3', latency: 40 },
  { name: '4', latency: 85 },
  { name: '5', latency: 35 },
  { name: '6', latency: 60 },
  { name: '7', latency: 50 },
];

export default App
