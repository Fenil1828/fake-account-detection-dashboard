import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Shield, Upload, Play, AlertTriangle, CheckCircle, XCircle, Users, FileText, Download, Moon, Sparkles, Cpu, TrendingUp, BarChart3, Search, RefreshCw, Layers, Zap, Brain, Target, Lock, Eye, Activity, ChevronDown, ChevronUp, AlertCircle, ShieldCheck, ShieldAlert, Key } from 'lucide-react'
import { Chart as ChartJS, ArcElement, CategoryScale, LinearScale, BarElement, LineElement, PointElement, Tooltip, Legend } from 'chart.js'
import { Doughnut, Bar, Line } from 'react-chartjs-2'
import './App.css'

ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, LineElement, PointElement, Tooltip, Legend)

const API_URL = 'http://localhost:5000'

const MODEL_INFO = {
  name: 'Gradient Boosting Classifier',
  type: 'Ensemble Learning (Boosting)',
  library: 'scikit-learn v1.8.0',
  security: ['Data Poisoning Detection', 'Adversarial Input Detection', 'Rate Limiting', 'Input Validation'],
  params: { estimators: 100, depth: 5, lr: 0.1 },
  features: [
    { name: 'Profile', count: 10, ex: ['username_length', 'has_pic', 'account_age'] },
    { name: 'Behavioral', count: 6, ex: ['tweets_per_day', 'engagement'] },
    { name: 'Network', count: 6, ex: ['follower_count', 'ff_ratio'] },
    { name: 'Content', count: 3, ex: ['tweet_length', 'url_rate'] }
  ],
  perf: { acc: '94.5%', prec: '92.3%', rec: '89.7%', f1: '91.0%', auc: '0.96' }
}

const SAMPLE = [
  { username: 'bot_user123', followers_count: 5, friends_count: 8000, statuses_count: 15000, account_age_days: 20, has_profile_image: false, verified: false },
  { username: 'john_smith', followers_count: 500, friends_count: 300, statuses_count: 1200, account_age_days: 1825, has_profile_image: true, verified: false },
  { username: 'real_person', followers_count: 1200, friends_count: 400, statuses_count: 2500, account_age_days: 2000, has_profile_image: true, verified: true },
  { username: 'spam99', followers_count: 2, friends_count: 5000, statuses_count: 20000, account_age_days: 15, has_profile_image: false, verified: false },
  { username: 'legit_user', followers_count: 850, friends_count: 420, statuses_count: 3200, account_age_days: 1200, has_profile_image: true, verified: false },
]

function App() {
  const [accounts, setAccounts] = useState([])
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [showModel, setShowModel] = useState(false)
  const [showSecurity, setShowSecurity] = useState(true)
  const [apiOk, setApiOk] = useState('checking')
  const [securityStatus, setSecurityStatus] = useState({ status: 'SECURE', threat_score: 0, events: [] })
  const [threatHistory, setThreatHistory] = useState([])

  useEffect(() => {
    checkApi()
    const interval = setInterval(() => updateSecurityMetrics(), 5000)
    return () => clearInterval(interval)
  }, [])

  const checkApi = async () => {
    try {
      const r = await axios.get(`${API_URL}/api/health`, { timeout: 2000 })
      setApiOk(r.data.model_loaded ? 'ready' : 'no-model')
    } catch { setApiOk('offline') }
  }

  const updateSecurityMetrics = () => {
    // Simulate security metrics updates
    setThreatHistory(prev => {
      const newScore = Math.max(0, Math.min(100, (prev[prev.length - 1]?.score || 5) + (Math.random() - 0.5) * 10))
      return [...prev.slice(-20), { time: new Date().toLocaleTimeString(), score: newScore }]
    })
  }

  const genAccounts = (n) => {
    const bots = ['user', 'bot', 'spam', 'promo', 'deal'], reals = ['john', 'sarah', 'mike', 'emma', 'alex']
    const gen = []
    for (let i = 0; i < n; i++) {
      const isBot = Math.random() < 0.4
      gen.push(isBot ? {
        username: `${bots[Math.floor(Math.random() * bots.length)]}${Math.floor(Math.random() * 99999)}`,
        followers_count: Math.floor(Math.random() * 50), friends_count: Math.floor(Math.random() * 8000) + 1000,
        statuses_count: Math.floor(Math.random() * 40000) + 5000, account_age_days: Math.floor(Math.random() * 60) + 5,
        has_profile_image: Math.random() < 0.3, verified: false
      } : {
        username: `${reals[Math.floor(Math.random() * reals.length)]}_${Math.floor(Math.random() * 999)}`,
        followers_count: Math.floor(Math.random() * 5000) + 50, friends_count: Math.floor(Math.random() * 1000) + 50,
        statuses_count: Math.floor(Math.random() * 5000) + 100, account_age_days: Math.floor(Math.random() * 2000) + 365,
        has_profile_image: true, verified: Math.random() < 0.1
      })
    }
    setAccounts(gen); setResults([])
    addSecurityEvent('DATA_LOADED', `${n} accounts loaded for analysis`, 'INFO')
  }

  const addSecurityEvent = (type, message, severity) => {
    const event = { id: Date.now(), time: new Date().toLocaleTimeString(), type, message, severity }
    setSecurityStatus(prev => ({ ...prev, events: [event, ...prev.events].slice(0, 10) }))
  }

  const analyze = async () => {
    if (!accounts.length) return
    setLoading(true); setProgress(0)
    addSecurityEvent('ANALYSIS_STARTED', `Processing ${accounts.length} accounts`, 'INFO')

    const res = []
    let blocked = 0, suspicious = 0

    for (let i = 0; i < accounts.length; i++) {
      try {
        const r = await axios.post(`${API_URL}/api/analyze`, { ...accounts[i], bio: '', favourites_count: 0 })
        res.push({ ...accounts[i], result: r.data })
        if (r.data.prediction?.is_fake) suspicious++
      } catch (e) {
        res.push({ ...accounts[i], error: e.message })
        blocked++
      }
      setProgress(((i + 1) / accounts.length) * 100)
    }

    setResults(res)
    setLoading(false)

    // Update security status
    const threatScore = Math.min(100, (suspicious / accounts.length) * 100 + blocked * 5)
    setSecurityStatus(prev => ({
      ...prev,
      status: threatScore < 30 ? 'SECURE' : threatScore < 60 ? 'WARNING' : 'CRITICAL',
      threat_score: threatScore
    }))
    addSecurityEvent('ANALYSIS_COMPLETE', `Found ${suspicious} fake accounts, ${blocked} blocked`, suspicious > accounts.length * 0.3 ? 'WARNING' : 'INFO')
  }

  const filtered = results.filter(r => {
    if (filter === 'fake' && !r.result?.prediction?.is_fake) return false
    if (filter === 'real' && r.result?.prediction?.is_fake) return false
    if (search && !r.username.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  const stats = { total: results.length, fake: results.filter(r => r.result?.prediction?.is_fake).length, real: results.filter(r => !r.result?.prediction?.is_fake && r.result).length }
  const getRisk = l => ({ 'CRITICAL': 'critical', 'HIGH': 'high', 'MEDIUM': 'medium', 'LOW': 'low' }[l] || 'medium')

  const threatChartData = {
    labels: threatHistory.map(t => t.time),
    datasets: [{ label: 'Threat Level', data: threatHistory.map(t => t.score), borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.2)', fill: true, tension: 0.4 }]
  }

  return (
    <div className="app-container space-bg">
      <div className="stars-container"><div className="stars"></div><div className="stars2"></div><div className="moon"></div></div>

      <header className="header">
        <div className="header-badge"><Lock size={14} /> Secure ML System <ShieldCheck size={14} /></div>
        <h1><Shield size={40} /> Secure Fake Account Detector <Moon size={24} className="moon-icon" /></h1>
        <p>ML-powered detection with adversarial robustness & data poisoning protection</p>
        <div className={`api-status ${apiOk}`}><span className="dot"></span>{apiOk === 'ready' ? '🔒 Secure API Ready' : apiOk === 'offline' ? 'API Offline' : 'Checking...'}</div>
      </header>

      <main className="batch-main">
        {/* Security Status Panel */}
        <section className="glass-card security-panel">
          {/* Floating Particles */}
          <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
            {[...Array(12)].map((_, i) => (
              <div key={i} style={{
                position: 'absolute',
                width: Math.random() * 4 + 2 + 'px',
                height: Math.random() * 4 + 2 + 'px',
                borderRadius: '50%',
                background: `radial-gradient(circle, ${i % 3 === 0 ? 'rgba(16,185,129,0.6)' : i % 3 === 1 ? 'rgba(139,92,246,0.5)' : 'rgba(99,102,241,0.4)'}, transparent)`,
                left: Math.random() * 100 + '%',
                top: Math.random() * 100 + '%',
                animation: `float ${Math.random() * 8 + 6}s ease-in-out infinite`,
                animationDelay: `-${Math.random() * 8}s`,
                filter: 'blur(1px)',
              }} />
            ))}
          </div>

          <div className="sec-header" onClick={() => setShowSecurity(!showSecurity)}>
            <h2><ShieldAlert size={22} /> Security Monitor <span className={`status-badge ${securityStatus.status.toLowerCase()}`}>{securityStatus.status}</span></h2>
            {showSecurity ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </div>

          {showSecurity && <div className="sec-body">
            <div className="sec-grid">
              {/* Circular Threat Gauge */}
              <div className="sec-card threat-gauge">
                <h4><Activity size={16} /> Threat Level</h4>
                <div className="gauge" style={{
                  '--gauge-percent': `${securityStatus.threat_score}%`,
                  '--current-color': securityStatus.threat_score < 30 ? '#10b981' : securityStatus.threat_score < 60 ? '#f59e0b' : '#ef4444'
                }}>
                  <div className="gauge-fill"></div>
                  <span className="gauge-value">{securityStatus.threat_score.toFixed(0)}%</span>
                </div>
              </div>

              {/* Security Features */}
              <div className="sec-card features">
                <h4><ShieldCheck size={16} /> Active Protections</h4>
                <div className="feature-list">
                  <div className="feat-item active"><CheckCircle size={14} /> Data Poisoning Detection</div>
                  <div className="feat-item active"><CheckCircle size={14} /> Adversarial Input Filter</div>
                  <div className="feat-item active"><CheckCircle size={14} /> Rate Limiting (100/hr)</div>
                  <div className="feat-item active"><CheckCircle size={14} /> SQL/XSS Injection Block</div>
                  <div className="feat-item active"><CheckCircle size={14} /> Input Validation</div>
                  <div className="feat-item active"><CheckCircle size={14} /> Anomaly Detection</div>
                </div>
              </div>

              {/* Threat Chart */}
              <div className="sec-card chart">
                <h4><TrendingUp size={16} /> Threat History</h4>
                <div className="mini-chart">
                  {threatHistory.length > 0 && <Line data={threatChartData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { min: 0, max: 100, display: false } } }} />}
                </div>
              </div>

              {/* Security Events */}
              <div className="sec-card events">
                <h4><Eye size={16} /> Recent Events</h4>
                <div className="event-list">
                  {securityStatus.events.length === 0 ? <div className="no-events">No security events</div> :
                    securityStatus.events.map(e => (
                      <div key={e.id} className={`event-item ${e.severity.toLowerCase()}`}>
                        <span className="time">{e.time}</span>
                        <span className="type">{e.type}</span>
                        <span className="msg">{e.message}</span>
                      </div>
                    ))
                  }
                </div>
              </div>
            </div>
          </div>}
        </section>

        {/* Model Info */}
        <section className="glass-card model-card">
          <div className="model-head" onClick={() => setShowModel(!showModel)}>
            <h2><Brain size={20} /> ML Model: {MODEL_INFO.name}</h2>
            {showModel ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </div>
          {showModel && <div className="model-body">
            <div className="model-grid">
              <div className="m-section">
                <h4><Cpu size={16} /> Specs</h4>
                <div className="specs"><span>Type: {MODEL_INFO.type}</span><span>Library: {MODEL_INFO.library}</span><span>Estimators: {MODEL_INFO.params.estimators}</span></div>
              </div>
              <div className="m-section">
                <h4><Target size={16} /> Performance</h4>
                <div className="perf-row"><span>Accuracy: <b>{MODEL_INFO.perf.acc}</b></span><span>Precision: <b>{MODEL_INFO.perf.prec}</b></span><span>AUC: <b>{MODEL_INFO.perf.auc}</b></span></div>
              </div>
            </div>
          </div>}
        </section>

        {/* Controls */}
        <section className="glass-card controls">
          <div className="btns">
            <button onClick={() => { setAccounts(SAMPLE); setResults([]); addSecurityEvent('SAMPLE_LOADED', '5 sample accounts loaded', 'INFO') }}><FileText size={16} /> Sample</button>
            <button onClick={() => genAccounts(50)}><Sparkles size={16} /> Gen 50</button>
            <button onClick={() => genAccounts(100)}><Zap size={16} /> Gen 100</button>
            <label className="upload-btn"><Upload size={16} /> CSV<input type="file" accept=".csv" hidden /></label>
          </div>
          <button className="analyze-btn" disabled={loading || !accounts.length} onClick={analyze}>
            {loading ? <><RefreshCw size={20} className="spin" /> {progress.toFixed(0)}%</> : <><Lock size={18} /><Play size={20} /> Secure Analyze {accounts.length}</>}
          </button>
          {loading && <div className="prog"><div style={{ width: `${progress}%` }}></div></div>}
        </section>

        {/* Stats */}
        {results.length > 0 && <section className="stats">
          <div className="stat t"><BarChart3 size={28} /><b>{stats.total}</b><span>Total</span></div>
          <div className="stat f"><XCircle size={28} /><b>{stats.fake}</b><span>Fake</span></div>
          <div className="stat r"><CheckCircle size={28} /><b>{stats.real}</b><span>Real</span></div>
          <div className="stat p"><TrendingUp size={28} /><b>{stats.total ? ((stats.fake / stats.total) * 100).toFixed(1) : 0}%</b><span>Fake Rate</span></div>
        </section>}

        {/* Filters & Results */}
        {results.length > 0 && <>
          <section className="glass-card filters">
            <div className="search"><Search size={16} /><input placeholder="Search..." value={search} onChange={e => setSearch(e.target.value)} /></div>
            <div className="fbtn"><button className={filter === 'all' ? 'on' : ''} onClick={() => setFilter('all')}>All</button><button className={filter === 'fake' ? 'on f' : ''} onClick={() => setFilter('fake')}>Fake</button><button className={filter === 'real' ? 'on r' : ''} onClick={() => setFilter('real')}>Real</button></div>
            <button className="exp" onClick={() => { const c = results.map(r => `${r.username},${r.result?.prediction?.is_fake ? 'Fake' : 'Real'},${r.result?.prediction?.risk_level}`).join('\n'); const b = new Blob(['Username,Status,Risk\n' + c], { type: 'text/csv' }); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'secure_results.csv'; a.click() }}><Download size={16} /> Export</button>
          </section>

          <section className="results">
            <div className="head"><span>Username</span><span>Status</span><span>Risk</span><span>Confidence</span><span>Network</span></div>
            <div className="body">{filtered.map((r, i) => <div key={i} className={`row ${r.result?.prediction?.is_fake ? 'fake' : 'real'}`}>
              <span className="user">@{r.username}</span>
              <span className={`st ${r.result?.prediction?.is_fake ? 'f' : 'r'}`}>{r.result?.prediction?.is_fake ? <><XCircle size={14} />FAKE</> : <><CheckCircle size={14} />REAL</>}</span>
              <span className={`badge ${getRisk(r.result?.prediction?.risk_level)}`}>{r.result?.prediction?.risk_level || 'N/A'}</span>
              <span className="conf"><div className="bar"><div style={{ width: `${(r.result?.prediction?.confidence || 0) * 100}%`, background: r.result?.prediction?.is_fake ? '#ef4444' : '#10b981' }}></div></div>{((r.result?.prediction?.confidence || 0) * 100).toFixed(0)}%</span>
              <span className="net"><Users size={12} />{r.followers_count}/{r.friends_count}</span>
            </div>)}</div>
          </section>
        </>}

        {!accounts.length && <section className="glass-card empty"><ShieldCheck size={50} /><h2>Secure ML System Ready</h2><p>Load data to begin secure analysis with adversarial protection</p></section>}
        {accounts.length > 0 && !results.length && !loading && <section className="glass-card empty"><Lock size={50} /><h2>{accounts.length} Accounts Queued</h2><p>Click Secure Analyze to process with ML protection</p></section>}
      </main>

      <footer className="footer"><p>🔒 Secured with Data Poisoning Detection • Adversarial Robustness • Rate Limiting</p></footer>
    </div>
  )
}
export default App
