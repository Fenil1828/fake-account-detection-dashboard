// import React from 'react'
// import { Shield, Lock, AlertCircle, ShieldCheck, Activity, AlertTriangle } from 'lucide-react'
// import '../security-monitor.css'

// export const SecurityMonitor = ({ threatStatus = 'WARNING', threatScore = 40, events = [] }) => {
//   const isSecure = threatScore < 30
//   const isWarning = threatScore >= 30 && threatScore < 60
//   const isCritical = threatScore >= 60

//   return (
//     <div className="security-monitor">
//       {/* HEADER */}
//       <div className="monitor-header">
//         <div className="monitor-title">
//           <Shield size={28} strokeWidth={1.5} />
//           <span>Security Monitor</span>
//         </div>
//         <div className={`status-badge ${isCritical ? 'critical' : isWarning ? 'warning' : ''}`}>
//           <span>{threatStatus}</span>
//         </div>
//       </div>

//       {/* STAT CARDS */}
//       <div className="stats-grid">
//         <div className="stat-card">
//           <span className="stat-label">Threat Assessment</span>
//           <span className="stat-value">{threatScore}%</span>
//           <span className="stat-badge-icon">🟡 WARNING</span>
//         </div>
//         <div className="stat-card">
//           <span className="stat-label">Fake Accounts</span>
//           <span className="stat-value">2</span>
//           <span className="stat-badge-icon">⚠️</span>
//         </div>
//         <div className="stat-card">
//           <span className="stat-label">Real Accounts</span>
//           <span className="stat-value">3</span>
//           <span className="stat-badge-icon">✓</span>
//         </div>
//         <div className="stat-card">
//           <span className="stat-label">Analyzed</span>
//           <span className="stat-value">5</span>
//           <span className="stat-badge-icon">📊</span>
//         </div>
//       </div>

//       {/* RISK LEVELS */}
//       <div className="risk-levels">
//         <div className="risk-item secure">
//           <div className="risk-emoji">🟢</div>
//           <div className="risk-label">Secure</div>
//           <div className="risk-range">0-30%</div>
//         </div>
//         <div className="risk-item warning">
//           <div className="risk-emoji">🟡</div>
//           <div className="risk-label">Warning</div>
//           <div className="risk-range">30-60%</div>
//         </div>
//         <div className="risk-item critical">
//           <div className="risk-emoji">🔴</div>
//           <div className="risk-label">Critical</div>
//           <div className="risk-range">60-100%</div>
//         </div>
//       </div>

//       {/* STATUS MESSAGE */}
//       <div className={`status-message ${isCritical ? 'critical' : isWarning ? 'warning' : ''}`}>
//         <AlertCircle size={20} strokeWidth={1.5} />
//         <span className="status-text">
//           💡 Warning: Moderate threat detected. Review suspicious accounts.
//         </span>
//       </div>

//       {/* ACTIVE PROTECTIONS */}
//       <div className="protections-section">
//         <div className="protections-header">
//           <ShieldCheck size={20} strokeWidth={1.5} />
//           <span>Active Protections</span>
//           <span className="protection-count">6/6</span>
//         </div>
//         <div className="protections-grid">
//           <div className="protection-item">
//             <Lock size={20} strokeWidth={1.5} className="protection-icon" />
//             <span className="protection-name">🔒 Data Poisoning Detection</span>
//           </div>
//           <div className="protection-item">
//             <Shield size={20} strokeWidth={1.5} className="protection-icon" />
//             <span className="protection-name">⚔️ Adversarial Input Filter</span>
//           </div>
//           <div className="protection-item">
//             <Activity size={20} strokeWidth={1.5} className="protection-icon" />
//             <span className="protection-name">⏱️ Rate Limiting (100/hr)</span>
//           </div>
//           <div className="protection-item">
//             <AlertTriangle size={20} strokeWidth={1.5} className="protection-icon" />
//             <span className="protection-name">🚫 SQL/XSS Injection Block</span>
//           </div>
//           <div className="protection-item">
//             <ShieldCheck size={20} strokeWidth={1.5} className="protection-icon" />
//             <span className="protection-name">✓ Input Validation</span>
//           </div>
//           <div className="protection-item">
//             <Activity size={20} strokeWidth={1.5} className="protection-icon" />
//             <span className="protection-name">🔍 Anomaly Detection</span>
//           </div>
//         </div>
//       </div>

//       {/* THREAT HISTORY */}
//       <div className="threat-section">
//         <div className="threat-header">
//           <div className="threat-title">
//             <AlertTriangle size={20} strokeWidth={1.5} />
//             <span>Threat History</span>
//             <span className="threat-count">21 events</span>
//           </div>
//         </div>

//         <div className="threat-list">
//           <div className="threat-item">
//             <div className="threat-dot"></div>
//             <div className="threat-details">
//               <span className="threat-time">ANALYSIS_COMPLETE</span>
//               <span className="threat-time-sub">4:55:57 PM</span>
//               <span className="threat-desc">Found 2 fake accounts, 0 blocked</span>
//             </div>
//           </div>
//           <div className="threat-item">
//             <div className="threat-dot"></div>
//             <div className="threat-details">
//               <span className="threat-time">ANALYSIS_STARTED</span>
//               <span className="threat-time-sub">4:55:56 PM</span>
//               <span className="threat-desc">Processing 5 accounts</span>
//             </div>
//           </div>
//           <div className="threat-item">
//             <div className="threat-dot"></div>
//             <div className="threat-details">
//               <span className="threat-time">SAMPLE_LOADED</span>
//               <span className="threat-time-sub">4:55:55 PM</span>
//               <span className="threat-desc">5 sample accounts loaded</span>
//             </div>
//           </div>
//         </div>

//         {/* RECENT EVENTS SECTION */}
//         <div style={{ marginTop: '2.5rem', paddingTop: '2rem', borderTop: '1px solid rgba(255, 184, 0, 0.15)' }}>
//           <div style={{ fontSize: '1rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
//             <Activity size={18} strokeWidth={1.5} />
//             Recent Events
//             <span className="threat-count">3</span>
//           </div>
//           <div style={{ backgroundColor: 'rgba(255, 184, 0, 0.05)', border: '1px dashed rgba(255, 184, 0, 0.25)', borderRadius: '8px', padding: '2.5rem 2rem', textAlign: 'center', minHeight: '280px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
//             <div style={{ fontSize: '3.5rem', marginBottom: '1.25rem', opacity: '0.8' }}>📋</div>
//             <div style={{ fontSize: '1.1rem', color: 'var(--text-primary)', fontWeight: '600', marginBottom: '0.75rem' }}>3 Recent Security Events</div>
//             <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: '1.6' }}>
//               Analysis completed with 2 fake accounts detected
//             </div>
//             <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: '1.6' }}>
//               5 social media accounts were processed and analyzed
//             </div>
//             <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', borderTop: '1px solid rgba(255, 184, 0, 0.15)', paddingTop: '1.5rem', marginTop: '1.5rem', maxWidth: '300px', opacity: '0.7' }}>
//               ⏱️ Last event: 4:55:57 PM<br/>
//               📊 Threat Level: 40% (WARNING)<br/>
//               ✓ All protections active (6/6)
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   )
// }

// export default SecurityMonitor

