import React, { useMemo } from 'react'
import { BarChart3, PieChart } from 'lucide-react'

export const ThreatChart = ({ results }) => {
  const stats = useMemo(() => {
    if (!results || results.length === 0) {
      return {
        total: 0,
        fake: 0,
        real: 0,
        byRisk: { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 },
        avgConfidence: 0
      }
    }

    const fake = results.filter(r => r.result?.prediction?.is_fake).length
    const real = results.length - fake
    const byRisk = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 }
    let totalConfidence = 0

    results.forEach(r => {
      const risk = r.result?.prediction?.risk_level
      if (risk && byRisk.hasOwnProperty(risk)) {
        byRisk[risk]++
      }
      totalConfidence += r.result?.prediction?.confidence || 0
    })

    return {
      total: results.length,
      fake,
      real,
      byRisk,
      avgConfidence: results.length > 0 ? (totalConfidence / results.length * 100).toFixed(1) : 0
    }
  }, [results])

  const riskColors = {
    CRITICAL: '#ff4757',
    HIGH: '#ffa502',
    MEDIUM: '#ffb800',
    LOW: '#10b981'
  }

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '2rem',
      marginTop: '2.5rem',
      marginBottom: '2.5rem'
    }}>
      {/* Fake vs Real */}
      <div style={{
        backgroundColor: 'rgba(0, 229, 255, 0.02)',
        border: '1px solid rgba(0, 229, 255, 0.15)',
        borderRadius: '12px',
        padding: '2rem',
        textAlign: 'center'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '1.5rem',
          fontSize: '1.1rem',
          fontWeight: '600',
          color: 'var(--text-primary)'
        }}>
          <PieChart size={20} />
          Account Classification
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          <div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
              🔴 Fake Accounts
            </div>
            <div style={{
              fontSize: '2rem',
              fontWeight: '700',
              color: '#ff4757'
            }}>
              {stats.fake}
            </div>
            <div style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              marginTop: '0.5rem'
            }}>
              {stats.total > 0 ? ((stats.fake / stats.total * 100).toFixed(1)) : 0}%
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
              ✅ Real Accounts
            </div>
            <div style={{
              fontSize: '2rem',
              fontWeight: '700',
              color: '#10b981'
            }}>
              {stats.real}
            </div>
            <div style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              marginTop: '0.5rem'
            }}>
              {stats.total > 0 ? ((stats.real / stats.total * 100).toFixed(1)) : 0}%
            </div>
          </div>
        </div>
        <div style={{
          paddingTop: '1.5rem',
          borderTop: '1px solid rgba(0, 229, 255, 0.15)',
          fontSize: '0.9rem',
          color: 'var(--text-secondary)'
        }}>
          Total Analyzed: <strong style={{ color: '#00e5ff' }}>{stats.total}</strong>
        </div>
      </div>

      {/* Risk Distribution */}
      <div style={{
        backgroundColor: 'rgba(0, 229, 255, 0.02)',
        border: '1px solid rgba(0, 229, 255, 0.15)',
        borderRadius: '12px',
        padding: '2rem'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '1.5rem',
          fontSize: '1.1rem',
          fontWeight: '600',
          color: 'var(--text-primary)'
        }}>
          <BarChart3 size={20} />
          Risk Level Distribution
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(risk => (
            <div key={risk}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '0.5rem',
                fontSize: '0.9rem'
              }}>
                <span style={{ color: riskColors[risk], fontWeight: '600' }}>
                  {risk === 'CRITICAL' && '🔴'} 
                  {risk === 'HIGH' && '🟠'} 
                  {risk === 'MEDIUM' && '🟡'} 
                  {risk === 'LOW' && '🟢'} 
                  {' '}{risk}
                </span>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {stats.byRisk[risk]} accounts
                </span>
              </div>
              <div style={{
                width: '100%',
                height: '8px',
                backgroundColor: 'rgba(0, 229, 255, 0.1)',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${stats.total > 0 ? (stats.byRisk[risk] / stats.total * 100) : 0}%`,
                  height: '100%',
                  backgroundColor: riskColors[risk],
                  transition: 'width 0.3s ease'
                }} />
              </div>
            </div>
          ))}
        </div>
        <div style={{
          paddingTop: '1.5rem',
          marginTop: '1.5rem',
          borderTop: '1px solid rgba(0, 229, 255, 0.15)',
          fontSize: '0.9rem',
          color: 'var(--text-secondary)'
        }}>
          Avg Confidence: <strong style={{ color: '#00e5ff' }}>{stats.avgConfidence}%</strong>
        </div>
      </div>
    </div>
  )
}
