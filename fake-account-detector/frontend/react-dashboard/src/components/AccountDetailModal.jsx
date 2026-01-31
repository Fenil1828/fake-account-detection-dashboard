import React from 'react'
import { X, AlertTriangle, CheckCircle, Users, Activity, TrendingUp } from 'lucide-react'
import { NetworkGraph } from './NetworkGraph'

export const AccountDetailModal = ({ account, isOpen, onClose }) => {
  if (!isOpen || !account) return null

  const prediction = account.result?.prediction || {}
  const riskFactors = account.result?.risk_factors || []
  const behaviorAnalysis = account.result?.behavioral_analysis || {}
  const networkAnalysis = account.result?.network_analysis || {}

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'CRITICAL': return '#ff4757'
      case 'HIGH': return '#ffa502'
      case 'MEDIUM': return '#ffb800'
      case 'LOW': return '#10b981'
      default: return 'var(--text-secondary)'
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          zIndex: 999,
          backdropFilter: 'blur(4px)'
        }}
      />

      {/* Modal */}
      <div style={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: 'var(--bg-void)',
        border: '1px solid rgba(0, 229, 255, 0.2)',
        borderRadius: '12px',
        padding: '2rem',
        zIndex: 1000,
        maxWidth: '600px',
        maxHeight: '90vh',
        overflowY: 'auto',
        width: '90%'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '2rem',
          paddingBottom: '1rem',
          borderBottom: '1px solid rgba(0, 229, 255, 0.15)'
        }}>
          <div>
            <div style={{
              fontSize: '1.5rem',
              fontWeight: '700',
              color: '#00e5ff',
              marginBottom: '0.5rem'
            }}>
              {account.username}
            </div>
            <div style={{
              display: 'flex',
              gap: '1rem',
              fontSize: '0.9rem'
            }}>
              <span style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                color: prediction.is_fake ? '#ff4757' : '#10b981'
              }}>
                {prediction.is_fake ? <AlertTriangle size={16} /> : <CheckCircle size={16} />}
                {prediction.is_fake ? 'FAKE' : 'REAL'}
              </span>
              <span style={{
                padding: '0.25rem 0.75rem',
                backgroundColor: getRiskColor(prediction.risk_level),
                color: 'white',
                borderRadius: '4px',
                fontWeight: '600',
                fontSize: '0.85rem'
              }}>
                {prediction.risk_level}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#00e5ff',
              cursor: 'pointer',
              padding: '0.5rem',
              display: 'flex'
            }}
          >
            <X size={24} />
          </button>
        </div>

        {/* Prediction Details */}
        <div style={{
          marginBottom: '2rem',
          padding: '1rem',
          backgroundColor: 'rgba(0, 229, 255, 0.05)',
          borderRadius: '8px'
        }}>
          <div style={{
            fontSize: '0.9rem',
            fontWeight: '600',
            marginBottom: '1rem',
            color: 'var(--text-secondary)'
          }}>
            📊 Analysis Result
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '1rem',
            fontSize: '0.9rem'
          }}>
            <div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Confidence</div>
              <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#00e5ff' }}>
                {((prediction.confidence || 0) * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Fake Probability</div>
              <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#ff4757' }}>
                {((prediction.fake_probability || 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Network Graph Analysis */}
        <NetworkGraph account={account} />

        {/* Network Analysis */}
        <div style={{
          marginBottom: '2rem',
          padding: '1rem',
          backgroundColor: 'rgba(0, 229, 255, 0.05)',
          borderRadius: '8px'
        }}>
          <div style={{
            fontSize: '0.9rem',
            fontWeight: '600',
            marginBottom: '1rem',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Users size={16} />
            Network Profile
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: '1rem',
            fontSize: '0.85rem'
          }}>
            <div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Followers</div>
              <div style={{ fontSize: '1.15rem', fontWeight: '600', color: '#00e5ff' }}>
                {account.followers_count || 0}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Following</div>
              <div style={{ fontSize: '1.15rem', fontWeight: '600', color: '#00e5ff' }}>
                {account.friends_count || 0}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>F/F Ratio</div>
              <div style={{ fontSize: '1.15rem', fontWeight: '600', color: '#00e5ff' }}>
                {account.followers_count > 0 ? ((account.friends_count / account.followers_count).toFixed(2)) : 'N/A'}
              </div>
            </div>
          </div>
          {networkAnalysis.assessment && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              backgroundColor: 'rgba(255, 184, 0, 0.1)',
              borderRadius: '6px',
              borderLeft: '3px solid #ffb800',
              fontSize: '0.85rem',
              color: 'var(--text-secondary)'
            }}>
              <strong>Assessment:</strong> {networkAnalysis.assessment}
            </div>
          )}
        </div>

        {/* Behavioral Analysis */}
        {behaviorAnalysis.posting_frequency && (
          <div style={{
            marginBottom: '2rem',
            padding: '1rem',
            backgroundColor: 'rgba(0, 229, 255, 0.05)',
            borderRadius: '8px'
          }}>
            <div style={{
              fontSize: '0.9rem',
              fontWeight: '600',
              marginBottom: '1rem',
              color: 'var(--text-secondary)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <Activity size={16} />
              Behavioral Patterns
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '1rem',
              fontSize: '0.85rem'
            }}>
              <div>
                <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Posting Frequency</div>
                <div style={{ color: '#00e5ff', fontWeight: '600' }}>
                  {behaviorAnalysis.posting_frequency}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Activity Level</div>
                <div style={{ color: '#00e5ff', fontWeight: '600' }}>
                  {behaviorAnalysis.account_activity}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Total Posts</div>
                <div style={{ color: '#00e5ff', fontWeight: '600' }}>
                  {behaviorAnalysis.total_tweets || account.statuses_count || 0}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Posts/Day</div>
                <div style={{ color: '#00e5ff', fontWeight: '600' }}>
                  {behaviorAnalysis.tweets_per_day || 0}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Risk Factors */}
        {riskFactors && riskFactors.length > 0 && (
          <div style={{
            marginBottom: '2rem',
            padding: '1rem',
            backgroundColor: 'rgba(255, 71, 87, 0.05)',
            borderRadius: '8px'
          }}>
            <div style={{
              fontSize: '0.9rem',
              fontWeight: '600',
              marginBottom: '1rem',
              color: '#ff4757',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <AlertTriangle size={16} />
              Risk Factors ({riskFactors.length})
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {riskFactors.map((factor, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.75rem',
                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '6px',
                    borderLeft: `3px solid ${
                      factor.severity === 'critical' ? '#ff4757' :
                      factor.severity === 'high' ? '#ffa502' :
                      factor.severity === 'medium' ? '#ffb800' :
                      '#10b981'
                    }`,
                    fontSize: '0.85rem'
                  }}
                >
                  <div style={{
                    fontWeight: '600',
                    color: 'var(--text-primary)',
                    marginBottom: '0.25rem'
                  }}>
                    {factor.factor}
                  </div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                    {factor.description}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timestamp */}
        <div style={{
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
          textAlign: 'center',
          paddingTop: '1rem',
          borderTop: '1px solid rgba(0, 229, 255, 0.15)'
        }}>
          Analyzed on {new Date(account.result?.timestamp).toLocaleString()}
        </div>
      </div>
    </>
  )
}
