import React, { useMemo } from 'react'
import { Users, TrendingUp, AlertCircle, CheckCircle, Star } from 'lucide-react'

export const NetworkGraph = ({ account }) => {
  const analysis = useMemo(() => {
    const followers = account?.followers_count || 0
    const following = account?.friends_count || 0
    const verified = account?.verified || false
    const accountAge = account?.account_age_days || 1
    
    // Calculate ratios
    const followRatio = followers > 0 ? (following / followers).toFixed(2) : 0
    const followerRatio = following > 0 ? (followers / following).toFixed(2) : followers
    
    // Determine account type and reasoning
    const isCelebrity = followers > 100000 && following < followers * 0.1
    const isBot = followers < 100 && following > 1000
    const isInfluencer = followers > 10000 && followers < 1000000 && following < followers * 0.2
    const isNormal = followers < 10000 && following > followers * 0.5 && following < followers * 2
    
    // Generate detailed reasoning
    let analysis = {
      type: 'UNKNOWN',
      description: '',
      reasoning: [],
      suspicion: 0,
      verification: 'PENDING'
    }
    
    if (isCelebrity) {
      analysis.type = 'CELEBRITY/VERIFIED'
      analysis.description = 'High-profile account with typical celebrity pattern'
      analysis.verification = 'LOW_RISK'
      analysis.suspicion = 0.1
      analysis.reasoning = [
        `✓ Massive followers (${(followers / 1000000).toFixed(1)}M) - Expected for celebrities`,
        `✓ Low following ratio (${followRatio}:1) - Typical celeb pattern of selective follows`,
        `✓ ${verified ? 'Account is verified ✓' : 'Not verified but high followers'}`,
        `✓ Healthy engagement asymmetry - followers >> following is normal`
      ]
    } else if (isInfluencer) {
      analysis.type = 'INFLUENCER'
      analysis.description = 'Mid-tier influencer or content creator'
      analysis.verification = 'LOW_RISK'
      analysis.suspicion = 0.15
      analysis.reasoning = [
        `✓ Strong follower base (${(followers / 1000).toFixed(0)}K) - Influencer tier`,
        `✓ Selective following (${followRatio}:1) - Typical influencer pattern`,
        `✓ Follows <10% of followers - Focused audience engagement`,
        `✓ ${verified ? '✓ Verified account' : 'Unverified but consistent pattern'}`
      ]
    } else if (isNormal) {
      analysis.type = 'REGULAR USER'
      analysis.description = 'Normal social media user with balanced following'
      analysis.verification = 'LOW_RISK'
      analysis.suspicion = 0.1
      analysis.reasoning = [
        `✓ Balanced followers/following - ${followers} / ${following}`,
        `✓ Healthy engagement ratio - Realistic social interaction`,
        `✓ Follow 1:1 pattern - Typical of regular users`,
        `✓ Natural growth pattern expected`
      ]
    } else if (isBot) {
      analysis.type = 'LIKELY BOT/FAKE'
      analysis.description = 'Suspicious account showing bot characteristics'
      analysis.verification = 'HIGH_RISK'
      analysis.suspicion = 0.85
      analysis.reasoning = [
        `⚠️ Very low followers (${followers}) but massive following (${following})`,
        `⚠️ Following ratio (${followRatio}:1) - Extreme bot pattern`,
        `⚠️ Account follows ${((following / followers) * 100).toFixed(0)}% more accounts than followers`,
        `⚠️ Classic bot strategy: Follow many to gain visibility, get few followers back`
      ]
    } else {
      // Anomalous pattern
      analysis.type = 'SUSPICIOUS PATTERN'
      analysis.description = 'Unusual follower/following relationship detected'
      analysis.verification = 'MEDIUM_RISK'
      analysis.suspicion = 0.55
      analysis.reasoning = [
        `⚠️ Followers: ${followers} | Following: ${following}`,
        `⚠️ Ratio (${followRatio}:1) doesn't match expected patterns`,
        `⚠️ Not celebrity, not normal, not typical influencer pattern`,
        `⚠️ Possible fake followers or suspicious engagement strategy`
      ]
    }
    
    return {
      ...analysis,
      followers,
      following,
      followRatio: parseFloat(followRatio),
      followerRatio: parseFloat(followerRatio),
      verified,
      accountAge
    }
  }, [account])

  const getTypeColor = (type) => {
    switch (type) {
      case 'CELEBRITY/VERIFIED':
        return { bg: 'rgba(255, 184, 0, 0.1)', border: '#ffb800', text: '#ffb800', icon: '👑' }
      case 'INFLUENCER':
        return { bg: 'rgba(0, 255, 157, 0.1)', border: '#00ff9d', text: '#00ff9d', icon: '⭐' }
      case 'REGULAR USER':
        return { bg: 'rgba(0, 229, 255, 0.1)', border: '#00e5ff', text: '#00e5ff', icon: '👤' }
      case 'LIKELY BOT/FAKE':
        return { bg: 'rgba(255, 71, 87, 0.1)', border: '#ff4757', text: '#ff4757', icon: '🤖' }
      default:
        return { bg: 'rgba(255, 193, 7, 0.1)', border: '#ffc107', text: '#ffc107', icon: '❓' }
    }
  }

  const colors = getTypeColor(analysis.type)
  const suspicionPercent = (analysis.suspicion * 100).toFixed(0)

  return (
    <div style={{
      marginTop: '2rem',
      marginBottom: '2rem',
      padding: '1.5rem',
      backgroundColor: colors.bg,
      border: `2px solid ${colors.border}`,
      borderRadius: '12px'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        marginBottom: '1.5rem',
        paddingBottom: '1rem',
        borderBottom: `1px solid ${colors.border}`
      }}>
        <div style={{ fontSize: '2rem' }}>{colors.icon}</div>
        <div>
          <div style={{
            fontSize: '1.1rem',
            fontWeight: '700',
            color: colors.text
          }}>
            {analysis.type}
          </div>
          <div style={{
            fontSize: '0.85rem',
            color: 'var(--text-secondary)',
            marginTop: '0.25rem'
          }}>
            {analysis.description}
          </div>
        </div>
      </div>

      {/* Network Visualization */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '2rem',
        marginBottom: '2rem',
        alignItems: 'center'
      }}>
        {/* Left: Follower/Following Visual */}
        <div style={{ textAlign: 'center' }}>
          <div style={{
            position: 'relative',
            width: '200px',
            height: '200px',
            margin: '0 auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {/* Outer circle (followers) */}
            <svg
              width="200"
              height="200"
              style={{
                position: 'absolute',
                filter: 'drop-shadow(0 0 10px rgba(0, 229, 255, 0.3))'
              }}
            >
              <circle
                cx="100"
                cy="100"
                r="95"
                fill="none"
                stroke="#00e5ff"
                strokeWidth="2"
                opacity="0.3"
              />
              <circle
                cx="100"
                cy="100"
                r="70"
                fill="none"
                stroke="#00ff9d"
                strokeWidth="2"
                opacity="0.5"
              />
            </svg>

            {/* Center info */}
            <div style={{
              position: 'relative',
              zIndex: 10,
              textAlign: 'center'
            }}>
              <div style={{
                fontSize: '0.75rem',
                color: 'var(--text-secondary)',
                marginBottom: '0.5rem'
              }}>
                Ratio
              </div>
              <div style={{
                fontSize: '1.5rem',
                fontWeight: '700',
                color: colors.text
              }}>
                {analysis.followRatio}:1
              </div>
            </div>
          </div>

          <div style={{
            marginTop: '1rem',
            fontSize: '0.85rem',
            color: 'var(--text-secondary)'
          }}>
            <div>👥 Followers: <strong style={{ color: '#00e5ff' }}>{analysis.followers.toLocaleString()}</strong></div>
            <div>📤 Following: <strong style={{ color: '#00ff9d' }}>{analysis.following.toLocaleString()}</strong></div>
          </div>
        </div>

        {/* Right: Pattern Analysis */}
        <div>
          <div style={{
            fontSize: '0.9rem',
            fontWeight: '600',
            marginBottom: '1rem',
            color: 'var(--text-secondary)'
          }}>
            📊 Network Pattern Analysis
          </div>

          {/* Suspicion meter */}
          <div style={{
            marginBottom: '1.5rem',
            padding: '1rem',
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            borderRadius: '8px'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: '0.5rem',
              fontSize: '0.85rem'
            }}>
              <span>Suspicion Level</span>
              <span style={{ color: analysis.suspicion > 0.5 ? '#ff4757' : '#00ff9d', fontWeight: '600' }}>
                {suspicionPercent}%
              </span>
            </div>
            <div style={{
              width: '100%',
              height: '6px',
              backgroundColor: 'rgba(0, 229, 255, 0.1)',
              borderRadius: '3px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${suspicionPercent}%`,
                height: '100%',
                backgroundColor: analysis.suspicion > 0.5 ? '#ff4757' : '#00ff9d',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>

          {/* Quick stats */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0.75rem',
            fontSize: '0.85rem'
          }}>
            <div style={{
              padding: '0.75rem',
              backgroundColor: 'rgba(0, 0, 0, 0.2)',
              borderRadius: '6px'
            }}>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Account Age</div>
              <div style={{ color: '#00e5ff', fontWeight: '600' }}>{analysis.accountAge} days</div>
            </div>
            <div style={{
              padding: '0.75rem',
              backgroundColor: 'rgba(0, 0, 0, 0.2)',
              borderRadius: '6px'
            }}>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Verification</div>
              <div style={{ color: analysis.verified ? '#00ff9d' : 'var(--text-secondary)', fontWeight: '600' }}>
                {analysis.verified ? '✓ Verified' : 'Unverified'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Reasoning */}
      <div style={{
        backgroundColor: 'rgba(0, 0, 0, 0.2)',
        borderRadius: '8px',
        padding: '1rem'
      }}>
        <div style={{
          fontSize: '0.9rem',
          fontWeight: '600',
          marginBottom: '1rem',
          color: colors.text,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          💡 Analysis & Reasoning
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem'
        }}>
          {analysis.reasoning.map((reason, idx) => (
            <div
              key={idx}
              style={{
                fontSize: '0.85rem',
                color: 'var(--text-secondary)',
                lineHeight: '1.4',
                paddingLeft: '0.5rem',
                borderLeft: `2px solid ${colors.border}`,
                paddingBottom: '0.5rem'
              }}
            >
              {reason}
            </div>
          ))}
        </div>
      </div>

      {/* Risk Badge */}
      <div style={{
        marginTop: '1rem',
        padding: '0.75rem 1rem',
        borderRadius: '6px',
        backgroundColor: analysis.verification === 'LOW_RISK' ? 'rgba(0, 255, 157, 0.1)' :
                        analysis.verification === 'HIGH_RISK' ? 'rgba(255, 71, 87, 0.1)' :
                        'rgba(255, 184, 0, 0.1)',
        border: `1px solid ${
          analysis.verification === 'LOW_RISK' ? '#00ff9d' :
          analysis.verification === 'HIGH_RISK' ? '#ff4757' :
          '#ffb800'
        }`,
        fontSize: '0.85rem',
        color: analysis.verification === 'LOW_RISK' ? '#00ff9d' :
               analysis.verification === 'HIGH_RISK' ? '#ff4757' :
               '#ffb800',
        fontWeight: '600',
        textAlign: 'center'
      }}>
        Risk: {analysis.verification.replace(/_/g, ' ')}
      </div>
    </div>
  )
}

export default NetworkGraph
