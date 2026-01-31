import React, { useMemo, useState } from 'react'
import { Users, TrendingUp, AlertTriangle, CheckCircle, XCircle, Search, Download } from 'lucide-react'

export const ResultsTable = ({ results, filter = 'all', searchTerm = '' }) => {
  const [sortBy, setSortBy] = useState('risk')
  const [sortDesc, setSortDesc] = useState(true)

  const filteredResults = useMemo(() => {
    let filtered = results

    // Filter by type
    if (filter === 'fake') {
      filtered = filtered.filter(r => r.result?.prediction?.is_fake)
    } else if (filter === 'real') {
      filtered = filtered.filter(r => !r.result?.prediction?.is_fake)
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(r => 
        r.username.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal, bVal

      switch (sortBy) {
        case 'risk':
          const riskOrder = { CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 }
          aVal = riskOrder[a.result?.prediction?.risk_level] || 0
          bVal = riskOrder[b.result?.prediction?.risk_level] || 0
          break
        case 'confidence':
          aVal = a.result?.prediction?.confidence || 0
          bVal = b.result?.prediction?.confidence || 0
          break
        case 'followers':
          aVal = a.followers_count || 0
          bVal = b.followers_count || 0
          break
        default:
          aVal = a.username
          bVal = b.username
      }

      return sortDesc ? (aVal > bVal ? -1 : 1) : (aVal < bVal ? -1 : 1)
    })

    return filtered
  }, [results, filter, searchTerm, sortBy, sortDesc])

  const stats = useMemo(() => {
    const total = results.length
    const fake = results.filter(r => r.result?.prediction?.is_fake).length
    const real = total - fake
    return {
      total,
      fake,
      real,
      fakeRate: total > 0 ? ((fake / total) * 100).toFixed(1) : 0
    }
  }, [results])

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'CRITICAL': return '#ff4757'
      case 'HIGH': return '#ffa502'
      case 'MEDIUM': return '#ffb800'
      case 'LOW': return '#10b981'
      default: return 'var(--text-secondary)'
    }
  }

  const exportToCSV = () => {
    if (filteredResults.length === 0) return

    const headers = ['Username', 'Status', 'Risk Level', 'Confidence', 'Followers', 'Following']
    const rows = filteredResults.map(r => [
      r.username,
      r.result?.prediction?.is_fake ? 'FAKE' : 'REAL',
      r.result?.prediction?.risk_level || 'N/A',
      ((r.result?.prediction?.confidence || 0) * 100).toFixed(0) + '%',
      r.followers_count || 0,
      r.friends_count || 0
    ])

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `analysis-results-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
  }

  if (results.length === 0) {
    return (
      <div style={{
        padding: '3rem 2rem',
        textAlign: 'center',
        color: 'var(--text-secondary)',
        backgroundColor: 'rgba(0, 229, 255, 0.02)',
        borderRadius: '12px',
        border: '1px dashed rgba(0, 229, 255, 0.2)'
      }}>
        <Search size={32} style={{ opacity: 0.5, marginBottom: '1rem' }} />
        <p>No results yet. Upload a CSV file to analyze accounts.</p>
      </div>
    )
  }

  return (
    <div style={{ marginTop: '2.5rem' }}>
      {/* Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem'
      }}>
        {[
          { label: 'Total', value: stats.total, icon: <Users size={18} /> },
          { label: 'Fake', value: stats.fake, icon: <XCircle size={18} />, color: '#ff4757' },
          { label: 'Real', value: stats.real, icon: <CheckCircle size={18} />, color: '#10b981' },
          { label: 'Fake Rate', value: `${stats.fakeRate}%`, icon: <TrendingUp size={18} /> }
        ].map((stat, idx) => (
          <div
            key={idx}
            style={{
              backgroundColor: 'rgba(0, 229, 255, 0.05)',
              border: '1px solid rgba(0, 229, 255, 0.15)',
              borderRadius: '8px',
              padding: '1.25rem',
              textAlign: 'center'
            }}
          >
            <div style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              marginBottom: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}>
              <span style={{ color: stat.color || 'var(--text-secondary)' }}>
                {stat.icon}
              </span>
              {stat.label}
            </div>
            <div style={{
              fontSize: '1.75rem',
              fontWeight: '700',
              color: stat.color || '#00e5ff'
            }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Table Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1rem',
        paddingBottom: '1rem',
        borderBottom: '1px solid rgba(0, 229, 255, 0.15)'
      }}>
        <div style={{
          fontSize: '0.95rem',
          color: 'var(--text-secondary)'
        }}>
          Showing {filteredResults.length} of {results.length} results
        </div>
        <button
          onClick={exportToCSV}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            backgroundColor: 'rgba(0, 229, 255, 0.1)',
            border: '1px solid rgba(0, 229, 255, 0.2)',
            borderRadius: '6px',
            color: '#00e5ff',
            cursor: 'pointer',
            fontSize: '0.9rem',
            fontWeight: '500',
            transition: 'all 0.3s ease'
          }}
        >
          <Download size={16} />
          Export CSV
        </button>
      </div>

      {/* Table */}
      <div style={{
        overflowX: 'auto',
        backgroundColor: 'rgba(0, 229, 255, 0.02)',
        borderRadius: '8px',
        border: '1px solid rgba(0, 229, 255, 0.1)'
      }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.95rem'
        }}>
          <thead>
            <tr style={{ borderBottom: '2px solid rgba(0, 229, 255, 0.2)' }}>
              {[
                { key: 'username', label: 'Username' },
                { key: 'status', label: 'Status' },
                { key: 'risk', label: 'Risk' },
                { key: 'confidence', label: 'Confidence' },
                { key: 'followers', label: 'Network' }
              ].map(col => (
                <th
                  key={col.key}
                  onClick={() => {
                    setSortBy(col.key)
                    setSortDesc(!sortDesc)
                  }}
                  style={{
                    padding: '1rem',
                    textAlign: 'left',
                    fontWeight: '600',
                    cursor: 'pointer',
                    userSelect: 'none',
                    color: sortBy === col.key ? '#00e5ff' : 'var(--text-secondary)',
                    transition: 'color 0.2s ease'
                  }}
                >
                  {col.label} {sortBy === col.key && (sortDesc ? '▼' : '▲')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredResults.map((result, idx) => (
              <tr
                key={idx}
                style={{
                  borderBottom: '1px solid rgba(0, 229, 255, 0.1)',
                  backgroundColor: idx % 2 === 0 ? 'transparent' : 'rgba(0, 229, 255, 0.02)',
                  transition: 'background-color 0.2s ease'
                }}
              >
                <td style={{ padding: '1rem', color: '#00e5ff', fontWeight: '500' }}>
                  {result.username}
                </td>
                <td style={{ padding: '1rem' }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    color: result.result?.prediction?.is_fake ? '#ff4757' : '#10b981'
                  }}>
                    {result.result?.prediction?.is_fake ? (
                      <><XCircle size={16} /> FAKE</>
                    ) : (
                      <><CheckCircle size={16} /> REAL</>
                    )}
                  </div>
                </td>
                <td style={{
                  padding: '1rem',
                  color: getRiskColor(result.result?.prediction?.risk_level),
                  fontWeight: '600'
                }}>
                  {result.result?.prediction?.risk_level || 'N/A'}
                </td>
                <td style={{ padding: '1rem', color: 'var(--text-secondary)' }}>
                  {((result.result?.prediction?.confidence || 0) * 100).toFixed(0)}%
                </td>
                <td style={{ padding: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  {result.followers_count}/{result.friends_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
