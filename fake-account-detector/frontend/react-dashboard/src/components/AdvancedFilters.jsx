import React, { useState } from 'react'
import { Filter, X } from 'lucide-react'

export const AdvancedFilters = ({ onFilterChange, onReset }) => {
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState({
    riskLevel: 'all',
    confidenceMin: 0,
    confidenceMax: 100,
    followersMin: 0,
    followersMax: 100000
  })

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value }
    setFilters(newFilters)
    onFilterChange(newFilters)
  }

  const handleReset = () => {
    const defaultFilters = {
      riskLevel: 'all',
      confidenceMin: 0,
      confidenceMax: 100,
      followersMin: 0,
      followersMax: 100000
    }
    setFilters(defaultFilters)
    onFilterChange(defaultFilters)
    onReset?.()
  }

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <button
        onClick={() => setShowFilters(!showFilters)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.75rem 1.5rem',
          backgroundColor: 'rgba(0, 229, 255, 0.1)',
          border: '1px solid rgba(0, 229, 255, 0.2)',
          borderRadius: '8px',
          color: '#00e5ff',
          cursor: 'pointer',
          fontWeight: '500',
          transition: 'all 0.3s ease'
        }}
      >
        <Filter size={18} />
        Advanced Filters {showFilters ? '▲' : '▼'}
      </button>

      {showFilters && (
        <div style={{
          marginTop: '1rem',
          padding: '1.5rem',
          backgroundColor: 'rgba(0, 229, 255, 0.02)',
          border: '1px solid rgba(0, 229, 255, 0.15)',
          borderRadius: '8px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '1.5rem'
        }}>
          {/* Risk Level */}
          <div>
            <label style={{
              display: 'block',
              fontSize: '0.9rem',
              fontWeight: '600',
              marginBottom: '0.75rem',
              color: 'var(--text-secondary)'
            }}>
              Risk Level
            </label>
            <select
              value={filters.riskLevel}
              onChange={(e) => handleFilterChange('riskLevel', e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem',
                backgroundColor: 'rgba(0, 229, 255, 0.05)',
                border: '1px solid rgba(0, 229, 255, 0.15)',
                borderRadius: '6px',
                color: '#00e5ff',
                fontFamily: 'inherit',
                cursor: 'pointer'
              }}
            >
              <option value="all">All Risk Levels</option>
              <option value="CRITICAL">🔴 Critical</option>
              <option value="HIGH">🟠 High</option>
              <option value="MEDIUM">🟡 Medium</option>
              <option value="LOW">🟢 Low</option>
            </select>
          </div>

          {/* Confidence Range */}
          <div>
            <label style={{
              display: 'block',
              fontSize: '0.9rem',
              fontWeight: '600',
              marginBottom: '0.75rem',
              color: 'var(--text-secondary)'
            }}>
              Confidence: {filters.confidenceMin}% - {filters.confidenceMax}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={filters.confidenceMin}
              onChange={(e) => handleFilterChange('confidenceMin', parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
            <input
              type="range"
              min="0"
              max="100"
              value={filters.confidenceMax}
              onChange={(e) => handleFilterChange('confidenceMax', parseInt(e.target.value))}
              style={{ width: '100%', marginTop: '0.5rem' }}
            />
          </div>

          {/* Followers Range */}
          <div>
            <label style={{
              display: 'block',
              fontSize: '0.9rem',
              fontWeight: '600',
              marginBottom: '0.75rem',
              color: 'var(--text-secondary)'
            }}>
              Followers: {filters.followersMin} - {filters.followersMax}
            </label>
            <input
              type="number"
              min="0"
              value={filters.followersMin}
              onChange={(e) => handleFilterChange('followersMin', parseInt(e.target.value) || 0)}
              placeholder="Min followers"
              style={{
                width: '100%',
                padding: '0.75rem',
                backgroundColor: 'rgba(0, 229, 255, 0.05)',
                border: '1px solid rgba(0, 229, 255, 0.15)',
                borderRadius: '6px',
                color: '#00e5ff',
                marginBottom: '0.5rem'
              }}
            />
            <input
              type="number"
              value={filters.followersMax}
              onChange={(e) => handleFilterChange('followersMax', parseInt(e.target.value) || 100000)}
              placeholder="Max followers"
              style={{
                width: '100%',
                padding: '0.75rem',
                backgroundColor: 'rgba(0, 229, 255, 0.05)',
                border: '1px solid rgba(0, 229, 255, 0.15)',
                borderRadius: '6px',
                color: '#00e5ff'
              }}
            />
          </div>

          {/* Reset Button */}
          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button
              onClick={handleReset}
              style={{
                width: '100%',
                padding: '0.75rem',
                backgroundColor: 'rgba(255, 71, 87, 0.1)',
                border: '1px solid rgba(255, 71, 87, 0.3)',
                borderRadius: '6px',
                color: '#ff4757',
                cursor: 'pointer',
                fontWeight: '500',
                transition: 'all 0.3s ease'
              }}
            >
              <X size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
              Reset Filters
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
