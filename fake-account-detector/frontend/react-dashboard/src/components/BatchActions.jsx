import React, { useState, useCallback } from 'react'
import { CheckSquare, Copy, Download, Trash2, AlertCircle } from 'lucide-react'

export const BatchActions = ({ selectedAccounts, results, onDelete, onExport }) => {
  const [action, setAction] = useState(null)

  const selectedResults = selectedAccounts.size > 0
    ? results.filter(r => selectedAccounts.has(r.username))
    : []

  const fakeCount = selectedResults.filter(r => r.result?.prediction?.is_fake).length
  const realCount = selectedResults.length - fakeCount

  const handleBatchDelete = () => {
    selectedAccounts.forEach(username => {
      onDelete?.(username)
    })
    setAction(null)
  }

  const handleBatchExport = () => {
    const headers = ['Username', 'Status', 'Risk Level', 'Confidence', 'Followers', 'Following']
    const rows = selectedResults.map(r => [
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
    a.download = `selected-accounts-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    setAction(null)
  }

  const handleCopyUsernames = () => {
    const usernames = Array.from(selectedAccounts).join('\n')
    navigator.clipboard.writeText(usernames)
    setAction('copied')
    setTimeout(() => setAction(null), 2000)
  }

  if (selectedAccounts.size === 0) {
    return null
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '2rem',
      left: '50%',
      transform: 'translateX(-50%)',
      backgroundColor: 'rgba(5, 5, 10, 0.95)',
      border: '1px solid rgba(0, 229, 255, 0.3)',
      borderRadius: '12px',
      padding: '1.5rem 2rem',
      backdropFilter: 'blur(10px)',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
      zIndex: 1000,
      minWidth: '300px',
      maxWidth: '600px'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1rem'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.95rem',
          fontWeight: '600',
          color: '#00e5ff'
        }}>
          <CheckSquare size={18} />
          {selectedAccounts.size} Selected
        </div>
        <div style{{
          display: 'flex',
          gap: '0.5rem',
          fontSize: '0.85rem',
          color: 'var(--text-secondary)'
        }}>
          <span>🔴 {fakeCount} Fake</span>
          <span>✅ {realCount} Real</span>
        </div>
      </div>

      {/* Status Message */}
      {action === 'copied' && (
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'rgba(0, 255, 157, 0.1)',
          border: '1px solid rgba(0, 255, 157, 0.3)',
          borderRadius: '6px',
          color: '#00ff9d',
          fontSize: '0.85rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          ✓ Usernames copied to clipboard
        </div>
      )}

      {/* Action Buttons */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
        gap: '0.75rem'
      }}>
        <button
          onClick={handleCopyUsernames}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            padding: '0.75rem',
            backgroundColor: 'rgba(0, 229, 255, 0.1)',
            border: '1px solid rgba(0, 229, 255, 0.3)',
            borderRadius: '6px',
            color: '#00e5ff',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: '500',
            transition: 'all 0.3s ease'
          }}
          title="Copy all selected usernames"
        >
          <Copy size={14} />
          Copy List
        </button>

        <button
          onClick={handleBatchExport}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            padding: '0.75rem',
            backgroundColor: 'rgba(0, 255, 157, 0.1)',
            border: '1px solid rgba(0, 255, 157, 0.3)',
            borderRadius: '6px',
            color: '#00ff9d',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: '500',
            transition: 'all 0.3s ease'
          }}
          title="Export selected accounts as CSV"
        >
          <Download size={14} />
          Export
        </button>

        <button
          onClick={() => {
            if (window.confirm(`Delete ${selectedAccounts.size} account(s)?`)) {
              handleBatchDelete()
            }
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            padding: '0.75rem',
            backgroundColor: 'rgba(255, 71, 87, 0.1)',
            border: '1px solid rgba(255, 71, 87, 0.3)',
            borderRadius: '6px',
            color: '#ff4757',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: '500',
            transition: 'all 0.3s ease'
          }}
          title="Delete selected accounts from results"
        >
          <Trash2 size={14} />
          Delete
        </button>
      </div>
    </div>
  )
}
