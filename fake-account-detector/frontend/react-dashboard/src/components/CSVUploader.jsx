import React, { useRef, useState } from 'react'
import { Upload, FileText, AlertCircle } from 'lucide-react'

export const CSVUploader = ({ onFileSelect, onAnalyze, isLoading }) => {
  const fileInputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [error, setError] = useState('')

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      processFile(files[0])
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const processFile = (file) => {
    if (!file.name.endsWith('.csv')) {
      setError('Please select a CSV file')
      return
    }

    if (file.size > 5 * 1024 * 1024) {
      setError('File size must be less than 5MB')
      return
    }

    setSelectedFile(file)
    setError('')
    onFileSelect(file)
  }

  return (
    <div style={{ marginBottom: '2rem' }}>
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          border: dragActive ? '2px solid #00e5ff' : '2px dashed rgba(0, 229, 255, 0.3)',
          borderRadius: '12px',
          padding: '2rem',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: dragActive ? 'rgba(0, 229, 255, 0.05)' : 'rgba(0, 229, 255, 0.02)',
          transition: 'all 0.3s ease'
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        <Upload size={32} style={{ color: '#00e5ff', marginBottom: '1rem', opacity: 0.8 }} />
        <div style={{ fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
          {selectedFile ? selectedFile.name : 'Drag & drop CSV file here or click to upload'}
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          {selectedFile ? (
            <span style={{ color: '#00ff9d' }}>✓ Ready to analyze</span>
          ) : (
            'Supported format: CSV with columns like username, followers_count, friends_count, etc.'
          )}
        </div>
      </div>

      {error && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          backgroundColor: 'rgba(255, 71, 87, 0.1)',
          border: '1px solid #ff4757',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          color: '#ff4757'
        }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {selectedFile && (
        <button
          onClick={onAnalyze}
          disabled={isLoading}
          style={{
            marginTop: '1.5rem',
            padding: '0.75rem 2rem',
            backgroundColor: '#00e5ff',
            color: '#05050a',
            border: 'none',
            borderRadius: '8px',
            fontWeight: '600',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            opacity: isLoading ? 0.5 : 1,
            transition: 'all 0.3s ease'
          }}
        >
          {isLoading ? '⏳ Analyzing...' : '🔍 Secure Analyze'}
        </button>
      )}
    </div>
  )
}
