/**
 * CSV Parser utility for fake account detector
 */

export const parseCSV = (text) => {
  const lines = text.trim().split('\n')
  if (lines.length < 2) {
    return { error: 'CSV must have header and at least one row' }
  }

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase())
  const rows = []

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue // Skip empty lines

    const cells = line.split(',').map(c => c.trim())
    if (cells.length !== headers.length) continue

    const obj = {}
    headers.forEach((h, idx) => {
      const value = cells[idx]
      // Try to parse as number, otherwise keep as string
      obj[h] = !isNaN(value) && value !== '' ? Number(value) : value
    })

    // Require username field
    if (obj.username) {
      rows.push(obj)
    }
  }

  return rows.length > 0 ? rows : { error: 'No valid rows found' }
}

export const validateAccountData = (accounts) => {
  return accounts.filter(acc => acc.username && typeof acc.username === 'string')
}

export const exportResultsToCSV = (results, filename = 'analysis-results.csv') => {
  if (results.length === 0) return

  const headers = ['Username', 'Status', 'Risk Level', 'Confidence', 'Followers', 'Following', 'Is Fake', 'Fake Probability']
  const rows = results.map(r => [
    r.username,
    r.result?.prediction?.is_fake ? 'FAKE' : 'REAL',
    r.result?.prediction?.risk_level || 'N/A',
    ((r.result?.prediction?.confidence || 0) * 100).toFixed(0) + '%',
    r.followers_count || 0,
    r.friends_count || 0,
    r.result?.prediction?.is_fake ? 'Yes' : 'No',
    ((r.result?.prediction?.fake_probability || 0) * 100).toFixed(2) + '%'
  ])

  const csv = [headers, ...rows].map(row => row.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)

  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  link.style.visibility = 'hidden'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
