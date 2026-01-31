import { useState, useCallback } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:5000'

export const useBatchAnalysis = () => {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState([])
  const [error, setError] = useState('')

  const analyzeBatch = useCallback(async (accounts) => {
    if (!accounts || accounts.length === 0) {
      setError('No accounts to analyze')
      return
    }

    setLoading(true)
    setError('')
    setProgress(0)
    setResults([])

    try {
      // Validate API health first
      const healthRes = await axios.get(`${API_URL}/api/health`)
      if (!healthRes.data.model_loaded) {
        throw new Error('ML Model not loaded. Please train the model first.')
      }

      // Send batch analysis request
      const response = await axios.post(`${API_URL}/api/batch`, {
        accounts: accounts.map(acc => ({
          username: acc.username,
          followers_count: acc.followers_count || 0,
          friends_count: acc.friends_count || 0,
          statuses_count: acc.statuses_count || 0,
          account_age_days: acc.account_age_days || 1,
          has_profile_image: acc.has_profile_image !== 'false' && acc.has_profile_image !== false,
          verified: acc.verified === 'true' || acc.verified === true,
          bio: acc.bio || '',
          url: acc.url || ''
        }))
      }, {
        timeout: 30000
      })

      // Format results with original account data
      const formattedResults = response.data.results.map((result, idx) => ({
        ...accounts[idx],
        result: result
      }))

      setResults(formattedResults)
      setProgress(100)

      return {
        success: true,
        data: formattedResults,
        summary: {
          total: response.data.total_analyzed,
          fake: response.data.fake_accounts_detected,
          real: response.data.real_accounts,
          fakeRate: ((response.data.fake_accounts_detected / response.data.total_analyzed) * 100).toFixed(1)
        }
      }
    } catch (err) {
      const errorMsg = err.response?.data?.error || err.message || 'Analysis failed'
      setError(errorMsg)
      console.error('Batch analysis error:', err)
      return {
        success: false,
        error: errorMsg
      }
    } finally {
      setLoading(false)
    }
  }, [])

  const clearResults = useCallback(() => {
    setResults([])
    setError('')
    setProgress(0)
  }, [])

  return {
    loading,
    progress,
    results,
    error,
    analyzeBatch,
    clearResults
  }
}
