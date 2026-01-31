# Integration Guide - Quick Start

## What Was Added?

✅ **CSV Upload Component** - Drag & drop file upload with validation
✅ **Results Table** - Sortable, searchable results with export
✅ **Rate Limiting** - Backend security (100 req/hour)
✅ **Input Sanitization** - XSS/SQL injection prevention
✅ **Batch Analysis Hook** - React hook for API integration

## Adding to Your Dashboard (2 Steps)

### Step 1: Update App.jsx imports

Add these imports at the top of your `App.jsx`:

```jsx
import { CSVUploader } from './components/CSVUploader'
import { ResultsTable } from './components/ResultsTable'
import { useBatchAnalysis } from './hooks/useBatchAnalysis'
import { parseCSV } from './utils/csvParser'
```

### Step 2: Add to your component

```jsx
function App() {
  const { loading, results, error, analyzeBatch } = useBatchAnalysis()
  
  const handleFileSelect = async (file) => {
    try {
      const text = await file.text()
      const accounts = parseCSV(text)
      if (accounts.error) {
        alert(accounts.error)
        return
      }
      await analyzeBatch(accounts)
    } catch (err) {
      alert('Error reading file: ' + err.message)
    }
  }

  return (
    <div>
      {/* Your existing components */}
      <SecurityMonitor />
      
      {/* Add the new components */}
      <CSVUploader 
        onFileSelect={handleFileSelect}
        onAnalyze={() => {}}
        isLoading={loading}
      />
      
      {error && (
        <div style={{ 
          padding: '1rem', 
          background: '#ff4757', 
          color: 'white', 
          borderRadius: '8px',
          marginTop: '1rem'
        }}>
          Error: {error}
        </div>
      )}
      
      <ResultsTable results={results} filter="all" searchTerm="" />
    </div>
  )
}
```

## File Structure Created

```
frontend/react-dashboard/
├── src/
│   ├── components/
│   │   ├── SecurityMonitor.jsx (existing)
│   │   ├── CSVUploader.jsx          ✨ NEW
│   │   └── ResultsTable.jsx         ✨ NEW
│   ├── hooks/
│   │   └── useBatchAnalysis.js      ✨ NEW
│   ├── utils/
│   │   └── csvParser.js             ✨ NEW
│   └── App.jsx (update imports)
```

## Example CSV Format

Upload a CSV with these columns:

```csv
username,followers_count,friends_count,statuses_count,account_age_days,has_profile_image,verified
bot_user123,5,8000,15000,20,false,false
john_smith,500,300,1200,1825,true,false
real_person,1200,400,2500,2000,true,true
```

**Required:** username
**Optional:** All other fields default to safe values if missing

## Features Ready to Use

### CSVUploader
- Drag & drop interface
- File size validation (5MB max)
- Format validation
- Visual feedback

### ResultsTable
- Sort by Risk, Confidence, Followers, Username
- Filter: All / Fake / Real
- Search by username
- Export to CSV
- Shows statistics (total, fake, real, fake rate)

### Backend Security
- Rate limiting: 100 requests/hour per IP
- Input sanitization: removes XSS/SQL injection attempts
- Batch limit: max 100 accounts per request
- API health check before analysis

## Testing

1. **Backend** - Already has rate limiting & sanitization
2. **Frontend** - Components are self-contained, can be tested independently
3. **Integration** - Copy files, update App.jsx, refresh browser

## Minimal Impact

- ✅ No existing code changes needed (except imports)
- ✅ Works with current API endpoints
- ✅ SecurityMonitor component unchanged
- ✅ Can be used alongside existing functionality

## Next Steps

1. Copy the new files to your project
2. Update App.jsx with imports
3. Test CSV upload with sample data
4. View results with sorting and filtering
5. Export results as needed

**That's it! Everything is ready to go.**
