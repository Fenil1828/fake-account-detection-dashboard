/**
 * Fake Account Detection Dashboard - Main JavaScript
 * Handles form submission, results display, and interactions
 */

// Load sample accounts on page load
document.addEventListener('DOMContentLoaded', function() {
    loadSampleAccounts();
    setupFormValidation();
});

/**
 * Load sample accounts from API
 */
async function loadSampleAccounts() {
    try {
        const response = await fetch('/api/sample-accounts');
        const data = await response.json();
        
        if (data.success) {
            const container = document.getElementById('sampleAccountsContainer');
            container.innerHTML = '';
            
            data.samples.forEach((sample, index) => {
                const col = document.createElement('div');
                col.className = 'col-md-6 col-lg-3';
                col.innerHTML = `
                    <div class="card h-100 sample-card" onclick="loadSampleToForm(${index})" style="cursor: pointer;">
                        <div class="card-body">
                            <h6 class="card-title">@${sample.username}</h6>
                            <p class="card-text small text-muted">
                                <i class="bi bi-people me-1"></i>${sample.followers_count.toLocaleString()} followers<br>
                                <i class="bi bi-image me-1"></i>${sample.posts_count} posts
                            </p>
                            <span class="badge ${sample.has_profile_pic ? 'bg-success' : 'bg-secondary'}">
                                ${sample.has_profile_pic ? 'Has Profile Pic' : 'No Profile Pic'}
                            </span>
                        </div>
                    </div>
                `;
                container.appendChild(col);
            });
            
            window.sampleAccounts = data.samples;
        }
    } catch (error) {
        console.error('Error loading samples:', error);
    }
}

/**
 * Setup form validation and submission
 */
function setupFormValidation() {
    const form = document.getElementById('quickAnalysisForm');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            await handleAnalysisFormSubmit(this);
        });
    }
}

/**
 * Handle the quick analysis form submission
 */
async function handleAnalysisFormSubmit(form) {
    // Validate form
    if (!form.checkValidity()) {
        form.classList.add('was-validated');
        return;
    }

    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalButtonText = submitBtn.innerHTML;
    
    // Disable button and show loading
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.result);
            showSuccessNotification('Analysis completed successfully!');
        } else {
            showErrorNotification('Error: ' + (data.error || 'Unknown error occurred'));
        }
    } catch (error) {
        console.error('Error:', error);
        showErrorNotification('Error analyzing account: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalButtonText;
    }
}

/**
 * Display analysis results
 */
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    
    // Update risk score
    const riskPercent = Math.round(result.risk_score * 100);
    document.getElementById('riskScoreValue').textContent = riskPercent + '%';
    document.getElementById('riskLevelText').textContent = result.risk_level + ' Risk';
    document.getElementById('confidenceValue').textContent = Math.round(result.confidence * 100) + '%';
    
    // Update risk card styling
    const riskCard = document.getElementById('riskScoreCard');
    riskCard.className = 'text-center p-4 rounded-3 mb-3';
    
    if (result.risk_score < 0.4) {
        riskCard.classList.add('bg-success-subtle', 'text-success');
    } else if (result.risk_score < 0.7) {
        riskCard.classList.add('bg-warning-subtle', 'text-warning');
    } else {
        riskCard.classList.add('bg-danger-subtle', 'text-danger');
    }
    
    // Update classification badge
    const badge = document.getElementById('classificationBadge');
    badge.textContent = result.classification;
    badge.className = 'badge fs-6 ' + (result.is_fake ? 'bg-danger' : 'bg-success');
    
    // Update summary
    document.getElementById('summaryText').textContent = result.explanation.summary;
    
    // Update suspicious factors
    const suspiciousList = document.getElementById('suspiciousFactors');
    suspiciousList.innerHTML = '';
    if (result.suspicious_attributes && result.suspicious_attributes.length > 0) {
        result.suspicious_attributes.forEach(factor => {
            const li = document.createElement('li');
            li.className = 'mb-2 p-2 rounded bg-light';
            li.innerHTML = `
                <strong class="text-danger"><i class="bi bi-exclamation-circle me-1"></i>${factor.factor}</strong>
                <br>
                <span class="text-muted">${factor.detail}</span>
            `;
            suspiciousList.appendChild(li);
        });
    } else {
        suspiciousList.innerHTML = '<li class="text-muted">No suspicious factors detected</li>';
    }
    
    // Update positive factors
    const positiveList = document.getElementById('positiveFactors');
    positiveList.innerHTML = '';
    if (result.positive_attributes && result.positive_attributes.length > 0) {
        result.positive_attributes.forEach(factor => {
            const li = document.createElement('li');
            li.className = 'mb-2 p-2 rounded bg-light';
            li.innerHTML = `
                <strong class="text-success"><i class="bi bi-check-circle me-1"></i>${factor.factor}</strong>
                <br>
                <span class="text-muted">${factor.detail}</span>
            `;
            positiveList.appendChild(li);
        });
    } else {
        positiveList.innerHTML = '<li class="text-muted">No positive factors identified</li>';
    }
    
    // Render charts if available
    if (result.charts) {
        try {
            if (result.charts.engagement) {
                const engagementData = JSON.parse(result.charts.engagement);
                Plotly.newPlot('engagementChart', engagementData.data, engagementData.layout, {
                    responsive: true,
                    displayModeBar: false
                });
            }
        } catch (e) {
            console.error('Error rendering engagement chart:', e);
        }
        
        try {
            if (result.charts.followers) {
                const followersData = JSON.parse(result.charts.followers);
                Plotly.newPlot('followersChart', followersData.data, followersData.layout, {
                    responsive: true,
                    displayModeBar: false
                });
            }
        } catch (e) {
            console.error('Error rendering followers chart:', e);
        }
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Load sample account data into the form
 */
function loadSampleToForm(index) {
    if (!window.sampleAccounts || !window.sampleAccounts[index]) {
        showErrorNotification('Sample account not found');
        return;
    }
    
    const sample = window.sampleAccounts[index];
    const form = document.getElementById('quickAnalysisForm');
    
    // Populate form fields
    form.querySelector('input[name="username"]').value = sample.username;
    form.querySelector('input[name="followers_count"]').value = sample.followers_count;
    form.querySelector('input[name="following_count"]').value = sample.following_count;
    form.querySelector('input[name="posts_count"]').value = sample.posts_count;
    form.querySelector('input[name="account_age_days"]').value = sample.account_age_days;
    form.querySelector('select[name="has_profile_pic"]').value = sample.has_profile_pic ? 'true' : 'false';
    form.querySelector('textarea[name="bio"]').value = sample.bio || '';
    
    // Scroll to form
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showSuccessNotification('Sample account loaded!');
}

/**
 * Load a random sample account
 */
function loadSample() {
    if (!window.sampleAccounts || window.sampleAccounts.length === 0) {
        showErrorNotification('No sample accounts available');
        return;
    }
    
    const randomIndex = Math.floor(Math.random() * window.sampleAccounts.length);
    loadSampleToForm(randomIndex);
}

/**
 * Show success notification
 */
function showSuccessNotification(message) {
    showNotification(message, 'success');
}

/**
 * Show error notification
 */
function showErrorNotification(message) {
    showNotification(message, 'danger');
}

/**
 * Show generic notification
 */
function showNotification(message, type = 'info') {
    const notificationHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 9999; min-width: 300px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <i class="bi bi-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    const container = document.createElement('div');
    container.innerHTML = notificationHtml;
    document.body.appendChild(container);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
            alert.remove();
        }
        container.remove();
    }, 5000);
}

/**
 * Export results as CSV
 */
async function exportResults() {
    try {
        const response = await fetch('/api/export-results', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Export failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis_results_${new Date().getTime()}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        
        showSuccessNotification('Results exported successfully!');
    } catch (error) {
        showErrorNotification('Error exporting results: ' + error.message);
    }
}

/**
 * Clear form fields
 */
function clearForm() {
    const form = document.getElementById('quickAnalysisForm');
    if (form) {
        form.reset();
        form.classList.remove('was-validated');
        document.getElementById('resultsSection').style.display = 'none';
        showSuccessNotification('Form cleared');
    }
}

/**
 * Format large numbers with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Format percentage
 */
function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}
