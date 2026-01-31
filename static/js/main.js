/**
 * Fake Account Detection Dashboard - Main JavaScript
 * Handles UI interactions, API calls, and chart rendering
 */

// ==================== Utility Functions ====================

/**
 * Format a number with commas for thousands
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Format percentage value
 */
function formatPercent(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Get risk level class based on score
 */
function getRiskClass(score) {
    if (score < 0.4) return 'success';
    if (score < 0.7) return 'warning';
    return 'danger';
}

/**
 * Get risk level text based on score
 */
function getRiskLevel(score) {
    if (score < 0.2) return 'Very Low';
    if (score < 0.4) return 'Low';
    if (score < 0.6) return 'Medium';
    if (score < 0.8) return 'High';
    return 'Very High';
}

/**
 * Show loading spinner
 */
function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3 text-muted">Loading...</p>
            </div>
        `;
    }
}

/**
 * Show error message
 */
function showError(elementId, message) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
}

/**
 * Debounce function for rate limiting
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ==================== API Functions ====================

/**
 * Analyze a single account
 */
async function analyzeAccount(accountData) {
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(accountData)
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        return data.result;
    } catch (error) {
        console.error('Error analyzing account:', error);
        throw error;
    }
}

/**
 * Analyze batch of accounts from CSV
 */
async function analyzeBatch(formData) {
    try {
        const response = await fetch('/api/analyze-batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Batch analysis failed');
        }
        
        return data;
    } catch (error) {
        console.error('Error analyzing batch:', error);
        throw error;
    }
}

/**
 * Get sample accounts
 */
async function getSampleAccounts() {
    try {
        const response = await fetch('/api/sample-accounts');
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to load samples');
        }
        
        return data.samples;
    } catch (error) {
        console.error('Error loading samples:', error);
        throw error;
    }
}

/**
 * Get model metrics
 */
async function getModelMetrics() {
    try {
        const response = await fetch('/api/model-metrics');
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to load metrics');
        }
        
        return data;
    } catch (error) {
        console.error('Error loading metrics:', error);
        throw error;
    }
}

/**
 * Export results to CSV
 */
async function exportResultsToCSV(results) {
    try {
        const response = await fetch('/api/export-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ results })
        });
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fake_account_report_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        return true;
    } catch (error) {
        console.error('Error exporting results:', error);
        throw error;
    }
}

// ==================== Chart Functions ====================

/**
 * Render a Plotly chart from JSON data
 */
function renderChart(elementId, chartJson, options = {}) {
    try {
        const chartData = typeof chartJson === 'string' ? JSON.parse(chartJson) : chartJson;
        const config = {
            responsive: true,
            displayModeBar: options.showModeBar !== false,
            ...options
        };
        
        Plotly.newPlot(elementId, chartData.data, chartData.layout, config);
    } catch (error) {
        console.error(`Error rendering chart ${elementId}:`, error);
    }
}

/**
 * Create a simple gauge chart
 */
function createGaugeChart(elementId, value, title = 'Risk Score') {
    const data = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: value * 100,
        title: { text: title, font: { size: 16 } },
        gauge: {
            axis: { range: [0, 100], tickwidth: 1 },
            bar: { color: value < 0.4 ? '#28a745' : value < 0.7 ? '#ffc107' : '#dc3545' },
            steps: [
                { range: [0, 40], color: 'rgba(40, 167, 69, 0.2)' },
                { range: [40, 70], color: 'rgba(255, 193, 7, 0.2)' },
                { range: [70, 100], color: 'rgba(220, 53, 69, 0.2)' }
            ],
            threshold: {
                line: { color: 'black', width: 2 },
                thickness: 0.75,
                value: 50
            }
        }
    }];
    
    const layout = {
        margin: { t: 40, r: 25, l: 25, b: 25 },
        paper_bgcolor: 'transparent',
        font: { family: 'Segoe UI, system-ui, sans-serif' }
    };
    
    Plotly.newPlot(elementId, data, layout, { responsive: true, displayModeBar: false });
}

// ==================== UI Helper Functions ====================

/**
 * Create a factor list item
 */
function createFactorItem(factor, type = 'suspicious') {
    const weightClass = factor.weight === 'high' 
        ? (type === 'suspicious' ? 'danger' : 'success')
        : factor.weight === 'medium' 
            ? (type === 'suspicious' ? 'warning' : 'info')
            : 'secondary';
    
    return `
        <li class="mb-3">
            <div class="d-flex align-items-start">
                <span class="badge bg-${weightClass} me-2">${factor.weight}</span>
                <div>
                    <strong>${factor.factor}</strong>
                    <br><small class="text-muted">${factor.detail}</small>
                </div>
            </div>
        </li>
    `;
}

/**
 * Create a result table row
 */
function createResultRow(result, index) {
    const riskPercent = Math.round(result.risk_score * 100);
    const riskClass = getRiskClass(result.risk_score);
    
    return `
        <tr data-index="${index}" 
            data-classification="${result.is_fake ? 'fake' : 'genuine'}"
            data-risk-level="${result.risk_score < 0.4 ? 'low' : result.risk_score < 0.7 ? 'medium' : 'high'}">
            <td><strong>@${escapeHtml(result.username)}</strong></td>
            <td>
                <div class="progress" style="height: 20px; min-width: 100px;">
                    <div class="progress-bar bg-${riskClass}" style="width: ${riskPercent}%">
                        ${riskPercent}%
                    </div>
                </div>
            </td>
            <td>
                <span class="badge bg-${result.is_fake ? 'danger' : 'success'}">
                    ${result.is_fake ? 'Fake/Bot' : 'Genuine'}
                </span>
            </td>
            <td>${Math.round(result.confidence * 100)}%</td>
            <td>
                <span class="badge bg-secondary">${result.suspicious_attributes.length} factors</span>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="showAccountDetail(${index})">
                    <i class="bi bi-eye"></i>
                </button>
            </td>
        </tr>
    `;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '1100';
        document.body.appendChild(container);
    }
    
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    container.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

// ==================== Form Validation ====================

/**
 * Validate account form data
 */
function validateAccountForm(formData) {
    const errors = [];
    
    const followers = parseInt(formData.get('followers_count'));
    const following = parseInt(formData.get('following_count'));
    
    if (isNaN(followers) || followers < 0) {
        errors.push('Followers count must be a non-negative number');
    }
    
    if (isNaN(following) || following < 0) {
        errors.push('Following count must be a non-negative number');
    }
    
    const username = formData.get('username');
    if (!username || username.trim() === '') {
        errors.push('Username is required');
    }
    
    return errors;
}

// ==================== Event Listeners ====================

// Initialize tooltips and popovers when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

// Handle window resize for charts
window.addEventListener('resize', debounce(function() {
    // Resize all Plotly charts
    document.querySelectorAll('[id$="Chart"]').forEach(chart => {
        if (chart.data) {
            Plotly.Plots.resize(chart);
        }
    });
}, 250));

// Export functions for global use
window.FakeDetector = {
    analyzeAccount,
    analyzeBatch,
    getSampleAccounts,
    getModelMetrics,
    exportResultsToCSV,
    renderChart,
    createGaugeChart,
    formatNumber,
    formatPercent,
    getRiskClass,
    getRiskLevel,
    showToast,
    showLoading,
    showError
};
