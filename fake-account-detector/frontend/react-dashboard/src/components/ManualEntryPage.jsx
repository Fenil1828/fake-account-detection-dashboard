import React, { useState, useEffect } from 'react';
import { ArrowLeft, Save, RefreshCw, AlertCircle, CheckCircle2, Eye, EyeOff, Zap } from 'lucide-react';
import '../styles/manual-entry-page.css';

export function ManualEntryPage({ onSave, onCancel, onGenerate }) {
  const [formData, setFormData] = useState({
    username: '',
    followers_count: '',
    friends_count: '',
    statuses_count: '',
    account_age_days: '',
    bio: '',
    location: '',
    has_profile_image: true,
    verified: false,
    default_profile: false,
    default_profile_image: false,
  });

  const [errors, setErrors] = useState({});
  const [showPreview, setShowPreview] = useState(false);
  const [touched, setTouched] = useState({});

  const calculateMetrics = () => {
    const followers = parseInt(formData.followers_count) || 0;
    const following = parseInt(formData.friends_count) || 0;
    const posts = parseInt(formData.statuses_count) || 0;
    const age = parseInt(formData.account_age_days) || 1;

    return {
      posts_per_day: (posts / age).toFixed(2),
      follow_ratio: followers > 0 ? (following / followers).toFixed(2) : '0',
      avg_likes: Math.floor(followers * 0.15),
      engagement_rate: followers > 0 ? ((posts / (age * followers)) * 100).toFixed(2) : '0'
    };
  };

  const detectAccountType = () => {
    const followers = parseInt(formData.followers_count) || 0;
    const following = parseInt(formData.friends_count) || 0;

    if (following > 2000 && followers < 100) return { type: '🤖 Bot', color: '#ef4444', risk: 'HIGH' };
    if (followers > 100000 && following < followers * 0.1) return { type: '👑 Celebrity', color: '#fbbf24', risk: 'LOW' };
    if (followers > 10000 && following < followers * 0.2) return { type: '⭐ Influencer', color: '#8b5cf6', risk: 'LOW' };
    if (followers >= 100 && followers <= 10000 && following >= 50 && following <= 1000) return { type: '👤 Regular', color: '#06b6d4', risk: 'LOW' };
    return { type: '❓ Unknown', color: '#6b7280', risk: 'MEDIUM' };
  };

  const validateField = (name, value) => {
    const newErrors = { ...errors };

    switch (name) {
      case 'username':
        if (!value.trim()) {
          newErrors.username = 'Username is required';
        } else if (value.length < 2) {
          newErrors.username = 'Username must be at least 2 characters';
        } else if (!/^[a-zA-Z0-9_]+$/.test(value)) {
          newErrors.username = 'Username can only contain letters, numbers, and underscores';
        } else {
          delete newErrors.username;
        }
        break;
      case 'followers_count':
      case 'friends_count':
      case 'statuses_count':
      case 'account_age_days':
        const num = parseInt(value);
        if (value && (isNaN(num) || num < 0)) {
          newErrors[name] = 'Must be a positive number';
        } else {
          delete newErrors[name];
        }
        break;
      default:
        break;
    }

    setErrors(newErrors);
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;

    setFormData(prev => ({
      ...prev,
      [name]: newValue
    }));

    if (touched[name]) {
      validateField(name, newValue);
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    setTouched(prev => ({
      ...prev,
      [name]: true
    }));
    validateField(name, value);
  };

  const isFormValid = () => {
    return formData.username &&
      formData.followers_count &&
      formData.friends_count &&
      formData.account_age_days &&
      Object.keys(errors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validate all required fields
    ['username', 'followers_count', 'friends_count', 'account_age_days'].forEach(field => {
      setTouched(prev => ({ ...prev, [field]: true }));
      if (field === 'username' && !formData.username) {
        setErrors(prev => ({ ...prev, username: 'Username is required' }));
      }
    });

    if (isFormValid()) {
      const account = {
        username: formData.username,
        followers_count: parseInt(formData.followers_count),
        friends_count: parseInt(formData.friends_count),
        statuses_count: parseInt(formData.statuses_count) || 0,
        account_age_days: parseInt(formData.account_age_days),
        bio: formData.bio,
        location: formData.location,
        has_profile_image: formData.has_profile_image,
        verified: formData.verified,
        default_profile: formData.default_profile,
        default_profile_image: formData.default_profile_image,
      };

      onSave([account]);
    }
  };

  const handleReset = () => {
    setFormData({
      username: '',
      followers_count: '',
      friends_count: '',
      statuses_count: '',
      account_age_days: '',
      bio: '',
      location: '',
      has_profile_image: true,
      verified: false,
      default_profile: false,
      default_profile_image: false,
    });
    setErrors({});
    setTouched({});
  };

  const metrics = calculateMetrics();
  const accountType = detectAccountType();

  return (
    <div className="mep-container">
      {/* Header */}
      <div className="mep-header">
        <button className="mep-back-btn" onClick={onCancel}>
          <ArrowLeft size={20} />
          <span>Back</span>
        </button>
        <h1>Add Account Manually</h1>
        <div className="mep-header-spacer"></div>
      </div>

      {/* Quick Generate Buttons */}
      <div className="mep-quick-generate">
        <button 
          type="button"
          className="mep-generate-btn"
          onClick={() => {
            const accounts = [];
            for (let i = 0; i < 50; i++) {
              const is_bot = Math.random() < 0.4;
              const followers = is_bot ? Math.floor(Math.random() * 50) : Math.floor(Math.random() * 2000) + 50;
              const following = is_bot ? Math.floor(Math.random() * 9000) + 1000 : Math.floor(Math.random() * 950) + 50;
              accounts.push({
                username: is_bot ? `bot_${i}_${Date.now()}` : `user_${i}_${Date.now()}`,
                followers_count: followers,
                friends_count: following,
                statuses_count: is_bot ? Math.floor(Math.random() * 45000) + 5000 : Math.floor(Math.random() * 4900) + 100,
                account_age_days: is_bot ? Math.floor(Math.random() * 89) + 1 : Math.floor(Math.random() * 3285) + 365,
                verified: !is_bot && Math.random() < 0.05,
                has_profile_image: is_bot ? Math.random() < 0.4 : Math.random() < 0.9,
                bio: is_bot ? 'Follow for deals!' : 'Tech enthusiast | Coffee lover',
                location: is_bot ? '' : ['New York', 'San Francisco', 'London'][Math.floor(Math.random() * 3)]
              });
            }
            onGenerate(accounts);
            onCancel();
          }}
        >
          <Zap size={18} />
          Generate 50 Accounts
        </button>
        <button 
          type="button"
          className="mep-generate-btn"
          onClick={() => {
            const accounts = [];
            for (let i = 0; i < 100; i++) {
              const is_bot = Math.random() < 0.4;
              const followers = is_bot ? Math.floor(Math.random() * 50) : Math.floor(Math.random() * 2000) + 50;
              const following = is_bot ? Math.floor(Math.random() * 9000) + 1000 : Math.floor(Math.random() * 950) + 50;
              accounts.push({
                username: is_bot ? `bot_${i}_${Date.now()}` : `user_${i}_${Date.now()}`,
                followers_count: followers,
                friends_count: following,
                statuses_count: is_bot ? Math.floor(Math.random() * 45000) + 5000 : Math.floor(Math.random() * 4900) + 100,
                account_age_days: is_bot ? Math.floor(Math.random() * 89) + 1 : Math.floor(Math.random() * 3285) + 365,
                verified: !is_bot && Math.random() < 0.05,
                has_profile_image: is_bot ? Math.random() < 0.4 : Math.random() < 0.9,
                bio: is_bot ? 'Follow for deals!' : 'Tech enthusiast | Coffee lover',
                location: is_bot ? '' : ['New York', 'San Francisco', 'London'][Math.floor(Math.random() * 3)]
              });
            }
            onGenerate(accounts);
            onCancel();
          }}
        >
          <Zap size={18} />
          Generate 100 Accounts
        </button>
      </div>

      {/* Main Content */}
      <div className="mep-content">
        <div className="mep-form-container">
          <form onSubmit={handleSubmit} className="mep-form">
            {/* Left Column */}
            <div className="mep-column">
              {/* Basic Information */}
              <div className="mep-section">
                <h2 className="mep-section-title">
                  <span className="mep-icon">👤</span>
                  Basic Information
                </h2>
                
                <div className="mep-field">
                  <label className="mep-label">
                    Username
                    {touched.username && !errors.username && <CheckCircle2 size={16} className="mep-success" />}
                    {touched.username && errors.username && <AlertCircle size={16} className="mep-error-icon" />}
                  </label>
                  <input
                    type="text"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    placeholder="@username"
                    className={`mep-input ${touched.username && errors.username ? 'error' : ''}`}
                  />
                  {touched.username && errors.username && (
                    <span className="mep-error-text">{errors.username}</span>
                  )}
                </div>

                <div className="mep-field">
                  <label className="mep-label">Bio</label>
                  <textarea
                    name="bio"
                    value={formData.bio}
                    onChange={handleChange}
                    placeholder="Account bio or description..."
                    className="mep-textarea"
                    rows="2"
                  />
                </div>

                <div className="mep-field">
                  <label className="mep-label">Location</label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleChange}
                    placeholder="City, Country"
                    className="mep-input"
                  />
                </div>
              </div>

              {/* Audience Metrics */}
              <div className="mep-section">
                <h2 className="mep-section-title">
                  <span className="mep-icon">📊</span>
                  Audience Metrics
                </h2>

                <div className="mep-row">
                  <div className="mep-field">
                    <label className="mep-label">
                      Followers
                      {touched.followers_count && !errors.followers_count && <CheckCircle2 size={16} className="mep-success" />}
                      {touched.followers_count && errors.followers_count && <AlertCircle size={16} className="mep-error-icon" />}
                    </label>
                    <input
                      type="number"
                      name="followers_count"
                      value={formData.followers_count}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      placeholder="e.g., 5000"
                      min="0"
                      className={`mep-input ${touched.followers_count && errors.followers_count ? 'error' : ''}`}
                    />
                    {touched.followers_count && errors.followers_count && (
                      <span className="mep-error-text">{errors.followers_count}</span>
                    )}
                  </div>

                  <div className="mep-field">
                    <label className="mep-label">
                      Following
                      {touched.friends_count && !errors.friends_count && <CheckCircle2 size={16} className="mep-success" />}
                      {touched.friends_count && errors.friends_count && <AlertCircle size={16} className="mep-error-icon" />}
                    </label>
                    <input
                      type="number"
                      name="friends_count"
                      value={formData.friends_count}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      placeholder="e.g., 1000"
                      min="0"
                      className={`mep-input ${touched.friends_count && errors.friends_count ? 'error' : ''}`}
                    />
                    {touched.friends_count && errors.friends_count && (
                      <span className="mep-error-text">{errors.friends_count}</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Activity Metrics */}
              <div className="mep-section">
                <h2 className="mep-section-title">
                  <span className="mep-icon">📈</span>
                  Activity Metrics
                </h2>

                <div className="mep-row">
                  <div className="mep-field">
                    <label className="mep-label">
                      Total Posts
                      {touched.statuses_count && !errors.statuses_count && <CheckCircle2 size={16} className="mep-success" />}
                      {touched.statuses_count && errors.statuses_count && <AlertCircle size={16} className="mep-error-icon" />}
                    </label>
                    <input
                      type="number"
                      name="statuses_count"
                      value={formData.statuses_count}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      placeholder="e.g., 15000"
                      min="0"
                      className={`mep-input ${touched.statuses_count && errors.statuses_count ? 'error' : ''}`}
                    />
                  </div>

                  <div className="mep-field">
                    <label className="mep-label">
                      Account Age (Days)
                      {touched.account_age_days && !errors.account_age_days && <CheckCircle2 size={16} className="mep-success" />}
                      {touched.account_age_days && errors.account_age_days && <AlertCircle size={16} className="mep-error-icon" />}
                    </label>
                    <input
                      type="number"
                      name="account_age_days"
                      value={formData.account_age_days}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      placeholder="e.g., 365"
                      min="1"
                      className={`mep-input ${touched.account_age_days && errors.account_age_days ? 'error' : ''}`}
                    />
                  </div>
                </div>
              </div>

              {/* Profile Flags */}
              <div className="mep-section">
                <h2 className="mep-section-title">
                  <span className="mep-icon">🚩</span>
                  Profile Flags
                </h2>

                <div className="mep-checkboxes">
                  <label className="mep-checkbox">
                    <input
                      type="checkbox"
                      name="has_profile_image"
                      checked={formData.has_profile_image}
                      onChange={handleChange}
                    />
                    <span className="mep-checkbox-label">Has Profile Image</span>
                  </label>

                  <label className="mep-checkbox">
                    <input
                      type="checkbox"
                      name="verified"
                      checked={formData.verified}
                      onChange={handleChange}
                    />
                    <span className="mep-checkbox-label">Verified Account</span>
                  </label>

                  <label className="mep-checkbox">
                    <input
                      type="checkbox"
                      name="default_profile"
                      checked={formData.default_profile}
                      onChange={handleChange}
                    />
                    <span className="mep-checkbox-label">Default Profile Design</span>
                  </label>

                  <label className="mep-checkbox">
                    <input
                      type="checkbox"
                      name="default_profile_image"
                      checked={formData.default_profile_image}
                      onChange={handleChange}
                    />
                    <span className="mep-checkbox-label">Default Profile Image</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Right Column - Preview */}
            <div className="mep-column mep-preview-column">
              {/* Account Type Card */}
              <div className="mep-card mep-type-card" style={{ borderTopColor: accountType.color }}>
                <div className="mep-type-header">
                  <span className="mep-type-emoji">
                    {accountType.type.split(' ')[0]}
                  </span>
                  <div>
                    <h3 className="mep-type-name">{accountType.type}</h3>
                    <span className={`mep-risk-badge mep-risk-${accountType.risk.toLowerCase()}`}>
                      {accountType.risk} RISK
                    </span>
                  </div>
                </div>
              </div>

              {/* Metrics Card */}
              <div className="mep-card">
                <h3 className="mep-card-title">Calculated Metrics</h3>
                <div className="mep-metrics">
                  <div className="mep-metric">
                    <span className="mep-metric-label">Posts/Day</span>
                    <span className="mep-metric-value">{metrics.posts_per_day}</span>
                  </div>
                  <div className="mep-metric">
                    <span className="mep-metric-label">Follow Ratio</span>
                    <span className="mep-metric-value">{metrics.follow_ratio}x</span>
                  </div>
                  <div className="mep-metric">
                    <span className="mep-metric-label">Engagement Rate</span>
                    <span className="mep-metric-value">{metrics.engagement_rate}%</span>
                  </div>
                  <div className="mep-metric">
                    <span className="mep-metric-label">Est. Likes/Post</span>
                    <span className="mep-metric-value">{metrics.avg_likes}</span>
                  </div>
                </div>
              </div>

              {/* Summary Card */}
              <div className="mep-card">
                <h3 className="mep-card-title">Account Summary</h3>
                <div className="mep-summary">
                  <div className="mep-summary-row">
                    <span className="mep-summary-label">Username:</span>
                    <span className="mep-summary-value">
                      {formData.username ? `@${formData.username}` : 'Not entered'}
                    </span>
                  </div>
                  <div className="mep-summary-row">
                    <span className="mep-summary-label">Followers:</span>
                    <span className="mep-summary-value">
                      {formData.followers_count ? parseInt(formData.followers_count).toLocaleString() : '0'}
                    </span>
                  </div>
                  <div className="mep-summary-row">
                    <span className="mep-summary-label">Following:</span>
                    <span className="mep-summary-value">
                      {formData.friends_count ? parseInt(formData.friends_count).toLocaleString() : '0'}
                    </span>
                  </div>
                  <div className="mep-summary-row">
                    <span className="mep-summary-label">Total Posts:</span>
                    <span className="mep-summary-value">
                      {formData.statuses_count ? parseInt(formData.statuses_count).toLocaleString() : '0'}
                    </span>
                  </div>
                  <div className="mep-summary-row">
                    <span className="mep-summary-label">Account Age:</span>
                    <span className="mep-summary-value">
                      {formData.account_age_days ? `${formData.account_age_days} days` : 'Not entered'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Status Card */}
              <div className="mep-card mep-status-card">
                <div className="mep-status">
                  <div className="mep-status-indicator" style={{
                    background: isFormValid() ? '#10b981' : '#ef4444'
                  }}></div>
                  <div>
                    <p className="mep-status-title">
                      {isFormValid() ? '✓ Ready to Analyze' : '✗ Complete Required Fields'}
                    </p>
                    <p className="mep-status-text">
                      {isFormValid()
                        ? 'All required fields are filled and valid'
                        : `${Object.keys(errors).length} validation error(s)`
                      }
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="mep-footer">
        <button
          type="button"
          onClick={handleReset}
          className="mep-btn mep-btn-secondary"
        >
          <RefreshCw size={18} />
          Reset
        </button>

        <div className="mep-footer-spacer"></div>

        <button
          type="button"
          onClick={onCancel}
          className="mep-btn mep-btn-cancel"
        >
          Cancel
        </button>

        <button
          type="submit"
          onClick={handleSubmit}
          disabled={!isFormValid()}
          className="mep-btn mep-btn-primary"
        >
          <Save size={18} />
          Generate & Analyze
        </button>
      </div>
    </div>
  );
}
