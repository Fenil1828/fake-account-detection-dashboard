"""
Secure ML Security Module for Fake Account Detection
Implements: Data Poisoning Detection, Adversarial Robustness, Input Validation
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from datetime import datetime
import hashlib
import json
import os

class SecureDataValidator:
    """Validates training data to prevent poisoning attacks"""
    
    def __init__(self, contamination=0.1):
        self.anomaly_detector = IsolationForest(contamination=contamination, random_state=42)
        self.baseline_stats = None
        self.security_log = []
    
    def establish_baseline(self, clean_data):
        """Store baseline statistics from clean training data"""
        self.baseline_stats = {
            'mean': clean_data.mean(),
            'std': clean_data.std(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'shape': clean_data.shape
        }
        self.anomaly_detector.fit(clean_data)
        self.log_event("BASELINE_ESTABLISHED", "Clean data baseline recorded")
    
    def validate_training_data(self, data):
        """Multi-layer validation for training data"""
        results = {'valid': True, 'checks': [], 'anomalies': []}
        
        # Check 1: Statistical Outliers (Z-score)
        z_outliers = self.detect_z_score_outliers(data)
        results['checks'].append({
            'name': 'Z-Score Outliers',
            'passed': len(z_outliers) < len(data) * 0.1,
            'outlier_count': len(z_outliers)
        })
        
        # Check 2: IQR Outliers
        iqr_outliers = self.detect_iqr_outliers(data)
        results['checks'].append({
            'name': 'IQR Outliers',
            'passed': len(iqr_outliers) < len(data) * 0.15,
            'outlier_count': len(iqr_outliers)
        })
        
        # Check 3: Isolation Forest Anomalies
        if hasattr(self.anomaly_detector, 'estimators_'):
            anomaly_scores = self.anomaly_detector.predict(data)
            anomaly_count = np.sum(anomaly_scores == -1)
            results['checks'].append({
                'name': 'Isolation Forest',
                'passed': anomaly_count < len(data) * 0.15,
                'anomaly_count': int(anomaly_count)
            })
            results['anomalies'] = np.where(anomaly_scores == -1)[0].tolist()[:10]
        
        # Check 4: Distribution Consistency
        if self.baseline_stats:
            dist_valid = self.check_distribution_consistency(data)
            results['checks'].append({
                'name': 'Distribution Consistency',
                'passed': dist_valid,
                'details': 'Matches baseline distribution'
            })
        
        # Overall validity
        results['valid'] = all(c['passed'] for c in results['checks'])
        
        if not results['valid']:
            self.log_event("POISONING_DETECTED", f"Data validation failed: {results['checks']}", "CRITICAL")
        
        return results
    
    def detect_z_score_outliers(self, data, threshold=3):
        """Detect outliers using Z-score method"""
        numeric_data = data.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_data, nan_policy='omit'))
        outliers = np.where(z_scores > threshold)
        return np.unique(outliers[0])
    
    def detect_iqr_outliers(self, data):
        """Detect outliers using IQR method"""
        numeric_data = data.select_dtypes(include=[np.number])
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
        return np.where(outliers)[0]
    
    def check_distribution_consistency(self, data):
        """Check if new data matches baseline distribution"""
        if not self.baseline_stats:
            return True
        
        numeric_data = data.select_dtypes(include=[np.number])
        new_mean = numeric_data.mean()
        
        # Check if means are within 2 standard deviations
        for col in new_mean.index:
            if col in self.baseline_stats['mean'].index:
                diff = abs(new_mean[col] - self.baseline_stats['mean'][col])
                if diff > 2 * self.baseline_stats['std'].get(col, 1):
                    return False
        return True
    
    def log_event(self, event_type, message, severity="INFO"):
        """Log security event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'message': message,
            'severity': severity
        }
        self.security_log.append(event)
        return event


class AdversarialDetector:
    """Detects adversarial inputs at prediction time"""
    
    def __init__(self):
        self.feature_bounds = {}
        self.training_stats = {}
        self.detection_log = []
    
    def fit(self, X_train):
        """Learn normal input characteristics"""
        self.training_stats = {
            'mean': np.mean(X_train, axis=0),
            'std': np.std(X_train, axis=0),
            'min': np.min(X_train, axis=0),
            'max': np.max(X_train, axis=0)
        }
        # Set bounds with some tolerance
        self.feature_bounds = {
            'min': self.training_stats['min'] - 2 * self.training_stats['std'],
            'max': self.training_stats['max'] + 2 * self.training_stats['std']
        }
    
    def is_adversarial(self, X):
        """Check if input appears adversarial"""
        results = {'is_adversarial': False, 'checks': [], 'score': 0}
        
        if len(self.training_stats) == 0:
            return results
        
        X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
        
        # Check 1: Out of bounds
        out_of_bounds = np.sum((X < self.feature_bounds['min']) | (X > self.feature_bounds['max']))
        results['checks'].append({
            'name': 'Bounds Check',
            'suspicious': out_of_bounds > 0,
            'out_of_bounds_features': int(out_of_bounds)
        })
        
        # Check 2: Mahalanobis-like distance
        z_scores = np.abs((X - self.training_stats['mean']) / (self.training_stats['std'] + 1e-10))
        max_z = np.max(z_scores)
        results['checks'].append({
            'name': 'Statistical Distance',
            'suspicious': max_z > 5,
            'max_z_score': float(max_z)
        })
        
        # Check 3: Unusual feature combinations
        extreme_features = np.sum(z_scores > 3)
        results['checks'].append({
            'name': 'Extreme Features',
            'suspicious': extreme_features > len(X[0]) * 0.3,
            'count': int(extreme_features)
        })
        
        # Calculate overall score
        suspicious_count = sum(1 for c in results['checks'] if c['suspicious'])
        results['score'] = suspicious_count / len(results['checks'])
        results['is_adversarial'] = results['score'] > 0.5
        
        if results['is_adversarial']:
            self.detection_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'ADVERSARIAL_INPUT',
                'details': results
            })
        
        return results


class SecureInputValidator:
    """Validates and sanitizes API inputs"""
    
    def __init__(self):
        self.rate_limits = {}
        self.blocked_ips = set()
        self.validation_log = []
    
    def validate_input(self, data, ip_address=None):
        """Comprehensive input validation"""
        results = {'valid': True, 'errors': [], 'sanitized': {}}
        
        # Check rate limiting
        if ip_address:
            if not self.check_rate_limit(ip_address):
                results['valid'] = False
                results['errors'].append('Rate limit exceeded')
                return results
        
        # Validate required fields
        required = ['username', 'followers_count', 'friends_count']
        for field in required:
            if field not in data:
                results['errors'].append(f'Missing required field: {field}')
        
        # Type and range validation
        numeric_fields = {
            'followers_count': (0, 1e9),
            'friends_count': (0, 1e9),
            'statuses_count': (0, 1e9),
            'account_age_days': (0, 10000)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in data:
                try:
                    value = float(data[field])
                    if value < min_val or value > max_val:
                        results['errors'].append(f'{field} out of valid range')
                    results['sanitized'][field] = max(min_val, min(max_val, value))
                except (ValueError, TypeError):
                    results['errors'].append(f'{field} must be numeric')
        
        # SQL Injection detection
        if self.detect_injection(data):
            results['valid'] = False
            results['errors'].append('Potential injection attack detected')
            self.log_security_event('INJECTION_ATTEMPT', data, ip_address)
        
        # String sanitization
        if 'username' in data:
            results['sanitized']['username'] = self.sanitize_string(str(data['username']))
        
        results['valid'] = len(results['errors']) == 0
        return results
    
    def detect_injection(self, data):
        """Detect SQL/NoSQL/XSS injection attempts"""
        dangerous_patterns = [
            '<script', 'javascript:', 'DROP TABLE', "'; --", 
            'UNION SELECT', '<iframe', 'onerror=', 'onload=',
            '${', '{{', '__proto__', 'constructor'
        ]
        
        for value in data.values():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if pattern.lower() in value.lower():
                        return True
        return False
    
    def sanitize_string(self, s, max_length=100):
        """Sanitize string input"""
        import re
        # Remove special characters, keep alphanumeric and basic punctuation
        sanitized = re.sub(r'[<>"\';(){}]', '', s)
        return sanitized[:max_length]
    
    def check_rate_limit(self, ip_address, max_requests=100, window_seconds=3600):
        """Simple rate limiting"""
        now = datetime.utcnow()
        
        if ip_address in self.blocked_ips:
            return False
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Clean old requests
        self.rate_limits[ip_address] = [
            t for t in self.rate_limits[ip_address]
            if (now - t).total_seconds() < window_seconds
        ]
        
        if len(self.rate_limits[ip_address]) >= max_requests:
            return False
        
        self.rate_limits[ip_address].append(now)
        return True
    
    def log_security_event(self, event_type, data, ip):
        """Log security events"""
        self.validation_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'ip': ip,
            'data_hash': hashlib.sha256(str(data).encode()).hexdigest()[:16]
        })


class SecurityMonitor:
    """Real-time security monitoring"""
    
    def __init__(self):
        self.events = []
        self.threat_scores = []
        self.metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'poisoning_attempts': 0,
            'adversarial_inputs': 0,
            'injection_attempts': 0
        }
    
    def log_event(self, event_type, details, severity="INFO"):
        """Log security event"""
        event = {
            'id': len(self.events) + 1,
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'details': details,
            'severity': severity
        }
        self.events.append(event)
        
        # Update metrics
        if event_type == 'POISONING_ATTEMPT':
            self.metrics['poisoning_attempts'] += 1
        elif event_type == 'ADVERSARIAL_INPUT':
            self.metrics['adversarial_inputs'] += 1
        elif event_type == 'INJECTION_ATTEMPT':
            self.metrics['injection_attempts'] += 1
        elif event_type == 'BLOCKED':
            self.metrics['blocked_requests'] += 1
        
        self.metrics['total_requests'] += 1
        
        # Calculate threat score
        self.calculate_threat_score()
        
        return event
    
    def calculate_threat_score(self):
        """Calculate current threat level (0-100)"""
        if self.metrics['total_requests'] == 0:
            score = 0
        else:
            threat_count = (
                self.metrics['poisoning_attempts'] * 3 +
                self.metrics['adversarial_inputs'] * 2 +
                self.metrics['injection_attempts'] * 3 +
                self.metrics['blocked_requests']
            )
            score = min(100, (threat_count / max(1, self.metrics['total_requests'])) * 100 * 10)
        
        self.threat_scores.append({
            'timestamp': datetime.utcnow().isoformat(),
            'score': score
        })
        return score
    
    def get_status(self):
        """Get current security status"""
        threat_score = self.threat_scores[-1]['score'] if self.threat_scores else 0
        
        return {
            'status': 'SECURE' if threat_score < 30 else 'WARNING' if threat_score < 70 else 'CRITICAL',
            'threat_score': threat_score,
            'metrics': self.metrics,
            'recent_events': self.events[-10:],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_threat_history(self, limit=50):
        """Get threat score history"""
        return self.threat_scores[-limit:]


# Initialize global security components
security_monitor = SecurityMonitor()
data_validator = SecureDataValidator()
adversarial_detector = AdversarialDetector()
input_validator = SecureInputValidator()
