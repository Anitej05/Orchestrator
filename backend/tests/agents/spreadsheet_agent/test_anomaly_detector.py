"""
Unit tests for Anomaly Detector

Tests detection of dtype drift, missing values, and outliers
"""

import pytest
import pandas as pd
import numpy as np
from backend.agents.spreadsheet_agent.anomaly_detector import (
    AnomalyDetector, Anomaly, AnomalyFix
)


@pytest.fixture
def detector():
    """Create an AnomalyDetector instance"""
    return AnomalyDetector(
        dtype_drift_threshold=0.8,
        missing_value_threshold=0.2,
        outlier_std_threshold=3.0
    )


class TestDtypeDriftDetection:
    """Test dtype drift detection"""
    
    def test_detect_numeric_as_text(self, detector):
        """Test detection of numeric data stored as text"""
        df = pd.DataFrame({
            'Revenue': ['100', '200', '300', '400', 'N/A'],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        anomalies = detector.detect_dtype_drift(df, ['Revenue'])
        
        assert len(anomalies) == 1
        assert anomalies[0].type == 'dtype_drift'
        assert 'Revenue' in anomalies[0].columns
        assert anomalies[0].metadata['numeric_percentage'] == 0.8
        assert 'N/A' in anomalies[0].sample_values['Revenue']
    
    def test_no_drift_for_pure_text(self, detector):
        """Test no drift detected for pure text columns"""
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        anomalies = detector.detect_dtype_drift(df, ['Name'])
        
        assert len(anomalies) == 0
    
    def test_no_drift_for_numeric_columns(self, detector):
        """Test no drift detected for already numeric columns"""
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45]
        })
        
        anomalies = detector.detect_dtype_drift(df, ['Age'])
        
        assert len(anomalies) == 0
    
    def test_drift_fix_suggestions(self, detector):
        """Test that dtype drift includes fix suggestions"""
        df = pd.DataFrame({
            'Revenue': ['100', '200', '300', '400', 'N/A']
        })
        
        anomalies = detector.detect_dtype_drift(df, ['Revenue'])
        
        assert len(anomalies) == 1
        fixes = anomalies[0].suggested_fixes
        assert len(fixes) >= 3
        
        # Check for expected fix actions
        fix_actions = [fix.action for fix in fixes]
        assert 'convert_numeric' in fix_actions
        assert 'ignore_rows' in fix_actions
        assert 'treat_as_text' in fix_actions


class TestMissingValueDetection:
    """Test missing value detection"""
    
    def test_detect_excessive_missing(self, detector):
        """Test detection of excessive missing values"""
        df = pd.DataFrame({
            'Age': [25, np.nan, np.nan, 40, np.nan],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        anomalies = detector.detect_missing_values(df, ['Age'])
        
        assert len(anomalies) == 1
        assert anomalies[0].type == 'missing_values'
        assert 'Age' in anomalies[0].columns
        assert anomalies[0].metadata['missing_percentage'] == 0.6
    
    def test_no_detection_below_threshold(self, detector):
        """Test no detection when missing values below threshold"""
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, np.nan]
        })
        
        anomalies = detector.detect_missing_values(df, ['Age'])
        
        # 20% missing is at threshold, should not trigger (>20% required)
        assert len(anomalies) == 0
    
    def test_missing_value_fix_suggestions(self, detector):
        """Test that missing values include fix suggestions"""
        df = pd.DataFrame({
            'Age': [25, np.nan, np.nan, 40, np.nan]
        })
        
        anomalies = detector.detect_missing_values(df, ['Age'])
        
        assert len(anomalies) == 1
        fixes = anomalies[0].suggested_fixes
        assert len(fixes) >= 4
        
        # Check for expected fix actions
        fix_actions = [fix.action for fix in fixes]
        assert 'drop_rows' in fix_actions
        assert 'fill_mean' in fix_actions
        assert 'fill_median' in fix_actions


class TestOutlierDetection:
    """Test outlier detection"""
    
    def test_detect_outliers(self, detector):
        """Test detection of outliers"""
        # Create data with clear outlier (need at least 10 points)
        df = pd.DataFrame({
            'Salary': [50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 200000]
        })
        
        anomalies = detector.detect_outliers(df, ['Salary'])
        
        assert len(anomalies) == 1
        assert anomalies[0].type == 'outliers'
        assert 'Salary' in anomalies[0].columns
        assert 200000 in anomalies[0].sample_values['Salary']
    
    def test_no_outliers_in_uniform_data(self, detector):
        """Test no outliers detected in uniform data"""
        df = pd.DataFrame({
            'Age': [25, 26, 27, 28, 29, 30]
        })
        
        anomalies = detector.detect_outliers(df, ['Age'])
        
        assert len(anomalies) == 0


class TestAnomalyFixApplication:
    """Test applying fixes to anomalies"""
    
    def test_apply_convert_numeric_fix(self, detector):
        """Test applying convert_numeric fix"""
        df = pd.DataFrame({
            'Revenue': ['100', '200', '300', 'N/A']
        })
        
        fix = AnomalyFix(
            action='convert_numeric',
            description='Convert to numeric',
            safe=True,
            parameters={'column': 'Revenue', 'method': 'coerce'}
        )
        
        anomaly = Anomaly(
            type='dtype_drift',
            columns=['Revenue'],
            sample_values={'Revenue': ['N/A']},
            suggested_fixes=[fix]
        )
        
        result_df = detector.apply_fix(df, anomaly, fix)
        
        assert result_df['Revenue'].dtype in [np.float64, np.int64, 'float64', 'int64']
        assert pd.isna(result_df['Revenue'].iloc[3])
    
    def test_apply_ignore_rows_fix(self, detector):
        """Test applying ignore_rows fix"""
        df = pd.DataFrame({
            'Revenue': ['100', '200', '300', 'N/A']
        })
        
        fix = AnomalyFix(
            action='ignore_rows',
            description='Filter non-numeric rows',
            safe=False,
            parameters={'column': 'Revenue', 'keep_numeric_only': True}
        )
        
        anomaly = Anomaly(
            type='dtype_drift',
            columns=['Revenue'],
            sample_values={'Revenue': ['N/A']},
            suggested_fixes=[fix]
        )
        
        result_df = detector.apply_fix(df, anomaly, fix)
        
        assert len(result_df) == 3  # One row removed
    
    def test_apply_fill_mean_fix(self, detector):
        """Test applying fill_mean fix"""
        df = pd.DataFrame({
            'Age': [25.0, 30.0, np.nan, 40.0]
        })
        
        fix = AnomalyFix(
            action='fill_mean',
            description='Fill with mean',
            safe=True,
            parameters={'column': 'Age', 'method': 'mean'}
        )
        
        anomaly = Anomaly(
            type='missing_values',
            columns=['Age'],
            sample_values={'Age': []},
            suggested_fixes=[fix]
        )
        
        result_df = detector.apply_fix(df, anomaly, fix)
        
        assert not result_df['Age'].isna().any()
        # Mean of [25, 30, 40] = 31.67
        assert abs(result_df['Age'].iloc[2] - 31.67) < 0.1


class TestDetectAllAnomalies:
    """Test detecting all anomalies at once"""
    
    def test_detect_multiple_anomalies(self, detector):
        """Test detection of multiple anomaly types"""
        # Create DataFrame with same length for all columns
        df = pd.DataFrame({
            'Revenue': ['100', '200', '300', 'N/A', '400', '500', '600', '700', '800', '900', '1000'],  # Dtype drift
            'Age': [25, np.nan, np.nan, 40, 35, 38, 42, 45, 48, 50, 52],  # Missing values (18%)
            'Salary': [50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 200000]  # Outlier
        })
        
        anomalies = detector.detect_anomalies(df)
        
        # Should detect dtype drift at minimum
        assert len(anomalies) >= 1
        
        anomaly_types = [a.type for a in anomalies]
        assert 'dtype_drift' in anomaly_types
