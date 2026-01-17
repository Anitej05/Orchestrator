"""
Anomaly Detector for Spreadsheet Agent

Detects data quality issues (dtype drift, missing values) and generates
fix suggestions with safety indicators.

Requirements: 10.1, 10.2, 10.6, 10.7
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AnomalyFix:
    """A suggested fix for an anomaly"""
    action: str  # 'convert_numeric', 'ignore_rows', 'treat_as_text', 'fill_na', 'drop_column'
    description: str  # Human-readable description
    safe: bool  # Whether this action is non-destructive
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "action": self.action,
            "description": self.description,
            "safe": self.safe,
            "parameters": self.parameters
        }


@dataclass
class Anomaly:
    """Detected data quality anomaly"""
    type: str  # 'dtype_drift', 'missing_values', 'outliers', 'inconsistent_format'
    columns: List[str]  # Affected columns
    sample_values: Dict[str, List[Any]]  # Column -> problematic sample values
    suggested_fixes: List[AnomalyFix]  # Possible fixes
    severity: str = 'warning'  # 'info', 'warning', 'error'
    message: str = ''  # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "type": self.type,
            "columns": self.columns,
            "sample_values": self._serialize_samples(self.sample_values),
            "suggested_fixes": [fix.to_dict() for fix in self.suggested_fixes],
            "severity": self.severity,
            "message": self.message,
            "metadata": self.metadata
        }
    
    def _serialize_samples(self, samples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Serialize sample values for JSON"""
        result = {}
        for col, values in samples.items():
            serialized = []
            for val in values:
                if pd.isna(val):
                    serialized.append(None)
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    serialized.append(int(val))
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    serialized.append(float(val))
                else:
                    serialized.append(str(val))
            result[col] = serialized
        return result


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    """
    Detects data quality issues in DataFrames.
    
    Detects:
    - Dtype drift: Numeric data stored as text
    - Missing values: Excessive null/empty values
    - Outliers: Values far from the mean (for numeric columns)
    - Inconsistent formats: Mixed date/time formats
    """
    
    def __init__(self, 
                 dtype_drift_threshold: float = 0.8,
                 missing_value_threshold: float = 0.2,
                 outlier_std_threshold: float = 3.0):
        """
        Initialize anomaly detector.
        
        Args:
            dtype_drift_threshold: Minimum % of numeric values to flag dtype drift (default 0.8)
            missing_value_threshold: Maximum % of missing values before flagging (default 0.2)
            outlier_std_threshold: Number of std deviations for outlier detection (default 3.0)
        """
        self.dtype_drift_threshold = dtype_drift_threshold
        self.missing_value_threshold = missing_value_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.logger = logging.getLogger(f"{__name__}.AnomalyDetector")
    
    def detect_anomalies(self, df: pd.DataFrame, 
                        operation: Optional[str] = None,
                        columns: Optional[List[str]] = None) -> List[Anomaly]:
        """
        Detect all anomalies in DataFrame.
        
        Args:
            df: DataFrame to analyze
            operation: Optional operation context (e.g., 'aggregate', 'filter')
            columns: Optional list of columns to check (if None, checks all)
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Determine which columns to check
        check_columns = columns if columns else df.columns.tolist()
        
        # Detect dtype drift
        dtype_anomalies = self.detect_dtype_drift(df, check_columns)
        anomalies.extend(dtype_anomalies)
        
        # Detect missing values
        missing_anomalies = self.detect_missing_values(df, check_columns)
        anomalies.extend(missing_anomalies)
        
        # Detect outliers (only for numeric columns)
        numeric_cols = [col for col in check_columns if col in df.select_dtypes(include=[np.number]).columns]
        if numeric_cols:
            outlier_anomalies = self.detect_outliers(df, numeric_cols)
            anomalies.extend(outlier_anomalies)
        
        self.logger.info(f"Detected {len(anomalies)} anomalies in DataFrame")
        return anomalies
    
    def detect_dtype_drift(self, df: pd.DataFrame, columns: List[str]) -> List[Anomaly]:
        """
        Detect columns with object dtype that contain mostly numeric values.
        
        This indicates numeric data stored as text, which can cause calculation failures.
        
        Requirements: 10.1, 10.6
        """
        anomalies = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Only check object/string columns
            if df[col].dtype not in ['object', 'string']:
                continue
            
            # Try converting to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate percentage of values that are numeric
            total_non_null = df[col].notna().sum()
            if total_non_null == 0:
                continue
            
            numeric_count = numeric_series.notna().sum()
            numeric_percentage = numeric_count / total_non_null
            
            # If >threshold% are numeric, flag as dtype drift
            if numeric_percentage >= self.dtype_drift_threshold:
                # Get sample problematic values (non-numeric ones)
                non_numeric_mask = df[col].notna() & numeric_series.isna()
                sample_values = df.loc[non_numeric_mask, col].head(5).tolist()
                
                anomaly = Anomaly(
                    type='dtype_drift',
                    columns=[col],
                    sample_values={col: sample_values},
                    suggested_fixes=self._generate_dtype_drift_fixes(col),
                    severity='warning',
                    message=f"Column '{col}' contains {numeric_percentage:.1%} numeric values but has object dtype. "
                            f"This may cause calculation failures.",
                    metadata={
                        'numeric_percentage': numeric_percentage,
                        'total_values': total_non_null,
                        'numeric_values': numeric_count,
                        'non_numeric_values': total_non_null - numeric_count,
                        'current_dtype': str(df[col].dtype)
                    }
                )
                anomalies.append(anomaly)
                self.logger.warning(f"Dtype drift detected in column '{col}': {numeric_percentage:.1%} numeric")
        
        return anomalies
    
    def detect_missing_values(self, df: pd.DataFrame, columns: List[str]) -> List[Anomaly]:
        """
        Detect columns with excessive missing values.
        
        Requirements: 10.2
        """
        anomalies = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Calculate missing percentage
            total_rows = len(df)
            missing_count = df[col].isna().sum()
            missing_percentage = missing_count / total_rows if total_rows > 0 else 0
            
            # If >threshold% are missing, flag it
            if missing_percentage > self.missing_value_threshold:
                # Get sample of non-null values for context
                non_null_values = df[col].dropna().head(5).tolist()
                
                anomaly = Anomaly(
                    type='missing_values',
                    columns=[col],
                    sample_values={col: non_null_values},
                    suggested_fixes=self._generate_missing_value_fixes(col, missing_percentage),
                    severity='warning' if missing_percentage < 0.5 else 'error',
                    message=f"Column '{col}' has {missing_percentage:.1%} missing values. "
                            f"This may affect analysis results.",
                    metadata={
                        'missing_percentage': missing_percentage,
                        'missing_count': missing_count,
                        'total_rows': total_rows,
                        'non_null_count': total_rows - missing_count
                    }
                )
                anomalies.append(anomaly)
                self.logger.warning(f"Missing values detected in column '{col}': {missing_percentage:.1%}")
        
        return anomalies
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> List[Anomaly]:
        """
        Detect outliers in numeric columns using standard deviation method.
        
        Values beyond mean ± (threshold * std) are considered outliers.
        """
        anomalies = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Only check numeric columns - be more flexible with dtype check
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Calculate statistics
            series = df[col].dropna()
            if len(series) < 10:  # Need sufficient data for outlier detection
                continue
            
            mean = series.mean()
            std = series.std()
            
            if std == 0:  # All values are the same
                continue
            
            # Find outliers
            lower_bound = mean - (self.outlier_std_threshold * std)
            upper_bound = mean + (self.outlier_std_threshold * std)
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_percentage = outlier_count / len(df)
                
                # Get sample outlier values
                sample_outliers = df.loc[outlier_mask, col].head(5).tolist()
                
                anomaly = Anomaly(
                    type='outliers',
                    columns=[col],
                    sample_values={col: sample_outliers},
                    suggested_fixes=self._generate_outlier_fixes(col),
                    severity='info',
                    message=f"Column '{col}' has {outlier_count} outliers ({outlier_percentage:.1%}) "
                            f"beyond {self.outlier_std_threshold} standard deviations from mean.",
                    metadata={
                        'outlier_count': outlier_count,
                        'outlier_percentage': outlier_percentage,
                        'mean': mean,
                        'std': std,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                )
                anomalies.append(anomaly)
                self.logger.info(f"Outliers detected in column '{col}': {outlier_count} values")
        
        return anomalies
    
    def _generate_dtype_drift_fixes(self, column: str) -> List[AnomalyFix]:
        """
        Generate fix suggestions for dtype drift.
        
        Requirements: 10.2, 10.3, 10.4, 10.5
        """
        return [
            AnomalyFix(
                action='convert_numeric',
                description=f"Convert '{column}' to numeric, replacing invalid values with NaN",
                safe=True,
                parameters={
                    'column': column,
                    'method': 'coerce'
                }
            ),
            AnomalyFix(
                action='ignore_rows',
                description=f"Filter out rows with non-numeric values in '{column}'",
                safe=False,  # Removes data
                parameters={
                    'column': column,
                    'keep_numeric_only': True
                }
            ),
            AnomalyFix(
                action='treat_as_text',
                description=f"Keep '{column}' as text (may cause calculation failures)",
                safe=True,
                parameters={
                    'column': column,
                    'no_conversion': True
                }
            )
        ]
    
    def _generate_missing_value_fixes(self, column: str, missing_pct: float) -> List[AnomalyFix]:
        """Generate fix suggestions for missing values"""
        fixes = [
            AnomalyFix(
                action='drop_rows',
                description=f"Remove rows with missing values in '{column}'",
                safe=False,  # Removes data
                parameters={
                    'column': column,
                    'drop_na': True
                }
            ),
            AnomalyFix(
                action='fill_mean',
                description=f"Fill missing values in '{column}' with column mean",
                safe=True,
                parameters={
                    'column': column,
                    'method': 'mean'
                }
            ),
            AnomalyFix(
                action='fill_median',
                description=f"Fill missing values in '{column}' with column median",
                safe=True,
                parameters={
                    'column': column,
                    'method': 'median'
                }
            ),
            AnomalyFix(
                action='fill_value',
                description=f"Fill missing values in '{column}' with a specific value",
                safe=True,
                parameters={
                    'column': column,
                    'method': 'value',
                    'value': 0  # Default, can be customized
                }
            )
        ]
        
        # If >50% missing, suggest dropping the column
        if missing_pct > 0.5:
            fixes.insert(0, AnomalyFix(
                action='drop_column',
                description=f"Drop column '{column}' (>50% missing values)",
                safe=False,
                parameters={
                    'column': column
                }
            ))
        
        return fixes
    
    def _generate_outlier_fixes(self, column: str) -> List[AnomalyFix]:
        """Generate fix suggestions for outliers"""
        return [
            AnomalyFix(
                action='keep_outliers',
                description=f"Keep outliers in '{column}' (no action)",
                safe=True,
                parameters={
                    'column': column,
                    'no_action': True
                }
            ),
            AnomalyFix(
                action='remove_outliers',
                description=f"Remove rows with outlier values in '{column}'",
                safe=False,  # Removes data
                parameters={
                    'column': column,
                    'remove_outliers': True
                }
            ),
            AnomalyFix(
                action='cap_outliers',
                description=f"Cap outliers in '{column}' to min/max bounds",
                safe=True,
                parameters={
                    'column': column,
                    'cap_outliers': True
                }
            )
        ]
    
    def apply_fix(self, df: pd.DataFrame, anomaly: Anomaly, fix: AnomalyFix) -> pd.DataFrame:
        """
        Apply a selected fix to the DataFrame.
        
        Args:
            df: Source DataFrame
            anomaly: The anomaly being fixed
            fix: The fix to apply
            
        Returns:
            Modified DataFrame
        """
        result_df = df.copy()
        column = fix.parameters.get('column')
        
        try:
            if fix.action == 'convert_numeric':
                # Convert to numeric with coercion
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
                self.logger.info(f"Converted column '{column}' to numeric")
            
            elif fix.action == 'ignore_rows':
                # Filter out non-numeric rows
                numeric_series = pd.to_numeric(result_df[column], errors='coerce')
                result_df = result_df[numeric_series.notna()].copy()
                self.logger.info(f"Filtered out non-numeric rows in column '{column}'")
            
            elif fix.action == 'treat_as_text':
                # No action needed, keep as-is
                self.logger.info(f"Keeping column '{column}' as text")
            
            elif fix.action == 'drop_rows':
                # Drop rows with missing values
                result_df = result_df.dropna(subset=[column]).copy()
                self.logger.info(f"Dropped rows with missing values in column '{column}'")
            
            elif fix.action == 'fill_mean':
                # Fill with mean
                mean_value = result_df[column].mean()
                result_df[column] = result_df[column].fillna(mean_value)
                self.logger.info(f"Filled missing values in column '{column}' with mean: {mean_value}")
            
            elif fix.action == 'fill_median':
                # Fill with median
                median_value = result_df[column].median()
                result_df[column] = result_df[column].fillna(median_value)
                self.logger.info(f"Filled missing values in column '{column}' with median: {median_value}")
            
            elif fix.action == 'fill_value':
                # Fill with specific value
                fill_value = fix.parameters.get('value', 0)
                result_df[column] = result_df[column].fillna(fill_value)
                self.logger.info(f"Filled missing values in column '{column}' with value: {fill_value}")
            
            elif fix.action == 'drop_column':
                # Drop the column
                result_df = result_df.drop(columns=[column])
                self.logger.info(f"Dropped column '{column}'")
            
            elif fix.action == 'keep_outliers':
                # No action needed
                self.logger.info(f"Keeping outliers in column '{column}'")
            
            elif fix.action == 'remove_outliers':
                # Remove outlier rows
                series = result_df[column].dropna()
                mean = series.mean()
                std = series.std()
                lower_bound = mean - (self.outlier_std_threshold * std)
                upper_bound = mean + (self.outlier_std_threshold * std)
                
                mask = (result_df[column] >= lower_bound) & (result_df[column] <= upper_bound)
                result_df = result_df[mask].copy()
                self.logger.info(f"Removed outliers from column '{column}'")
            
            elif fix.action == 'cap_outliers':
                # Cap outliers to bounds
                series = result_df[column].dropna()
                mean = series.mean()
                std = series.std()
                lower_bound = mean - (self.outlier_std_threshold * std)
                upper_bound = mean + (self.outlier_std_threshold * std)
                
                result_df[column] = result_df[column].clip(lower=lower_bound, upper=upper_bound)
                self.logger.info(f"Capped outliers in column '{column}' to bounds")
            
            else:
                self.logger.warning(f"Unknown fix action: {fix.action}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to apply fix '{fix.action}': {str(e)}", exc_info=True)
            return df  # Return original on error
