import numpy as np
from scipy.stats import ks_2samp, entropy
from typing import Dict, Any

def calculate_ks_drift(expected: list, actual: list, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Performs Kolmogorov-Smirnov (KS) test on two numerical distributions.
    Suitable for continuous numerical features and regression predictions.
    
    If p-value < alpha, we reject the null hypothesis (meaning distributions differ significantly).
    """
    if not expected or not actual:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}
    
    stat, p_value = ks_2samp(expected, actual)
    
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha)
    }

def calculate_psi(expected: list, actual: list, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and actual distributions.
    Suitable for classification class distributions and discrete categories.
    
    Interpretation:
    PSI < 0.1: Stable / No significant change
    0.1 <= PSI < 0.2: Moderate Change / Warning
    PSI >= 0.2: Significant Shift / Critical Drift
    """
    if not expected or not actual:
        return 0.0
    
    expected = np.array(expected)
    actual = np.array(actual)
    
    try:
        # Generate bin boundaries based on baseline percentiles
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, percentiles)
        bins = np.unique(bins)
        
        # Fallback if baseline is flat
        if len(bins) < 2:
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())
            if min_val == max_val:
                # All values are identical
                return 0.0
            bins = np.linspace(min_val, max_val, num_buckets + 1)
            bins = np.unique(bins)
            
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        
        # Convert counts to proportions
        expected_pcts = expected_counts / len(expected)
        actual_pcts = actual_counts / len(actual)
        
        # Add epsilon to prevent division by zero / log of zero
        expected_pcts = np.clip(expected_pcts, 0.0001, 1.0)
        actual_pcts = np.clip(actual_pcts, 0.0001, 1.0)
        
        # Formula: PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        psi_value = np.sum((expected_pcts - actual_pcts) * np.log(expected_pcts / actual_pcts))
        return float(psi_value)
    except Exception:
        # Fallback if any unexpected calculations occur
        return 0.0

def detect_zscore_anomaly(current_value: float, history: list) -> float:
    """
    Calculates the Z-score of a metric compared to its historical distribution.
    Useful for identifying latency anomalies.
    """
    if len(history) < 5:
        return 0.0
    
    mean = np.mean(history)
    std = np.std(history)
    
    if std == 0:
        return 0.0
    
    return float(abs((current_value - mean) / std))
