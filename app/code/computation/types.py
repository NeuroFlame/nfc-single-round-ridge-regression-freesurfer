"""Data structures exchanged by ridge workflow steps."""

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class RidgeInputs:
    """Hold validated covariates, outcomes, and ridge penalty."""

    X: pd.DataFrame
    y: pd.DataFrame
    lambda_value: float

    @property
    def covariate_labels(self) -> List[str]:
        """Return covariate column labels in model order."""
        return list(self.X.columns)

    @property
    def roi_labels(self) -> List[str]:
        """Return region-of-interest labels in model order."""
        return list(self.y.columns)


@dataclass
class CachedLocalState:
    """Persist validated site data between local workflow steps."""

    X: pd.DataFrame
    y: pd.DataFrame
    lambda_value: float


@dataclass
class LocalRoiStats:
    """Describe one site's fitted statistics for one region of interest."""

    coefficient: List[float]
    t_stat: List[float]
    p_value: List[float]
    r_squared: float
    degrees_of_freedom: float
    covariate_labels: List[str]
    sum_square_of_errors: float
    roi_label: str
    y_labels: List[str]
    mean_y_local: float
    num_subjects: int


@dataclass
class LocalModelSummary:
    """Collect local model statistics by region of interest."""

    roi_stats: Dict[str, LocalRoiStats]

    @property
    def roi_labels(self) -> List[str]:
        """Return region-of-interest labels represented by this summary."""
        if not self.roi_stats:
            return []
        first_roi = next(iter(self.roi_stats.values()))
        return list(first_roi.y_labels)


@dataclass
class GlobalRoiModel:
    """Describe the aggregated model for one region of interest."""

    variables: List[str]
    global_coefficients: List[float]
    global_degrees_of_freedom: float
    global_mean_y: float


@dataclass
class GlobalModelSummary:
    """Collect aggregated models by region of interest."""

    roi_models: Dict[str, GlobalRoiModel]


@dataclass
class LocalMetricSummary:
    """Hold site metrics calculated against the global model."""

    sse_local: List[float]
    sst_local: List[float]
    varx_matrix_local: List[List[List[float]]]


@dataclass
class AggregatorState:
    """Persist remote aggregation data between workflow rounds."""

    avg_coefficients: List[List[float]]
    global_mean_y: List[float]
    global_degrees_of_freedom: List[float]
    x_labels: List[str]
    y_labels: List[str]
    all_stats_local: Dict[str, List[Dict[str, Any]]]


@dataclass
class FinalResults:
    """Hold final serializable ridge result rows."""

    rows: List[Dict[str, Any]]
