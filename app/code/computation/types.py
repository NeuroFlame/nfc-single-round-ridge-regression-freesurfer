from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class RidgeInputs:
    X: pd.DataFrame
    y: pd.DataFrame
    lambda_value: float

    @property
    def covariate_labels(self) -> List[str]:
        return list(self.X.columns)

    @property
    def roi_labels(self) -> List[str]:
        return list(self.y.columns)


@dataclass
class CachedLocalState:
    X: pd.DataFrame
    y: pd.DataFrame
    lambda_value: float


@dataclass
class LocalRoiStats:
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
    roi_stats: Dict[str, LocalRoiStats]

    @property
    def roi_labels(self) -> List[str]:
        if not self.roi_stats:
            return []
        first_roi = next(iter(self.roi_stats.values()))
        return list(first_roi.y_labels)


@dataclass
class GlobalRoiModel:
    variables: List[str]
    global_coefficients: List[float]
    global_degrees_of_freedom: float
    global_mean_y: float


@dataclass
class GlobalModelSummary:
    roi_models: Dict[str, GlobalRoiModel]


@dataclass
class LocalMetricSummary:
    sse_local: List[float]
    sst_local: List[float]
    varx_matrix_local: List[List[List[float]]]


@dataclass
class AggregatorState:
    avg_coefficients: List[List[float]]
    global_mean_y: List[float]
    global_degrees_of_freedom: List[float]
    x_labels: List[str]
    y_labels: List[str]
    all_stats_local: Dict[str, List[Dict[str, Any]]]

@dataclass
class FinalResults:
    rows: List[Dict[str, Any]]
