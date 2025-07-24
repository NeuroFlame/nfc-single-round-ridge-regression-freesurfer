from enum import Enum, unique


class GlobalOutputMetricLabels(Enum):
    COEFFICIENT = "Coefficient"
    R_SQUARE = "R Squared"
    T_STAT = "t Stat"
    P_VALUE = "P-value"
    DEGREES_OF_FREEDOM = "Degrees of Freedom"
    SUM_OF_SQUARES_ERROR = "Sum Square of Errors"
    COVARIATE_LABELS = "covariate_labels"


@unique
class OutputDictKeyLabels(Enum):
    ROI = "ROI"
    GLOBAL_STATS = "global_stats"
    LOCAL_STATS = "local_stats"
