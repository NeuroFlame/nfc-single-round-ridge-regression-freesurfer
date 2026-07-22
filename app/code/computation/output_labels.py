from enum import Enum, unique


class GlobalOutputMetricLabels(Enum):
    """
      Holds the strings constants used for the displaying and gathering data
    """
    COEFFICIENT = "Coefficient"
    R_SQUARE = "R Squared"
    T_STAT = "t Stat"
    P_VALUE = "P-value"
    DEGREES_OF_FREEDOM = "Degrees of Freedom"
    SUM_OF_SQUARES_ERROR = "Sum Square of Errors"
    COVARIATE_LABELS = "covariate_labels"


@unique
class OutputDictKeyLabels(Enum):
    """
        Holds the strings constants used for the displaying the results
    """
    ROI = "ROI"
    GLOBAL_STATS = "global_stats"
    LOCAL_STATS = "local_stats"
