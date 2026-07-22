import os

import statsmodels.api as sm

from . import constants
from . import input_validation as iv

from .types import RidgeInputs


def load_ridge_inputs(data_dir, parameters, logger) -> RidgeInputs:
    return load_inputs(
        os.path.join(data_dir, "covariates.csv"),
        os.path.join(data_dir, "data.csv"),
        parameters,
        logger,
    )


def load_inputs(covariates_path, data_path, parameters, logger) -> RidgeInputs:
    logger.info(f"Computation parameters received: {parameters}")
    is_valid, covariates, data = iv.validate_and_get_inputs(
        covariates_path,
        data_path,
        parameters,
        logger,
    )
    if not is_valid:
        raise ValueError("Invalid run input. Check the site log for validation details")

    X = sm.add_constant(covariates)
    y = data
    lambda_value = parameters.get("Lambda", constants.DEFAULT_LAMBDA)
    return RidgeInputs(X=X, y=y, lambda_value=lambda_value)
