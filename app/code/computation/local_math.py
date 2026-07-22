import numpy as np
import pandas as pd
import statsmodels.api as sm

from framework import with_state

from .types import CachedLocalState, GlobalModelSummary, LocalMetricSummary, LocalModelSummary, LocalRoiStats, RidgeInputs


def fit_local_models(inputs: RidgeInputs):
    roi_stats = {}

    for column in inputs.roi_labels:
        curr_y = inputs.y[column]
        X_without_nans, y_without_nans = _ignore_nans(inputs.X, curr_y)
        mean_y_local = np.mean(y_without_nans)
        num_subjects = len(y_without_nans)

        model = _get_ridge_regression_model(X_without_nans, y_without_nans, inputs.lambda_value)
        coefficients = model.params
        sse = _get_sse(y_without_nans, model.predict(X_without_nans))

        roi_stats[column] = LocalRoiStats(
            coefficient=coefficients.tolist(),
            t_stat=model.tvalues.tolist(),
            p_value=model.pvalues.tolist(),
            r_squared=model.rsquared,
            degrees_of_freedom=model.df_resid,
            covariate_labels=inputs.covariate_labels,
            sum_square_of_errors=sse,
            roi_label=column,
            y_labels=inputs.roi_labels,
            mean_y_local=mean_y_local,
            num_subjects=num_subjects,
        )

    return with_state(
        LocalModelSummary(roi_stats=roi_stats),
        CachedLocalState(
            X=inputs.X,
            y=inputs.y,
            lambda_value=inputs.lambda_value,
        ),
    )


def compute_local_metrics(global_model: GlobalModelSummary, state: CachedLocalState) -> LocalMetricSummary:
    restored_inputs = RidgeInputs(
        X=state.X,
        y=state.y,
        lambda_value=state.lambda_value,
    )

    sse_local = []
    sst_local = []
    varx_matrix_local = []

    for column in restored_inputs.roi_labels:
        roi_global_model = global_model.roi_models[column]
        curr_y = restored_inputs.y[column]

        X_without_nans, y_without_nans = _ignore_nans(restored_inputs.X, curr_y)
        estimated_y = np.dot(roi_global_model.global_coefficients, np.matrix.transpose(X_without_nans))

        sse_local.append(_get_sse(y_without_nans, estimated_y))
        sst_local.append(np.sum(np.square(np.subtract(y_without_nans, roi_global_model.global_mean_y))))
        varx_matrix_local.append(np.dot(X_without_nans.T, X_without_nans).tolist())

    return LocalMetricSummary(
        sse_local=sse_local,
        sst_local=sst_local,
        varx_matrix_local=varx_matrix_local,
    )


def _get_sse(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2)


def _get_ridge_regression_model(X, y, lambda_value, use_regularization_fit=True):
    from statsmodels.tools.tools import pinv_extended

    model = sm.OLS(y, X.astype(float))
    if use_regularization_fit:
        reg_fit = model.fit_regularized(alpha=lambda_value, L1_wt=0)
        pinv_wexog, _ = pinv_extended(model.wexog)
        normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
        summary = sm.regression.linear_model.OLSResults(model, reg_fit.params, normalized_cov_params)
    else:
        summary = model.fit()

    if np.any(np.isnan(summary.params)):
        raise Exception("sm.OLS() failed to fit the regression model for this data")

    return summary


def _ignore_nans(X, y):
    if isinstance(X, pd.DataFrame):
        X_without_nans = X.values.astype("float64")
    else:
        X_without_nans = X

    if isinstance(y, pd.Series):
        y_without_nans = y.values.astype("float64")
    else:
        y_without_nans = y

    finite_x_idx = np.isfinite(X_without_nans).all(axis=1)
    finite_y_idx = np.isfinite(y_without_nans)
    finite_idx = finite_y_idx & finite_x_idx

    y_without_nans = y_without_nans[finite_idx]
    X_without_nans = X_without_nans[finite_idx, :]
    return X_without_nans, y_without_nans
