import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

import dominate
from dominate.tags import *

from utils.ancillary import *
from . import client_input_preprocessor as cip
from . import client_constants


def perform_client_step1(covariates_path, data_path, computation_parameters, log_path, cache_dict):
    # Validate the run inputs (covariates, dependent data, and parameters)
    is_valid, covariates, data = cip.validate_and_get_inputs(covariates_path, data_path, computation_parameters,
                                                             log_path)
    if not is_valid:
        # Halt execution if validation fails
        raise ValueError(f"Invalid run input. Check validation log at {log_path}")

    X = sm.add_constant(covariates)  # Add intercept
    X_labels = list(X.columns)

    y = data

    lamb = computation_parameters.get("Lambda", client_constants.DEFAULT_LAMBDA)

    # Initialize results storage
    output = {}

    # Calculate local statistics for each ROI
    for column in y.columns:
        curr_y = y[column]

        X_, y_ = cip.ignore_nans(X, curr_y)
        mean_y_local = np.mean(y_)
        num_subjects = len(y_)

        # Printing local stats as well
        # model = sm.OLS(y_, X_).fit()
        model = _get_ridge_regression_model(X_, y_, lamb)
        coefficients = model.params
        ssr = model.ssr
        sse = _get_SSE(y_, model.predict(X_))
        p_values = model.pvalues
        t_stats = model.tvalues
        r_squared = model.rsquared
        degrees_of_freedom = model.df_resid

        # Store the results, including the input and target data for global calculations
        output[column] = {
            GlobalOutputMetricLabels.COEFFICIENT.value: coefficients.tolist(),
            GlobalOutputMetricLabels.T_STAT.value: t_stats.tolist(),
            GlobalOutputMetricLabels.P_VALUE.value: p_values.tolist(),
            GlobalOutputMetricLabels.R_SQUARE.value: r_squared,
            GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value: degrees_of_freedom,
            GlobalOutputMetricLabels.COVARIATE_LABELS.value: X_labels,
            GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value: sse,
            "ROI Label": column,
            "mean_y_local": mean_y_local,
            "num_subjects": num_subjects
        }

    cache_dict = {
        "X": X.to_json(orient='split'),
        "y": y.to_json(orient='split'),
        "lambda": lamb
    }

    results = {'output': output, 'cache': cache_dict}

    return results


def perform_local_step2(agg_results, log_path, cache_dict):
    def get_y_estimate(coefficients, X):
        return np.dot(coefficients, np.matrix.transpose(X))

    X = pd.read_json(cache_dict["X"], orient='split')
    y = pd.read_json(cache_dict["y"], orient='split')
    lamb = cache_dict["lambda"]

    SSE_local, SST_local, varX_matrix_local = [], [], []
    for index, column in enumerate(y.columns):
        roi_agg_results = agg_results[column]
        global_coefficients = roi_agg_results["Global Coefficients"]
        mean_y_global = roi_agg_results["Global Mean Y"]

        curr_y = y[column]

        X_, y_ = cip.ignore_nans(X, curr_y)

        SSE_local.append(_get_SSE(y_, get_y_estimate(global_coefficients, X_)))
        SST_local.append(np.sum(np.square(np.subtract(y_, mean_y_global))))

        varX_matrix_local.append(np.dot(X_.T, X_).tolist())

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local,
    }

    results = {'output': output_dict, 'cache': {}}
    return results


def perform_local_step3(agg_results, log_path, cache_dict):
    import copy
    results = {'output': {'json': copy.deepcopy(agg_results),
                          'csv': _get_global_local_stats_df(copy.deepcopy(agg_results)),
                          'html': _get_html_from_results(copy.deepcopy(agg_results))
                          },
               'cache': {}}

    return results


def _get_global_local_stats_df(agg_results):
    import pandas as pd

    def _get_stats_df(temp_df, roi_names):
        covariate_labels = temp_df.pop(GlobalOutputMetricLabels.COVARIATE_LABELS.value)[0]
        col_names = temp_df.columns.tolist()
        cols_with_list_vals = []
        for idx, k in enumerate(temp_df.loc[0]):
            if type(k) == type([]):
                cols_with_list_vals.append(idx)

        new_df = pd.concat([temp_df[col_names[k]].apply(pd.Series) for k in cols_with_list_vals], axis=1)
        new_df.columns = [col_names[x] + "_" + y for x in cols_with_list_vals for y in covariate_labels]

        # Add remaining columns
        for col_idx in set(range(len(temp_df.columns.tolist()))) - set(cols_with_list_vals):
            new_df[col_names[col_idx]] = temp_df[col_names[col_idx]]

        # Add ROI names for row indexes
        new_df.index = roi_names
        return new_df

    result = {}
    rev_df = pd.DataFrame(agg_results)
    roi_names = rev_df[OutputDictKeyLabels.ROI.value].tolist()

    global_df = pd.json_normalize(rev_df[OutputDictKeyLabels.GLOBAL_STATS.value])
    result[OutputDictKeyLabels.GLOBAL_STATS.value] = _get_stats_df(global_df, roi_names)

    site_names = rev_df[OutputDictKeyLabels.LOCAL_STATS.value][0].keys()
    for site in site_names:
        local_df = pd.json_normalize(rev_df[OutputDictKeyLabels.LOCAL_STATS.value][0][site])
        for idx in range(1, len(roi_names)):
            local_df = pd.concat(
                [local_df, pd.json_normalize(rev_df[OutputDictKeyLabels.LOCAL_STATS.value][idx][site])],
                ignore_index=True, axis=0)

        result[f'{OutputDictKeyLabels.LOCAL_STATS.value}_{site}'] = _get_stats_df(local_df, roi_names)

    return result


def _get_html_from_results(agg_results):
    doc = dominate.document(title='Results')
    global_stats_label = OutputDictKeyLabels.GLOBAL_STATS.value
    local_stats_label = OutputDictKeyLabels.LOCAL_STATS.value

    # Add style
    with doc.head:
        style('''
            body {
                font-family: sans-serif;
            }
            hr {
                width: 100%;
            }
            table {
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 0.9em;
                font-family: sans-serif;
                min-width: 400px;
                width: 100%;
            }
            table thead tr {
                background-color: #009879;
                color: #ffffff;
                text-align: left;
            }
            table tr:nth-of-type(1) td {
                font-weight: bold;              
            }
            table tr td:nth-of-type(1) {
                background-color: white;
                font-weight: bold;
            }
            table tr td[colspan]{
                background-color: white;
            }
            table thead tr td { 
                text-transform: capitalize;
            }
            table th,
            table td {
                padding: 12px 15px;
                white-space: nowrap;
            }
            table tbody tr {
                border-bottom: 1px solid #dddddd;
            }

            table tbody tr:nth-of-type(even) {
                background-color: #efefef;
            }

            table tbody tr.active-row {
                font-weight: bold;
                color: #009879;
            }
        ''')

    # Add document body
    with doc:
        h1('Results')
        hr()
        for result in agg_results:
            h2(result[OutputDictKeyLabels.ROI.value])
            with table():
                with tbody():
                    with tr():
                        with td(rowspan=6):
                            h3('Globals')
                        global_labels = result[global_stats_label][GlobalOutputMetricLabels.COVARIATE_LABELS.value]
                        global_labels.insert(0, '')
                        for i in global_labels:
                            td(i)
                    with tr():
                        global_coefficient = result[global_stats_label][GlobalOutputMetricLabels.COEFFICIENT.value]
                        global_coefficient.insert(0, GlobalOutputMetricLabels.COEFFICIENT.value)
                        for i in global_coefficient:
                            td(i)
                    with tr():
                        global_tstat = result[global_stats_label][GlobalOutputMetricLabels.T_STAT.value]
                        global_tstat.insert(0, GlobalOutputMetricLabels.T_STAT.value)
                        for i in global_tstat:
                            td(i)
                    with tr():
                        global_pvalue = result[global_stats_label][GlobalOutputMetricLabels.P_VALUE.value]
                        global_pvalue.insert(0, GlobalOutputMetricLabels.P_VALUE.value)
                        for i in global_pvalue:
                            td(i)
                    with tr():
                        global_rsquared = result[global_stats_label][GlobalOutputMetricLabels.R_SQUARE.value]
                        td(GlobalOutputMetricLabels.R_SQUARE.value)
                        td(global_rsquared, colspan=5)
                    with tr():
                        global_degfree = result[global_stats_label][
                            GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value]
                        td(GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value)
                        td(global_degfree, colspan=5)
                for site in result[OutputDictKeyLabels.LOCAL_STATS.value]:
                    with tbody():
                        with tr():
                            with td(rowspan=6):
                                h3(site)
                            global_labels = result[global_stats_label][
                                GlobalOutputMetricLabels.COVARIATE_LABELS.value]
                            for j in global_labels:
                                td(j)
                        with tr():
                            local_coefficient = result[local_stats_label][site][
                                GlobalOutputMetricLabels.COEFFICIENT.value]
                            local_coefficient.insert(0, GlobalOutputMetricLabels.COEFFICIENT.value)
                            for i in local_coefficient:
                                td(i)
                        with tr():
                            local_tstat = result[local_stats_label][site][GlobalOutputMetricLabels.T_STAT.value]
                            local_tstat.insert(0, GlobalOutputMetricLabels.T_STAT.value)
                            for i in local_tstat:
                                td(i)
                        with tr():
                            local_pvalue = result[local_stats_label][site][GlobalOutputMetricLabels.P_VALUE.value]
                            local_pvalue.insert(0, GlobalOutputMetricLabels.P_VALUE.value)
                            for i in local_pvalue:
                                td(i)
                        with tr():
                            local_errors = result[local_stats_label][site][
                                GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value]
                            td(GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value)
                            td(local_errors, colspan=5)
                        with tr():
                            local_rsquared = result[local_stats_label][site][
                                GlobalOutputMetricLabels.R_SQUARE.value]
                            td(GlobalOutputMetricLabels.R_SQUARE.value)
                            td(local_rsquared, colspan=5)

    return str(doc)


def _get_SSE(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2)


def _get_ridge_regression_model(X, y, lamb, use_regularization_fit=True):
    """Performs ridge regression
    Args:
        X (float) : Training data of shape [n_samples, n_features]
        y (float) : Target values of shape [n_samples]
        lamb (float) : Regularization parameter lambda

    Returns:
        beta_vector (float) : weight vector of shape [n_features]
    """
    from statsmodels.tools.tools import pinv_extended

    model = sm.OLS(y, X.astype(float))
    if use_regularization_fit:
        reg_fit = model.fit_regularized(alpha=lamb, L1_wt=0)
        pinv_wexog, _ = pinv_extended(model.wexog)
        normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
        summary = sm.regression.linear_model.OLSResults(model, reg_fit.params, normalized_cov_params)
    else:
        summary = model.fit()

    if np.any(np.isnan(summary.params)):
        raise Exception('sm.OLS() failed to fit the regression model for this data')

    return summary
