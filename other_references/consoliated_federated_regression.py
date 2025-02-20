import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import simplejson as json

import dominate
from dominate.tags import *


AGGREGATOR_CACHE = {} # REMOTE CACHE
EXECUTOR_CACHE = {}  # LOCAL CACHE


from enum import Enum, unique
class GlobalOutputMetricLabels(Enum):
    COEFFICIENT= "Coefficient"
    R_SQUARE= "R Squared"
    T_STAT= "t Stat"
    P_VALUE = "P-value"
    DEGREES_OF_FREEDOM = "Degrees of Freedom"
    SUM_OF_SQUARES_ERROR = "Sum Square of Errors"
    COVARIATE_LABELS = "covariate_labels"

@unique
class OutputDictKeyLabels(Enum):
    ROI = "ROI"
    GLOBAL_STATS = "global_stats"
    LOCAL_STATS = "local_stats"


def save_results_to_json(global_results, sites):
    # Save global results
    with open('SNT_TEST_global_results.json', 'w') as f:
        json.dump(global_results, f, indent=4)

    # Save local results for each site
    for site_id in sites:
         with open(f'SNT_TEST_{site_id}_results.json', 'w') as f:
             json.dump(global_results, f, indent=4)

def save_results_to_html(output, sites):

    def generateOutput(output, outputfile):

        doc = dominate.document(title='Results')
        global_stats_label = OutputDictKeyLabels.GLOBAL_STATS.value
        local_stats_label = OutputDictKeyLabels.LOCAL_STATS.value

        with doc:
            h1('Results')
            hr()
            for result in output:
                h2(result['ROI'])
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
                            global_degfree = result[global_stats_label][GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value]
                            td(GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value)
                            td(global_degfree, colspan=5)
                    for site in result['local_stats']:
                        with tbody():
                            with tr():
                                with td(rowspan=6):
                                    h3(site)
                                global_labels = result[global_stats_label][GlobalOutputMetricLabels.COVARIATE_LABELS.value]
                                for j in global_labels:
                                    td(j)
                            with tr():
                                local_coefficient = result[local_stats_label][site][GlobalOutputMetricLabels.COEFFICIENT.value]
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
                                local_errors = result[local_stats_label][site][GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value]
                                td(GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value)
                                td(local_errors, colspan=5)
                            with tr():
                                local_rsquared = result[local_stats_label][site][GlobalOutputMetricLabels.R_SQUARE.value]
                                td(GlobalOutputMetricLabels.R_SQUARE.value)
                                td(local_rsquared, colspan=5)

        with open(outputfile, "a") as f:
            print(doc, file=f)

            print('<style>', file=f)

            styles = '''
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
            '''
            print(styles, file=f)
            print('</style>', file=f)

    # Save global results
    #with open('SNT_TEST_global_results.html', 'w') as f:
    generateOutput(output, 'SNT_TEST_global_results.html')

    # Save local results for each site
    for site_id in sites:
        generateOutput(output, f'SNT_TEST_{site_id}_results.html')

def ignore_nans(X, y):
    """Removing rows containing NaN's in X and y"""

    if type(X) is pd.DataFrame:
        X_ = X.values.astype('float64')
    else:
        X_ = X

    if type(y) is pd.Series:
        y_ = y.values.astype('float64')
    else:
        y_ = y

    finite_x_idx = np.isfinite(X_).all(axis=1)
    finite_y_idx = np.isfinite(y_)

    finite_idx = finite_y_idx & finite_x_idx

    y_ = y_[finite_idx]
    X_ = X_[finite_idx, :]

    return X_, y_

def get_SSE(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2)

def perform_local_step1(covariates_path, data_path):
    # Load data
    covariates = pd.read_csv(covariates_path)
    data = pd.read_csv(data_path)

    X = sm.add_constant(covariates)  # Add intercept
    X_labels = list(X.columns)

    """Calculate local statistics"""
    y_labels = list(data.columns)
    y=data


    # Initialize results storage
    results = {}

    for column in y.columns:
        curr_y = y[column]

        X_, y_ = ignore_nans(X, curr_y)
        mean_y_local = np.mean(y_)
        num_subjects = len(y_)

        # Printing local stats as well
        model = sm.OLS(y_, X_).fit()
        coefficients = model.params
        ssr = model.ssr
        sse = get_SSE(y_, model.predict(X_))
        p_values = model.pvalues
        t_stats = model.tvalues
        r_squared = model.rsquared
        degrees_of_freedom = model.df_resid


        # Store the results, including the input and target data for global calculations
        results[column] = {
            GlobalOutputMetricLabels.COEFFICIENT.value: coefficients.tolist(),
            GlobalOutputMetricLabels.T_STAT.value: t_stats.tolist(),
            GlobalOutputMetricLabels.P_VALUE.value: p_values.tolist(),
            GlobalOutputMetricLabels.R_SQUARE.value: r_squared,
            GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value: degrees_of_freedom,
            GlobalOutputMetricLabels.COVARIATE_LABELS.value: X_labels,
            GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value: sse,
            "ROI Label": column,
            "mean_y_local" : mean_y_local,
            "num_subjects" : num_subjects
        }

    global EXECUTOR_CACHE
    EXECUTOR_CACHE.update({'X': X, 'y': y})

    return results

def perform_remote_step1(site_results):
    global_results = {}
    num_sites = len(site_results.keys())
    roi_labels = []
    for roi_column in site_results[next(iter(site_results))].keys():
        # Initialize accumulators for weighted averaging
        total_sum_coefficients = None
        all_stats_local = []
        total_sum_mean_y_local = 0.0
        total_subjects = 0
        roi_labels.append(roi_column)
        #covariates_headers =

        for site, results in site_results.items():
            stats = results[roi_column]
            num_subjects = stats["num_subjects"]

            # Update the total number of subjects
            total_subjects += num_subjects

            # Aggregation of coefficients
            if total_sum_coefficients is None:
                total_sum_coefficients = np.array(stats[GlobalOutputMetricLabels.COEFFICIENT.value])
            else:
                total_sum_coefficients += np.array(stats[GlobalOutputMetricLabels.COEFFICIENT.value])

            # Aggregation of mean_y
            total_sum_mean_y_local += stats["mean_y_local"] * num_subjects

            # Aggregation of stats
            all_stats_local.append([{
                GlobalOutputMetricLabels.COEFFICIENT.value: stats[GlobalOutputMetricLabels.COEFFICIENT.value],
                GlobalOutputMetricLabels.T_STAT.value: stats[GlobalOutputMetricLabels.T_STAT.value],
                GlobalOutputMetricLabels.P_VALUE.value: stats[GlobalOutputMetricLabels.P_VALUE.value],
                GlobalOutputMetricLabels.R_SQUARE.value:stats[GlobalOutputMetricLabels.R_SQUARE.value],
                GlobalOutputMetricLabels.COVARIATE_LABELS.value: stats[GlobalOutputMetricLabels.COVARIATE_LABELS.value],
                GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value: stats[GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value],
                "ROI Label": roi_column
            }])

            covariates_headers = stats[GlobalOutputMetricLabels.COVARIATE_LABELS.value]

        # Compute weighted averages
        avg_coefficients = total_sum_coefficients / num_sites
        global_degrees_of_freedom = total_subjects - avg_coefficients.shape[0]
        global_mean_y = total_sum_mean_y_local / total_subjects
        # Store the aggregated global results
        global_results[roi_column] = {
            "Variables": covariates_headers,
            "Global Coefficients": avg_coefficients.tolist(),
            "Global Degrees of Freedom": global_degrees_of_freedom,
            "Global Mean Y" :  global_mean_y
        }

    global AGGREGATOR_CACHE
    AGGREGATOR_CACHE.update({
        "avg_coefficients":avg_coefficients.tolist(),
        "global_mean_y": global_mean_y,
        "global_degrees_of_freedom":global_degrees_of_freedom,
        "X_labels": covariates_headers,
        "y_labels" : roi_labels,
        "all_stats_local": all_stats_local
    })

    return global_results


def perform_local_step2(agg_results):
    def get_y_estimate(coefficients, X):
        return  np.dot(coefficients, np.matrix.transpose(X.to_numpy()))

    X = EXECUTOR_CACHE.get('X')
    y = EXECUTOR_CACHE.get('y')


    SSE_local, SST_local, varX_matrix_local = [], [], []
    for index, column in enumerate(y.columns):
        roi_agg_results = agg_results[column]
        global_coefficients = roi_agg_results["Global Coefficients"]
        mean_y_global = roi_agg_results["Global Mean Y"]

        curr_y = y[column]

        X_, y_ = ignore_nans(X, curr_y)

        SSE_local.append(get_SSE(y_, get_y_estimate(global_coefficients, X)))
        SST_local.append(np.sum(np.square(np.subtract(y_, mean_y_global))))

        varX_matrix_local.append(np.dot(X_.T, X_).tolist())

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local,
    }

    return output_dict

def perform_remote_step2(site_results):
    from itertools import repeat

    def get_stats_to_dict(a, *b):
        df = pd.DataFrame(list(zip(*b)), columns=a)
        dict_list = df.to_dict(orient='records')

        return dict_list

    def t_to_p(ts_beta, dof):
        """Returns the p-value for each t-statistic of the coefficient vector

        Args:
            dof (int)       : Degrees of Freedom
                                Given by len(y) - len(beta_vector)
            ts_beta (float) : t-statistic of shape [n_features +  1]

        Returns:
            p_values (float): of shape [n_features + 1]

        Comments:
            t to p value transformation(two tail)
        """
        return [2 * sp.stats.t.sf(np.abs(t), dof) for t in ts_beta]

    X_labels = AGGREGATOR_CACHE["X_labels"]
    y_labels = AGGREGATOR_CACHE["y_labels"]
    all_local_stats_dicts_old = AGGREGATOR_CACHE["all_stats_local"]
    avg_beta_vector = AGGREGATOR_CACHE["avg_coefficients"]
    dof_global = AGGREGATOR_CACHE["global_degrees_of_freedom"]
    #y_labels = args["cache"]["y_labels"]

    SSE_global = sum(
        [np.array(site_results[site]["SSE_local"]) for site in site_results])
    SST_global = sum(
        [np.array(site_results[site]["SST_local"]) for site in site_results])
    varX_matrix_global = sum([
        np.array(site_results[site]["varX_matrix_local"]) for site in site_results
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i in range(len(MSE)):
        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global[i])
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (avg_beta_vector[i] / se_beta_global).tolist()
        ps = t_to_p(ts, dof_global)
        ts_global.append(ts)
        ps_global.append(ps)

    # Block of code to print local stats as well
    sites = [site for site in site_results]

    all_local_stats_dicts = list(map(list, zip(*all_local_stats_dicts_old)))

    a_dict = [{key: value
               for key, value in zip(sites, stats_dict)}
              for stats_dict in all_local_stats_dicts]

    # Block of code to print just global stats
    keys1 = [s.value for s in GlobalOutputMetricLabels]

    # COEFFICIENT= "Coefficient"
    # R_SQUARE= "R Squared"
    # T_STAT= "t Stat"
    # P_VALUE = "P-value"
    # DEGREES_OF_FREEDOM = "Degrees of Freedom"
    # SUM_OF_SQUARES_ERROR = "Sum Square of Errors"
    # COVARIATE_LABELS = "covariate_labels"


    global_dict_list = get_stats_to_dict(keys1, [avg_beta_vector],
                                         r_squared_global, ts_global,
                                         ps_global, [dof_global], SSE_global.tolist(),
                                         repeat(X_labels, len(y_labels)))

    # Print Everything
    keys2 = [s.value for s in OutputDictKeyLabels]
    dict_list = get_stats_to_dict(keys2, y_labels, global_dict_list, a_dict)

    output_dict = {"output": dict_list}


    return output_dict


def run_federated_ssr():
    # Example usage for multiple sites
    sites = ['site1', 'site2']

    #Iteration - 1
    site_results = {}
    for site in sites:
        #covariates_path = os.path.join(f'../test_data/{site}/covariates.csv')
        #data_path = os.path.join(f'../test_data/{site}/data.csv')

        covariates_path = os.path.join(f'../test_data/{site}/{site}_ssr_fsl_repo_covariates.csv')
        data_path = os.path.join(f'../test_data/{site}/{site}_ssr_fsl_repo_data.csv')
        site_results[site] = perform_local_step1(covariates_path, data_path)

    remote_results = perform_remote_step1(site_results)

    #Iteration - 2
    site_results = {}
    for site in sites:
        site_results[site] = perform_local_step2(remote_results)

    remote_results = perform_remote_step2(site_results)

    # Save the results
    save_results_to_json(remote_results['output'], sites)
    save_results_to_html(remote_results['output'], sites)


run_federated_ssr()