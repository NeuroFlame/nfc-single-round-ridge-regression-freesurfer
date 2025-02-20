import numpy as np
import pandas as pd
import scipy as sp

from utils.ancillary import *

def perform_remote_step1(site_results, agg_cache_dict):
    global_results = {}
    num_sites = len(site_results.keys())
    roi_labels = []
    #print("\n\n\n\n Remote step1 input:")
    #print(site_results)
    #print("\n\n\n\n")
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

    agg_cache_dict.update({
        "avg_coefficients":avg_coefficients.tolist(),
        "global_mean_y": global_mean_y,
        "global_degrees_of_freedom":global_degrees_of_freedom,
        "X_labels": covariates_headers,
        "y_labels" : roi_labels,
        "all_stats_local": all_stats_local
    })

    results = {'output': global_results, 'cache': agg_cache_dict}
    return results



def perform_remote_step2(site_results, agg_cache_dict):
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

    X_labels = agg_cache_dict["X_labels"]
    y_labels = agg_cache_dict["y_labels"]
    all_local_stats_dicts_old = agg_cache_dict["all_stats_local"]
    avg_beta_vector = agg_cache_dict["avg_coefficients"]
    dof_global = agg_cache_dict["global_degrees_of_freedom"]

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

    results = {'output': dict_list, 'cache': agg_cache_dict}

    return results

