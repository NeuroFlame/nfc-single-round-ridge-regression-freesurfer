from itertools import repeat
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp

from framework import with_state

from .output_labels import GlobalOutputMetricLabels, OutputDictKeyLabels
from .types import AggregatorState, FinalResults, GlobalModelSummary, GlobalRoiModel, LocalMetricSummary, LocalModelSummary


def aggregate_global_model(site_results_payload: Dict[str, LocalModelSummary]):
    site_results = site_results_payload

    if not site_results:
        return with_state(
            GlobalModelSummary(roi_models={}),
            AggregatorState(
                avg_coefficients=[],
                global_mean_y=[],
                global_degrees_of_freedom=[],
                x_labels=[],
                y_labels=[],
                all_stats_local={},
            ),
        )

    num_sites = len(site_results)
    first_site = next(iter(site_results.values()))
    roi_labels = first_site.roi_labels

    roi_models = {}
    avg_coefficients_all_rois = []
    mean_y_global_all_rois = []
    dof_global = []

    for roi_label in roi_labels:
        total_sum_coefficients = None
        total_sum_mean_y_local = 0.0
        total_subjects = 0
        covariate_headers = []

        for local_model in site_results.values():
            stats = local_model.roi_stats[roi_label]
            total_subjects += stats.num_subjects

            if total_sum_coefficients is None:
                total_sum_coefficients = np.array(stats.coefficient)
            else:
                total_sum_coefficients += np.array(stats.coefficient)

            total_sum_mean_y_local += stats.mean_y_local * stats.num_subjects
            covariate_headers = stats.covariate_labels

        avg_coefficients = total_sum_coefficients / num_sites
        global_degrees_of_freedom = total_subjects - avg_coefficients.shape[0]
        global_mean_y = total_sum_mean_y_local / total_subjects

        roi_models[roi_label] = GlobalRoiModel(
            variables=covariate_headers,
            global_coefficients=avg_coefficients.tolist(),
            global_degrees_of_freedom=global_degrees_of_freedom,
            global_mean_y=global_mean_y,
        )

        avg_coefficients_all_rois.append(avg_coefficients.tolist())
        mean_y_global_all_rois.append(global_mean_y)
        dof_global.append(global_degrees_of_freedom)

    all_stats_local = {}
    for site_name in sorted(site_results.keys()):
        local_model = site_results[site_name]
        local_stats = []
        for roi_label in roi_labels:
            stats = local_model.roi_stats[roi_label]
            local_stats.append(
                {
                    GlobalOutputMetricLabels.COEFFICIENT.value: stats.coefficient,
                    GlobalOutputMetricLabels.T_STAT.value: stats.t_stat,
                    GlobalOutputMetricLabels.P_VALUE.value: stats.p_value,
                    GlobalOutputMetricLabels.R_SQUARE.value: stats.r_squared,
                    GlobalOutputMetricLabels.COVARIATE_LABELS.value: stats.covariate_labels,
                    GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value: stats.sum_square_of_errors,
                }
            )
        all_stats_local[site_name] = local_stats

    aggregator_state = AggregatorState(
        avg_coefficients=avg_coefficients_all_rois,
        global_mean_y=mean_y_global_all_rois,
        global_degrees_of_freedom=dof_global,
        x_labels=roi_models[roi_labels[0]].variables,
        y_labels=roi_labels,
        all_stats_local=all_stats_local,
    )

    return with_state(GlobalModelSummary(roi_models=roi_models), aggregator_state)


def aggregate_final_results(site_results_payload: Dict[str, LocalMetricSummary], state: AggregatorState) -> FinalResults:
    local_metric_summaries = site_results_payload

    sse_global = sum(np.array(summary.sse_local) for summary in local_metric_summaries.values())
    sst_global = sum(np.array(summary.sst_local) for summary in local_metric_summaries.values())
    varx_matrix_global = sum(np.array(summary.varx_matrix_local) for summary in local_metric_summaries.values())

    r_squared_global = 1 - (sse_global / sst_global)
    mse = sse_global / np.array(state.global_degrees_of_freedom)

    ts_global = []
    ps_global = []

    for index in range(len(mse)):
        var_covar_beta_global = mse[index] * sp.linalg.inv(varx_matrix_global[index])
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (np.array(state.avg_coefficients[index]) / se_beta_global).tolist()
        ps = _t_to_p(ts, state.global_degrees_of_freedom[index])
        ts_global.append(ts)
        ps_global.append(ps)

    site_names = sorted(state.all_stats_local.keys())
    local_stats = [
        {
            site_name: state.all_stats_local[site_name][roi_index]
            for site_name in site_names
        }
        for roi_index, _ in enumerate(state.y_labels)
    ]

    global_stats = _zip_records(
        [label.value for label in GlobalOutputMetricLabels],
        state.avg_coefficients,
        r_squared_global,
        ts_global,
        ps_global,
        state.global_degrees_of_freedom,
        sse_global.tolist(),
        repeat(state.x_labels, len(state.y_labels)),
    )

    rows = _zip_records(
        [label.value for label in OutputDictKeyLabels],
        state.y_labels,
        global_stats,
        local_stats,
    )

    return FinalResults(rows=_round_floats(rows, decimal_places=4))


def _zip_records(columns, *series_values):
    dataframe = pd.DataFrame(list(zip(*series_values)), columns=columns)
    return dataframe.to_dict(orient="records")


def _t_to_p(ts_beta, dof):
    return [2 * sp.stats.t.sf(np.abs(t_value), dof) for t_value in ts_beta]


def _round_floats(obj, decimal_places=4):
    if isinstance(obj, float):
        return round(obj, decimal_places)
    if isinstance(obj, dict):
        return {key: _round_floats(value, decimal_places) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(item, decimal_places) for item in obj]
    return obj
