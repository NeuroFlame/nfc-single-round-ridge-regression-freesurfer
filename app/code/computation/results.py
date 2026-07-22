import copy
import json

import dominate
import pandas as pd
from dominate import tags

from .output_labels import GlobalOutputMetricLabels, OutputDictKeyLabels
from .report_generator import generate_report_html
from .types import FinalResults


def build_output_payloads(final_results: FinalResults, parameters=None):
    parameters = parameters or {}
    payload = final_results.rows
    stats_dataframes = _build_stats_dataframes(copy.deepcopy(payload))
    outputs = {
        "global_regression_result.json": copy.deepcopy(payload),
        **{
            f"{file_stem}.csv": dataframe
            for file_stem, dataframe in stats_dataframes.items()
        },
        "index.html": generate_report_html(
            copy.deepcopy(payload),
            parameters,
            user_name=parameters.get("user_name"),
            user_id=parameters.get("user_id"),
        ),
    }
    return outputs


def _build_stats_dataframes(result_rows):
    def build_stats_dataframe(temp_df, roi_names):
        covariate_labels = temp_df.pop(GlobalOutputMetricLabels.COVARIATE_LABELS.value)[0]
        column_names = temp_df.columns.tolist()
        list_value_indexes = []
        for index, value in enumerate(temp_df.loc[0]):
            if isinstance(value, list):
                list_value_indexes.append(index)

        new_df = pd.concat([temp_df[column_names[idx]].apply(pd.Series) for idx in list_value_indexes], axis=1)
        new_df.columns = [
            column_names[column_idx] + "_" + covariate
            for column_idx in list_value_indexes
            for covariate in covariate_labels
        ]

        for column_idx in set(range(len(column_names))) - set(list_value_indexes):
            new_df[column_names[column_idx]] = temp_df[column_names[column_idx]]

        new_df.index = roi_names
        new_df.index.name = "ROI"
        return new_df

    result = {}
    reverse_df = pd.DataFrame(result_rows)
    roi_names = reverse_df[OutputDictKeyLabels.ROI.value].tolist()

    global_df = pd.json_normalize(reverse_df[OutputDictKeyLabels.GLOBAL_STATS.value])
    result[OutputDictKeyLabels.GLOBAL_STATS.value] = build_stats_dataframe(global_df, roi_names)

    site_names = reverse_df[OutputDictKeyLabels.LOCAL_STATS.value][0].keys()
    for site in site_names:
        local_df = pd.json_normalize(reverse_df[OutputDictKeyLabels.LOCAL_STATS.value][0][site])
        for index in range(1, len(roi_names)):
            local_df = pd.concat(
                [local_df, pd.json_normalize(reverse_df[OutputDictKeyLabels.LOCAL_STATS.value][index][site])],
                ignore_index=True,
                axis=0,
            )
        result[f"{OutputDictKeyLabels.LOCAL_STATS.value}_{site}"] = build_stats_dataframe(local_df, roi_names)

    return result


def _build_html(result_rows):
    doc = dominate.document(title="Results")
    global_stats_label = OutputDictKeyLabels.GLOBAL_STATS.value
    local_stats_label = OutputDictKeyLabels.LOCAL_STATS.value

    with doc.head:
        tags.style(
            '''
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
        )

    with doc:
        tags.h1("Results")
        tags.hr()
        for result in result_rows:
            tags.h2(result[OutputDictKeyLabels.ROI.value])
            with tags.table():
                with tags.tbody():
                    with tags.tr():
                        with tags.td(rowspan=6):
                            tags.h3("Globals")
                        global_labels = result[global_stats_label][GlobalOutputMetricLabels.COVARIATE_LABELS.value]
                        global_labels.insert(0, "")
                        for value in global_labels:
                            tags.td(value)
                    with tags.tr():
                        global_coefficient = result[global_stats_label][GlobalOutputMetricLabels.COEFFICIENT.value]
                        global_coefficient.insert(0, GlobalOutputMetricLabels.COEFFICIENT.value)
                        for value in global_coefficient:
                            tags.td(value)
                    with tags.tr():
                        global_tstat = result[global_stats_label][GlobalOutputMetricLabels.T_STAT.value]
                        global_tstat.insert(0, GlobalOutputMetricLabels.T_STAT.value)
                        for value in global_tstat:
                            tags.td(value)
                    with tags.tr():
                        global_pvalue = result[global_stats_label][GlobalOutputMetricLabels.P_VALUE.value]
                        global_pvalue.insert(0, GlobalOutputMetricLabels.P_VALUE.value)
                        for value in global_pvalue:
                            tags.td(value)
                    with tags.tr():
                        tags.td(GlobalOutputMetricLabels.R_SQUARE.value)
                        tags.td(result[global_stats_label][GlobalOutputMetricLabels.R_SQUARE.value], colspan=5)
                    with tags.tr():
                        tags.td(GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value)
                        tags.td(
                            result[global_stats_label][GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value],
                            colspan=5,
                        )
                for site in result[OutputDictKeyLabels.LOCAL_STATS.value]:
                    with tags.tbody():
                        with tags.tr():
                            with tags.td(rowspan=6):
                                tags.h3(site)
                            global_labels = result[global_stats_label][GlobalOutputMetricLabels.COVARIATE_LABELS.value]
                            for value in global_labels:
                                tags.td(value)
                        with tags.tr():
                            local_coefficient = result[local_stats_label][site][GlobalOutputMetricLabels.COEFFICIENT.value]
                            local_coefficient.insert(0, GlobalOutputMetricLabels.COEFFICIENT.value)
                            for value in local_coefficient:
                                tags.td(value)
                        with tags.tr():
                            local_tstat = result[local_stats_label][site][GlobalOutputMetricLabels.T_STAT.value]
                            local_tstat.insert(0, GlobalOutputMetricLabels.T_STAT.value)
                            for value in local_tstat:
                                tags.td(value)
                        with tags.tr():
                            local_pvalue = result[local_stats_label][site][GlobalOutputMetricLabels.P_VALUE.value]
                            local_pvalue.insert(0, GlobalOutputMetricLabels.P_VALUE.value)
                            for value in local_pvalue:
                                tags.td(value)
                        with tags.tr():
                            tags.td(GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value)
                            tags.td(
                                result[local_stats_label][site][GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value],
                                colspan=5,
                            )
                        with tags.tr():
                            tags.td(GlobalOutputMetricLabels.R_SQUARE.value)
                            tags.td(result[local_stats_label][site][GlobalOutputMetricLabels.R_SQUARE.value], colspan=5)

    return str(doc)
