import logging
import os
import json
from typing import Dict, Any

import pandas as pd
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.signal import Signal

from utils.utils import get_data_directory_path, get_output_directory_path
from utils.task_constants import *
from . import client_executor_methods as cem
from . import client_cache_store as ccs


class SRRExecutor(Executor):
    def __init__(self):
        """
        Initialize the SrrExecutor. This constructor sets up the logger.
        """
        logging.info("SrrExecutor initialized")

    def execute(
            self,
            task_name: str,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
    ) -> Shareable:
        """
        Main execution entry point. Routes tasks to specific methods based on the task name.
        
        Parameters:
            task_name: Name of the task to perform.
            shareable: Shareable object containing data for the task.
            fl_ctx: Federated learning context.
            abort_signal: Signal object to handle task abortion.
            
        Returns:
            A Shareable object containing results of the task.
        """
        cache_store = ccs.CacheSerialStore(get_output_directory_path(fl_ctx))
        # Prepare the Shareable object to send the result to other components
        outgoing_shareable = Shareable()

        if task_name == TASK_NAME_LOCAL_CLIENT_STEP1:
            client_result = self._do_task_perform_client_step1(shareable, fl_ctx, abort_signal,
                                                               cache_store.get_cache_dict())
            cache_store.update_cache_dict(client_result['cache'])
            outgoing_shareable['result'] = client_result['output']

        elif task_name == TASK_NAME_LOCAL_CLIENT_STEP2:
            client_result = self._do_task_perform_client_step2(shareable, fl_ctx, abort_signal,
                                                               cache_store.get_cache_dict())
            cache_store.update_cache_dict(client_result['cache'])
            outgoing_shareable['result'] = client_result['output']

        elif task_name == TASK_NAME_LOCAL_CLIENT_STEP3:
            client_result = self._do_task_perform_client_step3(shareable, fl_ctx, abort_signal,
                                                               cache_store.get_cache_dict())
            cache_store.remove_cache()
            # Sending empty sharable object as result

        else:
            # Raise an error if the task name is unknown
            raise ValueError(f"Unknown task name: {task_name}")

        # return client_result['output']

        return outgoing_shareable

    def _do_task_perform_client_step1(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict
    ) -> Dict:
        """
        Perform the ridge regression on the merged site data.

        This method assumes that data has been validated and is ready for regression analysis.
        It reads the covariates and dependent data, runs the local regression, and saves the results.

        Returns:
            A Shareable object with the regression results.
        """
        # Paths to data directories and logs
        data_directory = get_data_directory_path(fl_ctx)
        covariates_path = os.path.join(data_directory, "covariates.csv")
        data_path = os.path.join(data_directory, "data.csv")
        computation_parameters = fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS")
        log_path = os.path.join(get_output_directory_path(fl_ctx), "client_log.txt")

        # Perform ridge regression using the specified covariates and dependent variables
        result = cem.perform_client_step1_validate_inputs_and_compute_local_model(covariates_path, data_path, computation_parameters, log_path, cache_dict)

        # Prepare the Shareable object to send the result to other components
        # outgoing_shareable = Shareable()
        # outgoing_shareable["result"] = result
        # outgoing_shareable["result"]["site"] = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        # return outgoing_shareable

        return result

    def _do_task_perform_client_step2(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict
    ) -> Dict:
        """
        Receives the global regression parameters, uses these global regression model on local data
        to perform local metrics and sends it to remote to compute global metrics.

        This method retrieves the global regression results from the Shareable object,
         and returns a Shareable object with regression metrics with global regression params.
        """
        # Retrieve the global regression result from the Shareable object
        agg_result = shareable.get("result")

        # Paths to data directories and logs
        log_path = os.path.join(get_output_directory_path(fl_ctx), "client_log.txt")

        # Perform ridge regression using the specified covariates and dependent variables
        result = cem.perform_local_step2_compute_metrics_with_global_params(agg_result, log_path, cache_dict)

        # Prepare the Shareable object to send the result to other components
        # outgoing_shareable = Shareable()
        # outgoing_shareable["result"] = result
        # #outgoing_shareable["result"]["site"] = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        # return outgoing_shareable

        return result

    def _do_task_perform_client_step3(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict
    ) -> Dict:
        """
        Save the global regression results to a file.

        This method retrieves the global regression results from the Shareable object,
        saves them in JSON and HTML format, and returns a Shareable object.
        """
        # Retrieve the global regression result from the Shareable object
        agg_result = shareable.get("result")

        if agg_result == None:
            raise ("Empty aggregation result")
        # Paths to data directories and logs
        log_path = os.path.join(get_output_directory_path(fl_ctx), "client_log.txt")
        # Save the global regression results
        result = cem.perform_local_step3_persist_results(agg_result, log_path, cache_dict)
        for output_file_type, output_file_data in result.get('output').items():
            if output_file_type == 'json':
                self.save_json(output_file_data, "global_regression_result.json", fl_ctx)
            if output_file_type == 'html':
                self.save_html(output_file_data, "global_regression_result.html", fl_ctx)
            if output_file_type == 'csv':
                self.save_stats_csv(output_file_data, ".csv", fl_ctx)

        return result

    # Utility methods for saving JSON and HTML files
    def save_json(self, data: dict, filename: str, fl_ctx: FLContext) -> None:
        """
        Save a dictionary as a JSON file in the output directory.

        Parameters:
            data: The dictionary to be saved.
            filename: The name of the JSON file.
            fl_ctx: The federated learning context.
        """
        # Get the output directory path and save the JSON file
        output_dir = get_output_directory_path(fl_ctx)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    def save_html(self, data: str, filename: str, fl_ctx: FLContext) -> None:
        """
        Save a string as an HTML file in the output directory.

        Parameters:
            data: The string content to be saved.
            filename: The name of the HTML file.
            fl_ctx: The federated learning context.
        """
        # Get the output directory path and save the HTML file
        output_dir = get_output_directory_path(fl_ctx)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(data)

    def save_stats_csv(self, data: dict, filename: str, fl_ctx: FLContext) -> None:
        """
                Save a json output as an CSV file in the output directory.

                Parameters:
                    data: The string content to be saved.
                    filename: The name of the HTML file.
                    fl_ctx: The federated learning context.
                """
        # Get the output directory path and save the HTML file
        output_dir = get_output_directory_path(fl_ctx)
        for site_name, site_stats_df in data.items():
            output_path = os.path.join(output_dir, site_name + filename)
            site_stats_df.to_csv(output_path, index_label='ROI')
