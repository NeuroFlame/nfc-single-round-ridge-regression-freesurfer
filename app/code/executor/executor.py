# app/code/executor/executor.py

import logging
import os
from typing import Dict, Any

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils.logger import NFCLogger
from utils import task_constants as tc
from utils.utils import get_data_directory_path, get_output_directory_path

from . import client_cache_store as ccs
from . import client_executor_methods as cem


class SRRExecutor(Executor):
    def __init__(self):
        """
        Initialize the SRRExecutor. This constructor sets up the logger.
        """
        logging.info("SrrExecutor initialized")
        self.logger = None

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Main execution entry point. Routes tasks to specific methods based on the task name.

        IMPORTANT DESIGN NOTE (NeuroFLAME Results-Only):
        ------------------------------------------------
        NeuroFLAME/fileServer is the single source of truth for results distribution.
        Therefore, this executor does NOT receive or unpack any "results.zip" content
        over NVFLARE (no base64 transport, no unzip). The NVFLARE task "receive_results"
        is treated as a lifecycle signal only.

        Parameters:
            task_name: Name of the task to perform.
            shareable: Shareable object containing data for the task.
            fl_ctx: Federated learning context.
            abort_signal: Signal object to handle task abortion.

        Returns:
            A Shareable object containing results of the task (when applicable).
        """
        # Cache store is per-run and is rooted at the output directory.
        cache_store = ccs.CacheSerialStore(get_output_directory_path(fl_ctx))

        # Create run-scoped logger (writes into output dir).
        computation_parameters = fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS") or {}
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME) or "unknown_client"
        log_level = computation_parameters.get("log_level", "info")

        self.logger = NFCLogger(
            f"{client_name}.log",
            get_output_directory_path(fl_ctx),
            log_level,
        )

        # Prepare outgoing Shareable (returned back to server)
        outgoing_shareable = Shareable()

        try:
            if task_name == tc.TASK_NAME_LOCAL_CLIENT_STEP1:
                # Local compute step 1 (contributors only)
                client_result = self._do_task_perform_client_step1(
                    shareable, fl_ctx, abort_signal, cache_store.get_cache_dict()
                )
                cache_store.update_cache_dict(client_result.get("cache", {}))
                outgoing_shareable["result"] = client_result.get("output")

            elif task_name == tc.TASK_NAME_LOCAL_CLIENT_STEP2:
                # Local compute step 2 (contributors only) using global aggregated params from server
                client_result = self._do_task_perform_client_step2(
                    shareable, fl_ctx, abort_signal, cache_store.get_cache_dict()
                )
                cache_store.update_cache_dict(client_result.get("cache", {}))
                outgoing_shareable["result"] = client_result.get("output")

            elif task_name == tc.TASK_NAME_RECEIVE_RESULTS:
                # Results distribution is OUT-OF-BAND via NeuroFLAME fileServer.
                # This NVFLARE task is only a synchronization point / lifecycle signal.
                self._ack_results_available(shareable, fl_ctx)

                # After results are declared available, we can cleanup local cache for this run.
                cache_store.remove_cache()

                # Format / finalize logs if your NFCLogger supports it.
                self.logger.format_log()

                # Return an empty Shareable as ACK (server expects a response).
                # outgoing_shareable remains empty.

            else:
                raise ValueError(f"Unknown task name: {task_name}")

        except Exception as e:
            # Log the error clearly in the client log so debugging is straightforward.
            try:
                self.logger.error(f"Task '{task_name}' failed with error: {repr(e)}")
            except Exception:
                logging.exception("Failed to write to NFCLogger; falling back to std logging.")
                logging.exception(e)
            raise

        finally:
            # Always close logger handle.
            try:
                self.logger.close()
            except Exception:
                pass

        return outgoing_shareable

    def _do_task_perform_client_step1(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
        cache_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform the ridge regression on the merged site data.

        This method assumes that data has been validated and is ready for regression analysis.
        It reads the covariates and dependent data, runs the local regression, and returns results.

        Returns:
            Dict with:
              - 'output': local model outputs keyed by ROI
              - 'cache': items needed for subsequent steps
        """
        data_directory = get_data_directory_path(fl_ctx)
        covariates_path = os.path.join(data_directory, "covariates.csv")
        data_path = os.path.join(data_directory, "data.csv")
        computation_parameters = fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS") or {}

        self.logger.info(f"Step1: data_directory={data_directory}")
        self.logger.info(f"Step1: covariates_path={covariates_path}")
        self.logger.info(f"Step1: data_path={data_path}")

        result = cem.perform_client_step1_validate_inputs_and_compute_local_model(
            covariates_path,
            data_path,
            computation_parameters,
            self.logger,
            cache_dict,
        )
        return result

    def _do_task_perform_client_step2(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
        cache_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Receives the global regression parameters from the server, uses these global params
        on local data to compute local metrics and sends them to the server for global metrics.

        Contract:
            - Server MUST attach aggregated parameters under shareable["result"].
            - agg_result must be a dict keyed by ROI label, e.g. agg_result[roi_name][...].

        If this contract isn't satisfied, fail loudly with actionable context.
        """
        try:
            incoming_keys = list(shareable.keys())
        except Exception:
            incoming_keys = ["<unable_to_list_keys>"]

        self.logger.info(f"Step2: incoming shareable keys={incoming_keys}")

        # Controller implementations differ: some attach aggregated params under
        # "result" (common), others under "output" or "data".
        agg_result = shareable.get("result")
        if agg_result is None:
            agg_result = shareable.get("output")
        if agg_result is None:
            agg_result = shareable.get("data")

        # Guard rails to avoid cryptic "NoneType is not subscriptable" errors.
        if agg_result is None:
            raise ValueError(
                "Step2: missing aggregated result in shareable['result']. "
                f"Shareable keys={incoming_keys}. "
                "This usually means the controller/server did not attach global params "
                "when scheduling perform_local_client_step2."
            )

        if not isinstance(agg_result, dict):
            raise ValueError(
                f"Step2: expected dict for aggregated result, got {type(agg_result)}. "
                f"Shareable keys={incoming_keys}"
            )

        result = cem.perform_local_step2_compute_metrics_with_global_params(
            agg_result,
            self.logger,
            cache_dict,
        )
        return result


    def _ack_results_available(self, shareable: Shareable, fl_ctx: FLContext) -> None:
        """NVFLARE 'receive_results' task handler.

        Results distribution is handled by NeuroFLAME's fileServer (recommended for large artifacts).
        This handler is ACK-only: it does not expect or process a base64 zip in the shareable.
        """
        try:
            keys = list(shareable.keys())
        except Exception:
            keys = ["<unable_to_list_keys>"]

        self.logger.info(f"receive_results (ACK-only): shareable keys={keys}")
        self.logger.info(
            "receive_results (ACK-only): results distribution handled by NeuroFLAME fileServer."
        )

