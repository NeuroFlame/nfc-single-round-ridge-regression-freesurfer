import json
import os
import logging
from typing import Callable, Optional, List, Set

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils import task_constants as tc
from utils.utils import get_parameters_file_path

logger = logging.getLogger(__name__)


class SRRController(Controller):
    """
    SRRController defines the control flow for this application and broadcasts tasks to sites.

    Roles (general)
    --------------
    - Contributors: sites allowed to receive tasks that require local data and/or local compute.
    - Observers: sites that should not perform local data-dependent compute, but may still receive
      specific tasks (e.g., to persist global outputs locally).

    This controller assumes create_job.py wrote:
      <APP_ROOT>/config/participant_roles.json   (inside app_server)

    IMPORTANT:
      - min_clients is set to len(contributors) at runtime, so server config does not need
        to be manually kept in sync with generated runs.
      - This controller requires NVFlare broadcast_and_wait(targets=...) support.
    """

    def __init__(
        self,
        min_clients: int = 2,
        wait_time_after_min_received: int = 10,
        task_timeout: int = 0,
    ):
        super().__init__()
        self._task_timeout = task_timeout
        self._min_clients = int(min_clients)
        self._wait_time_after_min_received = int(wait_time_after_min_received)

        self._contributors: Set[str] = set()
        self._observers: Set[str] = set()
        self._all_sites: Optional[List[str]] = None

    # ----------------------------
    # NVFlare lifecycle hooks
    # ----------------------------

    def start_controller(self, fl_ctx: FLContext) -> None:
        """
        Called when the controller starts. Loads computation parameters and participant roles,
        and auto-syncs min_clients to the contributor count.
        """
        self.srr_aggregator = self._engine.get_component(tc.SRR_AGGREGATOR_ID)
        self._load_and_set_computation_parameters(fl_ctx)
        self._load_participant_roles_or_fail(fl_ctx)

        # Always wait for all contributors (single-round deterministic)
        self._min_clients = len(self._contributors)
        logger.info(
            "Controller started",
            extra={
                "min_clients": self._min_clients,
                "contributors": sorted(self._contributors),
                "observers": sorted(self._observers),
            },
        )

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """
        Main application workflow.

        Note: task names and which targets receive them are specific to this app.
        """
        fl_ctx.set_prop(key="CURRENT_ROUND", value=0)

        contributor_targets: List[str] = self._sorted(self._contributors)
        all_targets: List[str] = self._all_sites if self._all_sites is not None else contributor_targets

        # STEP1: contributors only
        self._broadcast_task(
            task_name=tc.TASK_NAME_LOCAL_CLIENT_STEP1,
            data=Shareable(),
            result_cb=self._accept_site_regression_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
            targets=contributor_targets,
        )

        aggregate_result = self.srr_aggregator.aggregate(fl_ctx)
        fl_ctx.set_prop(key="CURRENT_ROUND", value=1)

        # STEP2: contributors only
        self._broadcast_task(
            task_name=tc.TASK_NAME_LOCAL_CLIENT_STEP2,
            data=aggregate_result,
            result_cb=self._accept_site_regression_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
            targets=contributor_targets,
        )

        aggregate_result = self.srr_aggregator.aggregate(fl_ctx)
        fl_ctx.set_prop(key="CURRENT_ROUND", value=2)

        # STEP3: contributors + observers (e.g., persist global outputs locally)
        self._broadcast_task(
            task_name=tc.TASK_NAME_LOCAL_CLIENT_STEP3,
            data=aggregate_result,
            result_cb=None,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
            targets=all_targets,
        )

    # ----------------------------
    # Task plumbing
    # ----------------------------

    def _accept_site_regression_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        return self.srr_aggregator.accept(client_task.result, fl_ctx)

    def _broadcast_task(
        self,
        task_name: str,
        data: Shareable,
        result_cb: Optional[Callable[[ClientTask, FLContext], bool]],
        fl_ctx: FLContext,
        abort_signal: Signal,
        targets: Optional[List[str]] = None,
    ) -> None:
        task = Task(
            name=task_name,
            data=data,
            props={},
            timeout=self._task_timeout,
            result_received_cb=result_cb,
        )

        try:
            self.broadcast_and_wait(
                task=task,
                min_responses=self._min_clients,
                wait_time_after_min_received=self._wait_time_after_min_received,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
                targets=targets,
            )
        except TypeError as e:
            raise RuntimeError(
                "NVFlare build does not support broadcast_and_wait(targets=...). "
                "This controller requires targets for contributor/observer separation."
            ) from e

    # ----------------------------
    # Role loading
    # ----------------------------

    def _load_participant_roles_or_fail(self, fl_ctx: FLContext) -> None:
        """
        Loads contributor/observer roles from:
          <APP_ROOT>/config/participant_roles.json
        """
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        candidates: List[str] = []

        if app_root:
            candidates.append(os.path.join(str(app_root), "config", "participant_roles.json"))

        # Fallbacks (in case APP_ROOT is unset)
        candidates.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "participant_roles.json"))
        )
        candidates.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "participant_roles.json"))
        )

        cfg_path = next((p for p in candidates if os.path.exists(p)), None)
        if not cfg_path:
            raise FileNotFoundError(
                "participant_roles.json not found. Tried:\n  - " + "\n  - ".join(candidates)
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._contributors = set(map(str, data.get("contributors", [])))
        self._observers = set(map(str, data.get("observers", [])))

        if not self._contributors:
            raise ValueError("No contributors found in participant_roles.json")

        all_sites = list(self._contributors.union(self._observers))
        self._all_sites = self._sorted(all_sites) if all_sites else None

        logger.info(
            "Loaded participant roles",
            extra={
                "cfg_path": cfg_path,
                "contributors": sorted(self._contributors),
                "observers": sorted(self._observers),
            },
        )

    # ----------------------------
    # Computation parameters
    # ----------------------------

    def _load_and_set_computation_parameters(self, fl_ctx: FLContext) -> None:
        with open(get_parameters_file_path(fl_ctx), "r", encoding="utf-8") as f:
            fl_ctx.set_prop(
                key="COMPUTATION_PARAMETERS",
                value=json.load(f),
                private=False,
                sticky=True,
            )

    # ----------------------------
    # Required framework methods
    # ----------------------------

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        pass

    def stop_controller(self, fl_ctx: FLContext) -> None:
        pass

    # ----------------------------
    # Helpers
    # ----------------------------

    @staticmethod
    def _sorted(items) -> List[str]:
        return sorted(map(str, items))
