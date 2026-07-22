from typing import Dict, Any

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator

from .logger import close_computation_logger
from .serialization import deserialize_value, serialize_value
from .shared import build_runtime_context, load_computation_parameters, resolve_site_name
from .types import (
    ITERATION_STOP_KEY,
    ComputationSpec,
    IterativeWorkflow,
    SteppedWorkflow,
)


class ComputationAggregator(Aggregator):
    SPEC: ComputationSpec = None

    def __init__(self):
        super().__init__()
        if self.SPEC is None:
            raise ValueError("Aggregator SPEC must be defined")
        self.site_results: Dict[int, Dict[str, Any]] = {}
        self.remote_state: Any = None

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        site_id = site_result.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default=None)
        current_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)

        if current_round is None or site_id is None:
            return False

        parameters = fl_ctx.get_prop(key="COMPUTATION_PARAMETERS", default={})
        site_name = resolve_site_name(site_id, parameters)
        self.site_results.setdefault(current_round, {})
        self.site_results[current_round][site_name] = site_result["result"]
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        current_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)
        if current_round is None:
            return Shareable()

        workflow = self.SPEC.workflow
        remote_site_result_type = None
        if isinstance(workflow, SteppedWorkflow):
            step_definition = workflow.steps[current_round]
            remote_fn = step_definition.remote_fn
            remote_site_result_type = step_definition.remote_site_result_type
            if remote_fn is None:
                return Shareable()
        elif isinstance(workflow, IterativeWorkflow):
            remote_fn = workflow.iteration_step.remote_fn
            remote_site_result_type = workflow.iteration_step.remote_site_result_type
        else:
            raise ValueError(f"Unsupported workflow type: {type(workflow)!r}")

        parameters = load_computation_parameters(fl_ctx)
        runtime = build_runtime_context(
            self.SPEC,
            fl_ctx,
            current_round,
            parameters,
            logger_suffix=".remote.log",
        )

        try:
            site_results = {
                site_name: deserialize_value(
                    site_payload,
                    remote_site_result_type,
                    self.SPEC.codecs,
                    max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
                )
                for site_name, site_payload in self.site_results.get(current_round, {}).items()
            }
            step_result = remote_fn(
                site_results,
                parameters,
                self.remote_state,
                runtime,
            )
            should_stop = False
            if isinstance(workflow, IterativeWorkflow) and workflow.stop_fn is not None:
                stop_state = (
                    step_result.remote_state
                    if step_result.remote_state is not None
                    else self.remote_state
                )
                should_stop = workflow.stop_fn(
                    step_result.payload,
                    parameters,
                    stop_state,
                    runtime,
                )
        finally:
            if runtime.logger:
                close_computation_logger(runtime.logger)

        if step_result.remote_state is not None:
            self.remote_state = step_result.remote_state

        outgoing_shareable = Shareable()
        outgoing_shareable["result"] = serialize_value(
            step_result.payload,
            self.SPEC.codecs,
            max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
        )
        if isinstance(workflow, IterativeWorkflow):
            outgoing_shareable[ITERATION_STOP_KEY] = should_stop
        return outgoing_shareable

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.END_RUN:
            self.site_results.clear()
            self.remote_state = None


MultiRoundTabularAggregator = ComputationAggregator
