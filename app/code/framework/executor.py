from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from .cache import JsonStateStore
from .logger import close_computation_logger
from .serialization import deserialize_value, serialize_value
from .shared import build_runtime_context, load_computation_parameters, resolve_output_directory
from .types import (
    ITERATION_INDEX_KEY,
    ComputationSpec,
    IterativeWorkflow,
    SteppedWorkflow,
)
from .writers import write_standard_outputs


class ComputationExecutor(Executor):
    SPEC: ComputationSpec = None

    def __init__(self):
        super().__init__()
        self._state_stores = {}

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if self.SPEC is None:
            raise ValueError("Executor SPEC must be defined")

        parameters = load_computation_parameters(fl_ctx)
        output_dir = resolve_output_directory(fl_ctx)
        state_store = JsonStateStore(
            output_dir,
            self.SPEC.codecs,
            self.SPEC.max_inline_array_bytes,
        )
        self._state_stores[output_dir] = state_store
        current_round = fl_ctx.get_prop("CURRENT_ROUND", default=0)
        workflow = self.SPEC.workflow

        local_fn = None
        local_input_type = None
        expects_remote_result = False
        clear_state_after_step = False

        if isinstance(workflow, SteppedWorkflow):
            step_definition = next(
                (
                    step
                    for step in workflow.steps
                    if step.name == task_name
                ),
                None,
            )
            if step_definition is None:
                raise ValueError(f"Unknown task name: {task_name}")

            local_fn = step_definition.local_fn
            local_input_type = step_definition.local_input_type
            expects_remote_result = step_definition.remote_fn is not None
            clear_state_after_step = (
                step_definition.is_site_output
                and step_definition is workflow.steps[-1]
            )
        elif isinstance(workflow, IterativeWorkflow):
            if task_name == workflow.iteration_step.name:
                local_fn = workflow.iteration_step.local_fn
                local_input_type = workflow.iteration_step.local_input_type
                expects_remote_result = True
            elif task_name == workflow.output_step.name:
                local_fn = workflow.output_step.local_fn
                local_input_type = workflow.output_step.local_input_type
                clear_state_after_step = True
            else:
                raise ValueError(f"Unknown task name: {task_name}")
        else:
            raise ValueError(f"Unsupported workflow type: {type(workflow)!r}")

        local_state_type = workflow.local_state_type
        if isinstance(workflow, IterativeWorkflow):
            current_round = shareable.get(ITERATION_INDEX_KEY, current_round)
        local_state = state_store.load_state(local_state_type)
        incoming_payload = deserialize_value(
            shareable.get("result"),
            local_input_type,
            self.SPEC.codecs,
            max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
        )

        runtime = build_runtime_context(
            self.SPEC,
            fl_ctx,
            current_round,
            parameters,
            logger_suffix=".log",
        )

        try:
            step_result = local_fn(
                incoming_payload,
                parameters,
                local_state,
                runtime,
            )

            if step_result.local_state is not None:
                state_store.save_state(step_result.local_state)

            if step_result.outputs:
                write_standard_outputs(step_result.outputs, runtime)
        finally:
            if runtime.logger:
                close_computation_logger(runtime.logger)

        if clear_state_after_step:
            self._remove_state(output_dir)

        if not expects_remote_result:
            return Shareable()

        outgoing_shareable = Shareable()
        outgoing_shareable["result"] = serialize_value(
            step_result.payload,
            self.SPEC.codecs,
            max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
        )
        return outgoing_shareable

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.END_RUN:
            for output_dir in tuple(self._state_stores):
                self._remove_state(output_dir)

    def _remove_state(self, output_dir: str) -> None:
        state_store = self._state_stores.pop(output_dir, None)
        if state_store is not None:
            state_store.remove_state()


MultiRoundTabularExecutor = ComputationExecutor
