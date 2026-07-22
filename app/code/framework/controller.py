from typing import Callable

from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable

from .shared import load_computation_parameters
from .types import (
    ITERATION_INDEX_KEY,
    ITERATION_STOP_KEY,
    ComputationSpec,
    IterativeWorkflow,
    SteppedWorkflow,
)


_AGGREGATOR_COMPONENT_ID = "aggregator"


class ComputationController(Controller):
    SPEC: ComputationSpec = None

    def __init__(
        self,
        min_clients: int = 2,
        wait_time_after_min_received: int = 10,
        task_timeout: int = 0,
    ):
        super().__init__()
        if self.SPEC is None:
            raise ValueError("Controller SPEC must be defined")
        self._task_timeout = task_timeout
        self._min_clients = min_clients
        self._wait_time_after_min_received = wait_time_after_min_received

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.aggregator = self._engine.get_component(_AGGREGATOR_COMPONENT_ID)
        fl_ctx.set_prop(
            key="COMPUTATION_PARAMETERS",
            value=load_computation_parameters(fl_ctx),
            private=False,
            sticky=True,
        )

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        workflow = self.SPEC.workflow
        if isinstance(workflow, SteppedWorkflow):
            self._run_stepped_workflow(workflow, abort_signal, fl_ctx)
            return
        if isinstance(workflow, IterativeWorkflow):
            self._run_iterative_workflow(workflow, abort_signal, fl_ctx)
            return
        raise ValueError(f"Unsupported workflow type: {type(workflow)!r}")

    def _run_stepped_workflow(
        self,
        workflow: SteppedWorkflow,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> None:
        incoming_shareable = Shareable()

        for current_round, step_definition in enumerate(workflow.steps):
            fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round)
            expects_remote_result = step_definition.remote_fn is not None

            self._broadcast_task(
                task_name=step_definition.name,
                data=incoming_shareable,
                result_cb=self._accept_site_result if expects_remote_result else None,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            if expects_remote_result:
                incoming_shareable = self.aggregator.aggregate(fl_ctx)
            else:
                incoming_shareable = Shareable()

    def _run_iterative_workflow(
        self,
        workflow: IterativeWorkflow,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> None:
        incoming_shareable = Shareable()

        for current_round in range(workflow.max_iterations):
            fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round)
            incoming_shareable[ITERATION_INDEX_KEY] = current_round
            self._broadcast_task(
                task_name=workflow.iteration_step.name,
                data=incoming_shareable,
                result_cb=self._accept_site_result,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
            incoming_shareable = self.aggregator.aggregate(fl_ctx)
            if incoming_shareable.get(ITERATION_STOP_KEY, False):
                break

        fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round + 1)
        incoming_shareable[ITERATION_INDEX_KEY] = current_round + 1
        self._broadcast_task(
            task_name=workflow.output_step.name,
            data=incoming_shareable,
            result_cb=None,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _accept_site_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        return self.aggregator.accept(client_task.result, fl_ctx)

    def _broadcast_task(
        self,
        task_name: str,
        data: Shareable,
        result_cb: Callable[[ClientTask, FLContext], bool],
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> None:
        self.broadcast_and_wait(
            task=Task(
                name=task_name,
                data=data,
                props={},
                timeout=self._task_timeout,
                result_received_cb=result_cb,
            ),
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        pass

    def stop_controller(self, fl_ctx: FLContext) -> None:
        pass


MultiRoundTabularController = ComputationController
