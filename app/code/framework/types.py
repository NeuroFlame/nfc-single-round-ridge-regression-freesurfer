from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .serialization import DEFAULT_MAX_INLINE_ARRAY_BYTES


ITERATION_INDEX_KEY = "__neuroflame_iteration__"
ITERATION_STOP_KEY = "__neuroflame_iteration_stop__"


@dataclass
class RuntimeContext:
    fl_ctx: Any
    data_dir: str
    output_dir: str
    current_round: int
    logger: Any = None
    max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES


@dataclass
class StepResult:
    payload: Any = None
    local_state: Any = None
    remote_state: Any = None
    outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDefinition:
    name: str
    local_fn: Callable[[Any, Dict[str, Any], Dict[str, Any], RuntimeContext], StepResult]
    remote_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], RuntimeContext], StepResult]] = None
    local_input_type: Optional[type] = None
    remote_site_result_type: Optional[type] = None
    is_site_output: bool = False


@dataclass
class SteppedWorkflow:
    steps: List[StepDefinition]
    local_state_type: Optional[type] = None


@dataclass
class IterativeWorkflow:
    iteration_step: StepDefinition
    output_step: StepDefinition
    stop_fn: Optional[Callable[[Any, Dict[str, Any], Any, RuntimeContext], bool]] = None
    max_iterations: int = 50
    local_state_type: Optional[type] = None


WorkflowDefinition = Union[SteppedWorkflow, IterativeWorkflow]


@dataclass(init=False)
class ComputationSpec:
    workflow: WorkflowDefinition
    codecs: Dict[type, Any] = field(default_factory=dict)
    max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES

    def __init__(
        self,
        workflow: WorkflowDefinition,
        *,
        codecs: Optional[Mapping] = None,
        max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES,
    ):
        if not isinstance(workflow, (SteppedWorkflow, IterativeWorkflow)):
            raise TypeError(
                "ComputationSpec workflow must be created by stepped_workflow(...) "
                "or iterative_workflow(...)"
            )
        if codecs is not None and not isinstance(codecs, Mapping):
            raise TypeError("ComputationSpec codecs must be a type-to-codec mapping")

        resolved_codecs = dict(codecs or {})
        for value_type, codec in resolved_codecs.items():
            if not isinstance(value_type, type):
                raise TypeError("ComputationSpec codec keys must be Python types")
            if not callable(getattr(codec, "encode", None)) or not callable(
                getattr(codec, "decode", None)
            ):
                raise TypeError(
                    f"Codec for {value_type.__name__} must define callable encode() and decode()"
                )

        if isinstance(max_inline_array_bytes, bool) or not isinstance(
            max_inline_array_bytes, int
        ):
            raise TypeError("max_inline_array_bytes must be an integer byte count")
        if max_inline_array_bytes < 0:
            raise ValueError("max_inline_array_bytes cannot be negative")

        self.workflow = workflow
        self.codecs = resolved_codecs
        self.max_inline_array_bytes = max_inline_array_bytes
