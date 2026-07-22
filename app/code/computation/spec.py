from framework import ComputationSpec, local_step, remote_step, site_output_step, stepped_workflow

from .inputs import load_ridge_inputs
from .local_math import compute_local_metrics, fit_local_models
from .remote_math import aggregate_final_results, aggregate_global_model
from .results import build_output_payloads


SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=fit_local_models, input_fn=load_ridge_inputs),
        remote_step(fn=aggregate_global_model),
        local_step(fn=compute_local_metrics),
        remote_step(fn=aggregate_final_results),
        site_output_step(fn=build_output_payloads),
    ),
)
