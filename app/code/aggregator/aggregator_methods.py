# app/code/aggregator/aggregator_methods.py
#
# This module contains the server-side aggregation logic for SRR:
#   - Round 0: aggregate local ridge regression parameters into global parameters
#   - Round 1: aggregate local metric contributions into final global metrics
#
# NeuroFLAME integration (results-only):
#   - We persist human-friendly artifacts (results.json, CSV(s), index.html, results.zip)
#     to disk on the server at the end of Round 1.
#   - Results distribution to observers is handled OUTSIDE NVFLARE via NeuroFLAME's fileServer.
#
# IMPORTANT:
#   - Do NOT base64-embed results.zip in NVFlare shareables in results-only mode.
#   - The fileServer expects: <BASE_DIR>/<consortiumId>/<runId>/results/results.zip
#     (see fileServer route /download_results/:consortiumId/:runId)

import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from nvflare.apis.fl_context import FLContext

import numpy as np
import pandas as pd
import scipy as sp

from utils.ancillary import GlobalOutputMetricLabels, OutputDictKeyLabels

# HTML/CSV helpers (shared with client-side code)
from executor.client_executor_methods import _get_global_local_stats_df, _get_html_from_results

logger = logging.getLogger(__name__)


def _load_neuroflame_context_from_parameters() -> Dict[str, str]:
    # NVFlare server logs show parameters file path is supplied via env.
    candidates = [
        os.environ.get("NEUROFLAME_PARAMETERS_PATH"),
        os.environ.get("PARAMETERS_FILE"),
        "/workspace/runKit/parameters.json",
    ]
    for p in candidates:
        if not p:
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                nf = obj.get("neuroflame") or obj.get("NeuroFLAME") or {}
                if isinstance(nf, dict):
                    # normalize
                    out = {
                        "file_server_url": nf.get("file_server_url") or nf.get("fileServerUrl"),
                        "consortium_id": nf.get("consortium_id") or nf.get("consortiumId"),
                        "run_id": nf.get("run_id") or nf.get("runId"),
                        "token": nf.get("token") or nf.get("downloadToken") or nf.get("uploadToken"),
                    }
                    return {k: v for k, v in out.items() if v}
        except Exception:
            continue
    return {}



# ---------------------------------------------------------------------------
# Round 0: compute global regression parameters
# ---------------------------------------------------------------------------

def perform_remote_step1_compute_global_parameters(site_results: Dict[str, Any], agg_cache_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compute global beta vectors and other global parameters from client outputs.

    Args:
        site_results: dict[site_name -> per-ROI stats] from clients (Step 1)
        agg_cache_dict: aggregator cache to carry state across rounds

    Returns:
        Dict with:
          - 'output': global parameters keyed by ROI
          - 'cache': updated cache
    """
    global_results: Dict[str, Any] = {}
    num_sites = len(site_results.keys())
    if num_sites == 0:
        return {"output": {}, "cache": agg_cache_dict}

    # Infer ROI labels from first site payload.
    first_site = next(iter(site_results))
    first_roi = next(iter(site_results[first_site]))
    roi_labels = site_results[first_site][first_roi].get("y_labels") or list(site_results[first_site].keys())

    avg_coefficients_all_rois = []
    mean_y_global_all_rois = []
    dof_global_all_rois = []

    covariates_headers = None

    for roi_column in roi_labels:
        total_sum_coefficients = None
        total_sum_mean_y_local = 0.0
        total_subjects = 0

        for site, results in site_results.items():
            stats = results[roi_column]
            num_subjects = int(stats.get("num_subjects", 0))
            total_subjects += num_subjects

            coeff = np.array(stats[GlobalOutputMetricLabels.COEFFICIENT.value], dtype=float)
            total_sum_coefficients = coeff if total_sum_coefficients is None else (total_sum_coefficients + coeff)

            total_sum_mean_y_local += float(stats.get("mean_y_local", 0.0)) * num_subjects

            covariates_headers = stats.get(GlobalOutputMetricLabels.COVARIATE_LABELS.value)

        if total_sum_coefficients is None:
            continue

        # NOTE: original implementation averaged coefficients across sites (not subject-weighted).
        avg_coefficients = total_sum_coefficients / float(num_sites)

        # Degrees-of-freedom follows: total_subjects - number_of_parameters
        global_degrees_of_freedom = int(total_subjects - avg_coefficients.shape[0])

        # Mean of y is subject-weighted
        global_mean_y = (total_sum_mean_y_local / float(total_subjects)) if total_subjects > 0 else 0.0

        avg_coefficients_all_rois.append(avg_coefficients.tolist())
        mean_y_global_all_rois.append(global_mean_y)
        dof_global_all_rois.append(global_degrees_of_freedom)

        global_results[roi_column] = {
            "Variables": covariates_headers,
            "Global Coefficients": avg_coefficients.tolist(),
            "Global Degrees of Freedom": global_degrees_of_freedom,
            "Global Mean Y": global_mean_y,
        }

    # Store local stats for reporting later (Round 1 output packaging).
    all_local_stats_dicts = []
    for site in sorted(site_results.keys()):
        results = site_results[site]
        local_stats = []
        for roi in roi_labels:
            curr = results[roi]
            local_stats.append({
                GlobalOutputMetricLabels.COEFFICIENT.value: curr[GlobalOutputMetricLabels.COEFFICIENT.value],
                GlobalOutputMetricLabels.T_STAT.value: curr[GlobalOutputMetricLabels.T_STAT.value],
                GlobalOutputMetricLabels.P_VALUE.value: curr[GlobalOutputMetricLabels.P_VALUE.value],
                GlobalOutputMetricLabels.R_SQUARE.value: curr[GlobalOutputMetricLabels.R_SQUARE.value],
                GlobalOutputMetricLabels.COVARIATE_LABELS.value: curr[GlobalOutputMetricLabels.COVARIATE_LABELS.value],
                GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value: curr[GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value],
            })
        all_local_stats_dicts.append(local_stats)

    agg_cache_dict.update({
        "avg_coefficients": avg_coefficients_all_rois,
        "global_mean_y": mean_y_global_all_rois,
        "global_degrees_of_freedom": dof_global_all_rois,
        "X_labels": covariates_headers,
        "y_labels": roi_labels,
        "all_stats_local": all_local_stats_dicts,
    })

    return {"output": global_results, "cache": agg_cache_dict}


# ---------------------------------------------------------------------------
# Round 1: compute final global metrics
# ---------------------------------------------------------------------------

def perform_remote_step2(site_results: Dict[str, Any], agg_cache_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate local metrics and compute global metrics.

    Args:
        site_results: dict[site_name -> {SSE_local, SST_local, varX_matrix_local}]
        agg_cache_dict: cache from step 1

    Returns:
        Dict with:
          - 'output': list of ROI result dicts
          - 'cache': unchanged cache (but returned for consistency)
    """
    from itertools import repeat

    def get_stats_to_dict(a, *b):
        df = pd.DataFrame(list(zip(*b)), columns=a)
        return df.to_dict(orient="records")

    def t_to_p(ts_beta, dof):
        # Two-tailed p-values from t-stats.
        return [2 * sp.stats.t.sf(np.abs(t), dof) for t in ts_beta]

    X_labels = agg_cache_dict["X_labels"]
    y_labels = agg_cache_dict["y_labels"]
    all_local_stats_dicts_old = agg_cache_dict["all_stats_local"]
    avg_beta_vector = agg_cache_dict["avg_coefficients"]
    dof_global = agg_cache_dict["global_degrees_of_freedom"]

    SSE_global = sum([np.array(site_results[site]["SSE_local"]) for site in site_results])
    SST_global = sum([np.array(site_results[site]["SST_local"]) for site in site_results])
    varX_matrix_global = sum([np.array(site_results[site]["varX_matrix_local"]) for site in site_results])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i in range(len(MSE)):
        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global[i])
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (np.array(avg_beta_vector[i]) / se_beta_global).tolist()
        ps = t_to_p(ts, dof_global[i])
        ts_global.append(ts)
        ps_global.append(ps)

    # Local stats dicts grouped by ROI then by site
    sites = [site for site in site_results]
    all_local_stats_dicts = list(map(list, zip(*all_local_stats_dicts_old)))
    a_dict = [{key: value for key, value in zip(sites, stats_dict)} for stats_dict in all_local_stats_dicts]

    keys1 = [s.value for s in GlobalOutputMetricLabels]
    global_dict_list = get_stats_to_dict(
        keys1,
        avg_beta_vector,
        r_squared_global,
        ts_global,
        ps_global,
        dof_global,
        SSE_global.tolist(),
        repeat(X_labels, len(y_labels)),
    )

    keys2 = [s.value for s in OutputDictKeyLabels]
    dict_list = get_stats_to_dict(keys2, y_labels, global_dict_list, a_dict)

    return {"output": _round_floats_in_result(dict_list, decimal_places=4), "cache": agg_cache_dict}


def _perform_step2_aggregation(step2_results_by_site: Dict[str, Any], step1_agg: Any) -> Dict[str, Any]:
    """Compatibility helper used by perform_remote_step2_final_metric_aggregation.

    Some pipelines pass the full step1 aggregation dict (with keys like "output" and "cache"),
    while others pass just the cache dict. We normalize and delegate to perform_remote_step2.
    """

    if isinstance(step1_agg, dict) and "cache" in step1_agg:
        agg_cache_dict = step1_agg["cache"]
    else:
        # Assume it is already the cache dict.
        agg_cache_dict = step1_agg

    return perform_remote_step2(step2_results_by_site, agg_cache_dict)


def _round_floats_in_result(obj, decimal_places: int = 4):
    """Recursively rounds float values in nested dict/list structures."""
    if isinstance(obj, float):
        return round(obj, decimal_places)
    if isinstance(obj, dict):
        return {k: _round_floats_in_result(v, decimal_places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats_in_result(x, decimal_places) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Results artifacts (server-side, end of Round 1)
# ---------------------------------------------------------------------------

def _server_results_dir(fl_ctx) -> Path:
    """Return a default per-run results dir (next to runKit root)."""
    ws = fl_ctx.get_prop("WORKSPACE") or os.environ.get("WORKSPACE") or "/workspace/runKit"
    ws_path = Path(ws).resolve()

    # WORKSPACE is often .../runKit/server[/startup/..]
    if ws_path.name == "server":
        run_root = ws_path.parent
    elif ws_path.name.lower() == "runkit":
        run_root = ws_path.parent
    else:
        run_root = ws_path.parent

    return run_root / "results"


def _write_results_to_disk(results: Dict[str, Any], fl_ctx: Optional[FLContext] = None) -> str:
    """Persist aggregated results to a directory and return that directory path.

    This is used to create an artifact directory that can be zipped and sent back
    to clients/observers. We write a JSON payload plus a small human-readable
    summary so it's easy to sanity-check without parsing.
    """

    # Prefer the NVFlare output directory when fl_ctx is available.
    if fl_ctx is not None:
        out_dir = _server_results_dir(fl_ctx)
    else:
        # Fallback to a stable location under WORKSPACE/runKit.
        workspace = Path(os.environ.get("WORKSPACE", "/workspace/runKit")).resolve()
        out_dir = (workspace / "results")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write structured results.
    results_json = out_dir / "results.json"
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)


    # Human-friendly artifacts (HTML + CSV)
    try:
        # Results payload shape has changed over time:
        #  - legacy: a LIST of per-ROI dicts (written as global_regression_result.json)
        #  - current: a DICT with keys like {"output": [...], "cache": {...}} (written as results.json)
        # For backwards compatibility, always derive `agg_results` as the per-ROI LIST.
        agg_results = None
        if isinstance(results, list):
            agg_results = results
        elif isinstance(results, dict):
            if isinstance(results.get("output"), list):
                agg_results = results.get("output")
            elif isinstance(results.get("results"), list):
                agg_results = results.get("results")
        if agg_results is None:
            raise ValueError("Unable to determine aggregated results list for HTML/CSV generation")
    
        # Write legacy JSON filename expected by older clients/UI
        try:
            (out_dir / "global_regression_result.json").write_text(
                json.dumps(agg_results, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            pass
    
        # index.html (pretty report)
        html = _get_html_from_results(agg_results)
        (out_dir / "index.html").write_text(html, encoding="utf-8")
    
        # CSV(s) in the legacy naming scheme:
        #   - global_stats.csv
        #   - local_stats_<site>.csv
        # We build these directly from `agg_results` to avoid relying on helper return shapes.
        def _safe_float(x):
            try:
                return float(x)
            except Exception:
                return x
    
        # Build global_stats.csv rows
        global_rows = []
        local_rows_by_site = {}
    
        for r in agg_results:
            if not isinstance(r, dict):
                continue
            roi = r.get(OutputDictKeyLabels.ROI.value) or r.get("ROI")
            g = r.get(OutputDictKeyLabels.GLOBAL_STATS.value) or r.get("global_stats") or {}
            l = r.get(OutputDictKeyLabels.LOCAL_STATS.value) or r.get("local_stats") or {}
    
            cov_labels = list(g.get(GlobalOutputMetricLabels.COVARIATE_LABELS.value, []))
            # Normalize covariate labels to match the legacy CSV (e.g., sex -> sex_M)
            # The computation already emits "sex_M" in some contexts; we keep whatever is provided.
            coef = list(g.get(GlobalOutputMetricLabels.COEFFICIENT.value, []))
            tstat = list(g.get(GlobalOutputMetricLabels.T_STAT.value, []))
            pval = list(g.get(GlobalOutputMetricLabels.P_VALUE.value, []))
            rsq = g.get(GlobalOutputMetricLabels.R_SQUARE.value)
            dof = g.get(GlobalOutputMetricLabels.DEGREES_OF_FREEDOM.value)
            sse = g.get(GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value)
    
            # Create a flat row: ROI + per-covariate metrics + trailing scalars
            row = {"ROI": roi}
            for i, lab in enumerate(cov_labels):
                row[f"Coefficient_{lab}"] = _safe_float(coef[i]) if i < len(coef) else ""
            for i, lab in enumerate(cov_labels):
                row[f"t Stat_{lab}"] = _safe_float(tstat[i]) if i < len(tstat) else ""
            for i, lab in enumerate(cov_labels):
                row[f"P-value_{lab}"] = _safe_float(pval[i]) if i < len(pval) else ""
            row["R Squared"] = _safe_float(rsq)
            row["Degrees of Freedom"] = _safe_float(dof)
            row["Sum Square of Errors"] = _safe_float(sse)
            global_rows.append(row)
    
            # Local rows per site
            if isinstance(l, dict):
                for site, stats in l.items():
                    if not isinstance(stats, dict):
                        continue
                    site_row = {"ROI": roi}
                    lcoef = list(stats.get(GlobalOutputMetricLabels.COEFFICIENT.value, []))
                    ltstat = list(stats.get(GlobalOutputMetricLabels.T_STAT.value, []))
                    lpval = list(stats.get(GlobalOutputMetricLabels.P_VALUE.value, []))
                    lrsq = stats.get(GlobalOutputMetricLabels.R_SQUARE.value)
                    lsse = stats.get(GlobalOutputMetricLabels.SUM_OF_SQUARES_ERROR.value)
    
                    for i, lab in enumerate(cov_labels):
                        site_row[f"Coefficient_{lab}"] = _safe_float(lcoef[i]) if i < len(lcoef) else ""
                    for i, lab in enumerate(cov_labels):
                        site_row[f"t Stat_{lab}"] = _safe_float(ltstat[i]) if i < len(ltstat) else ""
                    for i, lab in enumerate(cov_labels):
                        site_row[f"P-value_{lab}"] = _safe_float(lpval[i]) if i < len(lpval) else ""
                    site_row["R Squared"] = _safe_float(lrsq)
                    site_row["Sum Square of Errors"] = _safe_float(lsse)
    
                    local_rows_by_site.setdefault(site, []).append(site_row)
    
        if global_rows:
            # Preserve column ordering like the legacy output as closely as possible
            df = pd.DataFrame(global_rows)
            df.to_csv(out_dir / "global_stats.csv", index=False)
    
        for site, rows in local_rows_by_site.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            safe_site = str(site).replace("/", "_").replace("\\", "_")
            df.to_csv(out_dir / f"local_stats_{safe_site}.csv", index=False)
    
    except Exception as e:
        logger.warning("Failed to build HTML/CSV artifacts: %s", e)
    
        # Write a quick summary for humans.
        summary_txt = out_dir / "summary.txt"
        try:
            keys = list(results.keys())
        except Exception:
            keys = []
    
        with summary_txt.open("w", encoding="utf-8") as f:
            f.write("Aggregated Results Summary\n")
            f.write("=========================\n")
            f.write(f"Top-level keys: {keys}\n")
            for k in keys:
                v = results.get(k)
                f.write(f"\n[{k}]\n")
                # Keep it short: show shape-like info.
                if isinstance(v, dict):
                    f.write(f"dict with {len(v)} keys\n")
                elif isinstance(v, list):
                    f.write(f"list with {len(v)} items\n")
                else:
                    f.write(f"type={type(v).__name__}\n")
    
    return str(out_dir)
    
    
    
    
def _zip_dir(src_dir: Optional[str], zip_path: str) -> None:
    """Create a zip file containing the contents of src_dir (recursively).

    Defensive behavior:
      - If src_dir is None/empty or doesn't exist, create an empty zip instead of crashing the run.
    """
    # Normalize inputs
    if not zip_path:
        raise ValueError('zip_path must be a non-empty string')

    if not src_dir:
        logger.error(f'_zip_dir: src_dir is empty; creating empty zip at: {zip_path}')
        import zipfile
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        return

    src_dir_abs = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir_abs):
        logger.error(f'_zip_dir: src_dir does not exist or is not a directory: {src_dir_abs}; creating empty zip at: {zip_path}')
        import zipfile
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        return

    # Ensure parent dir exists for zip file
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)

    import zipfile
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir_abs):
            for fn in files:
                full_path = os.path.join(root, fn)
                rel_path = os.path.relpath(full_path, src_dir_abs)
                zf.write(full_path, rel_path)


def write_results_artifacts(results: Dict[str, Any], fl_ctx: Optional[FLContext] = None) -> Dict[str, Any]:
    """Write server-side artifacts and a results.zip into the mounted output directory.

    NeuroFLAME mounts a per-run output directory into the server container at OUTPUT_DIRECTORY_PATH
    (typically /workspace/output). The NeuroFLAME fileServer serves files from that host directory.
    Therefore, this function must write the zip into OUTPUT_DIRECTORY_PATH rather than inside the runKit.
    """
    results_dir = _write_results_to_disk(results, fl_ctx=fl_ctx)

    # Safety: never let a missing results_dir crash the run
    if not results_dir:
        logger.error('write_results_artifacts: results_dir is empty; falling back to server results directory')
        try:
            fallback_dir = _server_results_dir(fl_ctx)
            fallback_dir.mkdir(parents=True, exist_ok=True)
            results_dir = str(fallback_dir)
        except Exception as e:
            logger.error(f'write_results_artifacts: failed to create fallback results dir: {e}')
            results_dir = ''
    if not results_dir:
        # Extremely defensive: never let artifact creation crash the run
        logger.error("write_results_artifacts: _write_results_to_disk returned empty/None; using server results dir")
        results_dir = str(_server_results_dir(fl_ctx))
        os.makedirs(results_dir, exist_ok=True)

    # Always zip into the mounted output directory
    output_dir = os.environ.get("OUTPUT_DIRECTORY_PATH", "/workspace/output")
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "results.zip")

    _zip_dir(results_dir, zip_path)

    # Upload to NeuroFLAME fileServer so observers/clients can download it.
    # This mirrors how NeuroFLAME distributes runKits (explicit upload, not filesystem watching).
    try:
        upload_results_to_fileserver(zip_path, fl_ctx=fl_ctx)
    except Exception as _e:
        logger.exception("NeuroFLAME results upload failed unexpectedly: %s", _e)

    logger.info("Wrote results zip for NeuroFLAME fileServer: %s", zip_path)
    return {"results_dir": results_dir, "results_zip_path": zip_path, "results_zip_name": "results.zip"}




# ---------------------------------------------------------------------------
# NeuroFLAME fileServer upload (results distribution)
# ---------------------------------------------------------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _truthy(v: Optional[str]) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _running_in_docker() -> bool:
    try:
        return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    except Exception:
        return False

def _normalize_fileserver_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    # Common container trap: localhost/127.0.0.1 points to *this* container.
    if (u.startswith("http://localhost") or u.startswith("https://localhost") or "://127.0.0.1" in u) and _running_in_docker():
        return (
            u.replace("localhost", "host.docker.internal")
             .replace("127.0.0.1", "host.docker.internal")
        )
    return u


def upload_results_to_fileserver(results_zip_path: str, fl_ctx: Optional[FLContext] = None) -> Dict[str, Any]:
    """Upload results.zip to NeuroFLAME fileServer so clients/observers can download it.

    Endpoint:
      POST {NEUROFLAME_FILESERVER_URL}/upload_results/{NEUROFLAME_CONSORTIUM_ID}/{NEUROFLAME_RUN_ID}
      multipart/form-data field name: 'file'
      header: x-access-token: {NEUROFLAME_ACCESS_TOKEN}

    Required env vars (ONLY these):
      - NEUROFLAME_FILESERVER_URL
      - NEUROFLAME_CONSORTIUM_ID
      - NEUROFLAME_RUN_ID
      - NEUROFLAME_ACCESS_TOKEN

    Optional debug env vars:
      - NEUROFLAME_RESULTS_UPLOAD_ENABLED (default: true)
      - NEUROFLAME_RESULTS_UPLOAD_DEBUG   (default: false)
      - NEUROFLAME_FILESERVER_PREFLIGHT   (default: false)  # GET /health before upload (non-fatal)
    """

    def _running_in_docker() -> bool:
        try:
            return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        except Exception:
            return False

    def _normalize_fileserver_url(u: Optional[str]) -> Optional[str]:
        """Inside Docker, rewrite localhost/127.0.0.1 to host.docker.internal (Mac Docker Desktop)."""
        if not u:
            return u
        s = u.strip()
        if not s:
            return s
        if _running_in_docker() and (
            s.startswith("http://localhost")
            or s.startswith("https://localhost")
            or "://127.0.0.1" in s
        ):
            return s.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
        return s

    enabled = _env("NEUROFLAME_RESULTS_UPLOAD_ENABLED", "true")
    debug = _env("NEUROFLAME_RESULTS_UPLOAD_DEBUG", "false")
    preflight = _env("NEUROFLAME_FILESERVER_PREFLIGHT", "false")

    if not _truthy(enabled):
        logger.info("NeuroFLAME results upload disabled via NEUROFLAME_RESULTS_UPLOAD_ENABLED=%s", enabled)
        return {"uploaded": False, "reason": "disabled"}

    # --- ONLY the env vars you specified ---
    file_server = _env("NEUROFLAME_FILESERVER_URL")
    consortium_id = _env("NEUROFLAME_CONSORTIUM_ID")
    run_id = _env("NEUROFLAME_RUN_ID")
    token = _env("NEUROFLAME_ACCESS_TOKEN")

    if not all([file_server, consortium_id, run_id, token]):
        missing = [
            k for k, v in {
                "NEUROFLAME_FILESERVER_URL": file_server,
                "NEUROFLAME_CONSORTIUM_ID": consortium_id,
                "NEUROFLAME_RUN_ID": run_id,
                "NEUROFLAME_ACCESS_TOKEN": token,
            }.items()
            if not v
        ]
        logger.warning("NeuroFLAME results upload skipped: missing env vars: %s", ", ".join(missing))
        return {"uploaded": False, "reason": "missing_env", "missing": missing}

    # Normalize localhost inside container -> host.docker.internal
    file_server = _normalize_fileserver_url(file_server)

    if not os.path.exists(results_zip_path):
        logger.error("NeuroFLAME results upload failed: results zip not found at %s", results_zip_path)
        return {"uploaded": False, "reason": "zip_missing", "results_zip_path": results_zip_path}

    try:
        file_size = os.path.getsize(results_zip_path)
    except Exception:
        file_size = None

    try:
        import requests
    except Exception as e:
        logger.error("NeuroFLAME results upload failed: requests not available: %s", e)
        return {"uploaded": False, "reason": "requests_missing", "error": str(e)}

    url = f"{file_server.rstrip('/')}/upload_results/{consortium_id}/{run_id}"
    headers = {"x-access-token": token}

    logger.info("Uploading results.zip to %s", url)

    if _truthy(debug):
        logger.info(
            "NeuroFLAME results upload debug",
            extra={
                "url": url,
                "results_zip_path": results_zip_path,
                "file_size": file_size,
                "running_in_docker": _running_in_docker(),
            },
        )

    if _truthy(preflight):
        try:
            health_url = f"{file_server.rstrip('/')}/health"
            r = requests.get(health_url, timeout=5)
            if _truthy(debug):
                logger.info("fileServer preflight GET /health status=%s", r.status_code)
        except Exception:
            if _truthy(debug):
                logger.info("fileServer preflight skipped/failed (non-fatal)", exc_info=True)

    try:
        with open(results_zip_path, "rb") as f:
            files = {"file": ("results.zip", f, "application/zip")}
            resp = requests.post(url, headers=headers, files=files, timeout=120)

        if 200 <= resp.status_code < 300:
            logger.info("NeuroFLAME results uploaded successfully (status=%s, size=%s)", resp.status_code, file_size)
            return {"uploaded": True, "status": resp.status_code, "url_used": url, "file_size": file_size}

        try:
            body = (resp.text or "")[:2000]
        except Exception:
            body = "<unreadable>"

        logger.error("NeuroFLAME results upload failed (status=%s): %s", resp.status_code, body)
        return {"uploaded": False, "status": resp.status_code, "url_used": url, "file_size": file_size, "body": body}

    except Exception as e:
        logger.exception("NeuroFLAME results upload exception: %s", e)
        return {"uploaded": False, "reason": "exception", "error": str(e), "url_used": url, "file_size": file_size}


def perform_remote_step2_final_metric_aggregation(
    step2_results_by_site: Dict[str, Dict[str, Any]],
    step1_agg: Dict[str, Any],
    fl_ctx: Optional[FLContext] = None,
) -> Dict[str, Any]:
    """Aggregate Step2 results and package artifacts for NeuroFLAME fileServer.

    IMPORTANT: We do NOT embed results.zip into the NVFlare shareable (no base64 shim).
    Large outputs should be transferred via NeuroFLAME's fileServer from the mounted OUTPUT_DIRECTORY_PATH.
    """
    # Step2 aggregation (existing behavior)
    final_output = _perform_step2_aggregation(step2_results_by_site, step1_agg)

    artifacts = write_results_artifacts(final_output, fl_ctx=fl_ctx)

    return {
        "output": final_output,
        "cache": {},
        "results_zip_path": artifacts.get("results_zip_path"),
        "results_zip_name": artifacts.get("results_zip_name"),
    }

