# app/code/aggregator/aggregator.py
import logging
from typing import Dict, Any

from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import FLContextKey

from . import aggregator_methods as am


class SRRAggregator:
    def __init__(self):
        self.site_results: Dict[int, Dict[str, Any]] = {}
        self.agg_cache_dict: Dict[str, Any] = {}

    def _get_round(self, fl_ctx: FLContext) -> int:
        r = fl_ctx.get_prop("CURRENT_ROUND")
        try:
            return int(r) if r is not None else 0
        except Exception:
            return 0

    def _get_site_name(self, fl_ctx: FLContext, round_id: int) -> str:
        site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        if site_name:
            return site_name

        # fallback: deterministic synthetic site id
        existing = self.site_results.get(round_id, {})
        return f"site_{len(existing) + 1}"

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        if not isinstance(shareable, Shareable):
            logging.error("Aggregator.accept: non-Shareable received")
            return False

        round_id = self._get_round(fl_ctx)
        self.site_results.setdefault(round_id, {})

        site_name = self._get_site_name(fl_ctx, round_id)

        try:
            payload = shareable.get("result") or shareable.get("data")
            if payload is None:
                logging.error("Aggregator.accept: missing result/data payload")
                return False

            self.site_results[round_id][site_name] = payload
            logging.info(
                "Aggregator.accept: stored result for round=%s site=%s",
                round_id,
                site_name,
            )
            return True

        except Exception as e:
            logging.exception("Aggregator.accept failed: %s", e)
            return False

    def aggregate(self, fl_ctx: FLContext) -> Dict[str, Any]:
        round_id = self._get_round(fl_ctx)
        round_results = self.site_results.get(round_id, {})

        if not round_results:
            logging.warning(
                "Aggregator.aggregate: no results for round=%s", round_id
            )
            return {"output": {}, "cache": self.agg_cache_dict}

        logging.info(
            "Aggregator.aggregate: aggregating results for round=%s", round_id
        )

        if round_id == 0:
            # STEP 1: server-side global aggregation
            agg_out = am.perform_remote_step1_compute_global_parameters(
                round_results,
                self.agg_cache_dict,
            )
        else:
            # STEP 2: final metric aggregation + build results.zip (csv + index.html)
            agg_out = am.perform_remote_step2_final_metric_aggregation(
                round_results,
                self.agg_cache_dict,
                fl_ctx,
            )

        if isinstance(agg_out, dict) and "cache" in agg_out:
            self.agg_cache_dict = agg_out["cache"]

        return agg_out