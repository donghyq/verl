"""Fixed, fictional local-life snapshot for an isolated agentic RL experiment.

The two tools are read-only. Search deliberately omits the facts needed to
verify all constraints, so a model has to request details before answering.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


DISTRICTS = ("江湾", "老城", "科创园", "东门")
TEMPLATES = (
    ("米馆", "中餐", 92, 8, "22:30", 6, True, True, True, "安静", "米饭套餐、菌菇煲和时令蔬菜"),
    ("炭火小馆", "烧烤", 118, 6, "23:30", 8, False, True, False, "热闹", "炭烤套餐与现场烤制"),
    ("茶点空间", "茶点", 72, 11, "21:30", 4, True, True, True, "安静", "茶点、豆乳甜品和轻食"),
    ("家常菜馆", "中餐", 85, 10, "22:00", 6, True, True, True, "普通", "家常热菜与独立点单的素食菜品"),
    ("夜粥铺", "粥品", 62, 5, "23:30", 3, True, False, False, "普通", "现煮粥品、素菜小食和夜间简餐"),
    ("面馆", "面食", 68, 7, "23:00", 4, True, True, False, "热闹", "汤面、拌面和素食浇头"),
)


def catalog() -> dict[str, dict[str, Any]]:
    """Return an immutable-by-convention snapshot constructed from fixed facts."""
    result = {}
    for district_index, district in enumerate(DISTRICTS):
        for template_index, row in enumerate(TEMPLATES):
            kind, cuisine, price, walk, closing, seats, vegetarian, accessible, quiet, noise, menu = row
            venue_id = f"V{district_index * len(TEMPLATES) + template_index + 1:03d}"
            result[venue_id] = {
                "id": venue_id,
                "name": f"{district}{kind}",
                "district": district,
                "cuisine": cuisine,
                "price_yuan": price + (district_index % 3) * 3,
                "walk_minutes": walk + (district_index % 2),
                "closing_time": closing,
                "seats": seats,
                "vegetarian_options": vegetarian,
                "wheelchair_accessible": accessible,
                "quiet": quiet,
                "noise": noise,
                "menu": menu,
                "reservation": "4 人及以下到店等位；5 人及以上建议提前预约",
                "source": f"虚构门店快照 2026-09-29 / {venue_id}",
            }
    return result


CATALOG = catalog()


def eligible(venue: dict[str, Any], constraints: dict[str, Any]) -> bool:
    return (
        venue["district"] == constraints["district"]
        and (constraints["cuisine"] == "不限" or venue["cuisine"] == constraints["cuisine"])
        and venue["price_yuan"] <= constraints["max_price_yuan"]
        and venue["walk_minutes"] <= constraints["max_walk_minutes"]
        and venue["seats"] >= constraints["party_size"]
        and venue["closing_time"] >= constraints["open_until"]
        and (not constraints["vegetarian_required"] or venue["vegetarian_options"])
        and (not constraints["wheelchair_required"] or venue["wheelchair_accessible"])
        and (not constraints["quiet_required"] or venue["quiet"])
    )


async def _fixed_delay() -> None:
    # One documented latency setting for both arms. No external service calls.
    delay_ms = int(os.environ.get("LOCAL_LIFE_TOOL_DELAY_MS", "500"))
    if not 0 <= delay_ms <= 5000:
        raise ValueError("LOCAL_LIFE_TOOL_DELAY_MS must be between 0 and 5000")
    await asyncio.sleep(delay_ms / 1000)


class SearchCandidatesTool(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(get_json_schema(self.search_candidates))

    def search_candidates(self, district: str, cuisine: str = "不限") -> dict[str, Any]:
        """Search fictional venue candidates in one district.

        Detail-only facts such as closing time, seating, access and noise are
        omitted. Call query_venue_detail before making a recommendation.

        Args:
            district: One of 江湾, 老城, 科创园 or 东门.
            cuisine: Optional cuisine, or 不限 for all cuisines.
        """
        matches = [
            {
                "id": item["id"],
                "name": item["name"],
                "cuisine": item["cuisine"],
                "price_yuan": item["price_yuan"],
                "walk_minutes": item["walk_minutes"],
                "menu_preview": item["menu"],
            }
            for item in CATALOG.values()
            if item["district"] == district and (cuisine == "不限" or item["cuisine"] == cuisine)
        ]
        return {"district": district, "cuisine": cuisine, "candidates": matches}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        await _fixed_delay()
        agent_data = kwargs.get("agent_data")
        result = self.search_candidates(**parameters)
        if agent_data is not None:
            agent_data.extra_fields["local_life_search_calls"] = agent_data.extra_fields.get("local_life_search_calls", 0) + 1
            queries = agent_data.extra_fields.setdefault("local_life_search_queries", [])
            queries.append([parameters["district"], parameters.get("cuisine", "不限")])
            seen = agent_data.extra_fields.setdefault("local_life_candidate_ids", [])
            seen.extend(item["id"] for item in result["candidates"])
        return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}


class QueryVenueDetailTool(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(get_json_schema(self.query_venue_detail))

    def query_venue_detail(self, venue_id: str) -> dict[str, Any]:
        """Get the complete operational details for one fictional venue.

        Args:
            venue_id: Candidate ID returned by search_candidates, for example V001.
        """
        return CATALOG.get(venue_id, {"error": "unknown venue_id", "venue_id": venue_id})

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        await _fixed_delay()
        agent_data = kwargs.get("agent_data")
        venue_id = str(parameters.get("venue_id", ""))
        result = self.query_venue_detail(**parameters)
        if agent_data is not None and "error" not in result:
            ids = agent_data.extra_fields.setdefault("local_life_detail_ids", [])
            ids.append(venue_id)
        return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}


def compute_score(data_source: str, solution_str: str, ground_truth: dict, extra_info: dict | None = None, **kwargs):
    """Score real tool use, a valid recommendation and matching evidence."""
    del data_source, kwargs
    extra_info = extra_info or {}
    search_calls = int(extra_info.get("local_life_search_calls") or 0)
    search_queries = extra_info.get("local_life_search_queries") or []
    candidate_ids = extra_info.get("local_life_candidate_ids") or []
    detail_ids = extra_info.get("local_life_detail_ids") or []
    match = re.findall(r"FINAL_ID\s*[:=]\s*(V\d{3})\b", solution_str)
    selected_id = match[-1] if match else ""
    selected = CATALOG.get(selected_id)
    constraints = ground_truth["constraints"]
    full_search = [constraints["district"], constraints["cuisine"]] in search_queries
    used_tools = search_calls >= 1 and full_search and selected_id in candidate_ids and selected_id in detail_ids
    correct = bool(used_tools and selected and eligible(selected, constraints))
    eligible_venues = [item for item in CATALOG.values() if eligible(item, constraints)]
    optimal = min(eligible_venues, key=lambda item: (item["price_yuan"], item["walk_minutes"], item["id"]))
    lower_candidates = [
        item["id"]
        for item in CATALOG.values()
        if item["district"] == constraints["district"]
        and (constraints["cuisine"] == "不限" or item["cuisine"] == constraints["cuisine"])
        and selected is not None
        and item["price_yuan"] < selected["price_yuan"]
        and item["walk_minutes"] <= constraints["max_walk_minutes"]
    ]
    checked_lower = sum(venue_id in detail_ids for venue_id in lower_candidates)
    coverage = checked_lower / len(lower_candidates) if lower_candidates else 1.0
    optimal_proof = bool(correct and selected_id == optimal["id"] and coverage == 1.0)
    # Only evidence after the final marker counts. Tool responses cannot earn it.
    tail = solution_str.rsplit("FINAL_ID", 1)[-1] if match else ""
    evidence = bool(
        correct
        and "EVIDENCE" in tail
        and str(selected["price_yuan"]) in tail
        and selected["closing_time"] in tail
        and str(selected["walk_minutes"]) in tail
    )
    # Give credit only for verified progress: a valid choice, matching final
    # evidence, examination of cheaper candidates, and a proven optimum.
    score = (
        (0.2 if correct else 0.0)
        + (0.3 if evidence else 0.0)
        + (0.3 * coverage if correct else 0.0)
        + (0.2 if optimal_proof else 0.0)
    )
    return {
        "score": score,
        "valid_recommendation": float(correct),
        "evidence_consistent": float(evidence),
        "optimal_proof": float(optimal_proof),
        "lower_price_candidate_coverage": coverage if correct else 0.0,
        "search_calls": search_calls,
        "detail_calls": len(detail_ids),
    }
