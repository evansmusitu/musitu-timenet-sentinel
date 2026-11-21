#!/usr/bin/env python3
"""Safe GPU planning and read-only Nebius inventory summarizer. Never launches VMs."""
from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, ROUND_UP
from pathlib import Path

GPU_RATE_USD = Decimal("1.80")  # Published indicative on-demand rate; not a quotation.
EXTRA_HEADROOM_USD = Decimal("0.50")
MAX_GPU_MINUTES = 60
MAX_SINGLE_RUN_USD = Decimal("3.00")
# Conservatively use the higher published B300 rate while Nebius price pages vary.
PROJECT_PLATFORM_RATES = {"gpu-b300-sxm": Decimal("9.50"), "gpu-rtx6000-a": Decimal("1.80")}


def estimate_cost(minutes: int, gpu_count: int = 1, rate: Decimal = GPU_RATE_USD) -> Decimal:
    if type(minutes) is not int or not 1 <= minutes <= MAX_GPU_MINUTES:
        raise ValueError("GPU runtime must be an integer from 1 to 60 minutes")
    if gpu_count != 1:
        raise ValueError("At most one GPU per run")
    if rate <= 0:
        raise ValueError("Hourly GPU rate must be positive")
    value = rate * Decimal(minutes) / Decimal(60) + EXTRA_HEADROOM_USD
    value = value.quantize(Decimal("0.01"), rounding=ROUND_UP)
    if value > MAX_SINGLE_RUN_USD:
        raise ValueError("Estimated cost exceeds the per-run planning threshold")
    return value


def _items(document: object) -> list[dict]:
    if isinstance(document, list):
        items = document
    elif isinstance(document, dict):
        items = document.get("items")
        if items is None:
            items = document.get("platforms")
        if items is None:
            items = document.get("subnets")
        if items is None:
            items = document.get("instances")
        if items is None and document == {}:
            items = []
        if items is None:
            raise ValueError("Unrecognized Nebius CLI JSON format")
    else:
        raise ValueError("CLI JSON must be a list or object")
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError("CLI collection must contain JSON objects")
    return items


def read_inventory(directory: Path) -> dict:
    def load(name):
        return _items(json.loads((directory / name).read_text(encoding="utf-8")))

    platforms = load("platforms.json")
    subnets = load("subnets.json")
    instances = load("instances.json")
    # Inspect advertised platform identity; not a resource availability or quota guarantee.
    gpu_names = []
    for item in platforms:
        meta = item.get("metadata") or {}
        text = " ".join(
            str(s) for s in [
                meta.get("name", ""),
                (item.get("spec") or {}).get("name", ""),
                (item.get("spec") or {}).get("description", ""),
            ]
        ).lower()
        if any(token in text for token in ("gpu", "h100", "h200", "rtx", "b200", "b300", "l40s", "a100")):
            gpu_names.append(meta.get("name") or "unnamed-gpu-platform")

    advertised_rates = {name: PROJECT_PLATFORM_RATES[name]
                        for name in gpu_names if name in PROJECT_PLATFORM_RATES}
    project_45m_estimates = {
        name: str((rate * Decimal(45) / Decimal(60) + EXTRA_HEADROOM_USD)
                  .quantize(Decimal("0.01"), rounding=ROUND_UP))
        for name, rate in advertised_rates.items()
    }
    return {
        "advertised_gpu_45m_indicative_cost_usd": project_45m_estimates,
        "advertised_gpu_45m_below_planning_threshold": all(
            Decimal(cost) <= MAX_SINGLE_RUN_USD
            for cost in project_45m_estimates.values()
        ) if project_45m_estimates else False,
        "platform_count": len(platforms),
        "gpu_platforms_advertised": sorted(set(gpu_names))[:15],
        "subnet_configured": bool(subnets),
        "existing_instance_count": len(instances),
        "gpu_capacity_confirmed": False,
        "actual_price_verified": False,
        "billing_hard_cap_configured": False,
        "ready_to_launch": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", type=Path)
    parser.add_argument("--gpu-minutes", type=int, default=45)
    args = parser.parse_args()
    estimate = estimate_cost(args.gpu_minutes)
    report = {
        "mode": "DRY_RUN_ONLY",
        "gpu_count": 1,
        "planned_max_runtime_minutes": args.gpu_minutes,
        "indicative_rate_usd_per_gpu_hour": str(GPU_RATE_USD),
        "headroom_usd": str(EXTRA_HEADROOM_USD),
        "indicative_run_estimate_usd": str(estimate),
        "max_single_run_planning_threshold_usd": str(MAX_SINGLE_RUN_USD),
        "disclaimer": "Estimate, not a billing hard cap. Disk, egress, spot variation and taxes can add costs.",
        "has_cloud_write_operations": False,
    }
    if args.inventory_dir is not None:
        report["inventory"] = read_inventory(args.inventory_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
