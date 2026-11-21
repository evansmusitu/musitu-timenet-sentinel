#!/usr/bin/env python3
"""Summarize Nebius project/region GPU capabilities from *read-only* CLI JSON.

Never creates resources; never implies that an advertised platform has quota.
"""
import json
import sys
from pathlib import Path

GPU_MARKERS=("gpu-", "h100", "h200", "rtx", "l40s", "a100", "b200", "b300")


def items_of(payload):
    if isinstance(payload, list):
        items=payload
    elif isinstance(payload, dict):
        items=payload.get("items", [])
    else:
        raise ValueError("Invalid Nebius response type")
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError("Invalid Nebius collection")
    return items


def summarize(payload):
    out=[]
    for item in items_of(payload):
        meta=item.get("metadata") or {}
        status=item.get("status") or {}
        spec=item.get("spec") or {}
        name=str(meta.get("name") or "")
        region=str(status.get("region") or spec.get("region") or "")
        # Project region is sometimes recorded as availability zone.
        zone=str(status.get("availability_zone") or spec.get("availability_zone") or "")
        out.append({"name":name,"region":region or zone or "unknown", "id":str(meta.get("id") or "")})
    return out


def summarize_platforms(payload):
    names=[]
    for item in items_of(payload):
        meta=item.get("metadata") or {}
        name=str(meta.get("name") or "")
        if any(token in name.lower() for token in GPU_MARKERS):
            names.append(name)
    return sorted(set(names))


if __name__=="__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: nebius_region_catalog.py project|platforms <jsonfile>")
    payload=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if sys.argv[1]=="project":
        print(json.dumps(summarize(payload),sort_keys=True))
    elif sys.argv[1]=="platforms":
        print(json.dumps(summarize_platforms(payload)))
    else:
        raise SystemExit("Unsupported action")
