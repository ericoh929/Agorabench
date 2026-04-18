#!/usr/bin/env python3
"""
After negotiation_single / negotiation_multi: run cal_buyer_send_to_llm then cal_buyer_metric
on each product output directory (where dialogue .txt files live) and write buyer_metric_summary.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from openai import OpenAI

from cal_buyer_metric import SINGLE_PRICE, compute_multi, compute_single, save_metric_summary_json
from cal_buyer_send_to_llm import PRODUCT_VARIANTS, extract_from_merged


def run_postprocess_single_merged(
    merged_dir: Path,
    *,
    buyer: str,
    seller: Optional[str],
    product: str,
    extract_api_model: str,
) -> Path:
    """Single-product market: TXT -> {buyer}_results.json -> buyer_metric_summary.json"""
    merged_s = str(merged_dir.resolve())
    client = OpenAI()
    extract_from_merged(
        merged_s,
        buyer,
        "single",
        client=client,
        api_model=extract_api_model,
    )
    pr = SINGLE_PRICE[product]
    m, t, dp, dr = compute_single(
        merged_s,
        buyer,
        float(pr["P_budget"]),
        float(pr["P_initial"]),
        float(pr["Cost"]),
        seller=seller,
    )
    out_json = merged_dir / "buyer_metric_summary.json"
    save_metric_summary_json(
        out_json,
        mode="single",
        buyer_model=buyer,
        merged_dir=merged_s,
        buyer_metric=m,
        avg_total=t,
        avg_deal_price=dp,
        deal_rate=dr,
        seller=seller,
        product=product,
    )
    print(f"[postprocess] {out_json}", flush=True)
    return out_json


def run_postprocess_multi_merged(
    merged_dir: Path,
    *,
    buyer: str,
    seller: Optional[str],
    product_category: str,
    extract_api_model: str,
) -> Path:
    """Several-style market: resolve allowed items by category (e.g. Camera), then same pipeline."""
    merged_s = str(merged_dir.resolve())
    variants = PRODUCT_VARIANTS.get(product_category)
    if not variants:
        raise ValueError(
            f"several extraction: unknown category '{product_category}' (not in PRODUCT_VARIANTS)."
        )
    client = OpenAI()
    extract_from_merged(
        merged_s,
        buyer,
        "several",
        client=client,
        api_model=extract_api_model,
        allowed_items=variants,
    )
    m, t, dp, dr = compute_multi(merged_s, buyer, seller=seller)
    out_json = merged_dir / "buyer_metric_summary.json"
    save_metric_summary_json(
        out_json,
        mode="multi",
        buyer_model=buyer,
        merged_dir=merged_s,
        buyer_metric=m,
        avg_total=t,
        avg_deal_price=dp,
        deal_rate=dr,
        seller=seller,
        product=product_category,
    )
    print(f"[postprocess] {out_json}", flush=True)
    return out_json
