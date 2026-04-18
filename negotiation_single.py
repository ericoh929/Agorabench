#!/usr/bin/env python3
"""
Single-product negotiation runner (vanilla / negative / monopoly / installment / deceptive).
Datasets: datasets/category_<market>.json (same scenario shape as the legacy gpt_vs_gemini_vanilla flow).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # type: ignore
        return x

from openai import OpenAI

from negotiation_core import (
    buyer_system_prompt,
    ensure_negotiation_credentials,
    negotiation_output_dir,
    negotiate_single_product,
)
from negotiation_postprocess import run_postprocess_single_merged

# market name -> JSON filename under datasets/
SINGLE_MARKET_FILES: Dict[str, str] = {
    "vanilla": "category_vanilla.json",
    "negative": "category_negative.json",
    "monopoly": "category_monopoly.json",
    "installment": "category_installment.json",
    "deceptive": "category_deceptive.json",
}


def load_scenarios(datasets_dir: Path, market: str) -> List[Dict[str, Any]]:
    fname = SINGLE_MARKET_FILES.get(market)
    if not fname:
        raise ValueError(f"Unknown single market: {market}")
    path = datasets_dir / fname
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Single-setting negotiation (one product segment per scenario)."
    )
    p.add_argument("--buyer", required=True, help="Buyer model id (e.g. gpt-4o, gemini-1.5-pro)")
    p.add_argument("--seller", required=True, help="Seller model id")
    p.add_argument("--epoch", type=int, default=10, help="Sessions per scenario")
    p.add_argument("--rounds", type=int, default=10, help="Max dialogue rounds per session")
    p.add_argument(
        "--method",
        choices=("ours", "og", "react"),
        default="ours",
        help="react: JSON scenario only | og: OG narrator (fixed offer per turn) | ours: reward terms, no OAR. "
        "Outputs: <results-root>/<method>/<dataset>/<product>/",
    )
    p.add_argument(
        "--markets",
        nargs="+",
        default=["vanilla"],
        help="One or more of: vanilla negative monopoly installment deceptive, or 'all'.",
    )
    p.add_argument(
        "--datasets-dir",
        type=Path,
        default=here / "datasets",
        help="Directory containing category_*.json",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("AGORA_RESULTS_ROOT", str(here / "results"))),
        help="Root folder for logs (override with AGORA_RESULTS_ROOT)",
    )
    p.add_argument(
        "--seller-reasoning",
        type=str,
        default="",
        help="OpenAI seller only: Responses API reasoning effort (high|medium|low). Empty = disabled.",
    )
    p.add_argument(
        "--products",
        nargs="*",
        default=None,
        metavar="PRODUCT",
        help="Run only these product names (e.g. Camera Jacket). Default: all scenarios in each market.",
    )
    p.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Save dialogue logs only; skip LLM extraction and buyer_metric_summary.json",
    )
    p.add_argument(
        "--extract-api-model",
        default=None,
        metavar="MODEL",
        help="Postprocess: OpenAI model id for extraction (default: same as --seller).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    markets = args.markets
    if len(markets) == 1 and markets[0].lower() == "all":
        markets = list(SINGLE_MARKET_FILES.keys())

    ensure_negotiation_credentials(args.buyer, args.seller)

    client = OpenAI()

    for market in markets:
        if market not in SINGLE_MARKET_FILES:
            print(f"[SKIP] unknown market '{market}'", file=sys.stderr)
            continue
        scenarios = load_scenarios(args.datasets_dir, market)
        dataset_folder = SINGLE_MARKET_FILES[market].replace(".json", "")
        selected = scenarios
        if args.products:
            selected = [s for s in scenarios if s["product"] in args.products]
        print(f"\n=== method={args.method} | Market: {market} ({dataset_folder}) — {len(selected)}/{len(scenarios)} scenarios ===")

        for scenario in tqdm(selected, desc=f"{market}"):
            product = scenario["product"]
            base_buyer = scenario["buyer"]["system_prompt"]
            seller_prompt = scenario["seller"]["system_prompt"]
            buyer_prompt = buyer_system_prompt(
                base_buyer, args.method, multi=False
            )

            out = negotiation_output_dir(
                args.results_root,
                args.method,
                dataset_folder,
                product,
            )
            negotiate_single_product(
                buyer_model=args.buyer,
                seller_model=args.seller,
                epoch=args.epoch,
                rounds=args.rounds,
                buyer_prompt=buyer_prompt,
                seller_prompt=seller_prompt,
                product=product,
                output_path=str(out),
                openai_client=client,
                method=args.method,
                seller_reasoning=(args.seller_reasoning.strip() or None),
            )
            if not args.skip_postprocess:
                extract_model = (args.extract_api_model or "").strip() or args.seller
                run_postprocess_single_merged(
                    Path(out),
                    buyer=args.buyer,
                    seller=args.seller,
                    product=product,
                    extract_api_model=extract_model,
                )

    print("\nAll single-setting sessions completed.")


if __name__ == "__main__":
    main()
