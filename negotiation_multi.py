#!/usr/bin/env python3
"""
Multi-segment negotiation runner (category_several* datasets).
Datasets: datasets/category_several*.json (buyer/seller inventory fields; same idea as legacy gpt_vs_gemini_several).
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
    negotiate_multi_product,
)
from negotiation_postprocess import run_postprocess_multi_merged

MULTI_MARKET_FILES: Dict[str, str] = {
    "several": "category_several.json",
    "several_negative": "category_several_negative.json",
    "several_monopoly": "category_several_monopoly.json",
    "several_installment": "category_several_installment.json",
}


def load_scenarios(datasets_dir: Path, market: str) -> List[Dict[str, Any]]:
    fname = MULTI_MARKET_FILES.get(market)
    if not fname:
        raise ValueError(f"Unknown multi market: {market}")
    path = datasets_dir / fname
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Multi-setting negotiation (inventory lists; several-item markets)."
    )
    p.add_argument("--buyer", required=True)
    p.add_argument("--seller", required=True)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument(
        "--method",
        choices=("ours", "og", "react"),
        default="ours",
        help="react: dataset prompt only | og: OG narrator | ours: reward + AR term (no OAR paragraph)",
    )
    p.add_argument(
        "--markets",
        nargs="+",
        default=["several"],
        help="several several_negative several_monopoly several_installment or all",
    )
    p.add_argument("--datasets-dir", type=Path, default=here / "datasets")
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("AGORA_RESULTS_ROOT", str(here / "results"))),
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
        help="Run only these product scenarios. Default: all scenarios in each market.",
    )
    p.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Save dialogue logs only; skip extraction and metrics",
    )
    p.add_argument(
        "--extract-api-model",
        default=None,
        metavar="MODEL",
        help="Postprocess extraction model id (default: same as --seller)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    markets = args.markets
    if len(markets) == 1 and markets[0].lower() == "all":
        markets = list(MULTI_MARKET_FILES.keys())

    ensure_negotiation_credentials(args.buyer, args.seller)

    client = OpenAI()

    for market in markets:
        if market not in MULTI_MARKET_FILES:
            print(f"[SKIP] unknown market '{market}'", file=sys.stderr)
            continue
        scenarios = load_scenarios(args.datasets_dir, market)
        dataset_folder = MULTI_MARKET_FILES[market].replace(".json", "")
        selected = scenarios
        if args.products:
            selected = [s for s in scenarios if s["product"] in args.products]
        print(f"\n=== method={args.method} | Multi market: {market} ({dataset_folder}) — {len(selected)}/{len(scenarios)} scenarios ===")

        for scenario in tqdm(selected, desc=market):
            product = scenario["product"]
            buyer_block = scenario["buyer"]
            seller_block = scenario["seller"]
            base_buyer = buyer_block["system_prompt"]
            buyer_inv = buyer_block["inventory"]
            seller_prompt = seller_block["system_prompt"]
            seller_inv = seller_block["inventory"]

            buyer_prompt = buyer_system_prompt(
                base_buyer, args.method, multi=True
            )

            out = negotiation_output_dir(
                args.results_root,
                args.method,
                dataset_folder,
                product,
            )
            negotiate_multi_product(
                buyer_model=args.buyer,
                seller_model=args.seller,
                epoch=args.epoch,
                rounds=args.rounds,
                buyer_prompt=buyer_prompt,
                seller_prompt=seller_prompt,
                buyer_inventory=buyer_inv,
                seller_inventory=seller_inv,
                product=product,
                output_path=str(out),
                openai_client=client,
                method=args.method,
                seller_reasoning=(args.seller_reasoning.strip() or None),
            )
            if not args.skip_postprocess:
                extract_model = (args.extract_api_model or "").strip() or args.seller
                run_postprocess_multi_merged(
                    Path(out),
                    buyer=args.buyer,
                    seller=args.seller,
                    product_category=product,
                    extract_api_model=extract_model,
                )

    print("\nAll multi-setting sessions completed.")


if __name__ == "__main__":
    main()
