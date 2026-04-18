#!/usr/bin/env python3
"""
Buyer metric from merged *_results.json — single-product vs multi-product formulas.
Single: 1.0139*u + 0.8812*n + 1.1049
Multi:  1.0139*u + 0.8812*n + 1.1049*item_sim(item)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Multi-product metric weights (legacy script alignment)
W_U, W_N, W_I = 1.0139, 0.8812, 1.1049

# Item name -> similarity (multi only)
ITEM_SIM: Dict[str, float] = {
    "Digital Camera": 0.7783,
    "Film Camera": 0.5748,
    "DSLR Camera": 1.0000,
    "Action Camera": 0.5867,
    "Leather Jacket": 0.7219,
    "Parka Jacket": 0.7832,
    "Parka": 0.7832,
    "Winter Jacket": 1.0000,
    "Rain Jacket": 0.7017,
    "Flagship Smartphone": 1.0000,
    "Mid-Range Smartphone": 0.7886,
    "Budget Smartphone": 0.7746,
    "Gaming Smartphone": 0.7399,
    "Designer Shoes": 1.0000,
    "Casual Shoes": 0.6474,
    "Athletic Shoes": 0.6505,
    "Sandals": 0.5953,
    "Mountain Bike": 1.0000,
    "Road Bike": 0.7819,
    "Hybrid Bike": 0.6950,
    "Folding Bike": 0.6043,
    "Professional Drone": 1.0000,
    "Recreational Drone": 0.7905,
    "Racing Drone": 0.7725,
    "Mini Drone": 0.7350,
    "Premium Soccer Ball": 1.0000,
    "Training Soccer Ball": 0.7015,
    "Recreational Soccer Ball": 0.7154,
    "Mini Soccer Ball": 0.6609,
    "Leather Bag": 1.0000,
    "Backpack": 0.6217,
    "Tote Bag": 0.6175,
    "Drawstring Bag": 0.6222,
    "Premium Wine": 1.0000,
    "Red Wine": 0.7406,
    "White Wine": 0.6230,
    "Sparkling Wine": 0.5618,
    "Ceramic Cup": 1.0000,
    "Glass Cup": 0.7451,
    "Travel Cup": 0.6377,
    "Plastic Cup": 0.6371,
    "None": 0.0,
}

MULTI_PRODUCT_DATA: Dict[str, Dict[str, float]] = {
    "Digital Camera": {"P_budget": 500, "Selling Price": 430, "Cost": 300},
    "Film Camera": {"P_budget": 500, "Selling Price": 380, "Cost": 250},
    "DSLR Camera": {"P_budget": 500, "Selling Price": 550, "Cost": 400},
    "Action Camera": {"P_budget": 500, "Selling Price": 250, "Cost": 150},
    "Leather Jacket": {"P_budget": 100, "Selling Price": 120, "Cost": 70},
    "Parka Jacket": {"P_budget": 100, "Selling Price": 100, "Cost": 60},
    "Winter Jacket": {"P_budget": 100, "Selling Price": 90, "Cost": 50},
    "Rain Jacket": {"P_budget": 100, "Selling Price": 80, "Cost": 45},
    "Flagship Smartphone": {"P_budget": 800, "Selling Price": 850, "Cost": 600},
    "Mid-Range Smartphone": {"P_budget": 800, "Selling Price": 600, "Cost": 400},
    "Budget Smartphone": {"P_budget": 800, "Selling Price": 350, "Cost": 250},
    "Gaming Smartphone": {"P_budget": 800, "Selling Price": 700, "Cost": 500},
    "Designer Shoes": {"P_budget": 150, "Selling Price": 160, "Cost": 100},
    "Casual Shoes": {"P_budget": 150, "Selling Price": 100, "Cost": 60},
    "Athletic Shoes": {"P_budget": 150, "Selling Price": 130, "Cost": 90},
    "Sandals": {"P_budget": 150, "Selling Price": 70, "Cost": 40},
    "Mountain Bike": {"P_budget": 400, "Selling Price": 450, "Cost": 300},
    "Road Bike": {"P_budget": 400, "Selling Price": 350, "Cost": 200},
    "Hybrid Bike": {"P_budget": 400, "Selling Price": 300, "Cost": 180},
    "Folding Bike": {"P_budget": 400, "Selling Price": 250, "Cost": 150},
    "Professional Drone": {"P_budget": 600, "Selling Price": 620, "Cost": 450},
    "Recreational Drone": {"P_budget": 600, "Selling Price": 400, "Cost": 300},
    "Racing Drone": {"P_budget": 600, "Selling Price": 300, "Cost": 200},
    "Mini Drone": {"P_budget": 600, "Selling Price": 150, "Cost": 100},
    "Premium Soccer Ball": {"P_budget": 50, "Selling Price": 55, "Cost": 30},
    "Training Soccer Ball": {"P_budget": 50, "Selling Price": 35, "Cost": 20},
    "Recreational Soccer Ball": {"P_budget": 50, "Selling Price": 25, "Cost": 15},
    "Mini Soccer Ball": {"P_budget": 50, "Selling Price": 15, "Cost": 10},
    "Leather Bag": {"P_budget": 80, "Selling Price": 90, "Cost": 50},
    "Backpack": {"P_budget": 80, "Selling Price": 70, "Cost": 40},
    "Tote Bag": {"P_budget": 80, "Selling Price": 50, "Cost": 30},
    "Drawstring Bag": {"P_budget": 80, "Selling Price": 30, "Cost": 20},
    "Premium Wine": {"P_budget": 100, "Selling Price": 110, "Cost": 70},
    "Red Wine": {"P_budget": 100, "Selling Price": 80, "Cost": 50},
    "White Wine": {"P_budget": 100, "Selling Price": 60, "Cost": 40},
    "Sparkling Wine": {"P_budget": 100, "Selling Price": 40, "Cost": 25},
    "Ceramic Cup": {"P_budget": 30, "Selling Price": 35, "Cost": 20},
    "Glass Cup": {"P_budget": 30, "Selling Price": 25, "Cost": 15},
    "Travel Cup": {"P_budget": 30, "Selling Price": 20, "Cost": 10},
    "Plastic Cup": {"P_budget": 30, "Selling Price": 10, "Cost": 5},
}

SINGLE_PRICE: Dict[str, Dict[str, int]] = {
    "Camera": {"P_budget": 500, "P_initial": 550, "Cost": 400},
    "Jacket": {"P_budget": 100, "P_initial": 120, "Cost": 70},
    "Smartphone": {"P_budget": 800, "P_initial": 850, "Cost": 600},
    "Shoes": {"P_budget": 150, "P_initial": 160, "Cost": 100},
    "Bicycle": {"P_budget": 400, "P_initial": 450, "Cost": 300},
    "Drone": {"P_budget": 600, "P_initial": 620, "Cost": 450},
    "Soccer Ball": {"P_budget": 50, "P_initial": 55, "Cost": 30},
    "Bag": {"P_budget": 80, "P_initial": 90, "Cost": 50},
    "Wine": {"P_budget": 100, "P_initial": 110, "Cost": 70},
    "Cup": {"P_budget": 30, "P_initial": 35, "Cost": 20},
}

SINGLE_PROFIT_BUDGET: Dict[str, int] = {
    "Camera": 500,
    "Jacket": 100,
    "Smartphone": 800,
    "Shoes": 150,
    "Bicycle": 400,
    "Drone": 600,
    "Soccer Ball": 50,
    "Bag": 80,
    "Wine": 100,
    "Cup": 30,
}


def _no_deal_clean(s: str) -> str:
    return (
        s.replace("No deal", "None")
        .replace("\u2018No deal\u2019", "None")
        .replace("No Deal", "None")
    )


def extract_deal_prices_from_text(text: str) -> List[Optional[float]]:
    m = re.search(r"deal_price:\s*\[.*?\]", text, re.DOTALL)
    raw = m.group(0) if m else text
    cleaned = raw.replace("deal_price:", "")
    cleaned = _no_deal_clean(cleaned)
    deal_prices = eval(cleaned)
    out: List[Optional[float]] = []
    for dp in deal_prices:
        if isinstance(dp, str):
            dp = dp.replace("$", "").strip()
        if dp == "None" or dp is None:
            out.append(None)
        else:
            out.append(float(dp))
    return out


def extract_dealt_items_from_text(text: str) -> List[Any]:
    m = re.search(r"dealt_item:\s*\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    cleaned = m.group(0).replace("dealt_item:", "")
    cleaned = _no_deal_clean(cleaned)
    return eval(cleaned)


def average_deal_price(values: List[Optional[float]]) -> float:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else 0.0


def seller_from_negotiation_log(filename: str, buyer_model: str) -> str:
    """Parse seller id from negotiation log ``{buyer}_{seller}.txt``."""
    if not filename.endswith(".txt"):
        return ""
    stem = filename[:-4]
    prefix = buyer_model + "_"
    return stem[len(prefix) :] if stem.startswith(prefix) else ""


def _filter_df(df: pd.DataFrame, buyer_model: str, seller: Optional[str]) -> pd.DataFrame:
    df = df[df["Buyer"] == buyer_model]
    if seller is not None:
        df = df[df["Seller"] == seller]
    return df


def _agg_return(df: pd.DataFrame, multi_nan_to_zero: bool) -> Tuple[float, pd.Series, pd.Series, pd.Series]:
    avg_total = df.groupby("Buyer")["Avg Total"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
    avg_deal = df.groupby("Buyer")["Avg Deal Price"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
    avg_rate = df.groupby("Buyer")["Deal Rate"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
    bm = pd.to_numeric(df["Avg Total"], errors="coerce").mean()
    if multi_nan_to_zero and pd.isna(bm):
        bm = 0.0
    return float(bm) if not pd.isna(bm) else float("nan"), avg_total, avg_deal, avg_rate


def compute_single(
    merged_dir: str,
    buyer_model: str,
    p_budget: float,
    p_initial: float,
    cost: float,
    seller: Optional[str] = None,
) -> Tuple[float, pd.Series, pd.Series, pd.Series]:
    """
    Single-product buyer metric (fixed budget / initial ask / cost per run).
    Prices vary across scenarios; budget, initial, and cost stay fixed per category.
    """

    def util(p_deal: float) -> Optional[float]:
        if p_deal <= cost:
            return 1.0
        if p_deal >= p_budget:
            return 0.0
        if p_deal == 0:
            return None
        return (p_deal - p_budget) / (cost - p_budget)

    def neg(p_deal: float) -> float:
        if p_deal <= cost:
            return 1.0
        if p_deal >= p_initial:
            return 0.0
        return (p_deal - p_initial) / (cost - p_initial)

    path = os.path.join(merged_dir, f"{buyer_model}_results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows: Dict[str, List[Any]] = {"file": [], "Avg Total": [], "Avg Deal Price": [], "Deal Rate": []}
    for entry in data:
        for file_name, blob in entry.items():
            m = re.search(r"deal_price:\s*\[.*?\]", blob)
            if not m:
                continue
            deal_prices = extract_deal_prices_from_text(m.group(0))
            avg_dp = average_deal_price(deal_prices)
            total_sum, valid_count, tot_num = 0.0, 0, 0
            for dp in deal_prices:
                tot_num += 1
                if dp is None:
                    continue
                u, n = util(dp), neg(dp)
                if u is None:
                    continue
                total_sum += W_U * u + W_N * n + W_I
                valid_count += 1
            avg_total = (total_sum / tot_num) if tot_num > 0 else None
            rows["file"].append(file_name)
            rows["Avg Total"].append(round(avg_total, 4) if avg_total is not None else "No deal")
            rows["Avg Deal Price"].append(round(avg_dp, 1) if avg_dp != 0 else "No deal")
            rows["Deal Rate"].append(round(100 * valid_count / tot_num, 1) if tot_num > 0 else 0.0)

    df = pd.DataFrame(rows)
    df["Buyer"] = buyer_model
    df["Seller"] = df["file"].apply(lambda fn: seller_from_negotiation_log(fn, buyer_model))
    df = _filter_df(df, buyer_model, seller)
    return _agg_return(df, multi_nan_to_zero=False)


def compute_multi(
    merged_dir: str,
    buyer_model: str,
    product_data: Optional[Dict[str, Dict[str, float]]] = None,
    seller: Optional[str] = None,
) -> Tuple[float, pd.Series, pd.Series, pd.Series]:
    """
    Multi-product buyer metric (per dealt item: budget, selling price, cost; item similarity).
    Lookup tables and similarity weights depend on ``dealt_item``.
    """

    pdata = product_data or MULTI_PRODUCT_DATA
    path = os.path.join(merged_dir, f"{buyer_model}_results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows: Dict[str, List[Any]] = {"file": [], "Avg Total": [], "Avg Deal Price": [], "Deal Rate": []}
    for entry in data:
        for file_name, blob in entry.items():
            mp = re.search(r"deal_price:\s*\[.*?\]", blob, re.DOTALL)
            mi = re.search(r"dealt_item:\s*\[.*?\]", blob, re.DOTALL)
            if not mp or not mi:
                continue
            deal_prices = extract_deal_prices_from_text(mp.group(0))
            deal_items = extract_dealt_items_from_text(mi.group(0))
            avg_dp = average_deal_price(deal_prices)
            total_sum, valid_count, tot_num = 0.0, 0, 0
            for dp, ditem in zip(deal_prices, deal_items):
                tot_num += 1
                if dp is None:
                    continue
                if ditem not in pdata:
                    continue
                pb = pdata[ditem]["P_budget"]
                p_init = pdata[ditem]["Selling Price"]
                c = pdata[ditem]["Cost"]
                if dp <= c:
                    u = 1.0
                elif dp >= pb:
                    u = 0.0
                elif dp == 0:
                    u = None  # type: ignore
                else:
                    u = (dp - pb) / (c - pb)
                if u is None:
                    continue
                if dp <= c:
                    n = 1.0
                elif dp >= p_init:
                    n = 0.0
                else:
                    n = (dp - p_init) / (c - p_init)
                cos = ITEM_SIM.get(ditem, 0.0)
                total_sum += W_U * float(u) + W_N * n + W_I * cos
                valid_count += 1

            if valid_count > 0 and tot_num > 0:
                rows["file"].append(file_name)
                rows["Avg Total"].append(round(total_sum / tot_num, 4))
                rows["Avg Deal Price"].append(round(avg_dp, 1))
                rows["Deal Rate"].append(round(100 * valid_count / tot_num, 1))
            else:
                rows["file"].append(file_name)
                rows["Avg Total"].append("No deal")
                rows["Avg Deal Price"].append("No deal")
                rows["Deal Rate"].append(0.0)

    df = pd.DataFrame(rows)
    df["Buyer"] = buyer_model
    df["Seller"] = df["file"].apply(lambda fn: seller_from_negotiation_log(fn, buyer_model))
    df = _filter_df(df, buyer_model, seller)
    return _agg_return(df, multi_nan_to_zero=True)


def series_to_scalar(series_or_value: Any) -> float:
    if hasattr(series_or_value, "iloc"):
        if len(series_or_value) == 0:
            return float("nan")
        return float(pd.to_numeric(series_or_value, errors="coerce").mean())
    return float(pd.to_numeric(pd.Series([series_or_value]), errors="coerce").mean())


def _series_to_jsonable(s: pd.Series) -> Dict[str, Any]:
    """Serialize a pandas Series to a JSON-friendly dict."""
    out: Dict[str, Any] = {}
    for k, v in s.items():
        if pd.isna(v):
            out[str(k)] = None
        elif isinstance(v, (np.floating, float)):
            out[str(k)] = float(v)
        else:
            out[str(k)] = v
    return out


def save_metric_summary_json(
    output_path: str | Path,
    *,
    mode: str,
    buyer_model: str,
    merged_dir: str,
    buyer_metric: float,
    avg_total: pd.Series,
    avg_deal_price: pd.Series,
    deal_rate: pd.Series,
    seller: Optional[str] = None,
    product: Optional[str] = None,
) -> None:
    """Write metric summary JSON (shared by negotiation postprocess and CLI)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bm_val: Optional[float]
    if isinstance(buyer_metric, float) and np.isnan(buyer_metric):
        bm_val = None
    else:
        bm_val = float(buyer_metric)
    payload = {
        "mode": mode,
        "merged_dir": os.path.abspath(merged_dir),
        "buyer_model": buyer_model,
        "seller_filter": seller,
        "product": product,
        "buyer_metric": bm_val,
        "avg_total_by_buyer": _series_to_jsonable(avg_total),
        "avg_deal_price_by_buyer": _series_to_jsonable(avg_deal_price),
        "deal_rate_by_buyer": _series_to_jsonable(deal_rate),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_batch_single(
    root: str,
    buyer_models: List[str],
    skip_products: Optional[set] = None,
    seller: Optional[str] = None,
) -> None:
    skip_products = skip_products or set()
    for dataset in sorted(os.listdir(root)):
        dpath = os.path.join(root, dataset)
        if not os.path.isdir(dpath):
            continue
        print(f"Dataset: {dataset}")
        for bm in buyer_models:
            overall_bm: List[float] = []
            for product in sorted(os.listdir(dpath)):
                if product in skip_products or product == "logs":
                    continue
                pdir = os.path.join(dpath, product)
                if not os.path.isdir(pdir):
                    continue
                if product not in SINGLE_PRICE:
                    continue
                pr = SINGLE_PRICE[product]
                m, _, _, dr = compute_single(
                    pdir, bm, pr["P_budget"], pr["P_initial"], pr["Cost"], seller=seller,
                )
                overall_bm.append(m)
                print(f"  {product} buyer_metric={m:.4f}" if not np.isnan(m) else f"  {product} buyer_metric=nan")
            if overall_bm:
                arr = [x for x in overall_bm if not np.isnan(x)]
                if arr:
                    print(f"  [{bm}] mean={np.mean(arr):.4f} std={np.std(arr):.4f}")


def _run_batch_multi(
    root: str,
    buyer_models: List[str],
    skip_products: Optional[set] = None,
    seller: Optional[str] = None,
) -> None:
    skip_products = skip_products or set()
    for dataset in sorted(os.listdir(root)):
        dpath = os.path.join(root, dataset)
        if not os.path.isdir(dpath):
            continue
        print(f"Dataset: {dataset}")
        for bm in buyer_models:
            overall_bm: List[float] = []
            for product in sorted(os.listdir(dpath)):
                if product in skip_products or product == "logs":
                    continue
                pdir = os.path.join(dpath, product)
                if not os.path.isdir(pdir):
                    continue
                m, _, _, _ = compute_multi(pdir, bm, seller=seller)
                overall_bm.append(m)
                print(f"  {product} buyer_metric={m:.4f}")
            if overall_bm:
                print(f"  [{bm}] mean={np.mean(overall_bm):.4f} std={np.std(overall_bm):.4f}")


def main_cli() -> None:
    p = argparse.ArgumentParser(description="Buyer metric: single vs multi product JSON.")
    p.add_argument("--mode", choices=("single", "multi"), required=True)
    p.add_argument(
        "--merged-dir",
        help="Product folder with {buyer}_results.json and dialogue .txt files",
    )
    p.add_argument(
        "--batch-root",
        help="Parent path whose children are datasets (each dataset/<product>/...)",
    )
    p.add_argument("--buyer-model", dest="buyer_models", nargs="+", default=["gpt-4o"])
    p.add_argument("--seller", default=None, help="e.g. deepseek-chat (often used in multi)")
    p.add_argument("--skip-product", nargs="*", default=[], help="Skip these product folder names in batch mode")
    p.add_argument("--product", type=str, help="single: key in SINGLE_PRICE (e.g. Jacket)")
    p.add_argument("--budget", type=float)
    p.add_argument("--initial", type=float)
    p.add_argument("--cost", type=float)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save summary JSON here (single-folder --merged-dir runs)",
    )
    args = p.parse_args()

    models = args.buyer_models
    if len(models) == 1 and "," in models[0]:
        models = [x.strip() for x in models[0].split(",") if x.strip()]
    skip = set(args.skip_product)

    if args.batch_root:
        if args.mode == "single":
            _run_batch_single(args.batch_root, models, skip_products=skip, seller=args.seller)
        else:
            _run_batch_multi(args.batch_root, models, skip_products=skip, seller=args.seller)
        return

    if not args.merged_dir:
        raise SystemExit("Need --merged-dir or --batch-root.")

    for bm in models:
        if args.mode == "single":
            if args.budget is not None and args.initial is not None and args.cost is not None:
                b, i, c = args.budget, args.initial, args.cost
            elif args.product:
                if args.product not in SINGLE_PRICE:
                    raise SystemExit(f"Unknown --product: {args.product}")
                pr = SINGLE_PRICE[args.product]
                b, i, c = float(pr["P_budget"]), float(pr["P_initial"]), float(pr["Cost"])
            else:
                raise SystemExit(
                    "single mode: pass --budget/--initial/--cost, or --product, or use --batch-root"
                )
            m, t, dp, dr = compute_single(args.merged_dir, bm, b, i, c, seller=args.seller)
        else:
            m, t, dp, dr = compute_multi(args.merged_dir, bm, seller=args.seller)
        print(f"buyer_model={bm}  buyer_metric={m}")
        print(t)
        print(dp)
        print(dr)
        if args.output_json is not None:
            pk = args.product if args.mode == "single" else None
            save_metric_summary_json(
                args.output_json,
                mode=args.mode,
                buyer_model=bm,
                merged_dir=args.merged_dir,
                buyer_metric=m,
                avg_total=t,
                avg_deal_price=dp,
                deal_rate=dr,
                seller=args.seller,
                product=pk,
            )
            print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main_cli()
