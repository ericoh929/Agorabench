#!/usr/bin/env python3
"""
Read dialogue .txt under a product folder; use an LLM to extract deal_price (single) or
deal_price + dealt_item (several); write {buyer}_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Literal, Optional, Sequence

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    # tqdm optional: keep dependencies minimal
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

Mode = Literal["single", "several"]

# Product category -> allowed item names for "several" mode
PRODUCT_VARIANTS: Dict[str, List[str]] = {
    "Camera": ["Digital Camera", "Film Camera", "DSLR Camera", "Action Camera"],
    "Jacket": ["Leather Jacket", "Parka Jacket", "Winter Jacket", "Rain Jacket"],
    "Smartphone": ["Flagship Smartphone", "Mid-Range Smartphone", "Budget Smartphone", "Gaming Smartphone"],
    "Shoes": ["Designer Shoes", "Casual Shoes", "Athletic Shoes", "Sandals"],
    "Bicycle": ["Mountain Bike", "Road Bike", "Hybrid Bike", "Folding Bike"],
    "Drone": ["Professional Drone", "Recreational Drone", "Racing Drone", "Mini Drone"],
    "Soccer Ball": ["Premium Soccer Ball", "Training Soccer Ball", "Recreational Soccer Ball", "Mini Soccer Ball"],
    "Bag": ["Leather Bag", "Backpack", "Tote Bag", "Drawstring Bag"],
    "Wine": ["Premium Wine", "Red Wine", "White Wine", "Sparkling Wine"],
    "Cup": ["Ceramic Cup", "Glass Cup", "Travel Cup", "Plastic Cup"],
}


def build_base_path(
    results_base: str,
    method: str,
    dataset: str,
    *,
    sft_epochs: int = 62,
    kto_epochs: int = 486,
) -> str:
    """Result root for a method and dataset (same layout as negotiation: results_base/<method>/<dataset>/)."""
    if method in ("ours", "react", "og"):
        return os.path.join(results_base, method, dataset)
    if method == "sft":
        return os.path.join(results_base, "oss_vs_deepseek", f"sft{sft_epochs}", "gpt20b", dataset)
    if method == "kto":
        return os.path.join(results_base, "oss_vs_deepseek", f"kto{kto_epochs}", "gpt20b", dataset)
    return os.path.join(results_base, dataset)


def _prompt_single() -> str:
    return """You are a highly efficient and accurate assistant specialized in finding out deal prices given text.
Your goal is to find the deal price for the buyer in text.

The text is a deal price summary of 10 dialogues. Based on the above text, find the deal price. There must be 10 deal prices.
If there are fewer than 10, treat missing slots as no deal matched. If no deal is matched, denote as 'No deal'.
If you find installment wording like '600 over 3 months', use 600 only (ignore the installment phrase).
Answer strictly in this format only — no extra text, no $ sign. Be careful with decimals (e.g. 26.50 not 2650).

Example format:
deal_price: [475, 425, 425, 'No deal', 350, 440, 420, 'No deal', 400, 420]

Provide exactly 10 entries. Do not copy example numbers from this prompt; extract from the Text.
"""


def _prompt_several(allowed_items: Sequence[str]) -> str:
    items_joined = ", ".join(allowed_items)
    return f"""You are a highly efficient assistant specialized in extracting deal prices and dealt items from negotiation text.

There are 10 dialogues. Output exactly 10 deal_price values and 10 dealt_item values aligned by index.
If no deal: use 'No deal' for both that index's price and item as appropriate (item may be 'No deal').
Installment like '600 over 3 months' → use 600 as the numeric price.
No $ signs. Pick dealt_item ONLY from this allowed list (exact strings): {items_joined}

Strict output format (two lines, same file content style as before — only these keys):
deal_price: [ ...10 numbers or 'No deal' ... ]
dealt_item: [ ...10 strings from the allowed list or 'No deal' ... ]

Do not output JSON fences or markdown. Do not copy example numbers below.

Example:
deal_price: [475, 425, 425, 'No deal', 350, 440, 420, 'No deal', 400, 420]
dealt_item: ['DSLR Camera', 'Film Camera', 'DSLR Camera', 'No deal', 'Action Camera', 'DSLR Camera', 'DSLR Camera', 'No deal', 'DSLR Camera', 'DSLR Camera']
"""


def extract_from_merged(
    merged_dir: str,
    buyer_model: str,
    mode: Mode,
    *,
    client: OpenAI,
    api_model: str,
    allowed_items: Optional[Sequence[str]] = None,
) -> None:
    """
    Read .txt in merged_dir; only files starting with ``{buyer_model}_``; write {buyer_model}_results.json.
    allowed_items: required for mode "several".
    """
    merged_dir = os.path.normpath(merged_dir)

    if mode == "several" and not allowed_items:
        raise ValueError("mode 'several' requires allowed_items.")

    system_single = _prompt_single()
    system_several = _prompt_several(list(allowed_items or []))

    def call_api(user_text: str) -> str:
        if mode == "single":
            instructions, user = system_single, f"# Text:\n{user_text}\n"
        else:
            instructions, user = system_several, f"# Text:\n{user_text}\n"
        response = client.responses.create(
            model=api_model,
            instructions=instructions,
            input=user,
        )
        return response.output_text

    data: List[dict] = []
    txt_files = [f for f in os.listdir(merged_dir) if f.endswith(".txt")]
    label = os.path.basename(merged_dir.rstrip(os.sep)) or "out"
    # Log filenames: {buyer_model}_{seller_model}.txt (split on buyer prefix, not first underscore only)
    buyer_prefix = f"{buyer_model}_"
    for txt_file in tqdm(txt_files, desc=label):
        if not txt_file.startswith(buyer_prefix):
            continue
        fp = os.path.join(merged_dir.rstrip(os.sep), txt_file)
        with open(fp, encoding="utf-8") as fh:
            content = fh.read()
        data.append({txt_file: call_api(content)})

    out_path = os.path.join(merged_dir.rstrip(os.sep), f"{buyer_model}_results.json")
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(data, out, indent=4, ensure_ascii=False)
    print(f"Wrote {out_path} ({len(data)} files)")


def run_batch(
    *,
    mode: Mode,
    results_base: str,
    methods: Sequence[str],
    datasets: Sequence[str],
    models: Sequence[str],
    sft_epochs: int,
    kto_epochs: int,
    api_model: str,
    only_products: Optional[set],
) -> None:
    client = OpenAI()
    for method in methods:
        print(f"[METHOD] {method}")
        for dataset in datasets:
            directory_path = build_base_path(results_base, method, dataset, sft_epochs=sft_epochs, kto_epochs=kto_epochs)
            if not os.path.isdir(directory_path):
                print(f"[SKIP] missing: {directory_path}")
                continue
            folders = [n for n in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, n))]
            for product in folders:
                if product in ("logs",):
                    continue
                if only_products is not None and product not in only_products:
                    continue
                if product not in PRODUCT_VARIANTS and mode == "several":
                    continue
                dialogue_dir = os.path.join(directory_path, product)
                if not os.path.isdir(dialogue_dir):
                    continue
                print(f"  {dataset} / {product}")
                variants = PRODUCT_VARIANTS[product] if mode == "several" else None
                for buyer_model in models:
                    extract_from_merged(
                        dialogue_dir,
                        buyer_model,
                        mode,
                        client=client,
                        api_model=api_model,
                        allowed_items=variants,
                    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TXT → LLM → deal_price JSON (single | several)")
    p.add_argument("--mode", choices=("single", "several"), required=True)
    p.add_argument(
        "--merged-dir",
        help="Product output folder containing .txt (e.g. .../category_vanilla/Camera)",
    )
    p.add_argument("--buyer-model", default=None, help="Required for single-folder mode (must match log prefix)")
    p.add_argument("--batch", action="store_true", help="Walk datasets and products under results-base")
    p.add_argument(
        "--method",
        nargs="+",
        default=None,
        help="Batch: ours react og sft kto (multiple tokens; comma-separated in one token ok)",
    )
    p.add_argument("--datasets", nargs="+", default=None, help="Dataset folder names for batch mode")
    p.add_argument(
        "--results-base",
        default=os.environ.get("NEGOTIATION_RESULTS", "/home/ericoh929/Agorabench/results"),
        help="Results root (override with NEGOTIATION_RESULTS)",
    )
    p.add_argument("--sft-epochs", type=int, default=62)
    p.add_argument("--kto-epochs", type=int, default=486)
    p.add_argument(
        "--api-model",
        default="gpt-4o",
        help="OpenAI Responses model id (negotiation postprocess usually matches --seller)",
    )
    p.add_argument(
        "--only-product",
        nargs="*",
        default=None,
        help="Batch: only these product folder names (space-separated). Default: all",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.1"],
        help="Batch: buyer model ids to match log prefixes",
    )
    return p.parse_args(argv)


DEFAULT_DATASETS_SINGLE = (
    "category_vanilla",
    "category_deceptive",
    "category_monopoly",
    "category_installment",
    "category_negative",
)
DEFAULT_DATASETS_SEVERAL = (
    "category_several",
    "category_several_installment",
    "category_several_negative",
    "category_several_monopoly",
)


def cli(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    models = args.models
    if len(models) == 1 and "," in models[0]:
        models = [m.strip() for m in models[0].split(",") if m.strip()]

    only_set = set(args.only_product) if args.only_product else None

    if args.batch:
        if not args.method:
            sys.exit("--batch requires --method")
        datasets = args.datasets
        if not datasets:
            # Default dataset lists (same as legacy batch scripts)
            datasets = DEFAULT_DATASETS_SINGLE if args.mode == "single" else DEFAULT_DATASETS_SEVERAL
        methods = args.method
        if len(methods) == 1 and "," in methods[0]:
            methods = [m.strip() for m in methods[0].split(",") if m.strip()]
        run_batch(
            mode=args.mode,
            results_base=args.results_base,
            methods=methods,
            datasets=datasets,
            models=models,
            sft_epochs=args.sft_epochs,
            kto_epochs=args.kto_epochs,
            api_model=args.api_model,
            only_products=only_set,
        )
        return

    if not args.merged_dir or not args.buyer_model:
        sys.exit("Need --merged-dir and --buyer-model, or use --batch.")

    variants = None
    if args.mode == "several":
        # Folder name is the category key (Camera, …); legacy .../Camera/merged still supported
        _md = os.path.normpath(args.merged_dir)
        cat = (
            os.path.basename(os.path.dirname(_md))
            if os.path.basename(_md) == "merged"
            else os.path.basename(_md)
        )
        variants = PRODUCT_VARIANTS.get(cat)
        if not variants:
            sys.exit(
                f"several mode: unknown category '{cat}'. Valid keys: {list(PRODUCT_VARIANTS.keys())}"
            )

    extract_from_merged(
        args.merged_dir,
        args.buyer_model,
        args.mode,
        client=OpenAI(),
        api_model=args.api_model,
        allowed_items=variants,
    )


def main() -> None:
    """Compatibility entry point."""
    cli()


if __name__ == "__main__":
    cli()
