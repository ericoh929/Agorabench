# Agorabench

Tools to **run multi-turn buyer–seller negotiations**, optionally **extract deal prices** from dialogue logs with an LLM, and **compute buyer-side metrics**. Scenario JSON lives under `datasets/`.

---

## Contents

1. [Quick start](#quick-start)
2. [Pipeline & folder layout](#pipeline--folder-layout)
3. [Batch driver: `run_all_negotiations.sh`](#batch-driver-run_all_negotiationssh)
4. [Simulations: `negotiation_single.py` & `negotiation_multi.py`](#simulations-negotiation_singlepy--negotiation_multipy)
5. [Automatic post-processing](#automatic-post-processing)
6. [Manual extraction & metrics](#manual-extraction--metrics)

---

## Quick start

From the `Agorabench` directory (after installing dependencies and setting):

```bash
# One product, single market (vanilla), with extract + metric JSON next to logs
python3 negotiation_single.py \
  --buyer gpt-4o --seller gpt-4o \
  --method react \
  --markets vanilla \
  --products Camera \
  --epoch 10 --rounds 10
```

Check:

`results/<method>/category_vanilla/Camera/`

- `*_*.txt` — raw dialogues  
- `{buyer}_results.json` — extracted `deal_price` / `dealt_item` (if post-process ran)  
- `buyer_metric_summary.json` — scalar metric + aggregates (if post-process ran)

Skip LLM extract + metric (dialogue only):

```bash
python3 negotiation_single.py ... --skip-postprocess
```

---

## Pipeline & folder layout

```text
datasets/category_*.json
        │
        ▼
negotiation_single.py  OR  negotiation_multi.py
        │
        ▼  …/<ours|og|react>/<category_dataset>/<Product>/*.txt
        │
        ├── (optional) cal_buyer_send_to_llm  →  {buyer}_results.json
        │
        └── (optional) cal_buyer_metric       →  buyer_metric_summary.json
```

| Step | Role |
|------|------|
| Negotiation | Writes `<buyer>_<seller>.txt` in the product folder (sessions appended). |
| Extract | LLM parses TXT → `{buyer}_results.json` (`deal_price:` lines; **several** also `dealt_item:`). |
| Metric | Reads JSON → `buyer_metric`, deal rate, averages → **`buyer_metric_summary.json`**. |

By default, **`negotiation_*` runs extract + metric after each product** unless **`--skip-postprocess`**.

---

## Batch driver: `run_all_negotiations.sh`

Single script to run **all** single-product markets, then **all** multi-segment markets, with the same buyer/seller/method/epoch/rounds.

```bash
chmod +x run_all_negotiations.sh    # once
./run_all_negotiations.sh --help
./run_all_negotiations.sh
./run_all_negotiations.sh gpt-4o gemini-1.5-pro ours 10 10
```

| Positional (optional) | Meaning | Default |
|----------------------|---------|---------|
| 1 | Buyer | `gpt-4o` |
| 2 | Seller | `gemini-1.5-pro` |
| 3 | Method `ours\|og\|react` | `ours` |
| 4 | Epoch | `10` |
| 5 | Rounds | `10` |

Same values can be set via env: **`BUYER`**, **`SELLER`**, **`METHOD`**, **`EPOCH`**, **`ROUNDS`**.

---

## Simulations: `negotiation_single.py` & `negotiation_multi.py`

Shared implementation: **`negotiation_core.py`** (GPT / Gemini / DeepSeek / Claude / OpenAI Responses sellers, etc.).

### Methods (`--method`)

| Value | Buyer-side |
|-------|------------|
| **`react`** | Dataset `buyer.system_prompt` only. |
| **`og`** | OG narrator: scripted offer price per turn (`PRICE_DATA` budget by product). |
| **`ours`** | Adds private reward definition (single: constant AR; multi: AR via item similarity). |

Logs (same folder holds TXT + optional JSON):

`<results-root>/<method>/<category_dataset>/<Product>/`

### `negotiation_single.py`

Markets map to `datasets/category_*.json`: **`vanilla`**, **`negative`**, **`monopoly`**, **`installment`**, **`deceptive`** — or **`all`**.

```bash
python3 negotiation_single.py --buyer gpt-4o --seller gemini-1.5-pro \
  --method ours --markets vanilla --results-root ./results
```

### `negotiation_multi.py`

Markets: **`several`**, **`several_negative`**, **`several_monopoly`**, **`several_installment`** — or **`all`**.

```bash
python3 negotiation_multi.py --buyer gpt-4o --seller deepseek-chat \
  --method react --markets several_installment
```

### Useful flags

| Flag | Notes |
|------|--------|
| **`--markets`** | One or more market keys, or `all`. |
| **`--products`** | Restrict to product names (e.g. `Camera Jacket`). |
| **`--epoch`**, **`--rounds`** | Sessions per scenario and max turns per session. |
| **`--datasets-dir`**, **`--results-root`** | Paths (env **`AGORA_RESULTS_ROOT`** aliases default results root). |
| **`--seller-reasoning`** | OpenAI seller only (`high` / `medium` / `low`). |
| **`--skip-postprocess`** | Dialogue only; no `{buyer}_results.json` / `buyer_metric_summary.json`. |
| **`--extract-api-model`** | Override extraction model (default: **same id as `--seller`**). |

---

## Automatic post-processing

After each **product** finishes (unless **`--skip-postprocess`**):

1. **`extract_from_merged`** (same behavior as **`cal_buyer_send_to_llm.py`**) → **`{buyer}_results.json`**
2. **`compute_single` / `compute_multi`** (same as **`cal_buyer_metric.py`**) → **`buyer_metric_summary.json`**

### Extraction model

| Case | Model used for extraction |
|------|---------------------------|
| No **`--extract-api-model`** | **`--seller`** string (same as negotiation seller). |
| **`--extract-api-model MODEL`** | **`MODEL`**. |

Extraction uses the **OpenAI Responses API**. If **`--seller`** is Gemini / Claude / DeepSeek, pass **`--extract-api-model gpt-4o-mini`** (or another OpenAI id) for extraction only.

The batch script **`run_all_negotiations.sh`** disables post-processing unless **`RUN_POSTPROCESS=1`** to avoid huge API usage on the full grid.

### Metric JSON from CLI

```bash
python3 cal_buyer_metric.py --mode single \
  --merged-dir .../category_vanilla/Camera --buyer-model gpt-4o --product Camera \
  --output-json ./my_summary.json
```

---

## Manual extraction & metrics

Use when you already have **`*.txt`** in each product folder and skipped automatic post-processing.

### `cal_buyer_send_to_llm.py`

- Modes: **`single`** (`deal_price` list) or **`several`** (`deal_price` + `dealt_item`).
- **`--merged-dir`** points at the **product folder** (same layout as negotiation output). The flag name is legacy; it is not required to be named `merged`.
- TXT files must match **`{buyer}_{seller}.txt`** (buyer prefix = **`--buyer-model`**).
- **`--api-model`**: OpenAI Responses model (default **`gpt-4o`**). Align with your seller id when possible.
- **Several mode:** if you still have an old path ending in **`…/Product/merged`**, category detection still works.

```bash
python3 cal_buyer_send_to_llm.py --mode single \
  --merged-dir .../category_vanilla/Jacket \
  --buyer-model gpt-4o \
  --api-model gpt-4o
```

Batch mode: **`--batch`**, **`--results-base`**, **`--method`**, … — see **`--help`**.

### `cal_buyer_metric.py`

- **`--mode single`**: needs **`--product`** (built-in row) or **`--budget` / `--initial` / `--cost`**.
- **`--mode multi`**: uses per-item tables inside the script.

Metrics use:

- **Single:** \(1.0139\,u + 0.8812\,n + 1.1049\)
- **Multi:** same + \(1.1049 \times\) item similarity from built-in **`ITEM_SIM`**.

```bash
python3 cal_buyer_metric.py --mode single \
  --merged-dir .../category_vanilla/Jacket --buyer-model gpt-4o --product Jacket
```

---

### `datasets/human_preference_dataset_sft.jsonl`

This JSONL file is intended for **buyer-side supervised fine-tuning (SFT)**—pair it with your SFT pipeline when training the negotiation buyer model.

Happy experimenting!