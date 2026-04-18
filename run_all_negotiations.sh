#!/usr/bin/env bash
# Agorabench: run all single-product markets, then all multi-segment markets.
# Full usage, env vars, and examples:  ./run_all_negotiations.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
  cat <<'EOF'
Agorabench — run_all_negotiations.sh

Runs negotiation_single.py (--markets all), then negotiation_multi.py (--markets all).

Usage:
  ./run_all_negotiations.sh [BUYER] [SELLER] [METHOD] [EPOCH] [ROUNDS]
  ./run_all_negotiations.sh --help

Defaults: gpt-4o  gemini-1.5-pro  ours  10  10
          (overridable via BUYER, SELLER, METHOD, EPOCH, ROUNDS env)

Outputs:
  <results>/<method>/<category_dataset>/<Product>/*.txt  (+ JSONs if postprocess)

Environment:
  AGORA_RESULTS_ROOT    Results root (default: ./results)
  PYTHON                Interpreter (default: python3)
  SKIP_SINGLE=1         Skip single-product markets step
  SKIP_MULTI=1          Skip multi-segment markets step
  PRODUCTS="Camera ..." Limit to these product names (--products)
  RUN_POSTPROCESS=1     Run LLM extract + buyer_metric_summary.json per product folder
                        (default OFF — very many API calls if enabled on full grid)
  SELLER_REASONING=high Passed to negotiation_* (--seller-reasoning)

Examples:
  ./run_all_negotiations.sh gpt-4o gemini-1.5-pro ours 10 10
  PRODUCTS=Camera SKIP_MULTI=1 ./run_all_negotiations.sh gpt-4o gpt-4o react 10 10
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

PYTHON="${PYTHON:-python3}"

BUYER="${1:-${BUYER:-gpt-4o}}"
SELLER="${2:-${SELLER:-gemini-1.5-pro}}"
METHOD="${3:-${METHOD:-ours}}"
EPOCH="${4:-${EPOCH:-10}}"
ROUNDS="${5:-${ROUNDS:-10}}"

EXTRA_PY=()
if [[ -n "${SELLER_REASONING:-}" ]]; then
  EXTRA_PY+=(--seller-reasoning "$SELLER_REASONING")
fi

if [[ -n "${PRODUCTS:-}" ]]; then
  # shellcheck disable=SC2206
  PRODARR=(${PRODUCTS})
  EXTRA_PY+=(--products "${PRODARR[@]}")
fi

RESULTS_EXTRA=()
if [[ -n "${AGORA_RESULTS_ROOT:-}" ]]; then
  RESULTS_EXTRA+=(--results-root "$AGORA_RESULTS_ROOT")
fi

POST_EXTRA=()
if [[ -z "${RUN_POSTPROCESS:-}" ]]; then
  POST_EXTRA=(--skip-postprocess)
fi

echo "=========================================="
echo "Agorabench · run_all_negotiations.sh"
echo "Directory:  $SCRIPT_DIR"
echo "Buyer:      $BUYER"
echo "Seller:     $SELLER"
echo "Method:     $METHOD"
echo "Epoch:      $EPOCH   Rounds: $ROUNDS"
if [[ -n "${PRODUCTS:-}" ]]; then
  echo "Products:   $PRODUCTS"
fi
if [[ -n "${AGORA_RESULTS_ROOT:-}" ]]; then
  echo "Results:    $AGORA_RESULTS_ROOT"
else
  echo "Results:    $SCRIPT_DIR/results (default)"
fi
if [[ -n "${RUN_POSTPROCESS:-}" ]]; then
  echo "Postprocess: ON  (extract + buyer_metric_summary.json per product dir)"
else
  echo "Postprocess: OFF (set RUN_POSTPROCESS=1 to enable; see --help)"
fi
echo "=========================================="

run_single() {
  echo ""
  echo "[1/2] Single-product markets (vanilla, negative, monopoly, installment, deceptive)"
  "$PYTHON" negotiation_single.py \
    --buyer "$BUYER" \
    --seller "$SELLER" \
    --method "$METHOD" \
    --epoch "$EPOCH" \
    --rounds "$ROUNDS" \
    --markets all \
    "${EXTRA_PY[@]}" \
    "${RESULTS_EXTRA[@]}" \
    "${POST_EXTRA[@]}"
}

run_multi() {
  echo ""
  echo "[2/2] Multi-segment markets (several, several_*)"
  "$PYTHON" negotiation_multi.py \
    --buyer "$BUYER" \
    --seller "$SELLER" \
    --method "$METHOD" \
    --epoch "$EPOCH" \
    --rounds "$ROUNDS" \
    --markets all \
    "${EXTRA_PY[@]}" \
    "${RESULTS_EXTRA[@]}" \
    "${POST_EXTRA[@]}"
}

if [[ -z "${SKIP_SINGLE:-}" ]]; then
  run_single
else
  echo "[skip] SKIP_SINGLE is set — negotiation_single.py omitted"
fi

if [[ -z "${SKIP_MULTI:-}" ]]; then
  run_multi
else
  echo "[skip] SKIP_MULTI is set — negotiation_multi.py omitted"
fi

echo ""
echo "Done."
