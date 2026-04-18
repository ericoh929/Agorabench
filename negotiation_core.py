"""
Shared GPT/Gemini negotiation primitives for Agorabench (single-product & multi-product runners).

Methods: react (scenario JSON only) | og (OG narrator price schedule) | ours (reward text only, no OAR).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import google.generativeai as genai
from openai import OpenAI

warnings.filterwarnings("ignore")

try:
    import anthropic  # Claude seller/buyer (optional dependency)
except ImportError:
    anthropic = None

try:
    from google import genai as google_genai_mod  # Gemini 2+ API (optional dependency)
    from google.genai import types as google_genai_types
except ImportError:
    google_genai_mod = None  # type: ignore
    google_genai_types = None  # type: ignore

# DeepSeek OpenAI-compatible base URL (same family as rebuttal scripts)
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_deepseek_client: Optional[OpenAI] = None
_claude_client: Optional[Any] = None
_google_genai_client: Optional[Any] = None


def is_google_genai(model_name: str) -> bool:
    """Use google.genai client for Gemini 2+ (same rule as rebuttal scripts)."""
    n = model_name.lower()
    if "gemini" not in n:
        return False
    if "gemini-1.0" in n or "gemini-1.5" in n:
        return False
    return True


def is_legacy_gemini(model_name: str) -> bool:
    """Legacy google.generativeai SDK (gemini-1.0 / 1.5, etc.)."""
    return "gemini" in model_name.lower() and not is_google_genai(model_name)


def is_deepseek(model_name: str) -> bool:
    return "deepseek" in model_name.lower()


def is_claude(model_name: str) -> bool:
    return "claude" in model_name.lower()


def normalize_deepseek_model(_name: str) -> str:
    """Normalize to the DeepSeek API model id."""
    return "deepseek-chat"


def ensure_negotiation_credentials(buyer_model: str, seller_model: str) -> None:
    """Check required keys/packages before run (avoid unnecessary configure calls)."""
    b, s = buyer_model.lower(), seller_model.lower()
    if is_legacy_gemini(b) or is_legacy_gemini(s):
        configure_gemini_from_env()
    if is_google_genai(b) or is_google_genai(s):
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise RuntimeError(
                "Gemini 2+ (google.genai) requires GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        if google_genai_mod is None or google_genai_types is None:
            raise RuntimeError(
                "Install google-genai for Gemini 2+: pip install google-genai"
            )
    if is_deepseek(b) or is_deepseek(s):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("DeepSeek models require DEEPSEEK_API_KEY.")
    if is_claude(b) or is_claude(s):
        if anthropic is None:
            raise RuntimeError("Claude models require the anthropic package: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Claude models require ANTHROPIC_API_KEY.")

Method = Literal["ours", "og", "react"]

# Shared product keys -> OG narrator budgets (aligned with gpt_vs_gemini_vanilla_og.py)
PRICE_DATA: Dict[str, Dict[str, float]] = {
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


def configure_gemini_from_env() -> None:
    """Use GEMINI_API_KEY or GOOGLE_API_KEY."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini models."
        )
    genai.configure(api_key=key)


def clean_message(message_content: str) -> str:
    if "Thought:" in message_content:
        parts = message_content.split("Talk:")
        return "Talk:" + parts[1] if len(parts) > 1 else parts[0]
    return message_content.strip()


def create_gemini_model(model_name: str, system_instruction: Optional[str] = None):
    cfg: Dict[str, Any] = {
        "model_name": model_name,
        "generation_config": {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
    }
    if system_instruction and model_name != "gemini-1.0-pro":
        cfg["system_instruction"] = system_instruction
    return genai.GenerativeModel(**cfg)


def call_gemini_api(chat_session: Any, message: str, is_self: bool) -> str:
    try:
        response = chat_session.send_message(message)
        response_text = response.text.strip()

        if is_self:
            chat_session.history.append({"role": "user", "parts": [message]})
            chat_session.history.append({"role": "model", "parts": [response_text]})
        else:
            chat_session.history.append(
                {"role": "user", "parts": [clean_message(message)]}
            )
            chat_session.history.append(
                {"role": "model", "parts": [clean_message(response_text)]}
            )

        return response_text
    except genai.types.generation_types.StopCandidateException as exc:
        print(f"Caught StopCandidateException: {exc}")
        return call_gemini_api(chat_session, message, is_self)


def call_gpt_api(
    client: OpenAI, model: str, conversation: List[Dict[str, str]], is_self: bool
) -> str:
    response = client.chat.completions.create(model=model, messages=conversation)
    response_text = (response.choices[0].message.content or "").strip()

    if is_self:
        conversation.append({"role": "assistant", "content": response_text})
    else:
        conversation.append(
            {"role": "assistant", "content": clean_message(response_text)}
        )

    return response_text


def call_openai_responses_api(
    client: OpenAI,
    model: str,
    conversation: List[Dict[str, str]],
    is_self: bool,
    reasoning: Optional[str] = None,
) -> str:
    """OpenAI Responses API (gpt-4o / gpt-5.x); same path as default rebuttal seller."""
    system_text: Optional[str] = None
    input_msgs: List[Dict[str, str]] = []
    for msg in conversation:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            input_msgs.append(msg)
    kwargs: Dict[str, Any] = dict(
        model=model,
        instructions=system_text,
        input=input_msgs,
    )
    if reasoning:
        kwargs["reasoning"] = {"effort": reasoning}
    resp = client.responses.create(**kwargs)
    text = (resp.output_text or "").strip()
    if is_self:
        conversation.append({"role": "assistant", "content": text})
    else:
        conversation.append(
            {"role": "assistant", "content": clean_message(text)}
        )
    return text


def ensure_deepseek_client() -> OpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        _deepseek_client = OpenAI(api_key=key, base_url=_DEEPSEEK_BASE_URL)
    return _deepseek_client


def ensure_claude_client() -> Any:
    global _claude_client
    if _claude_client is None:
        if anthropic is None:
            raise RuntimeError("Install anthropic for Claude.")
        _claude_client = anthropic.Anthropic()
    return _claude_client


def ensure_google_genai_client() -> Any:
    global _google_genai_client
    if _google_genai_client is None:
        if google_genai_mod is None:
            raise RuntimeError("Install google-genai for Gemini 2+.")
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required.")
        _google_genai_client = google_genai_mod.Client(api_key=api_key)
    return _google_genai_client


def call_deepseek_api(
    model: str, conversation: List[Dict[str, str]], is_self: bool
) -> str:
    client = ensure_deepseek_client()
    resp = client.chat.completions.create(
        model=normalize_deepseek_model(model),
        messages=conversation,
        stream=False,
    )
    text = (resp.choices[0].message.content or "").strip()
    if is_self:
        conversation.append({"role": "assistant", "content": text})
    else:
        conversation.append(
            {"role": "assistant", "content": clean_message(text)}
        )
    return text


def call_claude_api(
    model: str, conversation: List[Dict[str, str]], is_self: bool
) -> str:
    client = ensure_claude_client()
    system_text = ""
    chat_msgs: List[Dict[str, str]] = []
    for msg in conversation:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            chat_msgs.append(msg)
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_text,
        messages=chat_msgs,
    )
    text = resp.content[0].text.strip()
    if is_self:
        conversation.append({"role": "assistant", "content": text})
    else:
        conversation.append(
            {"role": "assistant", "content": clean_message(text)}
        )
    return text


def call_google_genai_api(
    model: str, conversation: List[Dict[str, str]], is_self: bool
) -> str:
    """google.genai path for Gemini 2+ (same pattern as vanilla_rebuttal.py)."""
    assert google_genai_types is not None
    client = ensure_google_genai_client()
    gtypes = google_genai_types
    system_text: Optional[str] = None
    chat_msgs: List[Any] = []
    for msg in conversation:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            chat_msgs.append(
                gtypes.Content(
                    role="user", parts=[gtypes.Part(text=msg["content"])]
                )
            )
        elif msg["role"] == "assistant":
            chat_msgs.append(
                gtypes.Content(
                    role="model", parts=[gtypes.Part(text=msg["content"])]
                )
            )
    resp = client.models.generate_content(
        model=model,
        contents=chat_msgs,
        config=gtypes.GenerateContentConfig(
            system_instruction=system_text,
            max_output_tokens=8192,
            thinking_config=gtypes.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    text = (resp.text or "").strip()
    if is_self:
        conversation.append({"role": "assistant", "content": text})
    else:
        conversation.append(
            {"role": "assistant", "content": clean_message(text)}
        )
    return text


def interpolate_price(budget: float, max_turn: int, current_turn: int) -> float:
    """Linear ramp from 50% to 100% of budget over turns (same as vanilla_og)."""
    factor = 0.5 + 0.5 * current_turn / max_turn
    return round(budget * factor, 2)


def og_offer_instruction(suggest_price: float) -> str:
    """OG narrator: fixed offered price this turn (same text as gpt_vs_gemini_vanilla_og.py)."""
    return f"""Instructions for Offering Price:

When presenting the price to the seller, please follow these guidelines strictly:

The Specific Offer:

You are to offer exactly ${suggest_price}.
Do not offer any other amount, suggest a price range, or alter this price in any way.
No Other Prices:

Do not mention, discuss, or imply any other prices.
Your Negotiation Stance:

From your side, this price is firm. Do not initiate any negotiation regarding this price.
Responding to the Seller about Price:

If the seller questions your offer or suggests a different price, your response should be to state that ${suggest_price} is the price you are presenting.
If you have other specific instructions for handling seller responses or objections, follow those.
Importantly, do not independently agree to a different price or propose a new one.
Priority of these Instructions:

These instructions for offering ${suggest_price} are specific to this task. They take precedence over any general guidance you might have regarding price flexibility or negotiation strategy."""


# --- Buyer system prompt glue (no OAR) ---

_REWARD_SINGLE = """

Use the following private reward function to guide your negotiation strategy:

  reward = 1.0139 * CS + 0.8812 * NP + 1.1049 * AR
  where:
    - CS = (P_budget - P_deal) / (P_budget - P_cost)
    - NP = (P_initial - P_deal) / (P_initial - P_cost)
    - AR = 1 (constant)

If P_cost values are not given, make reasonable assumptions for them in your private reasoning. Your goal is to maximize “reward” by estimating P_cost as low as realistically possible and adjusting P_cost accordingly to be as low as possible while keeping the negotiation active and avoiding a failed deal. (Maximum reward is 3.)

Please follow these guidelines:

- **Private Reasoning**  
  - Think through multiple negotiation paths (in your hidden thought process).
  - Do not expose your exact calculations or chain-of-thought to the other party.
  - Continuously estimate CS and NP based on your assumptions.
"""

_REWARD_MULTI = """

Use the following private reward function to guide your negotiation strategy:

  reward = 1.0139 * CS + 0.8812 * NP + 1.1049 * AR
  where:
    - CS = (P_budget - P_deal) / (P_budget - P_cost)
    - NP = (P_initial - P_deal) / (P_initial - P_cost)
    - AR = How semantically similar the buyer's acquired items are to their desired items. (range [0, 1])

If P_cost values are not given, make reasonable assumptions for them in your private reasoning. Your goal is to maximize “reward” by estimating P_cost as low as realistically possible and adjusting P_cost accordingly to be as low as possible while keeping the negotiation active and avoiding a failed deal. (Maximum reward is 3.) Also, ensure that there is a high “cosine similarity” between the item you originally desired and the item you ultimately purchase. In other words, the closer the final purchased item is to your initial desired item, the higher the value for AR. Purchasing exactly the item you wanted from the start yields the maximum AR.

Please follow these guidelines:

- **Private Reasoning**  
  - Think through multiple negotiation paths (in your hidden thought process).
  - Do not expose your exact calculations or chain-of-thought to the other party.
  - Continuously estimate CS and NP based on your assumptions.
"""


def buyer_system_prompt(base_buyer: str, method: str, *, multi: bool) -> str:
    """
    react: scenario JSON only.
    og: scenario only (per-turn price via og_offer_instruction).
    ours: scenario + reward definition (no OAR).
    """
    m = method.lower().strip()
    if m == "react":
        return base_buyer
    if m == "og":
        return base_buyer
    if m == "ours":
        return base_buyer + (_REWARD_MULTI if multi else _REWARD_SINGLE)
    raise ValueError(f"Unknown method: {method}. Use ours, og, react.")


def negotiation_output_dir(
    results_root: Path | str,
    method: str,
    dataset_folder: str,
    product: str,
) -> Path:
    """results_root/<ours|og|react>/<dataset>/<product>/ — logs and JSON land here."""
    root = Path(results_root)
    m = method.lower().strip()
    if m not in ("ours", "og", "react"):
        raise ValueError(f"Unknown method: {method}")
    return root / m / dataset_folder / product


def negotiate_single_product(
    *,
    buyer_model: str,
    seller_model: str,
    epoch: int,
    rounds: int,
    buyer_prompt: str,
    seller_prompt: str,
    product: str,
    output_path: str,
    openai_client: OpenAI,
    method: str,
    seller_reasoning: Optional[str] = None,
) -> None:
    """react | ours: plain loop / og: compose per-turn offered-price schedule."""

    def mkdir(d: str) -> None:
        os.makedirs(d, exist_ok=True)

    m = method.lower().strip()
    mkdir(output_path)
    log_file = os.path.join(output_path, f"{buyer_model}_{seller_model}.txt")
    sm = seller_model.lower()

    # OG narrator needs per-category budget; react/ours need only scenario JSON
    budget: Optional[float] = None
    if m == "og":
        if product not in PRICE_DATA:
            raise KeyError(
                f"OG narrator: unknown product '{product}'. "
                f"Add to PRICE_DATA: {list(PRICE_DATA)}"
            )
        budget = float(PRICE_DATA[product]["P_budget"])

    with open(log_file, "a", encoding="utf-8") as f:
        for session in range(1, epoch + 1):
            initial_message = f"Hi, I wanna buy a nice {product}."

            if "gemini" in buyer_model:
                buyer_agent = create_gemini_model(buyer_model, buyer_prompt)
                buyer_history = (
                    [{"role": "model", "parts": [f"{buyer_prompt}\n\n{initial_message}"]}]
                    if buyer_model == "gemini-1.0-pro"
                    else [{"role": "model", "parts": [initial_message]}]
                )
                buyer_session = buyer_agent.start_chat(history=buyer_history)
            else:
                buyer_conversation = [
                    {"role": "system", "content": buyer_prompt},
                    {"role": "user", "content": initial_message},
                ]

            seller_session: Optional[Any] = None
            seller_conversation: Optional[List[Dict[str, str]]] = None

            if is_google_genai(sm):
                seller_conversation = [{"role": "system", "content": seller_prompt}]
            elif is_legacy_gemini(sm):
                seller_agent = create_gemini_model(seller_model, seller_prompt)
                seller_history = (
                    [{"role": "model", "parts": [seller_prompt]}]
                    if seller_model == "gemini-1.0-pro"
                    else [{"role": "model", "parts": [" "]}]
                )
                seller_session = seller_agent.start_chat(history=seller_history)
            else:
                seller_conversation = [{"role": "system", "content": seller_prompt}]

            print(f"########## Start of Session {session} ##########", file=f)
            print("Buyer:", initial_message, file=f)

            if is_google_genai(sm):
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_google_genai_api(
                    seller_model, seller_conversation, is_self=False
                )
            elif is_legacy_gemini(sm):
                assert seller_session is not None
                seller_response = call_gemini_api(
                    seller_session, initial_message, is_self=False
                )
            elif is_deepseek(sm):
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_deepseek_api(
                    seller_model, seller_conversation, is_self=False
                )
            elif is_claude(sm):
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_claude_api(
                    seller_model, seller_conversation, is_self=False
                )
            else:
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_openai_responses_api(
                    openai_client,
                    seller_model,
                    seller_conversation,
                    is_self=False,
                    reasoning=seller_reasoning,
                )

            print("Seller:", seller_response, file=f)

            for current_step in range(rounds):
                cleaned_seller_msg = clean_message(seller_response)

                if m == "og":
                    assert budget is not None
                    suggest_price = interpolate_price(
                        budget, max_turn=rounds, current_turn=current_step
                    )
                    add_on = og_offer_instruction(suggest_price)
                    system_prom = buyer_prompt + add_on
                    if "gemini" in buyer_model:
                        updated_agent = create_gemini_model(buyer_model, system_prom)
                        buyer_session = updated_agent.start_chat(
                            history=buyer_session.history
                        )
                        buyer_response = call_gemini_api(
                            buyer_session, cleaned_seller_msg, is_self=True
                        )
                    else:
                        buyer_conversation[0]["content"] = system_prom
                        buyer_conversation.append(
                            {"role": "user", "content": cleaned_seller_msg}
                        )
                        buyer_response = call_gpt_api(
                            openai_client,
                            buyer_model,
                            buyer_conversation,
                            is_self=True,
                        )
                else:
                    if "gemini" in buyer_model:
                        buyer_response = call_gemini_api(
                            buyer_session, cleaned_seller_msg, is_self=True
                        )
                    else:
                        buyer_conversation.append(
                            {"role": "user", "content": cleaned_seller_msg}
                        )
                        buyer_response = call_gpt_api(
                            openai_client,
                            buyer_model,
                            buyer_conversation,
                            is_self=True,
                        )

                print("Buyer:", buyer_response, file=f)
                cleaned_buyer_msg = clean_message(buyer_response)

                if is_google_genai(sm):
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_google_genai_api(
                        seller_model, seller_conversation, is_self=True
                    )
                elif is_legacy_gemini(sm):
                    assert seller_session is not None
                    seller_response = call_gemini_api(
                        seller_session, cleaned_buyer_msg, is_self=True
                    )
                elif is_deepseek(sm):
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_deepseek_api(
                        seller_model, seller_conversation, is_self=True
                    )
                elif is_claude(sm):
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_claude_api(
                        seller_model, seller_conversation, is_self=True
                    )
                else:
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_openai_responses_api(
                        openai_client,
                        seller_model,
                        seller_conversation,
                        is_self=True,
                        reasoning=seller_reasoning,
                    )

                print("Seller:", seller_response, file=f)

                if (
                    "[DEAL]" in buyer_response
                    or "[DEAL]" in seller_response
                    or "[QUIT]" in buyer_response
                    or "[QUIT]" in seller_response
                ):
                    break

            print("########## End of Session ##########", file=f)


def negotiate_multi_product(
    *,
    buyer_model: str,
    seller_model: str,
    epoch: int,
    rounds: int,
    buyer_prompt: str,
    seller_prompt: str,
    buyer_inventory: str,
    seller_inventory: str,
    product: str,
    output_path: str,
    openai_client: OpenAI,
    method: str,
    seller_reasoning: Optional[str] = None,
) -> None:
    """Multi inventory; OG looks up PRICE_DATA budget by category name in ``product``."""

    def mkdir(d: str) -> None:
        os.makedirs(d, exist_ok=True)

    m = method.lower().strip()
    mkdir(output_path)
    log_file = os.path.join(output_path, f"{buyer_model}_{seller_model}.txt")
    sm = seller_model.lower()

    budget: Optional[float] = None
    if m == "og":
        if product not in PRICE_DATA:
            raise KeyError(
                f"OG narrator: unknown product '{product}'. "
                f"Add to PRICE_DATA: {list(PRICE_DATA)}"
            )
        budget = float(PRICE_DATA[product]["P_budget"])

    buyer_full = f"{buyer_prompt}\n\n### Seller's Inventory List\n{buyer_inventory}"
    seller_full = f"{seller_prompt}\n\n### Inventory List\n{seller_inventory}"

    with open(log_file, "a", encoding="utf-8") as f:
        for session in range(1, epoch + 1):
            initial_message = f"Hi, I wanna buy a nice {product}."

            if "gemini" in buyer_model:
                buyer_agent = create_gemini_model(buyer_model, buyer_full)
                buyer_session = buyer_agent.start_chat(
                    history=[{"role": "model", "parts": [initial_message]}]
                )
            else:
                buyer_conversation = [
                    {"role": "system", "content": buyer_full},
                    {"role": "user", "content": initial_message},
                ]

            seller_session: Optional[Any] = None
            seller_conversation: Optional[List[Dict[str, str]]] = None

            if is_google_genai(sm):
                seller_conversation = [{"role": "system", "content": seller_full}]
            elif is_legacy_gemini(sm):
                seller_agent = create_gemini_model(seller_model, seller_full)
                seller_session = seller_agent.start_chat(
                    history=[{"role": "model", "parts": [" "]}]
                )
            else:
                seller_conversation = [{"role": "system", "content": seller_full}]

            print(f"########## Start of Session {session} ##########", file=f)
            print("Buyer:", initial_message, file=f)

            if is_google_genai(sm):
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_google_genai_api(
                    seller_model, seller_conversation, is_self=False
                )
            elif is_legacy_gemini(sm):
                assert seller_session is not None
                seller_response = call_gemini_api(
                    seller_session, initial_message, is_self=False
                )
            elif is_deepseek(sm):
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_deepseek_api(
                    seller_model, seller_conversation, is_self=False
                )
            elif is_claude(sm):
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_claude_api(
                    seller_model, seller_conversation, is_self=False
                )
            else:
                assert seller_conversation is not None
                seller_conversation.append(
                    {"role": "user", "content": initial_message}
                )
                seller_response = call_openai_responses_api(
                    openai_client,
                    seller_model,
                    seller_conversation,
                    is_self=False,
                    reasoning=seller_reasoning,
                )

            print("Seller:", seller_response, file=f)

            for current_step in range(rounds):
                cleaned_seller_msg = clean_message(seller_response)

                if m == "og":
                    assert budget is not None
                    suggest_price = interpolate_price(
                        budget, max_turn=rounds, current_turn=current_step
                    )
                    add_on = og_offer_instruction(suggest_price)
                    system_prom = buyer_full + add_on
                    if "gemini" in buyer_model:
                        updated_agent = create_gemini_model(buyer_model, system_prom)
                        buyer_session = updated_agent.start_chat(
                            history=buyer_session.history
                        )
                        buyer_response = call_gemini_api(
                            buyer_session, cleaned_seller_msg, is_self=True
                        )
                    else:
                        buyer_conversation[0]["content"] = system_prom
                        buyer_conversation.append(
                            {"role": "user", "content": cleaned_seller_msg}
                        )
                        buyer_response = call_gpt_api(
                            openai_client,
                            buyer_model,
                            buyer_conversation,
                            is_self=True,
                        )
                else:
                    if "gemini" in buyer_model:
                        buyer_response = call_gemini_api(
                            buyer_session, cleaned_seller_msg, is_self=True
                        )
                    else:
                        buyer_conversation.append(
                            {"role": "user", "content": cleaned_seller_msg}
                        )
                        buyer_response = call_gpt_api(
                            openai_client,
                            buyer_model,
                            buyer_conversation,
                            is_self=True,
                        )

                print("Buyer:", buyer_response, file=f)
                cleaned_buyer_msg = clean_message(buyer_response)

                if is_google_genai(sm):
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_google_genai_api(
                        seller_model, seller_conversation, is_self=True
                    )
                elif is_legacy_gemini(sm):
                    assert seller_session is not None
                    seller_response = call_gemini_api(
                        seller_session, cleaned_buyer_msg, is_self=True
                    )
                elif is_deepseek(sm):
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_deepseek_api(
                        seller_model, seller_conversation, is_self=True
                    )
                elif is_claude(sm):
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_claude_api(
                        seller_model, seller_conversation, is_self=True
                    )
                else:
                    assert seller_conversation is not None
                    seller_conversation.append(
                        {"role": "user", "content": cleaned_buyer_msg}
                    )
                    seller_response = call_openai_responses_api(
                        openai_client,
                        seller_model,
                        seller_conversation,
                        is_self=True,
                        reasoning=seller_reasoning,
                    )

                print("Seller:", seller_response, file=f)

                if (
                    "[DEAL]" in buyer_response
                    or "[DEAL]" in seller_response
                    or "[QUIT]" in buyer_response
                    or "[QUIT]" in seller_response
                ):
                    break

            print("########## End of Session ##########", file=f)
