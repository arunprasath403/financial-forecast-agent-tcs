# app/agent_orchestrator.py
"""
Auto-trimming orchestrator with transcript summarization (OpenRouter).
Replaces the previous agent_orchestrator. Run from project root:
    $env:OPENROUTER_API_KEY="sk-xxx"
    python -m app.agent_orchestrator
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

# your tool wrappers (must exist)
from app.cache_manager import CacheManager

# CONFIG
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
SUMMARIZE_MODEL = "openai/gpt-oss-20b:free"   # short-summary calls (you can change)
FORECAST_MODEL = "openai/gpt-oss-20b:free"    # final forecast call
# approximate token limit for model input (from error earlier set safely)
MAX_INPUT_TOKENS = 120000
# crude token estimate: characters / 4 (approx)
def estimate_tokens(s: str) -> int:
    return max(1, int(len(s) / 4))


# Developer requested the local file path be passed in payload (we include it)
TOOL_SOURCE_FILE = r"D:\Forecast_agent_TCS\app\tools\financial_extractor.py"

# ---------------------------
# Helpers: OpenRouter calls
# ---------------------------

def openrouter_client():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)


def summarize_text(client: OpenAI, text: str, max_words: int = 180) -> str:
    """
    Summarize a long text into <= max_words (approx).
    Returns a short summary (string). Uses a small prompt to reduce tokens.
    """
    prompt = (
        f"You are a concise summarizer. Produce a short summary (<= {max_words} words) "
        "of the following transcript. Focus on forward-looking statements, risks, "
        "opportunities, and any explicit management guidance. Return plain text only.\n\n"
        "Transcript:\n\n"
        + text
    )
    resp = client.chat.completions.create(
        model=SUMMARIZE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800  # allow enough tokens for the summary
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return text[:1000]  # fallback extremely short excerpt


def summarize_each_transcript(client: OpenAI, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    transcripts: list of dicts each with keys like 'path', 'text', 'snippets' or whatever qualitative tool returns.
    Returns processed list with 'summary' and 'top_snippets'
    """
    out = []
    for t in transcripts:
        # try to find a short precomputed summary from your qualitative tool output
        if isinstance(t, dict) and t.get("summary"):
            summary = t["summary"]
        else:
            # get raw_text or text fields
            raw = t.get("full_text") or t.get("text") or t.get("raw_text") or ""
            if not raw:
                summary = t.get("short", "") or ""
            else:
                # limit raw to a chunk size to avoid spending too many tokens summarizing
                CHUNK_LIMIT = 30_000  # characters per transcript chunk to summarize
                snippet = raw if len(raw) <= CHUNK_LIMIT else raw[:CHUNK_LIMIT]
                try:
                    summary = summarize_text(client, snippet, max_words=160)
                except Exception:
                    # fallback extremely short
                    summary = snippet[:800]
        # get top snippets if available
        top_snippets = []
        if isinstance(t, dict) and t.get("topic_hits"):
            # flatten first two hits per topic
            for k, hits in t["topic_hits"].items():
                for h in hits[:2]:
                    if isinstance(h, dict) and h.get("snippet"):
                        top_snippets.append(h["snippet"])
                    else:
                        top_snippets.append(str(h)[:400])
        # also check generic 'snippets' key
        if not top_snippets and isinstance(t, dict) and t.get("snippets"):
            top_snippets = [s for s in t["snippets"][:5]]
        out.append({"source": t.get("source") or t.get("path") or "<unknown>", "summary": summary, "top_snippets": top_snippets})
    return out


def shrink_payload_quick(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick aggressive trimming without extra LLM calls: fallback method
    Keeps numeric metrics + small evidence slices.
    """
    f = payload.get("financial", {})
    if "raw_evidence" in f and isinstance(f["raw_evidence"], list):
        f["raw_evidence"] = f["raw_evidence"][:5]
    payload["financial"] = f

    q = payload.get("qualitative", {})
    # remove heavy keys
    for big in ["full_text", "raw_text", "all_chunks"]:
        if big in q:
            q[big] = "<trimmed>"
    # shorten lists
    if "topic_hits" in q and isinstance(q["topic_hits"], dict):
        for k in list(q["topic_hits"].keys()):
            q["topic_hits"][k] = q["topic_hits"][k][:2]
    payload["qualitative"] = q
    return payload


# ----------------------------
# Main pipeline
# ----------------------------
def run_langchain_agent():
    client = openrouter_client()

    # 1) Run your two tools
    from app.cache_manager import CacheManager

    fin = CacheManager.get_financial()
    qual = CacheManager.get_qualitative()


    # build base payload
    payload = {
        "financial": fin,
        "qualitative": qual,
        "tool_source_file": TOOL_SOURCE_FILE,
        "generated_at": datetime.utcnow().isoformat()
    }

    # 2) Quick token estimate
    combined_text = json.dumps(payload, ensure_ascii=False)
    tokens = estimate_tokens(combined_text)

    if tokens > MAX_INPUT_TOKENS:
        # Prefer summarizing transcripts intelligently if qualitative tool provided transcripts list
        transcripts = []
        # many qualitative tools return a key like 'transcripts' or 'loaded_docs' or 'items'
        if isinstance(qual, dict):
            # try common keys
            for key in ["transcripts", "loaded_docs", "docs", "items"]:
                if key in qual and isinstance(qual[key], list):
                    transcripts = qual[key]
                    break
            # some tools store lists under 'selected' or 'topic_hits' (topic_hits contains hits per topic)
            if not transcripts and "topic_hits" in qual and isinstance(qual["topic_hits"], dict):
                # build minimal transcript entries from topic_hits if no transcripts list present
                transcripts = []
                for topic, hits in qual["topic_hits"].items():
                    # each hit may contain 'source' and 'snippet'; create a minimal transcript entry
                    transcripts.append({"source": hits[0].get("source") if isinstance(hits[0], dict) else "<unknown>", "full_text": " ".join([ (h.get("snippet") if isinstance(h, dict) else str(h))[:3000] for h in hits[:5] ])})
        # If we found transcripts, summarize each
        if transcripts:
            summarized = summarize_each_transcript(client, transcripts)
            # replace qualitative info with compact summary list + keep a tiny topic_hits sample
            payload["qualitative"] = {
                "summaries": summarized,
                "top_topics_sample": {k: v[:2] for k, v in (qual.get("topic_hits") or {}).items() if isinstance(v, list)}
            }
        else:
            # fallback: aggressive quick shrink in-memory
            payload = shrink_payload_quick(payload)

        # re-estimate tokens
        combined_text = json.dumps(payload, ensure_ascii=False)
        tokens = estimate_tokens(combined_text)

        # if still large, hard truncate the payload_text
        MAX_CHARS = MAX_INPUT_TOKENS * 4  # back to characters
        if len(combined_text) > MAX_CHARS:
            combined_text = combined_text[:MAX_CHARS]
            # try to keep payload parsable by wrapping in { "trimmed_payload": "<truncated>" }
            payload = {"trimmed_payload": combined_text, "generated_at": datetime.utcnow().isoformat()}

    # 3) Final call to OpenRouter to produce strict JSON forecast
    system_prompt = (
        "You are a financial forecast agent. Combine the extracted metrics and qualitative summaries "
        "and return ONLY valid JSON in this exact structure:\n\n"
        '{'
        '"summary": "", '
        '"financial_trends": {}, '
        '"management_outlook": "", '
        '"risks": [], '
        '"opportunities": [], '
        '"forecast": {"revenue_growth_next_quarter_pct": 0.0, "margin_trend": "", "confidence": ""}, '
        '"evidence": [], '
        '"generated_at": "ISO timestamp"'
        '}\n\nNO extra text. JSON only.'
    )

    user_prompt = "Here is the extracted data (already summarized if necessary):\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=FORECAST_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )

    raw = resp.choices[0].message.content
    try:
        final = json.loads(raw)
    except Exception as e:
        # save raw output for debugging and raise
        debug_file = Path("final_raw_output_debug.json")
        debug_file.write_text(json.dumps({"payload_sent": payload, "raw_output": raw}, indent=2, ensure_ascii=False))
        raise RuntimeError(f"LLM returned invalid JSON. Saved debug to {debug_file}. Error: {e}\n\nRaw:\n{raw}")

    return final


if __name__ == "__main__":
    result = run_langchain_agent()
    print(json.dumps(result, indent=2, ensure_ascii=False))
