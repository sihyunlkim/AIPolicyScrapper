from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
print("Loaded key:", os.getenv("GEMINI_API_KEY"))

assert GEMINI_KEY, "GEMINI_API_KEY not found"

client = genai.Client(api_key=GEMINI_KEY)

POLICY_QUERIES = [
    "{uni} generative AI policy",
    "{uni} AI policy",
    "{uni} AI guidance",
    "{uni} AI academic integrity",
    "{uni} academic integrity",
]

URL_RE = re.compile(r"https?://[^\s\]\)\}\"\'<>,]+", re.IGNORECASE)


def _extract_domain(website: str) -> str:
    if not website or not isinstance(website, str):
        return ""
    w = website.strip()
    if not w:
        return ""
    if not w.startswith(("http://", "https://")):
        w = "https://" + w
    try:
        host = urlparse(w).netloc.lower()
        return host.replace("www.", "")
    except Exception:
        return ""


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_urls_from_obj(obj: Any) -> list[str]:
    urls: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            if x.startswith(("http://", "https://")):
                urls.append(x)
            else:
                urls.extend(URL_RE.findall(x))

    walk(obj)
    return _dedupe_keep_order(urls)


def should_drop_url(url: str) -> bool:
    """Drop empty URLs and obvious Google/Grounding intermediate links."""
    if not url or not isinstance(url, str):
        return True

    u = url.strip()
    if not u:
        return True

    try:
        p = urlparse(u)
        host = p.netloc.lower().replace("www.", "")
        path = p.path.lower()
    except Exception:
        return True

    # Gemini grounding redirect
    if host == "vertexaisearch.cloud.google.com":
        return True

    # Google redirect / search / cache
    if host in {
        "google.com",
        "www.google.com",
        "webcache.googleusercontent.com",
        "translate.google.com",
    }:
        return True

    if "grounding-api-redirect" in path:
        return True

    return False

def _is_on_domain(url: str, domain: str) -> bool:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host == domain or host.endswith("." + domain)
    except Exception:
        return False

def _extract_urls_from_response(resp: Any) -> list[str]:
    urls: list[str] = []

    # Try structured response first
    try:
        if hasattr(resp, "model_dump"):
            urls.extend(_extract_urls_from_obj(resp.model_dump()))
        elif hasattr(resp, "to_dict"):
            urls.extend(_extract_urls_from_obj(resp.to_dict()))
        elif hasattr(resp, "__dict__"):
            urls.extend(_extract_urls_from_obj(resp.__dict__))
    except Exception:
        pass

    # Fallback to plain text
    try:
        text = getattr(resp, "text", "") or ""
        urls.extend(URL_RE.findall(text))
    except Exception:
        pass

    return _dedupe_keep_order(urls)


def search_policy_pages(
    university: str,
    website: str,
    num_results: int = 100,
    restrict_domain: bool = True,
    search_type: str = "deep",  # kept only for interface compatibility
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """
    Gemini version of your Exa function.
    Returns a deduped list of hits with a shape similar to the Exa output.
    """
    domain = _extract_domain(website)
    hits: list[dict] = []

    # Gemini grounding is tool-based; there is no Exa-style search_type parameter.
    _ = search_type

    for template in POLICY_QUERIES:
        q = template.format(uni=university)

        domain_instruction = ""
        if restrict_domain and domain:
            domain_instruction = (
                f"Prefer official pages on the university domain '{domain}'. "
                f"If a result is outside that domain, only include it if it is clearly an official university page."
            )

        prompt = f"""
Use Google Search to find official university pages relevant to this intent.

University: {university}
Known official website: {website}
Intent: {q}

{domain_instruction}

Return only a JSON object with this exact schema:
{{
  "results": [
    {{
      "url": "https://...",
      "title": "short page title",
      "why": "very short reason"
    }}
  ]
}}

Requirements:
- Focus on official university policy/guidance pages.
- Good targets include academic integrity, generative AI guidance, ChatGPT guidance, student handbook, provost/teaching center guidance.
- Return as many relevant results as you find.
- No markdown.
""".strip()

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            ),
        )

        parsed_results: list[dict] = []

        # Best case: model obeys JSON instruction
        raw_text = (getattr(resp, "text", "") or "").strip()
        if raw_text:
            try:
                obj = json.loads(raw_text)
                if isinstance(obj, dict) and isinstance(obj.get("results"), list):
                    for item in obj["results"]:
                        if isinstance(item, dict) and isinstance(item.get("url"), str):
                            parsed_results.append(item)
            except Exception:
                pass

        # Fallback: scrape URLs from response metadata/text
        if not parsed_results:
            urls = _extract_urls_from_response(resp)
            parsed_results = [{"url": u, "title": "", "why": ""} for u in urls]

        for item in parsed_results:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            if should_drop_url(url):
                continue
            if restrict_domain and domain and not _is_on_domain(url, domain):
                continue

            title = str(item.get("title", "")).strip()
            hits.append(
                {
                    "query": q,
                    "title": title,
                    "url": url,
                    "length": 0,
                    "domain_restricted": bool(restrict_domain and domain),
                    "university": university,
                    "domain": domain,
                    "model": model,
                    "source": "gemini_google_search_grounding",
                }
            )

    # URL dedupe across all query templates
    seen = set()
    deduped = []
    for h in hits:
        u = h["url"]
        if u not in seen:
            seen.add(u)
            deduped.append(h)

    return deduped