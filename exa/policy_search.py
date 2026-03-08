import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

EXA_KEY = os.getenv("EXA_API_KEY")
assert EXA_KEY, "EXA_API_KEY not found"

exa = Exa(EXA_KEY)

POLICY_QUERIES = [
    "{uni} generative AI policy",
    "{uni} AI policy academic integrity",
    "{uni} ChatGPT policy students",
    "{uni} provost generative AI guidance",
    "{uni} teaching and learning generative AI guidance",
]

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

def search_policy_pages(university: str, website: str, num_results: int=100, restrict_domain: bool = True, search_type: str = "deep", ):
    """
    If restrict_domain=True, prepend `site:{domain}` to each query to keep results on the official domain.
    """
    domain = _extract_domain(website)
    site_prefix = f"site:{domain} " if (restrict_domain and domain) else ""

    hits = []
    for template in POLICY_QUERIES:
        q = site_prefix + template.format(uni=university)
        kwargs = {
            "query": q,
            #"contents": {"text": True},
            "type": search_type,      # ✅ search type 고정
        }

        if num_results is not None:
            kwargs["num_results"] = num_results

        res = exa.search(**kwargs)
        
        for r in res.results:
            hits.append({
                "query": q,
                "title": r.title,
                "url": r.url,
                "length": len(r.text or ""),
                "domain_restricted": bool(site_prefix),
                "university": university,
                "domain": domain,
            })

    # URL dedupe
    seen = set()
    deduped = []
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"])
            deduped.append(h)

    return deduped
