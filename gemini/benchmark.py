"""
Gemini Search Grounding - Model Benchmark
Compares models on: speed, # of results, on-domain accuracy
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from policy_search_gemini import search_policy_pages

# ── Config ──────────────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]

TEST_CASES = [
    {"university": "NYU",      "website": "nyu.edu"},
    {"university": "MIT",      "website": "mit.edu"},
    {"university": "Stanford", "website": "stanford.edu"},
]

# ── Run ───────────────────────────────────────────────────────────────────────
results = []
total  = len(MODELS_TO_TEST) * len(TEST_CASES)
count  = 0

for model in MODELS_TO_TEST:
    for tc in TEST_CASES:
        count += 1
        print(f"[{count}/{total}] {model} × {tc['university']} ...", end=" ", flush=True)
        try:
            start = time.time()
            hits  = search_policy_pages(
                university=tc["university"],
                website=tc["website"],
                model=model,
            )
            elapsed = time.time() - start

            domain     = tc["website"].replace("www.", "")
            on_domain  = sum(1 for h in hits if domain in h.get("url", ""))
            total_urls = len(hits)

            print(f"{elapsed:.1f}s | {total_urls} URLs | {on_domain} on-domain")

            results.append({
                "model":      model,
                "university": tc["university"],
                "elapsed_s":  round(elapsed, 2),
                "total_urls": total_urls,
                "on_domain":  on_domain,
                "off_domain": total_urls - on_domain,
                "hits":       hits,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model":      model,
                "university": tc["university"],
                "elapsed_s":  0,
                "total_urls": 0,
                "on_domain":  0,
                "off_domain": 0,
                "hits":       [],
                "error":      str(e),
            })

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'MODEL':<35} {'AVG TIME':>9} {'AVG URLS':>9} {'ON-DOM%':>8}")
print("=" * 75)

grouped = defaultdict(list)
for r in results:
    grouped[r["model"]].append(r)

for model in MODELS_TO_TEST:
    rows = grouped[model]
    if not rows:
        continue
    avg_time   = sum(r["elapsed_s"]  for r in rows) / len(rows)
    avg_urls   = sum(r["total_urls"] for r in rows) / len(rows)
    on_dom_pct = (
        sum(r["on_domain"]  for r in rows) /
        max(sum(r["total_urls"] for r in rows), 1) * 100
    )
    print(f"{model:<35} {avg_time:>8.1f}s {avg_urls:>9.1f} {on_dom_pct:>7.0f}%")

print("=" * 75)

# ── Save full results ─────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nFull results saved to: {out_path}")