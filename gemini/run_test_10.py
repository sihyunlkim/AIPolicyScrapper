from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from policy_search_gemini import search_policy_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unis",    type=str, default="test_10_unis.csv",
                    help="CSV with UNITID, INSTNM, WEBADDR (no IPEDS matching needed).")
    ap.add_argument("--outfile", type=str, default="policy_links_10_unis_gemini.csv")
    ap.add_argument("--num-results",        type=int,  default=100)
    ap.add_argument("--restrict-domain",    action="store_true", default=True)
    ap.add_argument("--no-restrict-domain", action="store_false", dest="restrict_domain")
    ap.add_argument("--sleep",   type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.unis)
    df["WEBADDR"] = df["WEBADDR"].fillna("").astype(str)

    print(f"[OK] Universities to process: {len(df)}")

    out_path = Path(args.outfile)
    fieldnames = [
        "university", "landing_page", "domain",
        "query_used", "title", "policy_url",
        "method", "domain_restricted",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in df.iterrows():
            instnm = str(row["INSTNM"]).strip()
            web    = str(row["WEBADDR"]).strip()

            print(f"\n[{i+1}/{len(df)}] {instnm} | {web}")

            try:
                hits = search_policy_pages(
                    university=instnm,
                    website=web,
                    num_results=args.num_results,
                    restrict_domain=args.restrict_domain,
                )
            except Exception as e:
                print(f"  [ERROR] {e}")
                hits = []

            for h in hits:
                writer.writerow({
                    "university":       h.get("university", instnm),
                    "landing_page":     web,
                    "domain":           h.get("domain", ""),
                    "query_used":       h.get("query", ""),
                    "title":            h.get("title", ""),
                    "policy_url":       h.get("url", ""),
                    "method":           "gemini",
                    "domain_restricted": h.get("domain_restricted", args.restrict_domain),
                })

            print(f"  [OK] {len(hits)} hits written.")
            time.sleep(args.sleep)

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
