from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(path: Path, method_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_source"] = method_label
    df["policy_url"] = df["policy_url"].astype(str).str.strip().str.lower()
    df["university"] = df["university"].astype(str).str.strip()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exa",    type=str, default="exa/policy_links_10_unis_exa.csv")
    ap.add_argument("--gemini", type=str, default="gemini/policy_links_10_unis_gemini.csv")
    ap.add_argument("--outdir", type=str, default="comparison")
    args = ap.parse_args()

    exa_df    = load(Path(args.exa),    "exa")
    gemini_df = load(Path(args.gemini), "gemini")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Per-university URL counts
    # ------------------------------------------------------------------
    exa_counts    = exa_df.groupby("university")["policy_url"].nunique().rename("exa_urls")
    gemini_counts = gemini_df.groupby("university")["policy_url"].nunique().rename("gemini_urls")

    counts = pd.concat([exa_counts, gemini_counts], axis=1).fillna(0).astype(int)
    counts["diff"] = counts["gemini_urls"] - counts["exa_urls"]
    counts = counts.sort_values("diff", ascending=False)
    counts.to_csv(outdir / "url_counts_per_uni.csv")

    # ------------------------------------------------------------------
    # 2) Overlap: URLs found by BOTH vs only one
    # ------------------------------------------------------------------
    rows = []
    for uni in sorted(set(exa_df["university"]) | set(gemini_df["university"])):
        exa_urls    = set(exa_df[exa_df["university"] == uni]["policy_url"])
        gemini_urls = set(gemini_df[gemini_df["university"] == uni]["policy_url"])

        both       = exa_urls & gemini_urls
        exa_only   = exa_urls - gemini_urls
        gemini_only = gemini_urls - exa_urls

        rows.append({
            "university":       uni,
            "exa_total":        len(exa_urls),
            "gemini_total":     len(gemini_urls),
            "both":             len(both),
            "exa_only":         len(exa_only),
            "gemini_only":      len(gemini_only),
        })

    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(outdir / "overlap_per_uni.csv", index=False)

    # ------------------------------------------------------------------
    # 3) All URLs side by side (full detail)
    # ------------------------------------------------------------------
    combined = pd.concat([exa_df, gemini_df], ignore_index=True)
    combined.to_csv(outdir / "all_urls_combined.csv", index=False)

    # ------------------------------------------------------------------
    # 4) URLs only in Exa / only in Gemini
    # ------------------------------------------------------------------
    exa_urls_all    = set(exa_df["policy_url"])
    gemini_urls_all = set(gemini_df["policy_url"])

    exa_only_df    = exa_df[exa_df["policy_url"].isin(exa_urls_all - gemini_urls_all)].copy()
    gemini_only_df = gemini_df[gemini_df["policy_url"].isin(gemini_urls_all - exa_urls_all)].copy()

    exa_only_df.to_csv(outdir / "exa_only_urls.csv", index=False)
    gemini_only_df.to_csv(outdir / "gemini_only_urls.csv", index=False)

    # ------------------------------------------------------------------
    # 5) Print summary
    # ------------------------------------------------------------------
    total_exa    = exa_df["policy_url"].nunique()
    total_gemini = gemini_df["policy_url"].nunique()
    total_both   = len(exa_urls_all & gemini_urls_all)

    print("\n" + "=" * 55)
    print(f"{'COMPARISON SUMMARY':^55}")
    print("=" * 55)
    print(f"  {'Metric':<30} {'Exa':>8} {'Gemini':>8}")
    print("-" * 55)
    print(f"  {'Total universities':<30} {exa_df['university'].nunique():>8} {gemini_df['university'].nunique():>8}")
    print(f"  {'Total unique URLs':<30} {total_exa:>8} {total_gemini:>8}")
    print(f"  {'Avg URLs per university':<30} {total_exa/max(1,exa_df['university'].nunique()):>8.1f} {total_gemini/max(1,gemini_df['university'].nunique()):>8.1f}")
    print("-" * 55)
    print(f"  {'URLs in BOTH':<30} {total_both:>8}")
    print(f"  {'URLs only in Exa':<30} {len(exa_urls_all - gemini_urls_all):>8}")
    print(f"  {'URLs only in Gemini':<30} {len(gemini_urls_all - exa_urls_all):>8}")
    print("=" * 55)
    print(f"\nPer-university breakdown:")
    print(counts.to_string())
    print(f"\nOutputs saved to: {outdir}/")
    print(f"  - url_counts_per_uni.csv")
    print(f"  - overlap_per_uni.csv")
    print(f"  - all_urls_combined.csv")
    print(f"  - exa_only_urls.csv")
    print(f"  - gemini_only_urls.csv")


if __name__ == "__main__":
    main()