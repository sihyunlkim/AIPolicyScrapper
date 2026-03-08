from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

from policy_search import search_policy_pages


# --------------------------
# UNITID-based matching
# --------------------------

def match_common_to_ipeds(
    common_df: pd.DataFrame,
    ipeds_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match common_universities CSV to IPEDS by UNITID (exact join, no normalization).

    Returns
    -------
    matched   : rows enriched with INSTNM_IPEDS + WEBADDR from IPEDS
    unmatched : rows whose UNITID was not found in IPEDS (or WEBADDR missing)
    """
    common_df = common_df.copy()
    ipeds_df  = ipeds_df.copy()

    common_df["UNITID"] = common_df["UNITID"].astype(int)
    ipeds_df["UNITID"]  = ipeds_df["UNITID"].astype(int)
    ipeds_df["WEBADDR"] = ipeds_df["WEBADDR"].fillna("").astype(str)

    merged = common_df.merge(
        ipeds_df[["UNITID", "INSTNM", "WEBADDR"]],
        on="UNITID",
        how="left",
        suffixes=("_common", "_ipeds"),
    )

    has_web   = merged["WEBADDR"].notna() & (merged["WEBADDR"].str.strip() != "")
    matched   = merged[has_web].copy()
    unmatched = merged[~has_web].copy()

    matched["match_method"] = "unitid_exact"

    return matched, unmatched


# --------------------------
# Build selected list
# --------------------------

def build_selected(
    common_path: Path,
    ipeds_path: Path,
    out_csv: Path,
    *,
    reports_dir: Path,
) -> pd.DataFrame:
    """
    Match common_universities to IPEDS and write selected.csv.
    Also writes:
      reports/unmatched.csv  — UNITIDs not found / missing WEBADDR
    """
    common_df = pd.read_csv(common_path)
    ipeds_df  = pd.read_csv(ipeds_path)

    for col in ("UNITID", "INSTNM"):
        if col not in common_df.columns:
            raise ValueError(f"common CSV missing column: {col}")
    for col in ("UNITID", "INSTNM", "WEBADDR"):
        if col not in ipeds_df.columns:
            raise ValueError(f"IPEDS CSV missing column: {col}")

    reports_dir.mkdir(parents=True, exist_ok=True)

    matched, unmatched = match_common_to_ipeds(common_df, ipeds_df)

    if not unmatched.empty:
        unmatched.to_csv(reports_dir / "unmatched.csv", index=False)
        print(f"[WARN] {len(unmatched)} rows unmatched -> {reports_dir / 'unmatched.csv'}")

    # Rename for clarity
    if "INSTNM_ipeds" in matched.columns:
        matched = matched.rename(columns={"INSTNM_ipeds": "INSTNM_IPEDS"})
    if "INSTNM_common" in matched.columns:
        matched = matched.rename(columns={"INSTNM_common": "INSTNM"})
    elif "INSTNM" not in matched.columns and "INSTNM_IPEDS" in matched.columns:
        matched = matched.rename(columns={"INSTNM_IPEDS": "INSTNM"})

    selected = matched[["UNITID", "INSTNM", "WEBADDR"]].drop_duplicates("UNITID").copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False)

    return selected


# --------------------------
# HTML download
# --------------------------

def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def download_html(
    url: str,
    out_dir: Path,
    session: requests.Session,
    timeout: int = 25,
    max_retries: int = 2,
    sleep_s: float = 0.3,
) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; policy-research-bot/0.1; +https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            status = resp.status_code
            ctype = (resp.headers.get("content-type") or "").lower()

            if "text/html" not in ctype and "application/xhtml" not in ctype:
                return dict(url=url, ok=False, status_code=status, content_type=ctype, saved_path="", error="skipped_non_html")

            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{_url_hash(url)}.html"
            path.write_bytes(resp.content)

            return dict(url=url, ok=True, status_code=status, content_type=ctype, saved_path=str(path), error="")

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(sleep_s * (attempt + 1))
                continue
            return dict(url=url, ok=False, status_code=None, content_type="", saved_path="", error=last_err)


# --------------------------
# Main pipeline
# --------------------------

def html_already_done(html_dir: Path, unitid: int) -> bool:
    uni_html_dir = html_dir / str(unitid)
    if not (uni_html_dir / "manifest.csv").exists():
        return False
    return len(list(uni_html_dir.glob("*.html"))) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", type=str, default="common_universities_2019_2024.csv",
                    help="Path to common_universities CSV (must include UNITID, INSTNM).")
    ap.add_argument("--ipeds",  type=str, default="hd2024_data_stata 2.csv",
                    help="Path to IPEDS CSV (must include UNITID, INSTNM, WEBADDR).")

    ap.add_argument("--num-results",      type=int,   default=100)
    ap.add_argument("--search-type",      type=str,   default="deep",
                    choices=["neural", "keyword", "deep"])
    ap.add_argument("--restrict-domain",  action="store_true", default=True)
    ap.add_argument("--no-restrict-domain", action="store_false", dest="restrict_domain")

    ap.add_argument("--outdir",               type=str,   default="data/compilation")
    ap.add_argument("--skip-html",            action="store_true")
    ap.add_argument("--max-urls-per-uni",     type=int,   default=60)
    ap.add_argument("--sleep-between-unis",   type=float, default=0.7)
    ap.add_argument("--dry-sample",           action="store_true",
                    help="Only build selected.csv and exit (no Exa, no HTML).")

    # Resume flags
    ap.add_argument("--force-rebuild-sample",   action="store_true",
                    help="Rebuild selected.csv even if it already exists.")
    ap.add_argument("--force-rerun-exa",        action="store_true",
                    help="Re-run Exa search even if raw JSON exists.")
    ap.add_argument("--force-redownload-html",  action="store_true",
                    help="Re-download HTML even if already present.")

    args = ap.parse_args()

    common_path = Path(args.common)
    ipeds_path  = Path(args.ipeds)
    outdir      = Path(args.outdir)

    if not common_path.exists():
        raise FileNotFoundError(f"Common CSV not found: {common_path.resolve()}")
    if not ipeds_path.exists():
        raise FileNotFoundError(f"IPEDS CSV not found: {ipeds_path.resolve()}")

    sample_csv  = outdir / "sample" / "selected.csv"
    reports_dir = outdir / "reports"

    # ------------------------------------------------------------------
    # 0) Build or reuse selected.csv
    # ------------------------------------------------------------------
    if sample_csv.exists() and not args.force_rebuild_sample:
        print(f"[RESUME] Using existing sample: {sample_csv}")
        selected = pd.read_csv(sample_csv)
    else:
        if sample_csv.exists():
            print(f"[WARN] Forcing rebuild of sample: {sample_csv}")
        selected = build_selected(
            common_path=common_path,
            ipeds_path=ipeds_path,
            out_csv=sample_csv,
            reports_dir=reports_dir,
        )

    if args.dry_sample:
        print("\n[DRY] Sample preview:")
        print(selected.head(10).to_string(index=False))
        missing_web = (selected["WEBADDR"].astype(str).str.strip() == "").sum()
        print(f"\n[DRY] Total unis     : {len(selected)}")
        print(f"[DRY] Missing WEBADDR: {missing_web}")
        print(f"[DRY] Duplicate UNITID: {selected['UNITID'].duplicated().sum()}")
        print(f"[DRY] Reports dir    : {reports_dir}")
        print("[DRY] Exiting before Exa/HTML.")
        return

    print(f"[OK] Sample: {sample_csv} (n={len(selected)})")
    print(f"[OK] Reports: {reports_dir}")

    raw_dir  = outdir / "exa_raw"
    html_dir = outdir / "html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()

    # ------------------------------------------------------------------
    # 1) Main loop
    # ------------------------------------------------------------------
    for i, row in selected.iterrows():
        unitid = int(row["UNITID"])
        instnm = str(row["INSTNM"])
        web    = str(row["WEBADDR"]) if pd.notna(row["WEBADDR"]) else ""

        print(f"\n[{i+1}/{len(selected)}] {unitid} | {instnm}")

        raw_path = raw_dir / f"{unitid}.json"

        # (A) Resume: skip if fully done
        if (raw_path.exists() and not args.force_rerun_exa
                and (args.skip_html or html_already_done(html_dir, unitid)
                     or not args.force_redownload_html)):
            print(f"[SKIP] already processed: {raw_path}")
            continue

        # (B) Load or fetch Exa results
        hits = None
        if raw_path.exists() and not args.force_rerun_exa:
            try:
                hits = json.loads(raw_path.read_text(encoding="utf-8"))
                print(f"[RESUME] loaded raw json: {raw_path} (hits={len(hits)})")
            except Exception as e:
                print(f"[WARN] Could not parse raw json, rerunning Exa: {e}")
                hits = None

        if hits is None:
            try:
                hits = search_policy_pages(
                    university=instnm,
                    website=web,
                    num_results=args.num_results,
                    restrict_domain=args.restrict_domain,
                    search_type=args.search_type,
                )
            except Exception as e:
                print(f"[ERROR] Exa search failed for {instnm}: {e}")
                hits = []

            raw_path.write_text(
                json.dumps(hits, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[OK] raw json: {raw_path} (hits={len(hits)})")

        # (C) HTML download
        if args.skip_html:
            continue

        if html_already_done(html_dir, unitid) and not args.force_redownload_html:
            print(f"[SKIP] html already present for {unitid}")
            continue

        # Deduplicate URLs
        urls = list(dict.fromkeys(
            h.get("url", "") for h in hits if h.get("url", "")
        ))[:max(0, int(args.max_urls_per_uni))]

        uni_html_dir = html_dir / str(unitid)
        uni_html_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = uni_html_dir / "manifest.csv"

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "unitid", "instnm", "url", "ok",
                "status_code", "content_type", "saved_path", "error",
            ])
            w.writeheader()

            ok_count = 0
            for u in urls:
                fr = download_html(u, uni_html_dir, sess)
                if fr["ok"]:
                    ok_count += 1
                w.writerow({
                    "unitid":       unitid,
                    "instnm":       instnm,
                    "url":          fr["url"],
                    "ok":           fr["ok"],
                    "status_code":  fr["status_code"] if fr["status_code"] is not None else "",
                    "content_type": fr["content_type"],
                    "saved_path":   fr["saved_path"],
                    "error":        fr["error"],
                })

        print(f"[OK] html manifest: {manifest_path} | saved={ok_count}/{len(urls)}")
        time.sleep(max(0.0, float(args.sleep_between_unis)))

    print("\nDone.")
    print(f"- Sample  : {sample_csv}")
    print(f"- Reports : {reports_dir}")
    print(f"- Raw JSON: {raw_dir}")
    print(f"- HTML    : {html_dir}")


if __name__ == "__main__":
    main()