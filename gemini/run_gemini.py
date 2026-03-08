from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

from policy_search_gemini import search_policy_pages


# --------------------------
# Name normalization & matching
# --------------------------

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")

_SUBENTITY_RE = re.compile(
    r"\b("
    r"medical school|school of medicine|college of medicine|health sciences?|health and medicine|"
    r"medicine|hospital|medical center|health system|"
    r")\b",
    re.IGNORECASE,
)

_AT_HINT_RE = re.compile(r"\b(at|campus)\b", re.IGNORECASE)


def normalize_core(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("univ.", "university")
    s = s.replace("univ ", "university ")
    s = s.replace("*", "")
    s = _NORMALIZE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def saint_variants(s: str) -> list[str]:
    out = [s]
    if re.search(r"\bsaint\b", s):
        out.append(re.sub(r"\bsaint\b", "st", s))
    if re.search(r"\bst\b", s):
        out.append(re.sub(r"\bst\b", "saint", s))
    return list(dict.fromkeys(out))


def comma_at_variants(raw: str) -> list[str]:
    raw = str(raw).strip().replace("*", "")
    parts = [p.strip() for p in raw.split(",", 1)]
    if len(parts) == 2 and parts[0].lower().startswith("university of "):
        return [raw, f"{parts[0]} at {parts[1]}"]
    return [raw]


def hyphen_variants(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    out = [raw]
    if "-" in raw:
        out.append(raw.replace("-", " "))
        out.append(raw.replace("-", ""))
        out.append(raw.split("-", 1)[0].strip())
    return list(dict.fromkeys([x for x in out if x]))


def strip_subentity(raw: str) -> str:
    s = str(raw).strip()
    s = re.sub(_SUBENTITY_RE, " ", s)
    s = _WS_RE.sub(" ", s).strip(" ,")
    return s


def name_variants(raw: str) -> list[str]:
    out: list[str] = []
    raw = str(raw).strip()
    if not raw:
        return []

    for v0 in comma_at_variants(raw):
        for v1 in hyphen_variants(v0):
            core = normalize_core(v1)
            if core:
                out.extend(saint_variants(core))

    stripped = strip_subentity(raw)
    if stripped and stripped != raw:
        for v0 in comma_at_variants(stripped):
            for v1 in hyphen_variants(v0):
                core = normalize_core(v1)
                if core:
                    out.extend(saint_variants(core))

    return list(dict.fromkeys([x for x in out if x]))


def _split_aliases(alias: str) -> list[str]:
    if alias is None or (isinstance(alias, float) and pd.isna(alias)):
        return []
    s = str(alias).strip()
    if not s:
        return []
    parts = re.split(r"[;|/]", s)
    return [p.strip() for p in parts if p.strip()]


def build_name_map(ipeds_df: pd.DataFrame) -> dict[str, list[int]]:
    m: dict[str, list[int]] = {}
    has_ialias = "IALIAS" in ipeds_df.columns

    for idx, row in ipeds_df.iterrows():
        names = [row["INSTNM"]]
        if has_ialias:
            names.extend(_split_aliases(row.get("IALIAS", "")))

        for nm in names:
            for key in name_variants(nm):
                m.setdefault(key, []).append(idx)

    return m


def _token_set(s: str) -> set[str]:
    return set(normalize_core(s).split()) if s else set()


def _score_candidate(sc_key: str, campus_hint: str, ip_name: str, ip_web: str) -> float:
    score = 0.0
    ip_name_l = str(ip_name).lower()

    if campus_hint:
        ch = normalize_core(campus_hint)
        if ch and ch in normalize_core(ip_name_l):
            score += 2.5

    a = set(sc_key.split())
    b = _token_set(ip_name_l)
    if a and b:
        jacc = len(a & b) / max(1, len(a | b))
        score += 2.0 * jacc

    if ip_web and str(ip_web).strip():
        score += 0.2

    return score


def match_scimago_to_ipeds(
    scimago_df: pd.DataFrame,
    ipeds_df: pd.DataFrame,
    *,
    ambiguity_delta: float = 0.35,
    max_report_candidates: int = 8,
    manual_overrides: Optional[dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    name_map = build_name_map(ipeds_df)

    ipeds_key_idx_pairs: list[tuple[str, int]] = []
    for idx, row in ipeds_df.iterrows():
        for k in name_variants(row["INSTNM"]):
            ipeds_key_idx_pairs.append((k, idx))
        if "IALIAS" in ipeds_df.columns:
            for alias in _split_aliases(row.get("IALIAS", "")):
                for k in name_variants(alias):
                    ipeds_key_idx_pairs.append((k, idx))

    matched_rows = []
    unmatched_rows = []
    ambiguous_rows = []

    ip_inst = ipeds_df["INSTNM"]
    ip_unitid = ipeds_df["UNITID"]
    ip_webaddr = ipeds_df["WEBADDR"]

    for _, r in scimago_df.iterrows():
        inst_raw = str(r["Institution"]).strip()
        inst_clean = inst_raw.replace("*", " ").strip()

        parts = [p.strip() for p in inst_clean.split(",", 1)]
        inst_base = parts[0]
        campus_hint = parts[1] if len(parts) == 2 else ""

        if manual_overrides:
            k_override = normalize_core(inst_clean)
            if k_override in manual_overrides:
                best_idx = manual_overrides[k_override]
                matched_rows.append({
                    **r.to_dict(),
                    "UNITID": int(ip_unitid.at[best_idx]),
                    "INSTNM_IPEDS": str(ip_inst.at[best_idx]),
                    "WEBADDR": str(ip_webaddr.at[best_idx]) if pd.notna(ip_webaddr.at[best_idx]) else "",
                    "match_method": "manual_override",
                    "match_score": 999.0,
                })
                continue

        candidate_idxs: list[int] = []
        sc_keys = name_variants(inst_base)
        for sc_key in sc_keys:
            candidate_idxs.extend(name_map.get(sc_key, []))

        seen = set()
        candidate_idxs = [x for x in candidate_idxs if not (x in seen or seen.add(x))]

        if not candidate_idxs:
            primary = max(sc_keys, key=len) if sc_keys else normalize_core(inst_base)
            MIN_KEY_LEN = 16
            if primary and len(primary) >= MIN_KEY_LEN:
                cand = []
                for ip_key, ip_idx in ipeds_key_idx_pairs:
                    if primary in ip_key:
                        cand.append(ip_idx)
                seen2 = set()
                candidate_idxs = [x for x in cand if not (x in seen2 or seen2.add(x))]

        if not candidate_idxs:
            unmatched_rows.append(r.to_dict())
            continue

        primary_sc_key = max(sc_keys, key=len) if sc_keys else normalize_core(inst_base)
        scored = []
        for idx in candidate_idxs:
            sc = _score_candidate(
                sc_key=primary_sc_key,
                campus_hint=campus_hint,
                ip_name=str(ip_inst.at[idx]),
                ip_web=str(ip_webaddr.at[idx]) if pd.notna(ip_webaddr.at[idx]) else "",
            )
            scored.append((sc, idx))
        scored.sort(reverse=True, key=lambda x: x[0])

        best_score, best_idx = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0

        if len(scored) > 1 and (best_score - second_score) < ambiguity_delta:
            top_cands = scored[:max_report_candidates]
            ambiguous_rows.append({
                **r.to_dict(),
                "scimago_key": primary_sc_key,
                "campus_hint": campus_hint,
                "best_unitid": int(ip_unitid.at[best_idx]),
                "best_instnm": str(ip_inst.at[best_idx]),
                "best_webaddr": str(ip_webaddr.at[best_idx]) if pd.notna(ip_webaddr.at[best_idx]) else "",
                "best_score": best_score,
                "candidates": json.dumps([
                    {
                        "score": float(s),
                        "idx": int(i),
                        "UNITID": int(ip_unitid.at[i]),
                        "INSTNM": str(ip_inst.at[i]),
                        "WEBADDR": str(ip_webaddr.at[i]) if pd.notna(ip_webaddr.at[i]) else "",
                    }
                    for s, i in top_cands
                ], ensure_ascii=False),
            })
            method = "ambiguous_scored"
        else:
            method = "scored"

        matched_rows.append({
            **r.to_dict(),
            "UNITID": int(ip_unitid.at[best_idx]),
            "INSTNM_IPEDS": str(ip_inst.at[best_idx]),
            "WEBADDR": str(ip_webaddr.at[best_idx]) if pd.notna(ip_webaddr.at[best_idx]) else "",
            "match_method": method,
            "match_score": float(best_score),
        })

    return (
        pd.DataFrame(matched_rows),
        pd.DataFrame(unmatched_rows),
        pd.DataFrame(ambiguous_rows),
    )


def load_manual_overrides(path: Path, ipeds_df: pd.DataFrame) -> dict[str, int]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    need = {"scimago_institution", "unitid"}
    if not need.issubset(df.columns):
        raise ValueError(f"manual overrides must include columns {need}")

    u2idx = {int(u): int(i) for i, u in ipeds_df["UNITID"].astype(int).items()}

    out: dict[str, int] = {}
    for _, row in df.iterrows():
        k = normalize_core(str(row["scimago_institution"]))
        u = int(row["unitid"])
        if u not in u2idx:
            continue
        out[k] = u2idx[u]
    return out


# --------------------------
# HTML download
# --------------------------

@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int]
    content_type: str
    saved_path: str
    error: str


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def download_html(
    url: str,
    out_dir: Path,
    session: requests.Session,
    timeout: int = 25,
    max_retries: int = 2,
    sleep_s: float = 0.3,
) -> FetchResult:
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
                return FetchResult(url=url, ok=False, status_code=status, content_type=ctype, saved_path="", error="skipped_non_html")

            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_url_hash(url)}.html"
            path = out_dir / filename
            path.write_bytes(resp.content)

            return FetchResult(url=url, ok=True, status_code=status, content_type=ctype, saved_path=str(path), error="")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(sleep_s * (attempt + 1))
                continue
            return FetchResult(url=url, ok=False, status_code=None, content_type="", saved_path="", error=last_err)


# --------------------------
# Main pipeline
# --------------------------

def build_scimago_keyset(sc_df: pd.DataFrame) -> set[str]:
    keys = set()
    for inst in sc_df["Institution"].astype(str):
        base = inst.split(",", 1)[0].strip()
        for k in name_variants(base):
            keys.add(k)
        for k in name_variants(inst):
            keys.add(k)
    return keys


def ipeds_row_in_scimago(row: pd.Series, sc_keys: set[str]) -> bool:
    candidates = [row["INSTNM"]]
    if "IALIAS" in row.index:
        candidates.extend(_split_aliases(row.get("IALIAS", "")))
    for nm in candidates:
        for k in name_variants(nm):
            if k in sc_keys:
                return True
    return False


def build_selected_500(
    ipeds_path: Path,
    scimago_path: Path,
    top_n: int,
    random_n: int,
    seed: int,
    out_csv: Path,
    *,
    reports_dir: Path,
    manual_overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    ipeds = pd.read_csv(ipeds_path)

    needed = {"UNITID", "INSTNM", "WEBADDR"}
    missing = needed - set(ipeds.columns)
    if missing:
        raise ValueError(f"IPEDS file missing required columns: {missing}")

    ipeds = ipeds.copy()
    ipeds["UNITID"] = ipeds["UNITID"].astype(int)
    ipeds["INSTNM"] = ipeds["INSTNM"].astype(str)
    ipeds["WEBADDR"] = ipeds["WEBADDR"].fillna("").astype(str)
    if "IALIAS" in ipeds.columns:
        ipeds["IALIAS"] = ipeds["IALIAS"].fillna("").astype(str)

    sc = pd.read_csv(scimago_path, sep=",", encoding="latin1")
    if "Rank" not in sc.columns or "Institution" not in sc.columns:
        raise ValueError("SciMAGO file must contain 'Rank' and 'Institution' columns.")

    sc["Rank"] = pd.to_numeric(sc["Rank"], errors="coerce")
    sc = sc.dropna(subset=["Rank", "Institution"]).copy()
    sc["Rank"] = sc["Rank"].astype(int)
    sc["Institution"] = sc["Institution"].astype(str)

    reports_dir.mkdir(parents=True, exist_ok=True)

    manual_overrides = None
    if manual_overrides_path is not None:
        manual_overrides = load_manual_overrides(manual_overrides_path, ipeds_df=ipeds)

    sc_matched_all, sc_unmatched_all, sc_amb_all = match_scimago_to_ipeds(
        sc, ipeds,
        manual_overrides=manual_overrides,
    )
    if not sc_unmatched_all.empty:
        sc_unmatched_all.to_csv(reports_dir / "scimago_unmatched_all.csv", index=False)
    if not sc_amb_all.empty:
        sc_amb_all.to_csv(reports_dir / "scimago_ambiguous_all.csv", index=False)

    sc_top = sc.sort_values("Rank").head(top_n).copy()
    sc_matched_top, sc_unmatched_top, sc_amb_top = match_scimago_to_ipeds(
        sc_top, ipeds,
        manual_overrides=manual_overrides,
    )
    if not sc_unmatched_top.empty:
        sc_unmatched_top.to_csv(reports_dir / "scimago_unmatched_top.csv", index=False)
    if not sc_amb_top.empty:
        sc_amb_top.to_csv(reports_dir / "scimago_ambiguous_top.csv", index=False)

    if len(sc_matched_top) < top_n:
        print(f"[WARN] Only matched {len(sc_matched_top)}/{top_n} SciMAGO top rows to IPEDS.")

    top_df = pd.DataFrame({
        "UNITID": sc_matched_top["UNITID"].astype(int),
        "INSTNM": sc_matched_top["INSTNM_IPEDS"].astype(str),
        "WEBADDR": sc_matched_top["WEBADDR"].astype(str),
        "group": "scimago_top",
        "scimago_rank": sc_matched_top["Rank"].astype(int),
        "scimago_institution": sc_matched_top["Institution"].astype(str),
    })

    scimago_keys = build_scimago_keyset(sc)
    ipeds = ipeds.copy()
    ipeds["IN_SCIMAGO_600"] = ipeds.apply(lambda r: ipeds_row_in_scimago(r, scimago_keys), axis=1)

    pool = ipeds[~ipeds["IN_SCIMAGO_600"]].copy()
    pool = pool[pool["WEBADDR"].astype(str).str.strip() != ""].copy()

    if len(pool) < random_n:
        raise ValueError(f"Random pool too small ({len(pool)}) to sample {random_n} universities.")

    random.seed(seed)
    sampled_unitids = random.sample(pool["UNITID"].tolist(), random_n)
    rand_df = pool[pool["UNITID"].isin(sampled_unitids)].copy()
    rand_df["group"] = "random_not_in_scimago"
    rand_df["scimago_rank"] = ""
    rand_df["scimago_institution"] = ""

    selected = pd.concat(
        [top_df, rand_df[["UNITID", "INSTNM", "WEBADDR", "group", "scimago_rank", "scimago_institution"]]],
        ignore_index=True
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipeds", type=str, default="hd2024_data_stata 2.csv")
    ap.add_argument("--scimago", type=str, default="ScimagoIR 2025 - Overall Rank - Universities - USA.csv")
    ap.add_argument("--top-n", type=int, default=250)
    ap.add_argument("--random-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-results", type=int, default=20)
    ap.add_argument("--restrict-domain", action="store_true", default=True)
    ap.add_argument("--no-restrict-domain", action="store_false", dest="restrict_domain")

    ap.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--outdir", type=str, default="data/compilation_gemini")
    ap.add_argument("--skip-html", action="store_true")
    ap.add_argument("--max-urls-per-uni", type=int, default=60)
    ap.add_argument("--sleep-between-unis", type=float, default=1.0)
    ap.add_argument("--dry-sample", action="store_true")

    ap.add_argument("--manual-overrides", type=str, default="")
    ap.add_argument("--force-rebuild-sample", action="store_true")
    ap.add_argument("--force-rerun-gemini", action="store_true")
    ap.add_argument("--force-redownload-html", action="store_true")

    args = ap.parse_args()

    ipeds_path = Path(args.ipeds)
    scimago_path = Path(args.scimago)
    outdir = Path(args.outdir)

    sample_csv = outdir / "sample" / "selected_500.csv"
    reports_dir = outdir / "reports"
    manual_path = Path(args.manual_overrides) if args.manual_overrides.strip() else None

    # Only require IPEDS/SciMAGO if we actually need to rebuild the sample
    need_rebuild_sample = args.force_rebuild_sample or (not sample_csv.exists())

    if need_rebuild_sample:
        if not ipeds_path.exists():
            raise FileNotFoundError(f"IPEDS CSV not found: {ipeds_path.resolve()}")
        if not scimago_path.exists():
            raise FileNotFoundError(f"SciMAGO CSV not found: {scimago_path.resolve()}")

    if sample_csv.exists() and not args.force_rebuild_sample:
        print(f"[RESUME] Using existing sample: {sample_csv}")
        selected = pd.read_csv(sample_csv)
    else:
        selected = build_selected_500(
            ipeds_path=ipeds_path,
            scimago_path=scimago_path,
            top_n=args.top_n,
            random_n=args.random_n,
            seed=args.seed,
            out_csv=sample_csv,
            reports_dir=reports_dir,
            manual_overrides_path=manual_path,
        )

    if args.dry_sample:
        print(selected.head(5).to_string(index=False))
        return

    print(f"[OK] Sample: {sample_csv} (n={len(selected)})")

    raw_dir = outdir / "gemini_raw"
    html_dir = outdir / "html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()

    def html_already_done(unitid: int) -> bool:
        uni_html_dir = html_dir / str(unitid)
        manifest = uni_html_dir / "manifest.csv"
        if not manifest.exists():
            return False
        return len(list(uni_html_dir.glob("*.html"))) > 0

    for i, row in selected.iterrows():
        unitid = int(row["UNITID"])
        instnm = str(row["INSTNM"])
        web = str(row["WEBADDR"]) if pd.notna(row["WEBADDR"]) else ""
        group = str(row.get("group", ""))

        print(f"\n[{i+1}/{len(selected)}] {unitid} | {instnm} | {group}")

        raw_path = raw_dir / f"{unitid}.json"

        if raw_path.exists() and not args.force_rerun_gemini and (
            args.skip_html or html_already_done(unitid) or not args.force_redownload_html
        ):
            print(f"[SKIP] already processed: {raw_path}")
            if args.skip_html:
                continue

        hits = None
        if raw_path.exists() and not args.force_rerun_gemini:
            try:
                hits = json.loads(raw_path.read_text(encoding="utf-8"))
                print(f"[RESUME] loaded raw json: {raw_path} (hits={len(hits)})")
            except Exception as e:
                print(f"[WARN] Could not parse raw json, rerunning Gemini: {type(e).__name__}: {e}")
                hits = None

        if hits is None:
            try:
                hits = search_policy_pages(
                    university=instnm,
                    website=web,
                    num_results=args.num_results,
                    restrict_domain=args.restrict_domain,
                    model=args.gemini_model,
                )
            except Exception as e:
                print(f"[ERROR] Gemini search failed for {instnm}: {type(e).__name__}: {e}")
                hits = []

            raw_path.write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] raw json: {raw_path} (hits={len(hits)})")

        if args.skip_html:
            continue

        if html_already_done(unitid) and not args.force_redownload_html:
            print(f"[SKIP] html already present for {unitid}")
            continue

        urls = []
        seen = set()
        for h in hits:
            u = h.get("url", "")
            if not u or u in seen:
                continue
            seen.add(u)
            urls.append(u)
        urls = urls[: max(0, int(args.max_urls_per_uni))]

        uni_html_dir = html_dir / str(unitid)
        manifest_path = uni_html_dir / "manifest.csv"
        uni_html_dir.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["unitid", "instnm", "url", "ok", "status_code", "content_type", "saved_path", "error"],
            )
            w.writeheader()

            ok_count = 0
            for u in urls:
                fr = download_html(u, uni_html_dir, sess)
                if fr.ok:
                    ok_count += 1
                w.writerow({
                    "unitid": unitid,
                    "instnm": instnm,
                    "url": fr.url,
                    "ok": fr.ok,
                    "status_code": fr.status_code if fr.status_code is not None else "",
                    "content_type": fr.content_type,
                    "saved_path": fr.saved_path,
                    "error": fr.error,
                })

        print(f"[OK] html manifest: {manifest_path} | saved_html={ok_count}/{len(urls)}")
        time.sleep(max(0.0, float(args.sleep_between_unis)))

    print("\nDone.")
    print(f"- Sample: {sample_csv}")
    print(f"- Reports: {reports_dir}")
    print(f"- Raw JSON: {raw_dir}")
    print(f"- HTML: {html_dir}")


if __name__ == "__main__":
    main()