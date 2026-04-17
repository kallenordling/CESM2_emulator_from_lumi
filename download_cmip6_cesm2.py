#!/usr/bin/env python3
"""
Download all CESM2 CMIP6 members for a given scenario/variable from ESGF.

Example:
    python download_cmip6_cesm2.py --experiment ssp126
    python download_cmip6_cesm2.py --experiment ssp126 --variables tas pr
    python download_cmip6_cesm2.py --experiment historical --discover-only

Output layout:
    <outdir>/<experiment>/<variable>/<member>/<filename>

Members are discovered from ESGF (facets.member_id). Override with
--members to restrict to a specific list.
"""

import os
import sys
import time
import hashlib
import argparse
from urllib.parse import urlencode, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import SSLError, ConnectionError, ReadTimeout, JSONDecodeError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR = "cmip6_cesm2"
SOURCE_ID = "CESM2"

# variable_id -> table_id defaults
VARIABLE_TABLES = {
    "tas":  "Amon",
    "pr":   "Amon",
    "psl":  "Amon",
    "ts":   "Amon",
    "tos":  "Omon",
    "snw":  "LImon",
    "snd":  "LImon",
    "tsl":  "Lmon",
}

INDEX_NODES = [
    "https://esgf-data.dkrz.de/esg-search",
    "https://esgf.ceda.ac.uk/esg-search",
    "https://esgf-node.llnl.gov/esg-search",
]

PREFERRED_DATA_HOSTS = [
    "esgf-data.ucar.edu",       # NCAR — authoritative CESM2 host
    "aims3.llnl.gov",
    "aims2.llnl.gov",
    "esgf-data.dkrz.de",
    "esgf-data3.ceda.ac.uk",
    "esgf.ceda.ac.uk",
    "vesg.ipsl.upmc.fr",
]

SKIP_HOSTS = {"esgf.ichec.ie", "esg.camscma.cn"}


# ---------------------------------------------------------------------------
# Helpers (shared with download_cmip6.py template)
# ---------------------------------------------------------------------------

def first_str(val):
    if val is None:
        return None
    if isinstance(val, list):
        return str(val[0]) if val else None
    return str(val)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def md5_file(path, blocksize=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip_url(url):
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == h or host.endswith("." + h) for h in SKIP_HOSTS)


def preferred_url(doc):
    raw = doc.get("url", [])
    if isinstance(raw, str):
        raw = [raw]
    https_http, http_http, https_any, http_any = [], [], [], []
    for u in raw:
        parts = u.split("|")
        href = parts[0] if parts else ""
        svc = parts[2] if len(parts) > 2 else ""
        if href.startswith("https://"):
            https_any.append(href)
            if svc == "HTTPServer":
                https_http.append(href)
        elif href.startswith("http://"):
            http_any.append(href)
            if svc == "HTTPServer":
                http_http.append(href)
    for host in PREFERRED_DATA_HOSTS:
        for href in https_http:
            if host in href:
                return href
    if https_http:
        return https_http[0]
    for host in PREFERRED_DATA_HOSTS:
        for href in http_http:
            if host in href:
                return href
    return http_http[0] if http_http else (https_any[0] if https_any else (http_any[0] if http_any else None))


def make_session(timeout=60, verify=True):
    s = requests.Session()
    s.verify = verify
    retry = Retry(
        total=3, read=3, connect=3, status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def stream_download(session, url, dest, expected_md5=None, retries=3, timeout=60, verify=True):
    if should_skip_url(url):
        print(f"  Skipping blocked host: {url}")
        return False

    tmp = dest + ".part"
    ensure_dir(os.path.dirname(dest))
    resume_pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    for attempt in range(1, retries + 1):
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
        try:
            r = session.get(url, headers=headers, stream=True, timeout=timeout,
                            allow_redirects=True, verify=verify)
        except (SSLError, ConnectionError, ReadTimeout) as e:
            print(f"  [Attempt {attempt}/{retries}] Connection error: {e}")
            time.sleep(attempt * 2)
            continue

        if r.status_code not in (200, 206):
            print(f"  [Attempt {attempt}/{retries}] HTTP {r.status_code}: {url}")
            time.sleep(attempt * 2)
            continue

        total = int(r.headers.get("Content-Length", "0"))
        mode = "ab" if (resume_pos > 0 and r.status_code == 206) else "wb"
        initial = resume_pos if mode == "ab" else 0

        try:
            with open(tmp, mode) as f, tqdm(
                total=(initial + total if total > 0 else None),
                initial=initial, unit="B", unit_scale=True,
                desc=os.path.basename(dest),
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"  [Attempt {attempt}/{retries}] Write error: {e}")
            time.sleep(attempt * 2)
            continue

        if expected_md5:
            got = md5_file(tmp).lower()
            if got != expected_md5.lower():
                print(f"  [Attempt {attempt}/{retries}] Checksum mismatch, retrying...")
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                resume_pos = 0
                time.sleep(attempt * 2)
                continue

        os.replace(tmp, dest)
        return True

    print(f"  FAILED after {retries} attempts: {url}")
    return False


def esgf_search(facets, page_size=500, timeout=60, verify=True):
    """Search all INDEX_NODES; yield matching file docs (de-duplicated by id)."""
    # NOTE: do not pass replica=false — CESM2 originals are hosted at
    # esgf-data.ucar.edu which went offline in 2024. Only replicas remain.
    params_base = {
        "type": "File",
        "format": "application/solr+json",
        "limit": str(page_size),
        "distrib": "true",
    }
    params_base.update({k: first_str(v) for k, v in facets.items() if v is not None})

    bare = requests.Session()
    bare.verify = verify
    seen_ids: set = set()

    for node in INDEX_NODES:
        offset = 0
        consecutive_errors = 0
        while True:
            q = params_base | {"offset": str(offset)}
            url = f"{node}/search?{urlencode(q, doseq=True)}"
            try:
                r = bare.get(url, headers={"Accept": "application/solr+json"},
                             timeout=timeout, allow_redirects=True)
            except (SSLError, ConnectionError, ReadTimeout) as e:
                print(f"  Node unreachable ({type(e).__name__}): {node}")
                break

            if r.status_code in (500, 502, 503, 504):
                consecutive_errors += 1
                print(f"  Node HTTP {r.status_code} (error #{consecutive_errors}): {node}")
                if consecutive_errors >= 2:
                    print(f"  Giving up on node after repeated errors: {node}")
                    break
                time.sleep(consecutive_errors * 3)
                continue

            if r.status_code != 200:
                print(f"  Node error {r.status_code}: {node}")
                break

            consecutive_errors = 0

            try:
                data = r.json()
            except (ValueError, JSONDecodeError):
                print(f"  Bad JSON from: {node}")
                break

            resp = data.get("response", {})
            docs = resp.get("docs", [])
            if not docs:
                break

            for d in docs:
                fid = first_str(d.get("id")) or first_str(d.get("title"))
                if fid and fid in seen_ids:
                    continue
                if fid:
                    seen_ids.add(fid)
                yield d

            num_found = int(first_str(resp.get("numFound")) or "0")
            offset += len(docs)
            if offset >= num_found:
                break
            time.sleep(0.3)


def discover_members(experiment, variable_id, table_id, timeout=60, verify=True):
    """Query ESGF facet counts for all member_ids available for CESM2 under
    the given experiment/variable/table. Returns a sorted list."""
    params = {
        "type": "Dataset",
        "format": "application/solr+json",
        "limit": "0",
        "distrib": "true",
        "facets": "member_id",
        "project": "CMIP6",
        "source_id": SOURCE_ID,
        "experiment_id": experiment,
        "variable_id": variable_id,
        "table_id": table_id,
        "frequency": "mon",
    }
    bare = requests.Session()
    bare.verify = verify
    members: set = set()
    for node in INDEX_NODES:
        url = f"{node}/search?{urlencode(params, doseq=True)}"
        try:
            r = bare.get(url, timeout=timeout)
        except Exception as e:
            print(f"  discover_members: node unreachable {node}: {e}")
            continue
        if r.status_code != 200:
            print(f"  discover_members: HTTP {r.status_code} from {node}")
            continue
        try:
            data = r.json()
        except Exception:
            continue
        vals = data.get("facet_counts", {}).get("facet_fields", {}).get("member_id", [])
        for i in range(0, len(vals) - 1, 2):
            members.add(str(vals[i]))
        print(f"  discover_members: {len(members)} member(s) found via {node}")
        if members:
            break
    return sorted(members)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=f"Download all CESM2 CMIP6 members for a given scenario/variable from ESGF."
    )
    ap.add_argument("--outdir", default=OUTDIR,
                    help=f"Output directory (default: {OUTDIR})")
    ap.add_argument("--experiment", required=True,
                    help="CMIP6 experiment_id (e.g. ssp126, ssp370, historical)")
    ap.add_argument("--variables", nargs="+", default=["tas"],
                    help=f"Variables to download (default: tas). "
                         f"Known tables: {sorted(VARIABLE_TABLES)}")
    ap.add_argument("--members", nargs="+", metavar="MEMBER", default=None,
                    help="Restrict to specific member_ids (e.g. r1i1p1f1). "
                         "If omitted, all members found on ESGF are downloaded.")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--insecure", action="store_true",
                    help="Disable TLS verification (last-resort).")
    ap.add_argument("--discover-only", action="store_true",
                    help="Print what would be downloaded without downloading")
    args = ap.parse_args()

    verify = not args.insecure
    session = make_session(timeout=args.timeout, verify=verify)

    # Resolve (variable_id, table_id) pairs; allow unknown vars with user-supplied tables later.
    unknown = [v for v in args.variables if v not in VARIABLE_TABLES]
    if unknown:
        sys.exit(f"ERROR: no default table_id for variables {unknown}. "
                 f"Add to VARIABLE_TABLES at the top of the script.")
    var_table_pairs = [(v, VARIABLE_TABLES[v]) for v in args.variables]

    # Resolve member list (use first variable if we need to discover).
    if args.members:
        member_list = sorted(args.members)
        print(f"Using {len(member_list)} explicit member(s):")
    else:
        v0, t0 = var_table_pairs[0]
        print(f"Discovering CESM2 members for {args.experiment}/{v0} ({t0}) ...")
        member_list = discover_members(args.experiment, v0, t0,
                                       timeout=args.timeout, verify=verify)
        if not member_list:
            sys.exit(f"ERROR: no CESM2 members found on ESGF for experiment={args.experiment} "
                     f"variable={v0} table={t0}")
        print(f"Discovered {len(member_list)} member(s):")

    for m in member_list:
        print(f"  {m}")
    print()

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for variable_id, table_id in var_table_pairs:
        print(f"\n{'='*60}")
        print(f"Variable: {variable_id} ({table_id})  experiment={args.experiment}")
        print(f"{'='*60}")

        for member in member_list:
            print(f"\n  Member: {member}")
            facets = {
                "project": "CMIP6",
                "source_id": SOURCE_ID,
                "experiment_id": args.experiment,
                "variable_id": variable_id,
                "table_id": table_id,
                "member_id": member,
                "frequency": "mon",
            }

            # Dedupe by filename — the same file is often returned from multiple
            # mirrors. Keep the first (which preferred_url already ranks by host).
            by_name = {}
            for doc in esgf_search(facets, timeout=args.timeout, verify=verify):
                href = preferred_url(doc)
                if not href:
                    continue
                fname = os.path.basename(urlparse(href).path)
                if fname in by_name:
                    continue
                outpath = os.path.join(
                    args.outdir, args.experiment, variable_id, member, fname
                )
                csum = first_str(doc.get("checksum"))
                ctyp = first_str(doc.get("checksum_type"))
                checksum = csum if (csum and ctyp and ctyp.lower() == "md5") else None
                by_name[fname] = (href, outpath, checksum)
            found_files = list(by_name.values())

            if not found_files:
                print(f"    WARNING: no files on ESGF for {SOURCE_ID} {member} "
                      f"{args.experiment}/{variable_id}")
                continue

            print(f"    Found {len(found_files)} file(s)")

            for href, outpath, checksum in found_files:
                if args.discover_only:
                    print(f"    Would download: {os.path.basename(outpath)}  <- {href}")
                    continue

                if os.path.exists(outpath):
                    print(f"    Already exists: {os.path.basename(outpath)}")
                    total_skipped += 1
                    continue

                ensure_dir(os.path.dirname(outpath))
                ok = stream_download(session, href, outpath, expected_md5=checksum,
                                     retries=args.retries, timeout=args.timeout,
                                     verify=verify)
                if ok:
                    total_downloaded += 1
                else:
                    total_failed += 1

    print(f"\n{'='*60}")
    print(f"Done. Downloaded: {total_downloaded}  Skipped: {total_skipped}  Failed: {total_failed}")


if __name__ == "__main__":
    main()
