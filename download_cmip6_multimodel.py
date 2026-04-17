#!/usr/bin/env python3
"""
Download tas (Amon) and snw/snd/dfr (Lmon/LImon) from ESGF.

Models and members are specified via CLI (or use built-in defaults).
Uses one representative member per model (r1i1p1f1 preferred; with
automatic fallback to other r1i1p1fX variants if not found on ESGF).

Output layout: <outdir>/<experiment>/<variable>/<model>/<filename>
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

OUTDIR = "cmip6_tas_snw"
EXPERIMENT = "historical"  # default; override with --experiment

# Default models/members if --models is not given.
# Tweak this list to match whatever set you want by default.
DEFAULT_MODEL_MEMBERS = {
    "MIROC-ES2L":     "r1i1p1f2",
    "UKESM1-0-LL":    "r1i1p1f2",
    "CanESM5":        "r1i1p1f1",
    "ACCESS-ESM1-5":  "r1i1p1f1",
    "NorESM2-LM":     "r1i1p1f1",
    "CESM2":          "r1i1p1f1",
    "IPSL-CM6A-LR":   "r1i1p1f1",
    "MPI-ESM1-2-LR":  "r1i1p1f1",
    "GFDL-ESM4":      "r1i1p1f1",
    "CNRM-ESM2-1":    "r1i1p1f2",
}

# Variables to download: (variable_id, table_id) pairs
# snw = snow amount (LImon), snd = snow depth (Lmon)
# dfr = depth of frozen soil (Lmon)
VARIABLES = [
    ("tas", "Amon"),
    ("snw", "LImon"),
    ("snd", "Lmon"),
    ("dfr", "Lmon"),
]

INDEX_NODES = [
    "https://esgf.ceda.ac.uk/esg-search",
    "https://esgf-data.dkrz.de/esg-search",
    "https://esgf-node.ipsl.upmc.fr/esg-search",
]

PREFERRED_DATA_HOSTS = [
    "esgf-data3.ceda.ac.uk",
    "esgf.ceda.ac.uk",
    "esgf-data.dkrz.de",
    "vesg.ipsl.upmc.fr",
    "aims2.llnl.gov",
]

SKIP_HOSTS = {"esgf.ichec.ie", "esg.camscma.cn"}

# Fallback member IDs to try if the requested member has no files on ESGF
FALLBACK_MEMBERS = ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3"]

# ---------------------------------------------------------------------------
# Helpers
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


def esgf_search(session, facets, page_size=500, timeout=60, verify=True):
    """Search all INDEX_NODES; yield matching file docs.

    Uses a bare session (no retry middleware) per request so that 500 errors
    are handled locally — a persistently broken node is skipped rather than
    causing urllib3 to raise RetryError up to the caller.
    Results are de-duplicated across nodes by file id.
    """
    params_base = {
        "type": "File",
        "format": "application/solr+json",
        "limit": str(page_size),
        "distrib": "true",
        "replica": "false",
    }
    params_base.update({k: first_str(v) for k, v in facets.items() if v is not None})

    # Bare session — no automatic retries; we decide when to give up per node.
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
                # De-duplicate across nodes by file id
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


# ---------------------------------------------------------------------------
# Model/member parsing
# ---------------------------------------------------------------------------

def parse_model_members(specs):
    """
    Parse a list of strings of the form 'Model:member' or just 'Model'.
    For 'Model' without ':member', default to r1i1p1f1.
    """
    result = {}
    for s in specs:
        if ":" in s:
            model, member = s.split(":", 1)
            result[model.strip()] = member.strip()
        else:
            result[s.strip()] = "r1i1p1f1"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Download tas + snw/snd/dfr from ESGF for a given set of CMIP6 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--experiment", default=EXPERIMENT,
                    help="CMIP6 experiment_id (e.g. historical, esm-hist, 1pctCO2-cdr)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Models to download, each as 'Model' or 'Model:member'. "
                         "Examples: 'MIROC-ES2L' or 'MIROC-ES2L:r1i1p1f2'. "
                         "If omitted, uses the built-in DEFAULT_MODEL_MEMBERS list.")
    ap.add_argument("--model", default=None,
                    help="Shortcut for a single model: 'Model' or 'Model:member'. "
                         "Equivalent to '--models <value>'. Overrides --models if both given.")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--discover-only", action="store_true",
                    help="Print what would be downloaded without downloading")
    ap.add_argument("--variables", nargs="+",
                    default=["tas", "snw", "snd", "dfr"],
                    help="Variables to download")
    args = ap.parse_args()

    verify = not args.insecure
    session = make_session(timeout=args.timeout, verify=verify)

    if args.model:
        model_members = parse_model_members([args.model])
    elif args.models:
        model_members = parse_model_members(args.models)
    else:
        model_members = dict(DEFAULT_MODEL_MEMBERS)

    print(f"Experiment: {args.experiment}")
    print(f"Models ({len(model_members)}):")
    for model, member in sorted(model_members.items()):
        print(f"  {model}  ->  {member}")
    print()

    # Filter variable list
    var_table_pairs = [(v, t) for v, t in VARIABLES if v in args.variables]

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for variable_id, table_id in var_table_pairs:
        print(f"\n{'='*60}")
        print(f"Variable: {variable_id} ({table_id})")
        print(f"{'='*60}")

        for model, member in sorted(model_members.items()):
            print(f"\n  Model: {model}  Member: {member}")

            def search_with_member(m):
                facets = {
                    "project": "CMIP6",
                    "experiment_id": args.experiment,
                    "variable_id": variable_id,
                    "table_id": table_id,
                    "source_id": model,
                    "member_id": m,
                }
                files = []
                for doc in esgf_search(session, facets, timeout=args.timeout, verify=verify):
                    href = preferred_url(doc)
                    if not href:
                        continue
                    fname = os.path.basename(urlparse(href).path)
                    outpath = os.path.join(args.outdir, args.experiment,
                                           variable_id, model, fname)
                    csum = first_str(doc.get("checksum"))
                    ctyp = first_str(doc.get("checksum_type"))
                    checksum = csum if (csum and ctyp and ctyp.lower() == "md5") else None
                    files.append((href, outpath, checksum))
                return files

            found_files = search_with_member(member)

            if not found_files:
                print(f"    No files found for {member}, trying fallback members...")
                for fb_member in FALLBACK_MEMBERS:
                    if fb_member == member:
                        continue
                    found_files = search_with_member(fb_member)
                    if found_files:
                        print(f"    Found {len(found_files)} file(s) for {fb_member}")
                        break

            if not found_files:
                print(f"    WARNING: No files found on any ESGF node for "
                      f"{model} {variable_id} ({args.experiment})")
                continue

            print(f"    Found {len(found_files)} file(s)")

            for href, outpath, checksum in found_files:
                if args.discover_only:
                    print(f"    Would download: {os.path.basename(outpath)}")
                    continue

                if os.path.exists(outpath):
                    print(f"    Already exists: {os.path.basename(outpath)}")
                    total_skipped += 1
                    continue

                ensure_dir(os.path.dirname(outpath))
                ok = stream_download(session, href, outpath, expected_md5=checksum,
                                     retries=args.retries, timeout=args.timeout, verify=verify)
                if ok:
                    total_downloaded += 1
                else:
                    total_failed += 1

    print(f"\n{'='*60}")
    print(f"Done. Downloaded: {total_downloaded}  Skipped: {total_skipped}  Failed: {total_failed}")


if __name__ == "__main__":
    main()
