#!/usr/bin/env python3
f"""
Download CMIP7 gridded emissions (BC, SO2, CO2) from ESGF input4MIPs.

CMIP7 forcing data lives under project=input4MIPs (NOT project=CMIP7); the
CMIP7-ness is the `mip_era` facet. Two source families matter here:

  historical  PNNL-JGCRI / CEDS-CMIP-2025-04-18   (target_mip=CMIP, 1750-2023)
  scenarios   IIASA-IAMC-<scen>-1-1-0             (target_mip=ScenarioMIP, 2022-2100)

CMIP7 ScenarioMIP scenarios are named by warming level, not RCP-style numbers:
vl (very low), l (low), ml (medium-low), m (medium), h (high), hl, ln.  As of
2026-08 only `h` and `vl` have gridded emissions published; use --list-sources
to see what is actually on ESGF now rather than trusting this comment.

Variables are `<SPECIES>_em_anthro` (surface anthropogenic, has a `sector`
dimension), with `_em_openburning` and `_em_AIR_anthro` (aircraft) available via
--kinds. Note SO2 is the sulfur species -- there is no "SUL" variable in
input4MIPs; SO2_em_anthro is the CMIP6 SUL inventory's successor.

Grid labels: `gn` = 0.5deg native global monthly (what you want), `gr` = a 0.1deg
regridded product covering only 1980-2023. Default is gn.

Output layout is chosen with --layout:
  nested (default)  <outdir>/CMIP7/<target_mip>/<source_id>/<variable_id>/<file>
  flat              <outdir>/<file>

Use `flat` to feed the existing cond-building pipeline: data/make_aerosol_files.py
and data/make_co2_files.py glob a single flat INPUT_DIR by filename
(default {L.DATA}/emission_data/inputs4mips/),
so nested subdirectories are invisible to them. Filenames are globally unique
(they encode source_id and date range), so flat has no collisions.

Examples
--------
# what's published right now
python download_input4mips_cmip7.py --list-sources

# default set: BC+SO2+CO2 anthro, CEDS historical + IIASA h/vl scenarios (~12 GB)
python download_input4mips_cmip7.py --outdir {L.DATA}/input4mips

# dry run first
python download_input4mips_cmip7.py --discover-only

# just historical sulfur
python download_input4mips_cmip7.py --species SO2 --sources CEDS-CMIP-2025-04-18
"""

import lumi_paths as L
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

OUTDIR = f"{L.DATA}/input4mips"

# Historical anthropogenic emissions (CEDS) + the published CMIP7 ScenarioMIP
# gridded scenarios. Verified present on ESGF 2026-08-06.
DEFAULT_SOURCES = [
    "CEDS-CMIP-2025-04-18",   # historical, 1750-2023
    "IIASA-IAMC-h-1-1-0",     # high scenario,      2022-2100
    "IIASA-IAMC-vl-1-1-0",    # very low scenario,  2022-2100
]

DEFAULT_SPECIES = ["BC", "SO2", "CO2"]

# variable_id suffix per emission kind
KIND_SUFFIX = {
    "anthro":      "_em_anthro",       # surface anthropogenic (sectored)
    "openburning": "_em_openburning",  # open biomass burning
    "air":         "_em_AIR_anthro",   # aircraft
}

# Only these nodes still serve the classic Solr esg-search API. Deliberately
# excluded: esgf-node.llnl.gov / aims2.llnl.gov now 302 to an ORNL
# "esgf-1-5-bridge" with a different response shape, and
# esgf-node.ipsl.upmc.fr returned HTTP 500 on every query (checked 2026-08-06).
# Both index nodes below are federated (distrib=true), so either alone sees the
# whole archive; the second is redundancy against one being down.
INDEX_NODES = [
    "https://esgf.ceda.ac.uk/esg-search",
    "https://esgf-data.dkrz.de/esg-search",
]

PREFERRED_DATA_HOSTS = [
    "esgf.ceda.ac.uk",
    "esgf-data3.ceda.ac.uk",
    "esgf1.dkrz.de",
    "esgf-data.dkrz.de",
    "vesg.ipsl.upmc.fr",
]

SKIP_HOSTS = {"esgf.ichec.ie", "esg.camscma.cn"}

# ---------------------------------------------------------------------------
# Helpers (same contract as download_cmip6_multimodel.py)
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


def sha256_file(path, blocksize=1024 * 1024):
    h = hashlib.sha256()
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
    """Pick the best HTTPServer href, preferring known-good hosts and https."""
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


def stream_download(session, url, dest, expected=None, retries=3, timeout=120, verify=True):
    """Resumable streaming download. `expected` is (algo, hexdigest) or None."""
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
                desc=os.path.basename(dest)[:50],
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"  [Attempt {attempt}/{retries}] Write error: {e}")
            resume_pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            time.sleep(attempt * 2)
            continue

        if expected:
            algo, digest = expected
            got = (md5_file(tmp) if algo == "md5" else sha256_file(tmp)).lower()
            if got != digest.lower():
                print(f"  [Attempt {attempt}/{retries}] {algo} mismatch, restarting file...")
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


def esgf_search(facets, search_type="File", page_size=500, timeout=90, verify=True,
                facet_fields=None):
    """Query INDEX_NODES for input4MIPs records.

    Yields file/dataset docs, de-duplicated across nodes by `title` (the
    filename) so ESGF replicas of the same file are only downloaded once.
    A node that errors repeatedly is skipped rather than aborting the run.
    If facet_fields is given, returns the merged facet counts dict instead.
    """
    params_base = {
        "type": search_type,
        "format": "application/solr+json",
        "limit": str(page_size),
        "distrib": "true",
    }
    params_base.update({k: first_str(v) for k, v in facets.items() if v is not None})
    if facet_fields:
        params_base.update({"facets": ",".join(facet_fields), "limit": "0"})

    bare = requests.Session()
    bare.verify = verify

    merged_facets: dict = {}
    seen_titles: set = set()

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

            if facet_fields:
                ff = data.get("facet_counts", {}).get("facet_fields", {})
                for key, flat in ff.items():
                    bucket = merged_facets.setdefault(key, {})
                    for name, count in zip(flat[::2], flat[1::2]):
                        # facets are per-node views of the same federation;
                        # take the max rather than summing replica counts
                        bucket[name] = max(bucket.get(name, 0), count)
                break

            resp = data.get("response", {})
            docs = resp.get("docs", [])
            if not docs:
                break

            for d in docs:
                title = first_str(d.get("title"))
                if title and title in seen_titles:
                    continue
                if title:
                    seen_titles.add(title)
                yield d

            num_found = int(first_str(resp.get("numFound")) or "0")
            offset += len(docs)
            if offset >= num_found:
                break
            time.sleep(0.3)

    if facet_fields:
        return merged_facets


def get_facets(facets, facet_fields, timeout=90, verify=True):
    """Non-generator wrapper for facet-count queries."""
    gen = esgf_search(facets, search_type="Dataset", timeout=timeout,
                      verify=verify, facet_fields=facet_fields)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value or {}


def doc_checksum(doc):
    """Return (algo, digest) if ESGF advertises one we can verify, else None."""
    csum = first_str(doc.get("checksum"))
    ctyp = (first_str(doc.get("checksum_type")) or "").lower()
    if csum and ctyp in ("md5", "sha256"):
        return (ctyp, csum)
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_sources(args, verify):
    """Print what CMIP7 emissions sources/variables ESGF currently holds."""
    print("Querying ESGF for published CMIP7 input4MIPs emissions ...\n")
    ff = get_facets({"project": "input4MIPs", "mip_era": "CMIP7"},
                    ["source_id", "target_mip", "grid_label", "frequency"],
                    timeout=args.timeout, verify=verify)

    srcs = ff.get("source_id", {})
    emissions = {s: c for s, c in srcs.items()
                 if c and (s.startswith("CEDS") or s.startswith("IIASA")
                           or "BB4CMIP" in s)}
    print("Emissions source_ids (mip_era=CMIP7):")
    for s, c in sorted(emissions.items()):
        mark = "  <-- in DEFAULT_SOURCES" if s in DEFAULT_SOURCES else ""
        print(f"  {s:34s} {c:5d} datasets{mark}")

    print("\ntarget_mip:", ", ".join(f"{k}({v})" for k, v in sorted(ff.get("target_mip", {}).items()) if v))
    print("grid_label:", ", ".join(f"{k}({v})" for k, v in sorted(ff.get("grid_label", {}).items()) if v))

    print("\nVariables available per default source:")
    for src in args.sources:
        vf = get_facets({"project": "input4MIPs", "mip_era": "CMIP7", "source_id": src},
                        ["variable_id"], timeout=args.timeout, verify=verify)
        got = sorted(v for v, c in vf.get("variable_id", {}).items()
                     if c and any(v.startswith(sp + "_") for sp in DEFAULT_SPECIES))
        print(f"  {src}:")
        print(f"    {', '.join(got) if got else '(none matching BC/SO2/CO2)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Download CMIP7 BC/SO2/CO2 gridded emissions from ESGF input4MIPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                    help="input4MIPs source_id values (historical CEDS and/or "
                         "IIASA-IAMC-<scen> scenarios). See --list-sources.")
    ap.add_argument("--species", nargs="+", default=DEFAULT_SPECIES,
                    help="Emission species prefixes, e.g. BC SO2 CO2 OC NOx NH3. "
                         "Note: sulfur is SO2 -- there is no 'SUL' in input4MIPs.")
    ap.add_argument("--kinds", nargs="+", default=["anthro"],
                    choices=sorted(KIND_SUFFIX),
                    help="Emission kind(s) fetched for EVERY species: anthro "
                         "(surface), openburning, air.")
    ap.add_argument("--air-species", nargs="+", default=["CO2"],
                    help="Species to additionally fetch _em_AIR_anthro (aircraft) "
                         "for. data/make_co2_files.py sums AIR+anthro for CO2 but "
                         "make_aerosol_files.py uses surface anthro ONLY for "
                         "SO2/BC, so the default mirrors that. AIR files carry a "
                         "level dim and are large (~5.5 GB for CO2). Pass 'none' "
                         "to skip.")
    ap.add_argument("--grid-label", default="gn",
                    help="gn = 0.5deg global monthly (recommended); "
                         "gr = 0.1deg regridded, 1980-2023 only.")
    ap.add_argument("--frequency", default="mon")
    ap.add_argument("--layout", default="nested", choices=("nested", "flat"),
                    help="nested = <outdir>/CMIP7/<target_mip>/<source_id>/<var>/; "
                         "flat = all files straight into <outdir> (what "
                         "data/make_{aerosol,co2}_files.py glob for).")
    ap.add_argument("--list-sources", action="store_true",
                    help="Print what ESGF currently publishes and exit.")
    ap.add_argument("--discover-only", action="store_true",
                    help="List what would be downloaded, with sizes, and exit.")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--no-checksum", action="store_true",
                    help="Skip checksum verification (faster, less safe).")
    args = ap.parse_args()

    verify = not args.insecure

    if args.list_sources:
        list_sources(args, verify)
        return 0

    session = make_session(timeout=args.timeout, verify=verify)

    variables = [f"{sp}{KIND_SUFFIX[k]}" for sp in args.species for k in args.kinds]
    # Aircraft emissions only for the species that actually consume them (CO2).
    air_species = [] if args.air_species == ["none"] else args.air_species
    for sp in air_species:
        if sp not in args.species:
            continue
        v = f"{sp}{KIND_SUFFIX['air']}"
        if v not in variables:
            variables.append(v)

    print(f"mip_era   : CMIP7 (project=input4MIPs)")
    print(f"sources   : {', '.join(args.sources)}")
    print(f"variables : {', '.join(variables)}")
    print(f"grid/freq : {args.grid_label} / {args.frequency}")
    print(f"outdir    : {args.outdir}  (layout={args.layout})\n")

    downloaded = skipped = failed = 0
    planned_bytes = 0
    plan = []

    for src in args.sources:
        print(f"\n{'='*66}\nSource: {src}\n{'='*66}")
        for variable_id in variables:
            facets = {
                "project": "input4MIPs",
                "mip_era": "CMIP7",
                "source_id": src,
                "variable_id": variable_id,
                "grid_label": args.grid_label,
                "frequency": args.frequency,
            }
            docs = list(esgf_search(facets, timeout=args.timeout, verify=verify))
            if not docs:
                print(f"  {variable_id}: no files found "
                      f"(grid_label={args.grid_label}, frequency={args.frequency})")
                continue

            target_mip = first_str(docs[0].get("target_mip")) or "unknown"
            print(f"  {variable_id}: {len(docs)} file(s), target_mip={target_mip}")

            for doc in docs:
                href = preferred_url(doc)
                if not href:
                    print(f"    WARNING: no HTTP URL for {first_str(doc.get('title'))}")
                    failed += 1
                    continue
                fname = os.path.basename(urlparse(href).path)
                if args.layout == "flat":
                    outpath = os.path.join(args.outdir, fname)
                else:
                    outpath = os.path.join(args.outdir, "CMIP7", target_mip, src,
                                           variable_id, fname)
                size = int(doc.get("size") or 0)
                checksum = None if args.no_checksum else doc_checksum(doc)
                plan.append((href, outpath, checksum, size))
                planned_bytes += size

    print(f"\n{'='*66}")
    print(f"Planned: {len(plan)} files, {planned_bytes/1e9:.2f} GB")
    print(f"{'='*66}\n")

    if args.discover_only:
        for _, outpath, _, size in plan:
            state = "HAVE" if os.path.exists(outpath) else "GET "
            print(f"  {state} {size/1e6:8.1f} MB  {outpath}")
        return 0

    for href, outpath, checksum, _ in plan:
        if os.path.exists(outpath):
            print(f"  Already exists: {os.path.basename(outpath)}")
            skipped += 1
            continue
        ok = stream_download(session, href, outpath, expected=checksum,
                             retries=args.retries, timeout=args.timeout, verify=verify)
        if ok:
            downloaded += 1
        else:
            failed += 1

    print(f"\n{'='*66}")
    print(f"Done. Downloaded: {downloaded}  Skipped: {skipped}  Failed: {failed}")
    print(f"{'='*66}")

    # Fail loudly so a wrapper/retry loop can see it (the LENS2 downloader's
    # silent-partial-failure mode cost us 268 missing PRECT files once).
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
