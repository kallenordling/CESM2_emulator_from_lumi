#!/usr/bin/env python3
"""
Download CESM2 Large Ensemble (LENS2, d651056) monthly timeseries.
OSDF base: https://osdf-director.osg-htc.org/ncar/gdex/d651056/CESM2-LE/

Files are chunked in 10-year increments:
  Historical : 185001-185912, 186001-186912, ..., 201001-201412  (17 chunks)
  SSP370     : 201501-202412, 202501-203412, ..., 209501-210012   (9 chunks)

LENS2 100-member structure:
  CMIP6 sub-ensemble (50 members, compsets BHISTcmip6 / BSSP370cmip6):
    LE2-1001.001 ... LE2-1181.010  (10 members, unique seeds 1001,1021,...,1181)
    LE2-1231.001 ... LE2-1301.010  (40 members, seeds 1231/1251/1281/1301, idx 001-010)

  SMBB sub-ensemble (50 members, compsets BHISTsmbb / BSSP370smbb):
    LE2-1011.001 ... LE2-1191.010  (10 members, unique seeds 1011,1031,...,1191)
    LE2-1231.011 ... LE2-1301.020  (40 members, seeds 1231/1251/1281/1301, idx 011-020)

Default: 30 members (15 CMIP6 + 15 SMBB) for maximum diversity.

Usage:
  python download_lens2.py --dry-run
  python download_lens2.py --output /scratch/project_462001328/emulator_data/lens2
  python download_lens2.py --variable TREFHT PRECT --output /scratch/.../lens2 --workers 4
  python download_lens2.py --subensemble smbb --n-members 30
  python download_lens2.py --check-urls --n-members 2

Requirements: pip install requests tqdm
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("Missing 'requests'. Run: pip install requests")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# LENS2 dataset constants
# ---------------------------------------------------------------------------

OSDF_BASE = "https://osdf-director.osg-htc.org/ncar/gdex/d651056/CESM2-LE"


def _decade_chunks(start_year, end_year):
    """Generate 10-year (YYYYMM, YYYYMM) chunk tuples inclusive of end_year."""
    chunks = []
    y = start_year
    while y <= end_year:
        y1 = min(y + 9, end_year)
        chunks.append((f"{y}01", f"{y1}12"))
        y += 10
    return chunks


HIST_CHUNKS = _decade_chunks(1850, 2014)   # 17 chunks; last = 201001-201412
SSP_CHUNKS  = _decade_chunks(2015, 2100)   #  9 chunks; last = 209501-210012


# ---------------------------------------------------------------------------
# Full 100-member table
# Each entry: (compset_hist, compset_ssp, seed, idx, subensemble)
# ---------------------------------------------------------------------------

def _all_members():
    members = []

    # CMIP6: group A — unique seeds, idx 001-010
    for i, seed in enumerate([1001,1021,1041,1061,1081,1101,1121,1141,1161,1181], start=1):
        members.append(("BHISTcmip6", "BSSP370cmip6", str(seed), f"{i:03d}", "cmip6"))
    # CMIP6: groups B-E — shared seeds, idx 001-010
    for seed in [1231, 1251, 1281, 1301]:
        for idx in range(1, 11):
            members.append(("BHISTcmip6", "BSSP370cmip6", str(seed), f"{idx:03d}", "cmip6"))

    # SMBB: group A — unique seeds, idx 001-010
    for i, seed in enumerate([1011,1031,1051,1071,1091,1111,1131,1151,1171,1191], start=1):
        members.append(("BHISTsmbb", "BSSP370smbb", str(seed), f"{i:03d}", "smbb"))
    # SMBB: groups B-E — shared seeds, idx 011-020
    for seed in [1231, 1251, 1281, 1301]:
        for idx in range(11, 21):
            members.append(("BHISTsmbb", "BSSP370smbb", str(seed), f"{idx:03d}", "smbb"))

    assert len(members) == 100
    return members


ALL_MEMBERS = _all_members()


def select_members(n, subensemble="mixed"):
    cmip6 = [m for m in ALL_MEMBERS if m[4] == "cmip6"]
    smbb  = [m for m in ALL_MEMBERS if m[4] == "smbb"]

    if subensemble == "cmip6":
        pool = cmip6
    elif subensemble == "smbb":
        pool = smbb
    else:  # mixed
        half = n // 2
        return cmip6[:half] + smbb[:n - half]

    if n > len(pool):
        sys.exit(f"ERROR: Only {len(pool)} {subensemble} members available, requested {n}")
    return pool[:n]


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def build_urls(members, variable, include_hist=True, include_ssp=True):
    results = []
    for compset_hist, compset_ssp, seed, idx, _ in members:
        label = f"LE2-{seed}.{idx}"
        if include_hist:
            for t0, t1 in HIST_CHUNKS:
                fname = f"b.e21.{compset_hist}.f09_g17.{label}.cam.h0.{variable}.{t0}-{t1}.nc"
                results.append((f"{OSDF_BASE}/atm/proc/tseries/month_1/{variable}/{fname}",
                                 fname, label))
        if include_ssp:
            for t0, t1 in SSP_CHUNKS:
                fname = f"b.e21.{compset_ssp}.f09_g17.{label}.cam.h0.{variable}.{t0}-{t1}.nc"
                results.append((f"{OSDF_BASE}/atm/proc/tseries/month_1/{variable}/{fname}",
                                 fname, label))
    return results


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session():
    session = requests.Session()
    retry = Retry(total=6, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["HEAD", "GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://",  HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "CESM2-LENS2-Downloader/1.0"
    return session


SESSION = make_session()


def head_check(url):
    try:
        r = SESSION.head(url, timeout=30, allow_redirects=True)
        return r.status_code == 200, int(r.headers.get("content-length", 0))
    except Exception:
        return False, 0


def download_file(url, dest, overwrite=False, dry_run=False):
    if dest.exists() and not overwrite:
        return True, f"SKIP (exists)  {dest.name}"
    if dry_run:
        return True, f"DRY-RUN  {url}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with SESSION.get(url, stream=True, timeout=600) as r:
            if r.status_code == 404:
                return False, f"NOT FOUND  {url}"
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            bar = tqdm(total=total, unit="B", unit_scale=True,
                       desc=dest.name[:50], ncols=90, leave=False) if HAS_TQDM else None
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    if bar: bar.update(len(chunk))
            if bar: bar.close()
        tmp.rename(dest)
        return True, f"OK ({dest.stat().st_size/1e6:.0f} MB)  {dest.name}"
    except Exception as e:
        if tmp.exists(): tmp.unlink()
        return False, f"ERROR  {dest.name}: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download CESM2 LENS2 variables (10-yr chunks, hist + SSP370)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable",     nargs="+", default=["TREFHT", "PRECT"],
                   help="Variable(s) to download (default: TREFHT PRECT)")
    p.add_argument("--output", "-o", type=Path, default=Path("./lens2_data"))
    p.add_argument("--n-members",    type=int,  default=30,
                   help="Members to download (default: 30)")
    p.add_argument("--subensemble",  choices=["cmip6","smbb","mixed"], default="mixed",
                   help="Sub-ensemble: cmip6 | smbb | mixed (default: mixed)")
    p.add_argument("--no-hist",      action="store_true")
    p.add_argument("--no-ssp",       action="store_true")
    p.add_argument("--workers",      type=int, default=2)
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--check-urls",   action="store_true")
    p.add_argument("--list-only",    action="store_true")
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--flat",         action="store_true",
                   help="No member subdirs — all files in output/LENS2/<variable>/")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_hist and args.no_ssp:
        sys.exit("ERROR: nothing to download (both --no-hist and --no-ssp set)")

    include_hist = not args.no_hist
    include_ssp  = not args.no_ssp
    members      = select_members(args.n_members, args.subensemble)

    # Build file list across all variables
    all_files = []   # (url, fname, label, variable)
    for variable in args.variable:
        for url, fname, label in build_urls(members, variable, include_hist, include_ssp):
            all_files.append((url, fname, label, variable))

    total  = len(all_files)
    est_gb = total * 14 / 1024

    print(f"\nCESM2 LENS2 downloader")
    print(f"  Variables : {args.variable}")
    print(f"  Members   : {args.n_members} ({args.subensemble})")
    print(f"  Hist      : {len(HIST_CHUNKS)} chunks × 10yr (1850-2014)" if include_hist else "  Hist      : skipped")
    print(f"  SSP370    : {len(SSP_CHUNKS)} chunks × 10yr (2015-2100)"  if include_ssp  else "  SSP370    : skipped")
    print(f"  Files     : {total}  (~{est_gb:.1f} GB)")
    print(f"  Output    : {args.output.resolve()}\n")

    if args.list_only:
        for i, (url, fname, _, variable) in enumerate(all_files, 1):
            print(f"[{i:4d}/{total}]  [{variable}]  {fname}")
        return

    if args.check_urls:
        ok = missing = 0
        for url, fname, _, variable in all_files:
            found, size = head_check(url)
            tag = f"OK  {size/1e6:6.1f} MB" if found else "MISSING      "
            print(f"  [{tag}]  [{variable}]  {fname}")
            ok += found; missing += (not found)
        print(f"\n{ok} found, {missing} missing")
        return

    if args.dry_run:
        print("DRY RUN — no files written\n")
        for _, fname, _, variable in all_files:
            print(f"  [{variable}]  {fname}")
        return

    def dest(fname, label, variable):
        base = args.output / "LENS2" / variable
        return base / fname if args.flat else base / label / fname

    def _task(item):
        url, fname, label, variable = item
        return download_file(url, dest(fname, label, variable), args.overwrite)

    success = fail = 0
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futs = {exe.submit(_task, item): item for item in all_files}
            done = 0
            for f in as_completed(futs):
                ok, msg = f.result(); done += 1
                print(f"[{done:4d}/{total}] {'V' if ok else 'X'}  {msg}")
                success += ok; fail += (not ok)
    else:
        for i, item in enumerate(all_files, 1):
            ok, msg = _task(item)
            print(f"[{i:4d}/{total}] {'V' if ok else 'X'}  {msg}")
            success += ok; fail += (not ok)
            time.sleep(0.05)

    print(f"\nDone: {success} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
