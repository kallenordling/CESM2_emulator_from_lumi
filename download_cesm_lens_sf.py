#!/usr/bin/env python3
"""
Download script for CESM2 Single Forcing Large Ensemble Project (d651055)
Dataset page:  https://gdex.ucar.edu/datasets/d651055/
OSDF base:     https://osdf-director.osg-htc.org/ncar/gdex/d651055/

Ensembles & members:
  AAER  - Only anthropogenic aerosols evolving      (members 001-020)
  BMB   - Only biomass burning aerosols evolving    (members 001-015)
  EE    - Everything else evolving                  (members 101-115)  <- note 1XX prefix
  GHG   - Only greenhouse gases evolving            (members 001-015)
  xAER  - Everything except anthropogenic aerosols  (members 001-010, starts 1920)

Period breakdown (monthly tseries files on OSDF):
  Historical  : 1850-2014  (xAER: 1920-2014)
  SSP370      : 2015-2050

Usage examples:
  # See what would be downloaded (no actual download)
  python download_cesm2_sf_lens.py --ensemble GHG AAER --variable TREFHT --dry-run

  # Download TREFHT for all GHG and AAER members
  python download_cesm2_sf_lens.py --ensemble GHG AAER --variable TREFHT \\
      --output /scratch/project_462001112/emulator_data/sf

  # Download multiple variables for specific members only
  python download_cesm2_sf_lens.py --ensemble GHG AAER BMB EE --variable TREFHT TS PRECL \\
      --members 001 002 003 --output /scratch/project_462001112/emulator_data/sf

  # Historical only
  python download_cesm2_sf_lens.py --ensemble GHG --variable TREFHT --no-ssp --output ./data

  # Check which URLs exist before downloading (HEAD requests, no data transfer)
  python download_cesm2_sf_lens.py --ensemble GHG --variable TREFHT --check-urls

  # 4 parallel workers
  python download_cesm2_sf_lens.py --ensemble GHG AAER --variable TREFHT --workers 4 --output ./data

Requirements:  pip install requests tqdm
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
    sys.exit("Missing 'requests'. Run:  pip install requests")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Dataset knowledge  (derived from CESM project page + provided sample URL)
# ---------------------------------------------------------------------------

OSDF_BASE = "https://osdf-director.osg-htc.org/ncar/gdex/d651055/CESM2-SF"

# Monthly files are split into 50-year chunks.
# Historical: 1850-2014 in four chunks; xAER starts 1920 so three chunks.
# SSP370:     2015-2050 as one chunk.
HIST_CHUNKS_1850 = [("185001", "189912"),
                    ("190001", "194912"),
                    ("195001", "199912"),
                    ("200001", "201412")]
HIST_CHUNKS_1920 = [("192001", "194912"),   # xAER starts here
                    ("195001", "199912"),
                    ("200001", "201412")]
SSP_CHUNKS       = [("201501", "205012")]

ENSEMBLES = {
    # key: (compset_hist, sim_label_hist, sim_label_ssp, n_members, member_fn, hist_chunks)
    # For xAER SSP the compset changes to BSSP370cmip6
    "GHG": {
        "compset_hist": "B1850cmip6",
        "sim_hist":     "CESM2-SF-GHG",
        "sim_ssp":      "CESM2-SF-GHG-SSP370",
        "compset_ssp":  "B1850cmip6",   # same compset prefix in SSP filename
        "n_members":    15,
        "member_fn":    lambda i: f"{i:03d}",   # 001-015
        "hist_chunks":  HIST_CHUNKS_1850,
        "ssp_chunks":   SSP_CHUNKS,
    },
    "AAER": {
        "compset_hist": "B1850cmip6",
        "sim_hist":     "CESM2-SF-AAER",
        "sim_ssp":      "CESM2-SF-AAER-SSP370",
        "compset_ssp":  "B1850cmip6",
        "n_members":    20,
        "member_fn":    lambda i: f"{i:03d}",   # 001-020
        "hist_chunks":  HIST_CHUNKS_1850,
        "ssp_chunks":   SSP_CHUNKS,
    },
    "BMB": {
        "compset_hist": "B1850cmip6",
        "sim_hist":     "CESM2-SF-BMB",
        "sim_ssp":      "CESM2-SF-BMB-SSP370",
        "compset_ssp":  "B1850cmip6",
        "n_members":    15,
        "member_fn":    lambda i: f"{i:03d}",   # 001-015
        "hist_chunks":  HIST_CHUNKS_1850,
        "ssp_chunks":   SSP_CHUNKS,
    },
    "EE": {
        "compset_hist": "B1850cmip6",
        "sim_hist":     "CESM2-SF-EE",
        "sim_ssp":      "CESM2-SF-EE-SSP370",
        "compset_ssp":  "B1850cmip6",
        "n_members":    15,
        "member_fn":    lambda i: f"1{i:02d}",  # 101-115  <- note 1XX
        "hist_chunks":  HIST_CHUNKS_1850,
        "ssp_chunks":   SSP_CHUNKS,
    },
    "xAER": {
        "compset_hist": "BHISTcmip6",
        "sim_hist":     "CESM2-SF-xAER",
        "sim_ssp":      "CESM2-SF-xAER",
        "compset_ssp":  "BSSP370cmip6",  # different compset for SSP!
        "n_members":    10,
        "member_fn":    lambda i: f"{i:03d}",   # 001-010
        "hist_chunks":  HIST_CHUNKS_1920,        # starts 1920
        "ssp_chunks":   SSP_CHUNKS,
    },
}

COMPONENTS = {
    # component: (folder, model_short_name, stream)
    "atm": ("atm", "cam",    "h0"),
    "lnd": ("lnd", "clm2",   "h0"),
    "ice": ("ice", "cice",   "h"),
    "ocn": ("ocn", "pop",    "h"),
    "rof": ("rof", "mosart", "h0"),
}

# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def build_urls(ensemble, variable, members=None, component="atm",
               include_hist=True, include_ssp=True):
    """Build list of (url, filename) for given ensemble/variable/members."""
    cfg = ENSEMBLES[ensemble]
    comp_folder, model_short, stream = COMPONENTS[component]

    if members is None:
        members = [cfg["member_fn"](i) for i in range(1, cfg["n_members"] + 1)]

    results = []

    for member in members:
        # Historical
        if include_hist:
            for (t0, t1) in cfg["hist_chunks"]:
                fname = (f"b.e21.{cfg['compset_hist']}.f09_g17"
                         f".{cfg['sim_hist']}.{member}"
                         f".{model_short}.{stream}.{variable}.{t0}-{t1}.nc")
                url = f"{OSDF_BASE}/{comp_folder}/proc/tseries/month_1/{variable}/{fname}"
                results.append((url, fname))

        # SSP370
        if include_ssp:
            for (t0, t1) in cfg["ssp_chunks"]:
                fname = (f"b.e21.{cfg['compset_ssp']}.f09_g17"
                         f".{cfg['sim_ssp']}.{member}"
                         f".{model_short}.{stream}.{variable}.{t0}-{t1}.nc")
                url = f"{OSDF_BASE}/{comp_folder}/proc/tseries/month_1/{variable}/{fname}"
                results.append((url, fname))

    return results


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session(retries=6, backoff=2.0):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["HEAD", "GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers["User-Agent"] = "CESM2-SF-Downloader/2.0 (research)"
    return session


SESSION = make_session()


def head_check(url):
    """Return (exists: bool, size_bytes: int)."""
    try:
        r = SESSION.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            return True, int(r.headers.get("content-length", 0))
        return False, 0
    except Exception:
        return False, 0


def download_file(url, dest, overwrite=False, dry_run=False, verify=False):
    """Download url -> dest. Returns (ok: bool, message: str)."""
    if dest.exists() and not overwrite:
        if not verify:
            return True, f"SKIP (exists)  {dest.name}"
        # --verify: size is the only cheap integrity signal the archive offers
        # (no checksums published), and it catches exactly the truncation this
        # script used to create.
        _, remote = head_check(url)
        local = dest.stat().st_size
        if remote and local != remote:
            print(f"  [verify] {dest.name}: {local} != {remote} — refetching")
        else:
            return True, f"SKIP (verified) {dest.name}"
    if dry_run:
        return True, f"DRY-RUN        {url}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        with SESSION.get(url, stream=True, timeout=600) as r:
            if r.status_code == 404:
                return False, f"NOT FOUND      {url}"
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            if HAS_TQDM:
                bar = tqdm(total=total, unit="B", unit_scale=True,
                           desc=dest.name[:50], ncols=90, leave=False)
            with open(tmp, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if HAS_TQDM:
                        bar.update(len(chunk))
                    elif total:
                        print(f"\r  {dest.name}: {downloaded/total*100:.0f}%  ",
                              end="", flush=True)
            if HAS_TQDM:
                bar.close()
            elif total:
                print()

            # A stream can end early without raising — the server closes the
            # connection and iter_content simply stops. Without this check the
            # short file is renamed into place and reported "OK (4 MB)", which
            # is how 112 of 525 files ended up truncated on 2026-08-19: valid
            # HDF5 header, so they open, then fail deep in a read with
            # "NetCDF: HDF error". Compare against Content-Length and refuse.
            if total and downloaded != total:
                raise IOError(f"truncated: got {downloaded} of {total} bytes")

        tmp.rename(dest)
        return True, f"OK ({dest.stat().st_size/1e6:.0f} MB)  {dest.name}"

    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return False, f"ERROR          {dest.name}: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download CESM2 Single Forcing Large Ensemble (d651055) via OSDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ensemble",   nargs="+", required=True,
                   choices=list(ENSEMBLES.keys()),
                   help="Ensemble(s): GHG AAER BMB EE xAER")
    p.add_argument("--variable",   nargs="+", required=True,
                   help="Variable name(s) e.g. TREFHT TS PRECL FLNT")
    p.add_argument("--members",    nargs="+", default=None,
                   help="Member string(s) e.g. 001 002 101 (default: all)")
    p.add_argument("--component",  default="atm", choices=list(COMPONENTS.keys()),
                   help="Model component (default: atm)")
    p.add_argument("--no-hist",    action="store_true",
                   help="Skip historical period (1850/1920-2014)")
    p.add_argument("--no-ssp",     action="store_true",
                   help="Skip SSP370 period (2015-2050)")
    p.add_argument("--output", "-o", type=Path, default=Path("./cesm2_sf_lens"),
                   help="Output directory (default: ./cesm2_sf_lens)")
    p.add_argument("--flat",       action="store_true",
                   help="Save all files in a flat output dir (no ensemble/variable subdirs)")
    p.add_argument("--workers",    type=int, default=2,
                   help="Parallel download threads (default: 2)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print URLs without downloading")
    p.add_argument("--check-urls", action="store_true",
                   help="HEAD-check each URL and print size (no download)")
    p.add_argument("--list-only",  action="store_true",
                   help="Print file list and exit")
    p.add_argument("--verify", action="store_true",
                   help="check existing files against the server's "
                        "Content-Length and refetch any whose size differs. "
                        "Use after an interrupted run: a truncated file still "
                        "has a valid HDF5 header, so it is only caught when "
                        "something reads deep into it.")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-download already-existing files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.no_hist and args.no_ssp:
        sys.exit("ERROR: --no-hist and --no-ssp together leaves nothing to download.")

    include_hist = not args.no_hist
    include_ssp  = not args.no_ssp

    # Build full URL list
    all_files = []   # (url, fname, ensemble, variable)
    for ens in args.ensemble:
        for var in args.variable:
            for url, fname in build_urls(ens, var, args.members, args.component,
                                         include_hist, include_ssp):
                all_files.append((url, fname, ens, var))

    total = len(all_files)
    print(f"\nCESM2 Single Forcing Large Ensemble downloader")
    print(f"  Ensembles : {args.ensemble}")
    print(f"  Variables : {args.variable}")
    print(f"  Component : {args.component}")
    print(f"  Period    : {'hist ' if include_hist else ''}{'ssp370' if include_ssp else ''}")
    print(f"  Files     : {total}")
    print(f"  Output    : {args.output.resolve()}")
    print()

    if args.list_only:
        for i, (url, fname, ens, var) in enumerate(all_files, 1):
            print(f"[{i:4d}/{total}]  {fname}")
            print(f"           {url}")
        return

    if args.check_urls:
        print("HEAD-checking URLs ...\n")
        ok = missing = 0
        for url, fname, ens, var in all_files:
            exists, size = head_check(url)
            tag = f"OK   {size/1e6:7.0f} MB" if exists else "MISSING          "
            print(f"  [{tag}]  {fname}")
            if exists: ok += 1
            else:       missing += 1
        print(f"\n{ok} found, {missing} missing")
        return

    if args.dry_run:
        print("DRY RUN – no files will be written\n")

    def dest_path(fname, ens, var):
        if args.flat:
            return args.output / fname
        return args.output / ens / var / fname

    def _task(item):
        url, fname, ens, var = item
        ok, msg = download_file(url, dest_path(fname, ens, var),
                                overwrite=args.overwrite, dry_run=args.dry_run,
                                verify=args.verify)
        return ok, msg

    success = fail = 0
    if args.workers > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_task, item): item for item in all_files}
            done = 0
            for future in as_completed(futures):
                ok, msg = future.result()
                done += 1
                print(f"[{done:4d}/{total}] {'V' if ok else 'X'}  {msg}")
                if ok: success += 1
                else:  fail += 1
    else:
        for i, item in enumerate(all_files, 1):
            ok, msg = _task(item)
            print(f"[{i:4d}/{total}] {'V' if ok else 'X'}  {msg}")
            if ok: success += 1
            else:  fail += 1
            if not args.dry_run:
                time.sleep(0.1)

    print(f"\nDone: {success} succeeded, {fail} failed")

    if fail > 0:
        print("\nSome files were missing. Possible reasons:")
        print("  1. The chunk date boundaries differ slightly on OSDF")
        print("  2. Not all variables are available for all ensembles")
        print("  3. OSDF may require a redirect / token for large files")
        print()
        print("To inspect what's actually available for a variable, try:")
        print("  curl -sL 'https://osdf-director.osg-htc.org/ncar/gdex/d651055/CESM2-SF/atm/proc/tseries/month_1/TREFHT/' | grep -o 'href=\"[^\"]*\\.nc\"'")


if __name__ == "__main__":
    main()
