#!/usr/bin/env python3
"""
Download the CEDS-2017-05-18 historical BC emissions from ESGF input4MIPs.

WHY THIS EXISTS
---------------
The BC conditioning channel has a +37.6% discontinuity at 2015 because its two
halves come from different CEDS vintages:

    BC  historical : CEDS-CMIP-2025-04-18   <- eight years newer
    BC  scenarios  : IAMC-*-ssp*-1-1        <- harmonised to CEDS-2017
    SO2 historical : CEDS-2017-05-18        <- matches its scenarios, so SO2 is fine

Measured directly on the 2015-2023 overlap where both sources exist: CEDS-2025
gives 5.730 Tg/yr global anthropogenic BC in 2015, the IAMC SSP files give
7.986 — a ratio of 1.394, which is the jump seen in the cond files. It is not a
constant offset either: by 2020 the ssp370 ratio has grown to 1.63, because
CEDS-2025 records the observed BC decline while the 2017-era SSP projections
assumed continued growth.

This script fetches the CEDS-2017-05-18 BC files so the historical can be
rebuilt on the same vintage the scenarios were harmonised to, exactly as SO2
already is.

AFTER DOWNLOADING
-----------------
Point data/make_aerosol_files.py at them (~line 46):

    ("BC", "hist"): "BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc"

and drop the CLIP_HIST_2014 special case at ~line 56 — it exists only because
CEDS-2025 runs to 2023, whereas CEDS-2017 already ends at 2014, with the same
file-level year ranges as the SO2 set.

CHECK BEFORE REBUILDING: make_aerosol_files.py sums over the `sector` dimension.
If CEDS-2017 and CEDS-2025 disagree on the number or order of sectors, that sum
changes silently. --verify-sectors compares the downloaded files against the SO2
CEDS-2017 file already on disk and reports any mismatch.

REGENERATING THE BC CHANNEL CHANGES ALL 165 HISTORICAL YEARS, not just the
junction, so it implies retraining. run_mseyb_BCprect_490 — the checkpoint the
paper figures rest on — was trained on the CEDS-2025 version.

WHY THE SEARCH LOOKS ODD
------------------------
CEDS-2017 datasets are published as bundles with variable_id "Multiple", so the
obvious query (variable_id=BC_em_anthro) returns NOTHING for them even though
the files are right there. The search therefore runs at type=File with a
free-text query and filters on the filename. Also note esgf-node.ornl.gov
answers with non-JSON (it is the 1.5-bridge) and is not in INDEX_NODES.

Usage
-----
    python scripts/download_ceds2017_bc.py --discover-only     # list, download nothing
    python scripts/download_ceds2017_bc.py                     # ~3.6 GB
    python scripts/download_ceds2017_bc.py --verify-sectors
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_input4mips_cmip7 import (        # noqa: E402  - path set above
    doc_checksum, esgf_search, first_str, make_session, preferred_url,
    stream_download,
)

try:
    import lumi_paths as L
    DEFAULT_OUT = f"{L.DATA}/emission_data/inputs4mips"
except Exception:                              # off-cluster / no mount
    DEFAULT_OUT = "emission_data/inputs4mips"

SOURCE_ID = "CEDS-2017-05-18"
PREFIX = "BC-em-anthro"
EXPECT_N = 7          # 1750-1799, 1800-1849, 1850, 1851-1899, 1900-1949,
                      # 1950-1999, 2000-2014 — same split as the SO2 set


def discover(verify=True):
    """File-level search, filtered on filename.

    A bare free-text query also matches records that merely MENTION the string,
    so the title check is load-bearing rather than belt-and-braces: the raw
    query returns ~114 hits, of which 7 are actually BC-em-anthro files.
    """
    docs = list(esgf_search({"project": "input4MIPs", "source_id": SOURCE_ID,
                             "query": PREFIX}, search_type="File", verify=verify))
    hits = [d for d in docs if (first_str(d.get("title")) or "").startswith(PREFIX)]
    hits.sort(key=lambda d: first_str(d.get("title")) or "")
    return hits


def verify_sectors(outdir):
    """Compare the sector dimension against the SO2 CEDS-2017 file on disk.

    make_aerosol_files.py sums over `sector`; a differing sector set changes
    that total without any error, which is the failure mode most likely to
    survive unnoticed into a training run.
    """
    import glob
    try:
        import xarray as xr
    except ImportError:
        print("[sectors] xarray not available — skipping", file=sys.stderr)
        return
    bc = sorted(glob.glob(os.path.join(outdir, f"{PREFIX}_*{SOURCE_ID}*.nc")))
    so2 = sorted(glob.glob(os.path.join(outdir, f"SO2-em-anthro_*{SOURCE_ID}*.nc")))
    if not bc:
        print("[sectors] no BC CEDS-2017 files downloaded yet", file=sys.stderr)
        return
    if not so2:
        print("[sectors] no SO2 CEDS-2017 file on disk to compare against",
              file=sys.stderr)
        return

    def sectors(p):
        with xr.open_dataset(p, decode_times=False) as ds:
            if "sector" not in ds.dims:
                return None, None
            n = int(ds.sizes["sector"])
            names = ds["sector"].attrs.get("ids") or ds["sector"].attrs.get("long_name")
            return n, names

    nb, namb = sectors(bc[0])
    ns, nams = sectors(so2[0])
    print(f"[sectors] BC  {os.path.basename(bc[0])}: {nb} sectors")
    print(f"[sectors] SO2 {os.path.basename(so2[0])}: {ns} sectors")
    if nb != ns:
        print(f"[sectors] MISMATCH: {nb} vs {ns}. make_aerosol_files.py sums over "
              f"sector, so the BC and SO2 channels would not be comparable. "
              f"Resolve before rebuilding.", file=sys.stderr)
    else:
        print("[sectors] match — the sector sum is comparable to the SO2 channel")
        if namb and nams and str(namb) != str(nams):
            print(f"[sectors] note: sector LABELS differ despite equal counts:\n"
                  f"          BC : {namb}\n          SO2: {nams}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=DEFAULT_OUT,
                    help="flat directory make_aerosol_files.py globs")
    ap.add_argument("--discover-only", action="store_true",
                    help="list what would be fetched and exit")
    ap.add_argument("--verify-sectors", action="store_true",
                    help="compare the sector dim against the SO2 CEDS-2017 file")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-verify-ssl", action="store_true",
                    help="some ESGF data nodes have expired certificates")
    args = ap.parse_args()
    verify = not args.no_verify_ssl

    print(f"[ceds2017-bc] searching ESGF for {PREFIX} / {SOURCE_ID} …")
    hits = discover(verify=verify)
    if not hits:
        print("[ceds2017-bc] nothing found. The CEDS-2017 datasets are published "
              "with variable_id 'Multiple', so this searches at file level — if "
              "that changed, check the query.", file=sys.stderr)
        return 1

    total = sum(float(first_str(d.get("size")) or 0) for d in hits)
    print(f"[ceds2017-bc] {len(hits)} file(s), {total / 1e9:.2f} GB")
    for d in hits:
        print(f"   {first_str(d.get('title'))}  "
              f"{float(first_str(d.get('size')) or 0) / 1e6:8.1f} MB")
    if len(hits) != EXPECT_N:
        print(f"[ceds2017-bc] NOTE: expected {EXPECT_N} files (the SO2 CEDS-2017 "
              f"split), got {len(hits)}. Check the list above covers 1750-2014 "
              f"with no gap before rebuilding the channel.", file=sys.stderr)

    if args.discover_only:
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    session = make_session(timeout=args.timeout, verify=verify)
    ok = failed = skipped = 0
    for d in hits:
        name = first_str(d.get("title"))
        dest = os.path.join(args.outdir, name)
        size = float(first_str(d.get("size")) or 0)
        if os.path.exists(dest) and abs(os.path.getsize(dest) - size) < 1024:
            print(f"  [have] {name}")
            skipped += 1
            continue
        url = preferred_url(d)
        if not url:
            print(f"  [no url] {name}", file=sys.stderr)
            failed += 1
            continue
        print(f"  [get ] {name}  ({size / 1e6:.1f} MB)")
        if stream_download(session, url, dest, expected=doc_checksum(d),
                           retries=args.retries, timeout=args.timeout,
                           verify=verify):
            ok += 1
        else:
            failed += 1

    print(f"\n[ceds2017-bc] downloaded {ok}, already present {skipped}, failed {failed}")
    print(f"[ceds2017-bc] outdir: {args.outdir}")
    if args.verify_sectors:
        verify_sectors(args.outdir)
    if failed == 0:
        print("\n[ceds2017-bc] next: point data/make_aerosol_files.py at these\n"
              "  (\"BC\", \"hist\"): "
              "\"BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc\"\n"
              "and drop CLIP_HIST_2014 — CEDS-2017 already ends at 2014.\n"
              "Rebuilding changes all 165 historical years, so retraining follows.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
