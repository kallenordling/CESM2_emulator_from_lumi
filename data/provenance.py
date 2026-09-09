"""Record in a generated file how it was made.

The cond files carry ZERO global attributes today, while the raw input4MIPs they
descend from carry 32 including `dataset_version_number` and `creation_date`.
Tracing the shipped `*_bc_co2fix.nc` back to a source therefore meant reading
file mtimes and script source, and it still could not settle which BC vintage
went in: the builder asks for CEDS-2017-05-18, only CEDS-CMIP-2025-04-18 is
present where it runs, and the file itself says nothing.

That ambiguity is worth ~37% of the historical BC signal, so this exists to
stop it recurring. `stamp()` writes, as CF-style global attributes:

    history / source        every input path, with the vintage read out of the
                            input's own metadata rather than parsed from its
                            filename
    creation_date           UTC, ISO
    created_by              script path and the git commit it ran at, so the
                            exact code is recoverable
    <script-specific>       whatever the caller passes as `extra` -- grid,
                            splice year, deflation factor, and so on

Use it at every to_netcdf in the cond pipeline. It is cheap, and the file
becomes self-describing instead of depending on someone remembering.
"""

import datetime as _dt
import os
import subprocess


# Attributes worth lifting from an input4MIPs source, in preference order.
_VINTAGE_KEYS = ("dataset_version_number", "source_id", "version",
                 "creation_date", "further_info_url")


def _git_commit(repo=None):
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        # A dirty tree means the recorded sha does NOT reproduce the file, so
        # say so rather than imply a clean provenance.
        return (sha + ("+dirty" if dirty.stdout.strip() else "")) if sha else "unknown"
    except Exception:
        return "unknown"


def describe_source(path, ds=None):
    """`path (key=value, ...)` using the source's OWN vintage metadata.

    Reading the vintage from the file rather than its name matters: the two
    CEDS vintages differ by ~37% post-1950 and a filename can be renamed,
    copied or staged into the wrong directory without the contents changing.
    """
    bits = []
    if ds is not None:
        for k in _VINTAGE_KEYS:
            v = ds.attrs.get(k)
            if v:
                bits.append(f"{k}={v}")
            if len(bits) >= 3:
                break
    return f"{path}" + (f" ({', '.join(bits)})" if bits else "")


def stamp(ds, script, sources, extra=None, note=None):
    """Attach provenance to `ds` in place and return it.

    ds      : xarray Dataset about to be written
    script  : the producing script, usually __file__
    sources : list of input paths, or (path, open_dataset) pairs so the
              vintage can be read out of the input itself
    extra   : dict of run-specific facts (grid, splice year, factors)
    note    : one line on WHY, for a reader who has only the file
    """
    src = []
    for s in sources:
        if isinstance(s, (tuple, list)):
            src.append(describe_source(s[0], s[1]))
        else:
            src.append(describe_source(s))

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rel = os.path.relpath(os.path.abspath(script),
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds.attrs["creation_date"] = now
    ds.attrs["created_by"] = f"{rel} @ git {_git_commit()}"
    ds.attrs["source"] = "\n".join(src)
    ds.attrs["history"] = (f"{now}: created by {rel} from {len(src)} source(s)"
                           + (f"; {ds.attrs['history']}" if "history" in ds.attrs else ""))
    if note:
        ds.attrs["comment"] = note
    for k, v in (extra or {}).items():
        ds.attrs[str(k)] = str(v)
    return ds
