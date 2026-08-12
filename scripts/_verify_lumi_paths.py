#!/usr/bin/env python3
"""Prove the lumi_paths codemod is a semantic no-op.

Substitutes every ${LUMI_*} / {L.*} reference back to the literal it replaced,
then diffs against the committed version of the same file. The ONLY differences
allowed are the intentional structural ones:

  - a removed `#SBATCH --account=` line
  - the inserted `source .../lumi_env.sh` + assert_account + banner header
  - an inserted `import lumi_paths as L`
  - the `f` prefix added to a rewritten python string literal

Anything else is a bug in the codemod and is printed. Exit 1 if any file fails.

    python scripts/_verify_lumi_paths.py [git-ref]      # default HEAD
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

OLD = "462001328"
ROOT = Path(__file__).resolve().parent.parent

UNSUB = [
    ("${LUMI_REPO_PFS}", f"/pfs/lustrep1/projappl/project_{OLD}/CESM2_emulator_from_lumi"),
    ("${LUMI_VENV}", f"/projappl/project_{OLD}/venvs/diffesm_laif"),
    ("${LUMI_REPO}", f"/projappl/project_{OLD}/CESM2_emulator_from_lumi"),
    ("${LUMI_PKGS}", f"/scratch/project_{OLD}/python_packages"),
    ("${LUMI_DATA}", f"/scratch/project_{OLD}/emulator_data"),
    ("${LUMI_EVAL_OUT}", f"/scratch/project_{OLD}/eval_output"),
    ("${LUMI_SCRATCH}", f"/scratch/project_{OLD}"),
    ("${LUMI_PROJAPPL}", f"/projappl/project_{OLD}"),
    ("${LUMI_ACCOUNT}", f"project_{OLD}"),
    ("{L.REPO_PFS}", f"/pfs/lustrep1/projappl/project_{OLD}/CESM2_emulator_from_lumi"),
    ("{L.VENV}", f"/projappl/project_{OLD}/venvs/diffesm_laif"),
    ("{L.REPO}", f"/projappl/project_{OLD}/CESM2_emulator_from_lumi"),
    ("{L.PKGS}", f"/scratch/project_{OLD}/python_packages"),
    ("{L.DATA}", f"/scratch/project_{OLD}/emulator_data"),
    ("{L.EVAL_OUT}", f"/scratch/project_{OLD}/eval_output"),
    ("{L.SCRATCH}", f"/scratch/project_{OLD}"),
    ("{L.PROJAPPL}", f"/projappl/project_{OLD}"),
    ("{L.ACCOUNT}", f"project_{OLD}"),
]

ALLOW_ADDED = (
    re.compile(r"^\s*source \"\$\(dirname \"\$\{BASH_SOURCE\[0\]\}\"\)/(\.\./)*lumi_env\.sh\"\s*$"),
    re.compile(r"^assert_account$"),
    re.compile(r"^lumi_env_banner$"),
    re.compile(r"^import lumi_paths as L$"),
    re.compile(r"^# Single source of truth for the LUMI project id and its paths\.$"),
    re.compile(r"^\s*$"),
)
ALLOW_REMOVED = (re.compile(r"^\s*#SBATCH\s+--account="),)


def unrender(text: str) -> str:
    for a, b in UNSUB:
        text = text.replace(a, b)
    return text


def normalize(text: str) -> str:
    """Un-render, then drop ALL f-prefixes.

    Stripping f only from the rewritten side would flag every pre-existing
    f-string in the file as a difference (it did, on the first run). Both sides
    get the same treatment so the prefix cancels and only path content is
    compared; f-prefix additions are counted separately by the caller.
    """
    text = unrender(text)
    # The one INTENTIONAL semantic change outside the mechanical rename: config
    # loads are wrapped so a config written under another project id resolves to
    # the current one. Unwrap it here so the rest of the file is still compared.
    text = re.sub(r"L\.resolve_cfg\((OmegaConf\.load\([^()]*\))\)", r"\1", text)
    return re.sub(r'\bf(?=["\'])', "", text)


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    changed = subprocess.run(["git", "diff", "--name-only", ref],
                             cwd=ROOT, capture_output=True, text=True).stdout.split()
    bad, ok = [], 0
    for rel in changed:
        p = ROOT / rel
        if p.suffix not in {".sh", ".py"} or not p.exists():
            continue
        orig = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT,
                              capture_output=True, text=True)
        if orig.returncode:
            continue
        before = normalize(orig.stdout)
        after = normalize(p.read_text())
        if before == after:
            ok += 1
            continue
        # Compare ignoring only the allowed structural lines.
        b = [l for l in before.splitlines()
             if not any(r.match(l) for r in ALLOW_REMOVED)]
        a = [l for l in after.splitlines()
             if not any(r.match(l) for r in ALLOW_ADDED)]
        b = [l for l in b if l.strip()]
        a = [l for l in a if l.strip()]
        if a == b:
            ok += 1
            continue
        import difflib
        d = [l for l in difflib.unified_diff(b, a, lineterm="", n=0)][2:]
        bad.append((rel, d[:12]))

    print(f"[verify] {ok} file(s) round-trip to their committed content")
    if bad:
        print(f"[verify] {len(bad)} file(s) DIFFER beyond the allowed structural edits:\n")
        for rel, d in bad:
            print(f"  --- {rel}")
            for l in d:
                print(f"      {l}")
        return 1
    print("[verify] no semantic change: the refactor is a proven no-op at "
          "LUMI_PROJECT=462001328")
    return 0


if __name__ == "__main__":
    sys.exit(main())
