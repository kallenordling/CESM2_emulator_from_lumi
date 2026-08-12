#!/usr/bin/env python3
"""One-shot codemod: hardcoded LUMI project paths -> lumi_env.sh / lumi_paths.py.

Kept in the repo because the transformation it performs is the thing a reviewer
needs to check, and because it is re-runnable if more literals creep back in.

    python scripts/_codemod_lumi_paths.py            # dry run, prints a summary
    python scripts/_codemod_lumi_paths.py --apply

Shell (.sh)
    /scratch/project_<id>            -> ${LUMI_SCRATCH}
    /projappl/project_<id>/venvs/... -> ${LUMI_VENV}
    ...and a `source lumi_env.sh` + `assert_account` header is inserted.
    `#SBATCH --account=` is REMOVED: SLURM does not expand variables in #SBATCH
    lines, so the account travels via SBATCH_ACCOUNT / lsubmit.sh instead, and
    assert_account() re-checks it inside the job.

Python (.py)
    "/scratch/project_<id>/x"  ->  f"{L.SCRATCH}/x"   (+ `import lumi_paths as L`)
    Strings already containing braces are SKIPPED and reported: turning them
    into f-strings would change their meaning.

YAML is deliberately untouched: lumi_paths.load_cfg() normalises the project id
in config values at load time, so the literals there act as recorded defaults.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OLD = "462001328"
ROOT = Path(__file__).resolve().parent.parent

# Longest first: /projappl/<id>/venvs/... must win over /projappl/<id>.
SH_SUBS = [
    (f"/pfs/lustrep1/projappl/project_{OLD}/CESM2_emulator_from_lumi", "${LUMI_REPO_PFS}"),
    (f"/projappl/project_{OLD}/venvs/diffesm_laif", "${LUMI_VENV}"),
    (f"/projappl/project_{OLD}/CESM2_emulator_from_lumi", "${LUMI_REPO}"),
    (f"/scratch/project_{OLD}/python_packages", "${LUMI_PKGS}"),
    (f"/scratch/project_{OLD}/emulator_data", "${LUMI_DATA}"),
    (f"/scratch/project_{OLD}/eval_output", "${LUMI_EVAL_OUT}"),
    (f"/scratch/project_{OLD}", "${LUMI_SCRATCH}"),
    (f"/projappl/project_{OLD}", "${LUMI_PROJAPPL}"),
    (f"project_{OLD}", "${LUMI_ACCOUNT}"),
]
PY_SUBS = [
    (f"/pfs/lustrep1/projappl/project_{OLD}/CESM2_emulator_from_lumi", "{L.REPO_PFS}"),
    (f"/projappl/project_{OLD}/venvs/diffesm_laif", "{L.VENV}"),
    (f"/projappl/project_{OLD}/CESM2_emulator_from_lumi", "{L.REPO}"),
    (f"/scratch/project_{OLD}/python_packages", "{L.PKGS}"),
    (f"/scratch/project_{OLD}/emulator_data", "{L.DATA}"),
    (f"/scratch/project_{OLD}/eval_output", "{L.EVAL_OUT}"),
    (f"/scratch/project_{OLD}", "{L.SCRATCH}"),
    (f"/projappl/project_{OLD}", "{L.PROJAPPL}"),
    (f"project_{OLD}", "{L.ACCOUNT}"),
]

SKIP = {"lumi_env.sh", "lumi_paths.py", "_codemod_lumi_paths.py",
        "migrate_lumi_project.sh"}

STR_RE = re.compile(r"(?P<pre>[frbFRB]*)(?P<q>\"\"\"|'''|\"|')(?P<body>.*?)(?<!\\)(?P=q)",
                    re.DOTALL)


def do_shell(text: str, rel: Path) -> tuple[str, list[str]]:
    notes = []
    lines = text.splitlines(keepends=True)
    out, dropped = [], False
    for ln in lines:
        if re.match(r"\s*#SBATCH\s+--account=", ln):
            dropped = True
            notes.append("removed #SBATCH --account (now via SBATCH_ACCOUNT/lsubmit.sh)")
            continue
        out.append(ln)
    text = "".join(out)
    for a, b in SH_SUBS:
        text = text.replace(a, b)
    if "lumi_env.sh" not in text:
        depth = len(rel.parts) - 1
        up = "/".join([".."] * depth) + "/" if depth else ""
        header = (
            '\n# Single source of truth for the LUMI project id and its paths.\n'
            f'source "$(dirname "${{BASH_SOURCE[0]}}")/{up}lumi_env.sh"\n'
        )
        if dropped or "#SBATCH" in text:
            header += "assert_account\nlumi_env_banner\n"
        idx = [i for i, l in enumerate(text.splitlines(keepends=True))
               if l.startswith("#SBATCH")]
        ls = text.splitlines(keepends=True)
        at = (idx[-1] + 1) if idx else (1 if ls and ls[0].startswith("#!") else 0)
        text = "".join(ls[:at]) + header + "".join(ls[at:])
        notes.append("inserted lumi_env.sh source header")
    return text, notes


def do_python(text: str, rel: Path) -> tuple[str, list[str]]:
    notes, changed = [], False

    def repl(m: re.Match) -> str:
        nonlocal changed, notes
        pre, q, body = m.group("pre"), m.group("q"), m.group("body")
        if OLD not in body:
            return m.group(0)
        if ("{" in body or "}" in body) and "f" not in pre.lower():
            notes.append(f"MANUAL: string with braces at offset {m.start()}")
            return m.group(0)
        new = body
        for a, b in PY_SUBS:
            new = new.replace(a, b)
        if new == body:
            return m.group(0)
        changed = True
        if "f" not in pre.lower():
            pre = "f" + pre
        return f"{pre}{q}{new}{q}"

    text2 = STR_RE.sub(repl, text)
    # Comments / docstext outside string literals: plain textual swap, no f-string.
    if OLD in text2:
        for a, b in SH_SUBS:
            text2 = text2.replace(a, b.replace("${", "${"))
        notes.append("swapped remaining occurrences in comments")
    if changed and "import lumi_paths" not in text2:
        ls = text2.splitlines(keepends=True)
        at = 0
        for i, l in enumerate(ls[:60]):
            if l.startswith(("import ", "from ")):
                at = i
                break
            if l.startswith(("#!", '"""', "'''", "#")) or not l.strip():
                at = i + 1
        ls.insert(at, "import lumi_paths as L\n")
        text2 = "".join(ls)
        notes.append("added `import lumi_paths as L`")
    return text2, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = [p for p in ROOT.rglob("*")
             if p.is_file() and p.suffix in {".sh", ".py"}
             and ".git" not in p.parts and "__pycache__" not in p.parts
             and p.name not in SKIP]
    total, manual = 0, []
    for p in sorted(files):
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        if OLD not in text:
            continue
        rel = p.relative_to(ROOT)
        new, notes = (do_shell if p.suffix == ".sh" else do_python)(text, rel)
        if new == text:
            continue
        total += 1
        left = new.count(OLD)
        flag = f"  [{left} LEFT]" if left else ""
        print(f"  {rel}{flag}")
        for n in notes:
            print(f"      - {n}")
            if n.startswith("MANUAL"):
                manual.append(f"{rel}: {n}")
        if args.apply:
            p.write_text(new)
    print(f"\n{total} files {'rewritten' if args.apply else 'would change'}")
    if manual:
        print(f"\n{len(manual)} need manual attention:")
        for m in manual:
            print(f"  {m}")
    if not args.apply:
        print("\nre-run with --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
