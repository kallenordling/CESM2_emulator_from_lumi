"""Python-side counterpart of ``lumi_env.sh``: the LUMI project id and its paths.

Set the project once in the environment and every script follows:

    export LUMI_PROJECT=462001112

Nothing here reads a config file, so it is safe to import from anywhere
(including modules imported before Hydra/OmegaConf is configured).

The default keeps the historical project id, so an environment that never sets
``LUMI_PROJECT`` behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import os

#: Digits only, e.g. ``462001328``. Everything else is derived from it.
LUMI_PROJECT: str = os.environ.get("LUMI_PROJECT", "462001328")

ACCOUNT: str = f"project_{LUMI_PROJECT}"
SCRATCH: str = f"/scratch/project_{LUMI_PROJECT}"
PROJAPPL: str = f"/projappl/project_{LUMI_PROJECT}"
REPO: str = f"{PROJAPPL}/CESM2_emulator_from_lumi"
VENV: str = f"{PROJAPPL}/venvs/diffesm_laif"
PKGS: str = f"{SCRATCH}/python_packages"
DATA: str = f"{SCRATCH}/emulator_data"

#: Eval output is a SEPARATE knob from LUMI_PROJECT — see lumi_env.sh. Results are
#: collected on LUMI_EVAL_PROJECT's scratch so they land in one place across runs,
#: while the data/venv stay on LUMI_PROJECT. ``expand()`` below leaves this path
#: alone; rewriting it to LUMI_PROJECT is exactly what it must not do.
EVAL_PROJECT: str = os.environ.get("LUMI_EVAL_PROJECT", "462001112")
EVAL_OUT: str = os.environ.get(
    "LUMI_EVAL_OUT", f"/scratch/project_{EVAL_PROJECT}/eval_output"
)

#: Container-internal view of projappl (singularity bind mount).
REPO_PFS: str = f"/pfs/lustrep1/projappl/project_{LUMI_PROJECT}/CESM2_emulator_from_lumi"

# Export for child processes and for OmegaConf's ${oc.env:...} resolver, so a
# config loaded by a script that only imported this module still resolves.
os.environ.setdefault("LUMI_PROJECT", LUMI_PROJECT)
for _k, _v in (
    ("LUMI_EVAL_PROJECT", EVAL_PROJECT),
    ("LUMI_ACCOUNT", ACCOUNT),
    ("LUMI_SCRATCH", SCRATCH),
    ("LUMI_PROJAPPL", PROJAPPL),
    ("LUMI_REPO", REPO),
    ("LUMI_VENV", VENV),
    ("LUMI_PKGS", PKGS),
    ("LUMI_DATA", DATA),
    ("LUMI_EVAL_OUT", EVAL_OUT),
):
    os.environ.setdefault(_k, _v)
del _k, _v


def expand(path: str) -> str:
    """Resolve ``${LUMI_*}`` placeholders in *path*.

    Also rewrites any surviving hardcoded project id, so a config or checkpoint
    written under a previous project still resolves against the current one
    instead of silently pointing at a directory this account cannot read.

    EXCEPTION: paths under ``EVAL_OUT`` are returned untouched. Eval output lives
    on LUMI_EVAL_PROJECT by design, and the blanket rewrite below would drag it
    back onto LUMI_PROJECT — the one place the "one project id" rule must not
    apply.
    """
    out = os.path.expandvars(path)
    if out.startswith(EVAL_OUT):
        return out
    for prefix in ("/scratch/project_", "/projappl/project_"):
        old = out.find(prefix)
        while old != -1:
            start = old + len(prefix)
            end = start
            while end < len(out) and out[end].isdigit():
                end += 1
            found = out[start:end]
            if found and found != LUMI_PROJECT:
                out = out[:start] + LUMI_PROJECT + out[end:]
            old = out.find(prefix, start)
    return out


def resolve_cfg(cfg):
    """Rewrite every project-id-bearing path inside a loaded config, in place.

    The YAML files keep literal ``/scratch/project_<id>/...`` paths — they are
    the recorded default and stay readable — and this rewrites them to the
    CURRENT ``LUMI_PROJECT`` at load time. Without it, setting LUMI_PROJECT would
    move the code but leave every data path pointing at the old allocation,
    which fails late (mid-job, on open) rather than at startup.

    No-op when LUMI_PROJECT matches what the config was written against.
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    def walk(node):
        if isinstance(node, DictConfig):
            for k in node:
                v = node[k]
                if isinstance(v, str):
                    node[k] = expand(v)
                else:
                    walk(v)
        elif isinstance(node, ListConfig):
            for i in range(len(node)):
                v = node[i]
                if isinstance(v, str):
                    node[i] = expand(v)
                else:
                    walk(v)

    was_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        walk(cfg)
    finally:
        OmegaConf.set_struct(cfg, was_struct)
    return cfg


def load_cfg(path: str):
    """Load a YAML config with ``${oc.env:...}`` interpolation resolved.

    Use this instead of ``yaml.safe_load`` wherever a config carries paths:
    plain ``yaml.safe_load`` returns the interpolation markers verbatim, which
    then reach ``open()`` as a literal ``${oc.env:LUMI_SCRATCH,...}`` directory.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg
