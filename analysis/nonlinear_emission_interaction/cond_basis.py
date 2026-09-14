"""One place to resolve a conditioning PCA basis by name.

Every script in this directory needs the same three things: apply a persisted
per-scenario basis, refuse the degenerate reference basis, or fit the joint
basis that is the only single coordinate system able to hold both ends of the
forcing rectangle. Keeping three copies of that logic is how
01_milestone_mixed_derivative.py ended up defaulting to `ssp370` long after
phase 1 had established that `ssp370` inverts the sign of N.

WHY `joint` EXISTS
------------------
A fitted PCA reconstructs as ``mu + V V^T (x - mu)``: a near-identity only near
the manifold it was fitted on. The pre-industrial state is nowhere near the
ssp370 manifold. Reconstruction error as a percentage of the field's own
cos-lat-weighted spatial sd, measured on ep0863 (phase1c_basis_fidelity.py):

                            CO2      SUL      BC
    1850-1900 via hist      0.1%     2.2%     9.8%
    1850-1900 via ssp370  1149%     207%     100%     <-- inverts N
    2041-2050 via hist      40%      33%      19%
    2041-2050 via ssp370    9.2%     0.2%     0.1%
    both ends via JOINT    12.5/3.2  2.5/5.2  0.9/2.4

``||x - mu_ssp370||`` is 110/120/85 for the 1850 baseline against 24/6.8/5.0
for the 2041-2050 target, and only 61-76% of that offset lies inside the ssp370
EOF span. So reconstructing 1850 through ssp370 does not return a
pre-industrial field; it returns something dragged most of the way to the
21st-century mean (SUL gmean -0.923 -> -0.772). The "aerosol-free" corner then
arrives carrying most of a modern aerosol load, T00 and T01 are not the corners
inclusion-exclusion assumes, and N changes sign:

    ep0863  N(hist baseline, ssp370 target) = +0.02818 +/- 0.00107
            N(ssp370 both ends)             = -0.01750 +/- 0.00067
            N(joint both ends)              = +0.02857 +/- 0.00081   <-- 1 sigma

The joint basis is fitted here rather than read from a checkpoint because no
training run ever fitted one. It is an analysis coordinate system, and it is
legitimate precisely because it reproduces BOTH ends to within a few percent of
what the model was actually fed.

This is NOT the doubled-CO2 basis defect. ep0817, whose basis matches the
corrected CO2 file, shows the same 1120%/206%/100% and the same sign flip.
"""

import sys

_JOINT_CACHE = {}


def resolve_basis(name, *, per_scenario, hist_path, ssp_path, cond_vars,
                  n_comp, sigmas, build_cond_tensor, verbose=True):
    """Return PCA objects for `name`, or None to skip PCA entirely.

    `name` may be "joint" (fit on hist <= 2014 + ssp370 >= 2015), None/"none"
    (no PCA), or a scenario key that must be present in `per_scenario`.
    """
    if name in (None, "none"):
        return None
    if name == "joint":
        return joint_basis(hist_path=hist_path, ssp_path=ssp_path,
                           cond_vars=cond_vars, n_comp=n_comp, sigmas=sigmas,
                           build_cond_tensor=build_cond_tensor, verbose=verbose)

    # NEVER fall back to the reference basis. ckpt["PCA"]["cond"] is
    # byte-identical to per_scenario["aaer"]["cond"], whose CO2 PCA was fitted
    # on a CONSTANT CO2 channel (explained_variance_ratio_ is NaN, confirmed on
    # both checkpoints in use). A silent fallback projects CO2 onto a
    # meaningless basis and still returns finite numbers, so the run looks fine
    # and is wrong.
    if name not in per_scenario or "cond" not in per_scenario[name]:
        sys.exit(f"[error] checkpoint has no per-scenario PCA basis '{name}' "
                 f"(has: {sorted(per_scenario)}). Refusing to fall back to the "
                 f"reference basis, whose CO2 PCA is degenerate.")
    return per_scenario[name]["cond"]


def joint_basis(*, hist_path, ssp_path, cond_vars, n_comp, sigmas,
                build_cond_tensor, verbose=True):
    """Fit (once) one basis on hist 1850-2014 plus ssp370 2015-2100."""
    key = (hist_path, ssp_path, tuple(cond_vars),
           tuple(n_comp) if n_comp else None, tuple(sigmas))
    if key not in _JOINT_CACHE:
        import torch
        from data.climate_dataset import pca_denoise_dataset

        rh, yh, _, _ = build_cond_tensor(hist_path, cond_vars, "time", None,
                                         n_comp, cond_smooth_sigma=sigmas,
                                         cond_smooth_method="gaussian")
        rs, ys, _, _ = build_cond_tensor(ssp_path, cond_vars, "time", None,
                                         n_comp, cond_smooth_sigma=sigmas,
                                         cond_smooth_method="gaussian")
        joint = torch.cat([rh[:, yh <= 2014], rs[:, ys >= 2015]], dim=1)
        _, pcas = pca_denoise_dataset(joint, n_components=n_comp,
                                      var_names=cond_vars, pca_objects=None)
        if verbose:
            print(f"[basis] fitted a joint basis on hist<=2014 + ssp370>=2015 "
                  f"({joint.shape[1]} years)")
        _JOINT_CACHE[key] = pcas
    return _JOINT_CACHE[key]
