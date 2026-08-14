# Figure data

The numbers behind the paper figures, written by the figure scripts themselves
via `--dump-data <dir>`, so these files are exactly what was plotted rather than
a re-derivation that could drift from it.

    main/     the four TRAINED scenarios (hist, ssp370, aaer, ghg)
    unseen/   the two OUT-OF-TRAINING scenarios (ssp126, ssp245)

All values are ANOMALIES vs each side's own 1850-1900 climatology — temperature
in degC, precipitation in percent change (global means) or mm/day (maps,
unless the figure was made in percent, which the `unit` column records).

## timeseries_<VAR>.csv

One row per (scenario, year, source, member). Long format.

| source | member | meaning |
|---|---|---|
| `cesm2` | member id | one reference member's annual global-mean anomaly |
| `emulator` | `m1`..`mN` | one diffusion member's annual global-mean anomaly |
| `emulator` | `ensemble_mean` | the thick line in panel (a) |
| `bias` | `emulator_minus_cesm2` | the line in the lower panels |
| `bias` | `cesm2_sigma` | 1 sigma of the CESM2 members about their own mean; the grey band is +/-2x this |

## histogram_<VAR>.csv

The pooled final-decade samples that form each histogram: one row per sample,
`n_members x n_years` per side (25x10 emulator, 3x10 or 10x10 CESM2).

## maps_gridded.csv

One row per grid point per panel: `emulator`, `cesm2`, their `difference` (what
the colour shows), the Welch `p_value`, and `significant` (1 where the
FDR-controlled test rejects — the hatching). `cesm2` and `p_value` are empty for
panels made with `--emulator-only`, which have no reference to difference
against.

Large: ~55k grid points per panel. Coordinates are the native CESM2 192x288
grid; `lon` runs 0-360.

## Regenerating

    python scripts/paper_fig_timeseries.py --var TREFHT --eval-dir <eval> \
        --n-ref-members 0 --dump-data figure_data/main
    python scripts/paper_fig_histograms.py --var TREFHT --eval-dir <eval> \
        --n-ref-members 0 --dump-data figure_data/main
    python scripts/paper_fig_maps.py --eval-dir <eval> \
        --n-ref-members 0 --dump-data figure_data/main

Add `--scenarios ssp126 ssp245` and point `--dump-data` at `figure_data/unseen`
for the out-of-training set.

CAVEATS worth carrying into any caption built from these files:

- The unseen scenarios have **3** CESM2 members (CMIP6 r4/r10/r11), against
  10/10/11/6 held-out LENS2 members for hist/ssp370/aaer/ghg. Statistics that
  divide by a member-count-dependent spread — `cesm2_sigma`, "% within spread" —
  are NOT comparable between the two directories.
- The emulator's members are independent diffusion samples, not independent
  climate realizations.
