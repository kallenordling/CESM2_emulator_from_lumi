plots/ — three series, three prefixes. Keep them apart.

figNN/   THE PAPER SET, in the order figures_overleaf/ uses (see its MANIFEST):
           01,02  global-mean temperature and precipitation
           03,04  their anomaly distributions
           05,06  ensemble-mean anomaly MAPS   (scripts/make_ensemble_mean_maps.py)
           07,08  city timeseries, T then P    (analysis/.../plot_city_series.py)
           09,10  city distributions, T then P (same script)
         Figures 11-21 of the paper (out-of-training, RAMIP, attribution) are
         built elsewhere and live in the eval output, not here.

supplement/  figS01-figS04, the ABSOLUTE counterparts of 01,02,05,06.

xaiNN/   The XAI / nonlinear-interaction report, its OWN sequence 09-21. It was
         briefly renumbered to track the paper on 2026-09-11, which made its
         numbers collide with paper slots meaning something else; the prefix
         ends that. Built by scripts/make_xai*.py.

auxNN/   Auxiliary figures not in either document: the climate-offset pair
         (make_fig5.py) and the running-mean pair (make_fig7.py). They used to
         name their outputs fig05-fig08 and would overwrite paper figures.

fig12_data/  Shared CSV dump for paper figures ONE and TWO — the name is a
             figure PAIR, not figure twelve. Same for make_fig12_csv.py,
             make_fig12_from_csv.py and make_fig34_from_csv.py.
