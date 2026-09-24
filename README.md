# Cross-library conformal benchmark for soil spectral prediction intervals

Analysis code and derived results for:

> Şahin, N., Arslan, S. **How reliable are prediction intervals from global soil
> spectral libraries? A cross-library conformal benchmark.** (in review)

The study asks whether a prediction interval calibrated in one soil spectral
library keeps its nominal coverage in another. Ten interval constructions are
compared across 42 mid-infrared and 28 Vis-NIR library × property × model cells
under a leave-one-library-out protocol.

## Data

The analysis uses the **Open Soil Spectral Library** (Safanelli et al., 2025,
PLoS ONE 20(1), e0296545), version 1.2, which is publicly available at
<https://soilspectroscopy.org>. The source data are not redistributed here.
Download them and place them in a `data/` folder at the root of this
repository before running the scripts.

Two arms are analysed separately and never pooled:

| Arm | Bands | Label track | Spectra | Libraries |
|---|---|---|---|---|
| MIR | 1,701 | method-matched (`__clean`) | 85,684 | KSSL, ICRAF-ISRIC, AFSIS1, AFSIS2, CAF, GARRETT, SERBIA |
| Vis-NIR | 1,051 | method-mixed (`__naive`) | 64,644 | LUCAS, LUCAS-WOODWELL, KSSL, ICRAF-ISRIC |

Properties: clay content, organic carbon, total nitrogen, pH in water.

## Pre-registered expectations

Each script contains a block of expectations, together with the rule by which
each would be judged, that was written **before that script was run**. Every
script prints its expectations and their outcomes at the end of its run; the
printed outcomes are reproduced in Table 4 of the manuscript. Twenty
expectations were registered across the five analyses; sixteen were met and
four were not. The four that failed are reported in the manuscript with the
same weight as those that held.

## Scripts

Run them in order from the repository root. Each writes its result tables to
`out/`.

| Script | What it does |
|---|---|
| `asama01_conformal_pilot.py` | Pilot on the MIR arm: does source-calibrated coverage hold under transfer at all? |
| `asama02_conformal_full.py` | Main analysis. All constructions, both arms, bootstrap intervals, inside/outside source range. |
| `asama03_weights_auc_seeds.py` | Weighted conformal, clipped weights, domain-classifier AUC, three random seeds. |
| `asama04_cqr_alpha.py` | Conformalized quantile regression and the sensitivity to the nominal level (0.80 / 0.90 / 0.95). |
| `asama06_cnn.py` | 1D-CNN model family under a reduced protocol. Requires PyTorch. |
| `asama07_eslesmis_temel.py` | PLSR and gradient boosting re-run under the identical CNN protocol, so the model is the only difference. |
| `asama05_figurler.py` | Figures 1–7 and the summary tables. Fits no models; reads the CSVs written by the scripts above, so run it last. |

Typical invocation:

```bash
python scripts/asama02_conformal_full.py --arm mir    --track clean
python scripts/asama02_conformal_full.py --arm visnir --track naive
python scripts/asama05_figurler.py          # both arms at once, no arguments
```

## Interval constructions

| Code | Construction | Target labels |
|---|---|---|
| C0 | Split conformal on absolute residuals, calibrated in the source | none |
| C0w | C0 inflated to the median half-width of C1 (width control) | none |
| C0n | Normalized conformal, score \|r\| / σ(x) with σ fitted in the source | none |
| C1 | Weighted conformal, density-ratio weights from a domain classifier | none |
| C1c | C1 with classifier probabilities clipped to [0.05, 0.95] | none |
| C4 | Conformalized quantile regression | none |
| C2 | Conformal threshold re-estimated on k target labels | k |
| C3 | Affine output correction on k labels, conformal on leave-one-out residuals | k |
| C3n | C3 with normalized scores | k |
| C4t | C4 re-conformalized on k target labels | k |

## Output

- `out/` — per-cell and summary result tables as CSV. These are the numbers
  quoted in the manuscript.
- `figures/` — Figures 1–7 as 300 dpi PNG and vector PDF.

## Environment

```bash
conda create -n ossl python=3.10
conda activate ossl
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # for asama06 only
```

Runtimes on a laptop CPU: the main analysis takes roughly 10 minutes per arm,
the seed and CQR analyses 15–25 minutes each, and the figure script under a
minute.

## License

Code is released under the MIT License (see `LICENSE`). The Open Soil Spectral
Library carries its own licence; see the link above.
