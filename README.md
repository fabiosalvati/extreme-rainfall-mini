# Extreme rainfall event analysis for the Netherlands in Python 3, with ERA5 environmental diagnostics and a small Fortran/Linux sidecar

## Project question
Can short-duration warm-season rainfall extremes in the Netherlands be converted from raw station observations into a defensible event catalogue, and can those events be linked to physically interpretable large-scale environmental diagnostics from ERA5 within a reproducible Python 3 workflow?

## Core objective
This project builds a process-oriented workflow for analysing warm-season short-duration rainfall extremes over the Netherlands using KNMI 10-minute station observations and ERA5 reanalysis.

The workflow is designed to:
1. preprocess KNMI 10-minute rainfall observations into analysis-ready accumulation series,
2. identify and merge station-level exceedances into event candidates,
3. quantify event characteristics such as intensity, duration, timing, and spatial footprint,
4. attach ERA5-based environmental diagnostics,
5. classify events into simple process-oriented regimes, and
6. demonstrate a small validated Fortran routine integrated into a Linux/Python workflow.

## Scope
- Region: Netherlands
- Season: May–September
- Event durations: 1 h and 3 h rainfall
- Analysis type: event-based analysis
- Environmental context: ERA5 large-scale diagnostics
- Fortran component: small rolling-accumulation routine with missing-value handling

## Out of scope
This project does not claim:
- climate trend attribution,
- convective-scale storm reconstruction from reanalysis,
- full atmospheric-model development,
- advanced Fortran or HPC expertise,
- machine learning classification.

## Data
### Required
- KNMI 10-minute station precipitation observations
- ERA5 reanalysis

### Optional stretch
- KNMI HARMONIE reforecast overlap analysis

## Planned outputs
- quality-controlled station rainfall accumulation series,
- event candidate catalogue,
- event diagnostics table,
- simple regime classification,
- validation notebook for the Fortran sidecar,
- short technical note and figures.

## Main caveats

- KNMI 10-minute archive handling must respect UTC timestamp semantics.
- Extreme-event definition depends on explicit threshold and clustering rules.
- ERA5 is used for environmental context, not convective-scale process diagnosis.
- HARMONIE is a stretch component and is not required for project success.

## Repository structure
- `src/`: Python source code
- `notebooks/`: analysis notebooks
- `fortran/`: small Fortran sidecar
- `data_raw/`: raw input data
- `data_processed/`: derived outputs
- `report/`: technical note, figures, slides
- `scripts/`: shell entry points

## Reproducibility
Environment definition is stored in `environment.yml`.
Fortran compilation is handled through the `Makefile`.
