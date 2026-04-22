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

## Data
### Required
- KNMI 10-minute station precipitation observations
- ERA5 reanalysis
