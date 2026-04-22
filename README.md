# Extreme rainfall event analysis for the Netherlands

This project builds a simple event-based workflow for short-duration rainfall analysis in the Netherlands using KNMI 10-minute station data and ERA5.

The goal is to:
- turn raw 10-minute rainfall observations into clean 1 h and 3 h accumulations
- build a catalogue of the strongest rainfall events in a selected 3-month window
- attach large-scale ERA5 context to those events
- test one small Fortran routine inside a Linux/Python workflow

## Scope

This is a **3-month event-based analysis**, not a climatology study.

Current analysis choices:
- domain: Netherlands
- analysis window: **May–July**
- rainfall focus: **1 h** and **3 h**
- event seed variable: **rolling 1 h rainfall**
- ERA5 role: **large-scale environmental context only**
- Fortran role: **one small validated side routine**

This project does **not** claim:
- climate attribution
- return periods
- long-term trend analysis
- storm-scale realism from ERA5
- advanced Fortran development

## Data

### KNMI 10-minute station data
- timestamps are in UTC
- each timestamp marks the **end** of the 10-minute interval
- rainfall variable used here: `rg`
- `rg` is treated as mean rain intensity over the 10-minute interval in `mm/h`
- 10-minute rainfall amount is computed as `rg / 6`

### ERA5
ERA5 is used only to describe the larger-scale environment around each event.

Variables used:
- total column water vapour
- CAPE
- mean sea-level pressure
- 850 hPa and 500 hPa wind
- 850 hPa humidity
- optional 700 hPa vertical velocity

## Event definition

A station-level 99th percentile is **not** used as the main threshold, because the analysis window is only 3 months.
Instead, the catalogue is built as a ranking of the strongest events in the selected block.

### Station seeds
For each station:
1. keep valid rolling 1 h rainfall values
2. detect local maxima
3. require at least **6 h** between selected peaks
4. keep the **top 5** peaks in the selected 3-month window

### National event merge
Station seeds are merged into one event when they are:
- within **6 h** in time
- within **75 km** in space

### Event attributes
Each event stores:
- start and end time
- number of stations involved
- peak 1 h rainfall
- peak station
- event-level peak 3 h rainfall
- simple footprint measures
- local time of peak
- nearby-station missing-data fraction

## ERA5 diagnostics

For each event, ERA5 is used to compute simple event-level diagnostics at the hourly ERA5 time nearest to the rainfall peak.

Diagnostics include:
- TCWV anomaly
- CAPE percentile within the analysis block
- 850 hPa wind speed
- low-level moisture transport proxy
- 850–500 hPa shear proxy
- mean sea-level pressure gradient proxy

These are **environmental proxies**, not full storm-process diagnostics.

## Preprocessing rules

- raw KNMI files are treated as 10-minute intervals ending at the file timestamp
- the full 10-minute timeline is rebuilt explicitly so gaps are visible
- negative rainfall values are treated as missing
- duplicate timestamps are treated as invalid for rolling sums
- 1 h rainfall requires 6 valid consecutive 10-minute steps
- 3 h rainfall requires 18 valid consecutive 10-minute steps
- incomplete rolling windows are masked out

## Regime labels

Events are given a simple rule-based label:
- localized afternoon convective
- organized convective
- widespread/frontal
- mixed or uncertain

These labels are first-pass interpretation only. They are not storm-truth labels.

## Fortran sidecar

The Fortran part of the project is one small routine for rolling rainfall accumulation with missing-value handling. It is used to show simple Python/Fortran workflow integration

## Repository layout

extreme-rainfall-mini/
├─ README.md
├─ environment.yml
├─ Makefile
├─ .gitignore
├─ data_raw/
├─ data_processed/
├─ notebooks/
├─ src/
├─ fortran/
├─ scripts/
└─ report/
