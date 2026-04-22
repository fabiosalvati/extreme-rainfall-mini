# Scope note

analysis window: exact 3 months used
primary seed variable: 1 h rolling rainfall
station seed rule: local maxima, 6 h minimum separation, top N retained
national merge rule: 6 h and 75 km
3 h rainfall used as diagnostic/event attribute
claim limited to strongest events in the selected window, not climatology

## Working title
Extreme rainfall event analysis for the Netherlands in Python 3, with ERA5 environmental diagnostics and a small Fortran/Linux sidecar

## Research focus
This project analyses warm-season short-duration rainfall extremes in the Netherlands using KNMI 10-minute station observations and ERA5 reanalysis.

The central aim is to build a defensible event-based workflow rather than a climatological trend study.

## Fixed scope
- Warm season only: May–September
- Short-duration rainfall focus: 1 h and 3 h accumulations
- Event-based analysis, not long-term trend attribution
- ERA5 used for large-scale environmental context
- One small Fortran routine integrated into a Linux/Python workflow

## Structural hinge
The project succeeds only if the workflow can do all of the following:
1. convert raw station observations into valid accumulation series,
2. construct a defensible event catalogue,
3. attach physically interpretable ERA5-based diagnostics,
4. demonstrate a real but limited Fortran/Linux component validated against Python.

## Intended claims
This project is intended to demonstrate:
- process-oriented rainfall-extremes analysis,
- physically grounded environmental diagnostics,
- reproducible Python 3 workflow design,
- basic Fortran/Linux integration relevant to numerical geoscience workflows.

## Non-claims
This project does not claim:
- climate attribution,
- storm-scale model realism from ERA5,
- advanced Fortran development,
- convection-permitting modelling expertise,
- machine learning methods.

## Immediate methodological constraints
The following must be made explicit later in the workflow:
- event threshold definition,
- treatment of missing intervals and incomplete windows,
- station-to-event clustering rules,
- ERA5 proxy definitions,
- regime-classification rules.

## Stretch component
HARMONIE evaluation is optional and should not be allowed to destabilize the core workflow.



##### UPDATE 1  KNMI 10-minute data

## Dataset
KNMI 10-minute in-situ meteorological observations

## File structure
- Station dimension: `station`
- Time dimension: `time`
- Sample file station count: 62

## Time semantics
- `time` denotes the end of the 10-minute measurement interval
- Example: file timestamp `201201010000` represents the interval 2011-12-31 23:50 UTC to 2012-01-01 00:00 UTC

## Precipitation variable
- Primary variable: `rg`
- Meaning: mean rain-gauge precipitation intensity over the 10-minute interval
- Units: `mm/h`
- Conversion to 10-minute rainfall amount: `precip_10min_amount_mm = rg / 6`

## Missing values
- Missing values appear as `NaN` after reading with xarray
- Missingness is station-dependent and must be propagated into later rolling-window calculations

## Sample coverage test
- Number of consecutive files inspected: 144
- End-time coverage: 2012-01-01 00:00 UTC to 2012-01-01 23:50 UTC
- Non-10-minute file gaps: 0

## Sample station series
- Station: 06260 (DE BILT AWS)
- Valid 10-minute precipitation values in first 144 files: 144/144

##### UPDATE 2
The KNMI 10-minute preprocessing layer was built using rg as the primary rainfall variable, converted from mm/h intensity to 10-minute rainfall amounts, with missing intervals and station-level missing values handled explicitly. Rolling 1 h and 3 h accumulations are only computed for complete windows, and downstream analysis must restrict to stations with sufficient rg availability.

##### UPDATE 3
For each station, I define an extreme 1 h rainfall exceedance as a valid warm-season rolling 1 h accumulation greater than or equal to the station-specific 99th percentile of valid May–September 1 h accumulations over the available record. Because rolling windows generate serially correlated exceedances, exceedances separated by 3 h or less at the same station are collapsed into one station episode. National event candidates are then constructed by clustering station episodes whose episode windows are separated by 4 h or less and whose station locations are within 75 km. This is an operational event-merging rule for catalogue construction, not a storm-tracking algorithm.



## Preprocessing rules
- Raw KNMI files are treated as 10-minute intervals ending at the file timestamp in UTC.
- The full 10-minute station timeline is rebuilt explicitly so gaps become visible rows.
- Negative rainfall values are treated as missing.
- Duplicate timestamps are considered invalid for rolling accumulations.
- 1 h rainfall requires 6 valid consecutive 10-minute intervals.
- 3 h rainfall requires 18 valid consecutive 10-minute intervals.
- Rolling sums are masked if the window is incomplete.
- For rg/pg, 10-minute rainfall amount = raw value / 6, assuming mm h^-1 mean intensity over the 10-minute interval.


## wha09_preprocess_summary.py is a quick QA script used after station preprocessing.
It reports missing intervals, duplicate timestamps, and basic counts of valid 1 h and 3 h windows.
It is a diagnostic check, not part of the scientific inference itself.

## Event definition used in this project

This project does **not** define extremes using a climatological percentile threshold, because the analysis window is limited to a selected 3-month block (May–July). A station-level 99th percentile estimated from only this short block would be too unstable to treat as a defensible extreme threshold.

Instead, the event catalogue is built as a **block-relative ranking of the strongest short-duration rainfall events** in the selected period.

### Analysis window
- Months used: **May–July**
- Domain: **Netherlands**
- Primary variable: **rolling 1 h rainfall**
- Secondary variable: **rolling 3 h rainfall**

### Station-level seed definition
For each station:
1. use only valid rolling 1 h rainfall values
2. detect **local maxima** in the 1 h series
3. require a minimum **6 h separation** between selected peaks at the same station
4. keep the **top 5 station peaks** in the selected 3-month block

These selected station peaks are used as **event seeds**.

### National event merging
Station seeds are merged into one national event candidate when they satisfy both conditions:
- peak times within **6 h**
- station distance within **75 km**

This is a simple operational clustering rule. It is used to group related station peaks into the same candidate event. It is **not** a storm-tracking algorithm.

### Event attributes
For each event candidate, the catalogue stores:
- start and end time based on the union of station seed windows
- number of stations involved
- peak 1 h rainfall
- peak station
- event-level peak 3 h rainfall

The **3 h rainfall** is treated as an **event attribute**, not as the primary detection variable.

## Interpretation of the catalogue

The resulting catalogue should be interpreted as:

> a catalogue of the strongest short-duration rainfall events in the selected May–July analysis block

It should **not** be interpreted as:
- a climatological extreme-value analysis
- a return-period analysis
- a long-term threshold-based national extreme-rainfall climatology
- a physically exact storm-object dataset

## Limitations
- The 3-month analysis window is too short to define stable climatological percentile thresholds.
- The event merge rule is heuristic and may merge nearby but distinct events, or split parts of the same meteorological system.
- Event start and end times are based on rolling-accumulation support windows, not exact rainfall onset and decay times.
- ERA5 diagnostics attached later are large-scale environmental context only, not resolved convective structure.

## Event catalogue construction

After building national event candidates from station 1 h rainfall peaks, the catalogue is extended with summary metrics for each event:
- event duration
- number of contributing stations
- peak 1 h rainfall
- event-level peak 3 h rainfall
- local time of peak
- simple spatial footprint proxies
- fraction of nearby stations with no valid 1 h data during the event window

The spatial footprint is described using simple station-based proxies, not storm-object geometry. These proxies are intended only to summarize how widespread the station response was within the selected event candidate.

## ERA5 data handling

ERA5 is used only for large-scale environmental context around the rainfall events.

The ERA5 processing step does the following:
- standardizes coordinate names across files
- converts all times to UTC
- subsets the data to a Netherlands-centred latitude/longitude box
- keeps only the variables needed for event diagnostics
- builds event-centred hourly windows around each rainfall event

Because ERA5 is hourly while the rainfall events are defined from 10-minute station data, each event is aligned to the nearest ERA5 hour before extracting the event-centred window.

## ERA5 context extraction

For each rainfall event in the catalogue, an event-centred ERA5 subset is extracted over a Netherlands-centred box.

The ERA5 context dataset contains:
- single-level variables: total column water vapour, CAPE, mean sea-level pressure
- pressure-level variables: 850 hPa and 500 hPa winds, 850 hPa humidity, and optionally 700 hPa vertical velocity

Each event is aligned to the nearest hourly ERA5 timestamp, and a fixed window from 12 hours before to 12 hours after the event peak is extracted.

## ERA5 event diagnostics

Event diagnostics are computed from the ERA5 field at the event anchor time, defined as the hourly ERA5 timestamp nearest to the rainfall peak.

For each event, the following box-mean environmental proxies are computed over the Netherlands-centred ERA5 domain:
- total column water vapour
- CAPE
- 850 hPa wind speed
- 850 hPa specific humidity
- low-level moisture-transport proxy: q850 × wind850
- 850–500 hPa shear proxy
- mean sea-level pressure gradient proxy

TCWV anomaly is defined relative to the mean TCWV over the full analysis-block ERA5 background.
CAPE is expressed as a percentile relative to the full analysis-block CAPE distribution over the same domain.


## Event-level ERA5 diagnostics

Event-level diagnostics are computed by combining:
- the rainfall event catalogue
- the event-centred ERA5 context dataset
- the full ERA5 background over the selected analysis block and Netherlands-centred box

The event-centred ERA5 dataset is used to extract environmental conditions at the event anchor time.
The full ERA5 background over the same spatial domain is used to define reference values for:
- mean TCWV over the analysis block
- the CAPE distribution over the analysis block