# How Many Events Does It Take to Value a Data Stream?

Code, result tables and LaTeX source for a cautionary pilot on **horizon-resolved attribution of data streams
for flash-flood warning in the haor wetlands of Bangladesh**, using the open 77-event inventory.

Author: Md Shafaq Uddin, Bangladesh University of Engineering and Technology (BUET) — mdshafaquddin8@gmail.com

## What is here
| Folder | Content |
|---|---|
| `code/` | `step5_baseline.py` (season baseline), `step6_leadtime.py` (lead-time ablation, set `HAOR_SITE=sylhet` or `sunamganj`), `step8_stats.py` (Holm p-values + power simulation), `compare_sites.py`, `step9_third_model.py` (robustness check with gradient boosting), `make_paper_assets.py` (builds all tables/figures of the paper from the CSVs), `make_map.py` and `make_study_area_map_v3.py` (study-area map; the v3 map uses Natural Earth, geoBoundaries, HydroRIVERS/HydroBASINS and AWS Terrain Tiles, downloaded when run) |
| `data/events.csv` | The 77-event inventory (not our data, see Credits) |
| `results/` | All result CSVs used in the paper |
| `paper/` | `paper.tex`, `refs.bib`, generated figures and tables |

## How to reproduce
1. Put `events.csv` next to the scripts (Python 3, `pip install numpy pandas scikit-learn matplotlib requests`).
2. `python step5_baseline.py`
3. `python step6_leadtime.py` (downloads ERA5/ERA5-Land/GloFAS series from Open-Meteo; needs internet). For the second site: `HAOR_SITE=sunamganj python step6_leadtime.py`.
4. `python step8_stats.py` (about 10 minutes), then `python compare_sites.py`.
   Robustness check with a third model: `python step9_third_model.py` (gradient boosting, settings fixed and not tuned; writes `outputs6/gb_*.csv`).
5. `python make_paper_assets.py` rebuilds `tables/` and `figures/`; compile `paper.tex` with `pdflatex` + `bibtex`.

## Important limits
The design is event-anchored and uses reanalysis, so lead-time results are an **upper bound** for a real forecast.
The cost–loss ratio is an assumed range, not a measured value. With 77 events the study has low power; see the paper.

## Credits and licences
* Event inventory: Koli & Jely (2026), Mendeley Data, DOI 10.17632/d72ny7rftc.1, licence CC BY 4.0.
* Weather data by Open-Meteo.com (CC BY 4.0). Contains modified Copernicus Climate Change Service information (ERA5, ERA5-Land, GloFAS).
* Code in this repository: MIT licence (see `LICENSE`). The dataset keeps its own CC BY 4.0 licence.
* Study-area map: Natural Earth (public domain); geoBoundaries (Runfola et al., 2020, CC BY 4.0); HydroRIVERS/HydroBASINS (Lehner & Grill, 2013, HydroSHEDS licence); terrain from AWS Terrain Tiles (SRTM-derived, Farr et al., 2007).
