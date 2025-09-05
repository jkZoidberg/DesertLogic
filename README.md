# DesertLogic: A Fuzzy–Supervaluation Pipeline for Desertness Classification

**DesertLogic** is the codebase for an MSc dissertation at the University of Leeds. It combines **fuzzy logic** and **supervaluation semantics** to evaluate “desertness” from open-source climate/land data. The pipeline ingests TerraClimate and MODIS NDVI (optionally DEM) to compute continuous membership, aggregate decisions, and export masks for **supertrue**, **superfalse**, and **undetermined** regions, with quick visualisation utilities.

## Features
- 📥 Open data only: TerraClimate (PPT, PET, Tmax/Tmin…), MODIS MOD13A3 NDVI; optional DEM.
- 🧠 Fuzzy membership functions (S/Z/Trapezoid) for gradual thresholds.
- 🧩 Supervaluation to handle vagueness/boundaries (supertrue/undetermined).
- 🗺️ Optional GeoTIFF export via `rioxarray` for GIS workflows.
- 🖼️ One-command quick plots (PNG) for inspection and reporting.

## Repository Structure
DesertLogic/
├── desertness_pipeline.py # Main end-to-end pipeline
├── fuzzy_membership.py # S-shaped/Z-shaped/Trapezoid functions & helpers
├── semantics_superval.py # Supervaluation operators (supertrue/undetermined)
├── rdr_apply.py # Optional Ripple-Down Rules extension
├── viz_quick.py # Quick map rendering to PNG
├── utils_terraclimate.py # Download/open/process TerraClimate & helpers
├── config.yml # Example configuration
└── README.md

bash
复制代码

## Data Sources (Open Access)
- **TerraClimate** – monthly climate data (precipitation, PET, temperature, etc.).  
- **MODIS MOD13A3** – monthly NDVI (vegetation index).  
- **Optional DEM (e.g., SRTM)** – if you extend to terrain/landform analyses.  
*Large raw datasets are **not** included in the repo. Scripts/notes indicate how to obtain them.*

## Quick Start
```bash
# 1) Clone
git clone https://github.com/username/DesertLogic.git
cd DesertLogic

# 2) (Recommended) Conda environment
conda create -n desertlogic python=3.10 -y
conda activate desertlogic

# 3) Core deps (consolidated on conda-forge for GDAL/HDF stack)
conda install -c conda-forge gdal rioxarray rasterio xarray netcdf4 h5py hdf5 hdf4 pyproj pandas numpy matplotlib pyyaml -y

# 4) (Optional) If you use pip for extras
pip install pydap

# 5) (Optional) If you keep a requirements file
# pip install -r requirements.txt
Configuration (example config.yml)
yaml
复制代码
# Geographic/temporal subset
region:
  lon_min: -20
  lon_max: 60
  lat_min: -35
  lat_max: 35
time:
  start: 2001-01
  end: 2020-12

# Variables to fetch/process
variables:
  - ppt        # precipitation
  - pet        # potential evapotranspiration
  - ndvi       # MODIS MOD13A3 (if integrated)

# Fuzzy thresholds (example placeholders)
fuzzy:
  ppt_low: [0, 250]      # mm/yr: desert → low PPT
  pet_high: [1200, 2000] # mm/yr: desert → high PET
  ndvi_low: [0.1, 0.2]   # unitless monthly mean

# Aggregation/operator for desertness (example)
aggregate:
  method: max            # e.g., max/min/weighted
  weights: [0.4, 0.4, 0.2]

# Supervaluation (grid of admissible thresholds)
supervaluation:
  n_scenarios: 20        # sample different cut pairs to evaluate supertrue/undetermined
  consensus: 0.67        # fraction of scenarios to mark as supertrue

# Output
output:
  dir: out/
  export_geotiff: true
Usage
bash
复制代码
# Run the pipeline (reads config.yml, writes NetCDF/GeoTIFF to out/)
python desertness_pipeline.py --config config.yml

# Quick-look maps (PNG)
python viz_quick.py
Expected outputs (examples):

out/desertness_mean.nc – continuous desertness score

out/supertrue_mask.nc, out/undetermined_mask.nc – supervaluation masks

desertness_mean.png – quick rendered map

Reproducibility Notes
Record environment: conda env export > environment.yml.

Fix random seeds where applicable (if any sampling is used).

Archive a release (e.g., GitHub Release + Zenodo DOI) referenced in the dissertation.

Citation
If this repository supports your research, please cite:

Wan, Y. (2025). DesertLogic: A Fuzzy–Supervaluation Approach for Desertness Classification. GitHub. Available at: https://github.com/username/DesertLogic
(Add Zenodo DOI here if archived.)

License
Distributed under the MIT License. See LICENSE for details.

Acknowledgements
Open data providers (TerraClimate, MODIS) and the open-source Python ecosystem (xarray/rioxarray/GDAL, etc.). This work forms part of the author’s MSc dissertation at the University of Leeds.

复制代码







询问 ChatGPT
