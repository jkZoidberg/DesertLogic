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
- **MODIS MOD13A3 NDVI (GEE)** – monthly NDVI (vegetation index).  
- **Altimeter Corrected Elevations, Version 2 (ACE2)** – Digital Elevation Model (DEM)*

Citation
If this repository supports your research, please cite:

Wan, Y. (2025). DesertLogic: A Fuzzy–Supervaluation Approach for Desertness Classification. GitHub. Available at: https://github.com/username/DesertLogic
(Add Zenodo DOI here if archived.)

License
Distributed under the Apache-2.0 license. See LICENSE for details.

Acknowledgements
Thanks to the providers of MODIS MOD13A3 NDVI (GEE), Altimeter Corrected Elevations, Version 2 (ACE2), TerraClimate, and to the open-source Python ecosystem (xarray/rioxarray/GDAL, etc.). This work forms part of the author’s MSc dissertation at the University of Leeds.








