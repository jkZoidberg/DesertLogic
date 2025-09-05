# DesertLogic: A Fuzzy-Supervaluation Pipeline for Desertness Classification

This repository contains the implementation of **DesertLogic**, a research project developed as part of the MSc Dissertation at the University of Leeds. The project combines **fuzzy logic**, **supervaluation semantics**, and **open-source climate/terrain datasets** to classify and analyse desert and dryland regions.

## 🌍 Overview
DesertLogic provides a reproducible pipeline for evaluating *desertness* using open-source climate datasets. Key features include:
- Integration of **TerraClimate** and MODIS data for precipitation, PET, NDVI, and temperature.
- Application of **fuzzy membership functions** for handling gradual thresholds.
- Use of **supervaluation semantics** to capture vagueness and boundary cases (supertrue, superfalse, undetermined).
- Optional support for **rioxarray** export to GeoTIFF for GIS workflows.

## 📂 Repository Structure
