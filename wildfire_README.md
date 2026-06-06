# Predicting California Wildfires

Using machine learning to predict wildfire occurrence across 3, 7, and 14-day horizons at the climate-division level in California.

## Overview

California wildfires cause billions in damage annually and are intensifying with climate change. This project explores whether ML models can predict not just whether a fire will occur, but which region of California it will occur in, using 40 years of weather and historical fire data.

The prediction task is an 8-class classification problem (7 CONUS climate divisions + no fire), with a 12.5% random baseline.

## Data

- **Weather data:** California Weather and Fire Prediction Dataset (1984-2025) from Zenodo
- **Fire records:** Cal Fire historical fire perimeters (1984-2025)
- **Localized weather:** NASA POWER meteorological data
- **Geographic boundaries:** CONUS climate divisions and US county shapefiles

Fire locations were geolocated to climate divisions using Shapely and GeoPandas. Final merged dataset: 20,710 rows, 108 features.

## Models

- **Logistic Regression** (baseline)
- **Feedforward Neural Network** (Optuna-tuned)
- **Random Forest** (with PCA and SMOTE variants)
- **XGBoost** (randomized search tuning)
- **ARIMA/SARIMAX** (time series)

## Results

All models converged around 50% accuracy for the 3-day horizon. Performance degraded at longer horizons across every model family. The convergence across different model architectures suggests the bottleneck is the available features rather than the model choice. Temperature and relative humidity were the strongest predictors.

## Stack

Python, XGBoost, scikit-learn, GeoPandas, Shapely, Optuna, SMOTE, SARIMAX

## Notebooks

- `00_baseline_EDA.ipynb` - Initial exploratory data analysis
- `01_data_merge.ipynb` - Merging weather, fire, and geographic data
- `02_EDA.ipynb` - Feature correlation and distribution analysis
- `03_preprocessing.ipynb` - Data preparation and temporal split
- `04_linear_regression(baseline).ipynb` - Baseline model
- `05_NN.ipynb` - Neural network with Optuna tuning
- `06_random_forest.ipynb` - Random Forest with PCA and SMOTE
- `07_XGBoost.ipynb` - XGBoost with hyperparameter tuning
- `08_ARIMA.ipynb` - Time series approach
