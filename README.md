# Predicting California Wildfires

Using machine learning to predict wildfire occurrence across 3, 7, and 14-day horizons at the climate-division level in California.

## 📌 Overview

California wildfires cause billions in damage annually and are intensifying with climate change. This project explores whether machine learning models can predict not just whether a fire will occur, but which region of California it will occur in, using more than 40 years of weather and historical fire data.

The prediction task is an 8-class classification problem: 7 CONUS climate divisions plus a no-fire class. A random classifier would achieve 12.5% accuracy. The project also examines how predictability breaks down as the forecasting horizon lengthens.

## 📊 Data Sources

- Weather data: California Weather and Fire Prediction Dataset (1984-2025) from Zenodo
- Fire records: Cal Fire historical fire perimeters (1984-2025)
- Localized weather: NASA POWER meteorological data
- Geographic boundaries: CONUS climate divisions and US county shapefiles

## ⚙️ Data Engineering

Fire locations were geolocated to climate divisions using Shapely and GeoPandas. Weather observations, fire history, climate-division assignments, and geographic information were merged into a unified forecasting dataset. Additional temporal and weather-based features were engineered, including lagged weather variables and derived environmental measurements.

Final merged dataset: 20,710 rows, 108 features.

## 🤖 Models

Five model families were compared, all evaluated with temporal train/validation/test splits to preserve chronological ordering and prevent information leakage:

- Logistic Regression (baseline)
- Feedforward Neural Network (Optuna-tuned)
- Random Forest (with PCA and SMOTE variants)
- XGBoost (randomized search tuning)
- ARIMA/SARIMAX (time series)


## 📈 Results

Across Logistic Regression, Neural Networks, Random Forest, XGBoost, and ARIMA/SARIMAX, accuracy at the 3-day horizon converged around similar levels and declined at longer horizons. The fact that very different architectures clustered at a similar ceiling points to the available feature signal and data quality as the binding constraint, not model choice.

XGBoost illustrated the horizon effect clearly:

| Horizon | Accuracy |
|---------|----------|
| 3-day   | 54.2%    |
| 7-day   | 40.6%    |
| 14-day  | 33.8%    |

Temperature and relative humidity emerged as the strongest predictors across multiple model families.

**Key findings:**
- Temperature and relative humidity emerged as the strongest predictors across multiple model families.
- Forecasting performance declined sharply as the prediction horizon increased.
- Temporal train/validation/test splits were essential for realistic estimates and for avoiding information leakage.
- Class imbalance was a persistent challenge given the low base rate of wildfire events; mitigation techniques including SMOTE were evaluated as part of the modeling process.
- The shared ~50% accuracy ceiling across architectures indicates the bottleneck is data and feature quality rather than model selection.

