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

Five model families were compared (Logistic Regression baseline, Feedforward Neural Network, Random Forest, XGBoost, and a time-series approach), all evaluated with temporal train/validation/test splits to preserve chronological ordering and prevent information leakage.

My contribution was the **XGBoost** model, tuned with randomized search (20 iterations per horizon) over `n_estimators`, `max_depth`, `learning_rate`, regularization, and subsampling parameters.

## 📈 Results

Across model families, accuracy at the 3-day horizon landed in the ~50% range, and performance degraded steadily as the horizon lengthened. The fact that very different architectures clustered at a similar ceiling points to the available features and data quality as the binding constraint, not the choice of model.

XGBoost showed this pattern clearly across horizons:

| Horizon | Accuracy |
|---------|----------|
| 3-day   | 54.2%    |
| 7-day   | 40.6%    |
| 14-day  | 33.8%    |

It outperformed the logistic-regression baseline at every horizon and produced the most balanced precision-recall tradeoff among the models tested. Temperature and relative humidity emerged as the strongest predictors.

**Key findings:**

- Forecasting performance declined sharply as the prediction horizon increased.
- Temporal validation was essential for realistic performance estimates.
- Class imbalance was a persistent challenge given the low base rate of wildfire events.
- The shared ~50% ceiling across architectures indicates the bottleneck is data and feature quality rather than model selection.

## 🛠️ Stack

Python, XGBoost, scikit-learn, GeoPandas, Shapely, Optuna, SMOTE

