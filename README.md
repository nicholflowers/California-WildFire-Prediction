Predicting California Wildfires

Using machine learning to predict wildfire occurrence across 3, 7, and 14 day horizons at the climate division level in California.

📌 Overview

California wildfires cause billions in damage annually and are intensifying with climate change. This project explores whether machine learning models can predict not just whether a fire will occur, but which region of California it will occur in, using more than 40 years of weather and historical fire data.

The project compares multiple model families, explores the impact of geospatial and meteorological features, and examines the limits of wildfire predictability across increasingly distant forecasting horizons.

The prediction task is an 8-class classification problem consisting of 7 CONUS climate divisions and a no-fire class. A random classifier would achieve 12.5% accuracy.

📊 Data Sources
Weather data: California Weather and Fire Prediction Dataset (1984-2025) from Zenodo
Fire records: Cal Fire historical fire perimeters (1984-2025)
Localized weather: NASA POWER meteorological data
Geographic boundaries: CONUS climate divisions and US county shapefiles
Data Engineering

Fire locations were geolocated to climate divisions using Shapely and GeoPandas. Weather observations, fire history, climate division assignments, and geographic information were merged into a unified forecasting dataset.

Additional temporal and weather-based features were engineered, including lagged weather variables and derived environmental measurements.

Final merged dataset: 20,710 rows, 108 features.

🤖 Models
Logistic Regression (baseline)
Feedforward Neural Network (Optuna-tuned)
Random Forest (with PCA and SMOTE variants)
XGBoost (randomized search tuning)
ARIMA/SARIMAX (time series)

All models were evaluated using temporal train, validation, and test splits to preserve chronological ordering and prevent information leakage.

📈 Results

All models converged around 50% accuracy for the 3 day horizon. Performance degraded at longer horizons across every model family.

The convergence across different model architectures suggests the bottleneck is the available features rather than the model choice. Temperature and relative humidity emerged as the strongest predictors across multiple model families.

Key findings:

Forecasting performance declined as prediction horizons increased.
Temporal validation was critical for obtaining realistic estimates of forecasting performance.
Class imbalance remained a significant challenge due to the relatively low frequency of wildfire events.
ARIMA/SARIMAX achieved the strongest overall performance among the evaluated approaches.

The findings highlight the importance of data quality, feature engineering, temporal validation, and geospatial context when building forecasting models for complex environmental forecasting tasks.

🛠️ Stack

Python, XGBoost, scikit-learn, GeoPandas, Shapely, Optuna, SMOTE, SARIMAX
