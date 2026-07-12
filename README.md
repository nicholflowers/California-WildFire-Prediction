# Predicting California Wildfires

**Can machine learning predict not just whether a wildfire will occur, but where in California it will occur and how far ahead?**

---

## 📌 Overview

California wildfires cause billions in damage annually and are intensifying with climate change. Short-term prediction has real operational value: it supports early warning, informs how firefighting resources are positioned and gives policymakers something to plan against.

This project asks whether machine learning can forecast wildfire occurrence at the **climate-division level** across 3, 7 and 14-day horizons, using more than 40 years of California weather and fire history.

The task is an **8-class classification problem**: seven CONUS climate divisions plus a no-fire class. A random classifier would land at 12.5%. Beyond raw accuracy, the project examines how quickly predictability decays as the forecast window lengthens and what ultimately limits it.

## 📊 Data Sources

| Source | Contents |
|---|---|
| California Weather & Fire Prediction Dataset (Zenodo) | 14,988 rows, 1984–2025 |
| Cal Fire historical fire perimeters | Fire locations and dates, 1984–2025 |
| NASA POWER | Localized meteorological observations |
| CONUS climate divisions + US county shapefiles | Geographic boundaries |

**Final merged dataset:** 20,710 rows, 108 features, spanning 1984–2025.

## ⚙️ Data Engineering

The core challenge was that these sources don't naturally align. Fire records are point locations; weather is gridded; climate divisions are polygons.

- **Shapely** was used to find one fire centroid per climate division
- **GeoPandas** geo-located labels and features to counties and climate divisions
- Weather observations, fire history and geographic assignments were merged into a single forecasting dataset
- Engineered features included lagged weather variables (precipitation, temperature, wind speed), daily temperature range and a wind-temperature ratio

**Target variables:** fire in 3 days, fire in 7 days, fire in 14 days.

## 🔍 What EDA Revealed

- **Temperature and relative humidity separate fire days from non-fire days most cleanly**, consistently across all seven climate regions.
- **Strong seasonality.** Fire days cluster in the summer months across every region.
- **Class imbalance is regional, not just binary.** The overall no-fire rate is 64%, a moderate imbalance. The harder problem is that fire prevalence varies nearly **9-fold across climate regions**: 178 examples in Northeast Interior Basins versus 1,562 in San Joaquin Drainage. Sparse regions risk being effectively invisible to the model.

## 🤖 Models

Five model families were compared. All used **temporal train/validation/test splits** to preserve chronological order and prevent information leakage. A random split would let the model see the future.

| Model | Tuning |
|---|---|
| Logistic Regression | Baseline, balanced class weights |
| Feedforward Neural Network | Optuna hyperparameter search |
| Random Forest | With PCA and SMOTE variants |
| XGBoost | Random search, 20 iterations per horizon |
| ARIMA/SARIMAX | Auto-ARIMA order selection |

## 📈 Results

Two reference points matter here. **Random guessing** across 8 classes lands at 12.5%, the floor any model must clear to be doing anything at all. **Logistic regression** is the modeling baseline, the simplest real approach, and what the more complex models had to beat to justify their added complexity.

| Model | 3-day | 7-day | 14-day |
|---|---|---|---|
| Logistic Regression (modeling baseline) | 45.3% | 32.0% | 23.2% |
| Feedforward Neural Network | 50.8% | 36.7% | 25.7% |
| Random Forest | 53.3% | 40.5% | 34.1% |
| XGBoost | 54.2% | 40.6% | 33.8% |

*Accuracy. Random guessing over 8 classes = 12.5%.*

Every model cleared random guessing by a wide margin, confirming there is real predictive signal in the data. But the more informative result is what happened at the top: **four architectures with very different inductive biases (a linear model, a neural network and two tree ensembles) all converged near the same ~50% ceiling at the 3-day horizon** and all degraded sharply as the horizon lengthened.

When approaches that different fail in the same place, the model isn't the problem. **The binding constraint is the data**: the available features and the scarcity of positive fire cases, not the choice of algorithm.

**Note:** ARIMA/SARIMAX reported substantially higher accuracy (67.2% / 65.9% / 65.0%) with almost no degradation across horizons. That pattern is inconsistent with every other model and with the underlying difficulty of the task. It pointed to a flaw in that model's evaluation, which the team was unable to resolve before the project deadline. Those results are excluded from the conclusions above.

**Key findings:**

- **Temperature and relative humidity** were the strongest predictors across multiple model families.
- **Predictability decays fast.** XGBoost fell from 54.2% at 3 days to 33.8% at 14. Fire risk further out is substantially harder to capture.
- **Temporal splits were essential.** Random splits would have leaked future information and produced misleadingly optimistic results.
- **SMOTE improved recall for the minority classes by over 23%** in Random Forest, at a small cost to overall accuracy. The right trade when the rare class is the one you actually care about.
- **The ceiling is a data problem.** Better fire incident data, more complete and more balanced across regions, would do more for this task than a better algorithm.

## 🔭 Future Work

- Time-series architectures better suited to temporal dependency (LSTM, Prophet)
- Additional feature sources: vegetation, terrain, fuel moisture, powerline proximity. These are features published wildfire models rely on that we did not have
- Resolving the ARIMA evaluation issue

## 🧰 Stack

Python, XGBoost, scikit-learn, TensorFlow/Keras, Optuna, statsmodels, GeoPandas, Shapely, pandas, Matplotlib, Seaborn.
