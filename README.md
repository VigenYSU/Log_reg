README.md

# Logistic Regression on Rain Prediction (weatherAUS Dataset)

## 📁 Dataset Overview

The dataset used is `weatherAUS.csv`, originally from the Australian Bureau of Meteorology, containing daily weather observations from multiple locations across Australia.

- **Original shape:** 145,460 rows × 23 columns
- **Goal:** Predict whether it will rain tomorrow (`RainTomorrow`)
- **Target column:** `RainTomorrow` (categorical: `Yes` or `No`)

---

## 📊 Column Data Types

| Column             | Type     | Description                                     |
|--------------------|----------|-------------------------------------------------|
| `Date`             | object   | Observation date                                |
| `Location`         | object   | Weather station location                        |
| `MinTemp`          | float64  | Minimum temperature (°C)                        |
| `MaxTemp`          | float64  | Maximum temperature (°C)                        |
| `Rainfall`         | float64  | Amount of rainfall (mm)                         |
| `Evaporation`      | float64  | Amount of evaporation (mm)                      |
| `Sunshine`         | float64  | Hours of sunshine                               |
| `WindGustSpeed`    | float64  | Speed of strongest wind gust (km/h)            |
| `WindSpeed9am`     | float64  | Wind speed at 9am (km/h)                        |
| `WindSpeed3pm`     | float64  | Wind speed at 3pm (km/h)                        |
| `Humidity9am`      | float64  | Humidity at 9am (%)                             |
| `Humidity3pm`      | float64  | Humidity at 3pm (%)                             |
| `Pressure9am`      | float64  | Atmospheric pressure at 9am (hPa)              |
| `Pressure3pm`      | float64  | Atmospheric pressure at 3pm (hPa)              |
| `Temp9am`          | float64  | Temperature at 9am (°C)                         |
| `Temp3pm`          | float64  | Temperature at 3pm (°C)                         |
| `RainToday`        | object   | Whether it rained today                         |
| `RainTomorrow`     | object   | **Target variable**                             |

---

## 📈 Statistical Summary (Selected Columns)

| Feature        | Mean   | Max    | 25%    | 50%    | 75%    |
|----------------|--------|--------|--------|--------|--------|
| MinTemp        | 12.19  | 28.0   | 6.8    | 12.3   | 17.5   |
| MaxTemp        | 23.26  | 48.1   | 17.9   | 22.9   | 28.3   |
| Rainfall       | 2.36   | 371.0  | 0.0    | 0.0    | 0.8    |
| Evaporation    | 4.66   | 145.0  | 3.2    | 4.4    | 5.8    |
| Sunshine       | 7.61   | 14.5   | 4.5    | 8.6    | 10.4   |
| WindGustSpeed  | 39.98  | 135.0  | 30.0   | 39.0   | 48.0   |
| Humidity3pm    | 46.3   | 100.0  | 32.0   | 44.0   | 59.0   |
| Pressure3pm    | 1015.3 | 1041.0 | 1011.2 | 1015.5 | 1019.6 |

> These values reflect the cleaned dataset used for training.

---

## 🧹 Cleaning & Preprocessing

- Dropped rows with missing values in:

- **Remaining rows after cleaning:** 65,434
- **Rows dropped:** ~80,000+ with NA in critical columns
- **Balanced dataset:** 14,430 "Yes" and 14,430 "No" samples (randomly sampled from original)

---

## 📌 Model Summary

- **Model:** Logistic Regression (implemented from scratch)
- **Learning rate:** 0.01
- **Epochs:** 2000
- **Final training loss:** 0.4455

### Coefficients (Feature Importance)

| Feature        | Coefficient |
|----------------|-------------|
| Sunshine       | **-0.7619** |
| Humidity3pm    | **+0.7158** |
| WindGustSpeed  | **+0.5209** |
| Pressure3pm    | **-0.3733** |
| Rainfall       | +0.1930     |
| Others         | moderate    |

---

## 📊 Evaluation

- **Test Accuracy:** 79.42%
- **Confusion Matrix:**



- **Loss Metrics:**
- MSE: 0.1420
- RMSE: 0.3769
- MAE: 0.2911

---

## 🔍 Correlation Matrix

| Feature1      | Feature2       | Correlation |
|---------------|----------------|-------------|
| Temp9am       | Temp3pm        | +0.98       |
| Humidity9am   | Humidity3pm    | +0.85       |
| Pressure9am   | Pressure3pm    | +0.99       |
| MinTemp       | MaxTemp        | +0.89       |
| Rainfall      | Humidity3pm    | +0.54       |
| Sunshine      | Humidity3pm    | **-0.67**   |

> Strong positive/negative correlations reflect high collinearity between measurements at 9am and 3pm.

---

## 📊 Visual Analysis

### 1. **Feature Importance Plot**
- Horizontal barplot of model coefficients (`seaborn`)

### 2. **Histograms**
- Distributions for selected features (e.g. Rainfall, Sunshine)

### 3. **Boxplots**
- RainTomorrow vs. Humidity3pm, Sunshine, Pressure3pm

### 4. **Heatmap**
- Correlation heatmap to detect multicollinearity

---

## ✅ Summary

- Logistic regression model from scratch achieved **~79.4% test accuracy** on a balanced dataset.
- Key features: **Humidity3pm**, **Sunshine**, **WindGustSpeed**
- All preprocessing, training, and evaluation steps were implemented manually.
- Final model provides interpretable results and robust performance.

---

