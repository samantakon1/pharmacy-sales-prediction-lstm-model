# LSTM Pharmacy Sales Forecasting

Deep Learning project for time-series forecasting using Long Short-Term Memory (LSTM) networks.
The project focuses on forecasting pharmaceutical sales using transactional Point-of-Sale (POS)
data aggregated at daily and weekly levels.

The repository is organized to reflect the end-to-end ML workflow, separating raw and processed data, 
exploratory analysis, preprocessing, baseline modeling, LSTM experimentation, and standardized evaluation 
outputs to ensure reproducibility and clear comparison.

---

## Project Structure
    pharmacy-sales-prediction-lstm-model/
    │
    ├── data/
    │ ├── raw/
    │ │ ├── sales_daily.csv # Daily aggregated pharmacy sales data
    │ │ └── sales_weekly.csv # Weekly aggregated pharmacy sales data
    │ │
    │ └── processed/
    │ ├── daily_processed.csv # Preprocessed daily dataset (scaled, windowed)
    │ └── weekly_processed.csv # Preprocessed weekly dataset
    │
    ├── notebooks/
    │ ├── 01_eda.ipynb # Exploratory Data Analysis
    │ ├── 02_preprocessing.ipynb # Data cleaning, scaling, windowing
    │ ├── 03_baseline_models.ipynb # Naive, Moving Average, ARIMA models
    │ └── 04_lstm_model.ipynb # LSTM model training and evaluation
    │
    ├── src/
    │ ├── preprocessing.py # Helper functions for preprocessing
    │ ├── baselines.py # Baseline forecasting models
    │ ├── lstm_model.py # LSTM model definition and training
    │ └── evaluation.py # Evaluation metrics and plotting utilities
    │
    ├── results/
    │ ├── figures/ # Generated plots and visualizations
    │ └── metrics/ # Model evaluation results (MAE, RMSE)
    │
    ├── requirements.txt # Python dependencies
    ├── README.md # Project documentation
    └── .gitignore

---

## Dataset

Source: Kaggle  
Dataset: *Wellness Pharmacy Sales Analysis*  
Link: https://www.kaggle.com/datasets/michaelhakim/wellness-pharmacy-sales-analysis

Used files:
- `sales_daily.csv` (primary dataset)
- `sales_weekly.csv` (secondary dataset for comparison)

The data originates from a pharmacy POS system and spans the period 2014–2019.

---

## Workflow

1. **Exploratory Data Analysis**
   - Trend and seasonality inspection
   - Missing value and outlier checks

2. **Preprocessing**
   - Time-series scaling
   - Sliding window transformation
   - Chronological train/validation/test split

3. **Baseline Models**
   - Naive forecast
   - Moving average
   - ARIMA / SARIMA

4. **LSTM Model**
   - Sequence-to-one LSTM architecture
   - Daily sales as main experiment
   - Weekly sales as comparative analysis

5. **Evaluation**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)

---

## How to Run

1. **Install dependencies:**
    pip install -r requirements.txt

2. **Run notebooks in order:**

    - 01_eda.ipynb
    - 02_preprocessing.ipynb
    - 03_baseline_models.ipynb
    - 04_lstm_model.ipynb
---

## Notes

- Time-based splitting is used to avoid data leakage.
- Daily data is used as the primary forecasting task.
- Weekly data is used to analyze the effect of temporal aggregation.
- The project is part of an MSc Deep Learning course assignment.

---

## Author

Samanta Koni