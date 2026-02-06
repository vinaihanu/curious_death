# curious_death
# 📈 Time Series Forecasting using SARIMAX and Attention-LSTM

## 📌 Project Overview
This project implements a **comprehensive time series forecasting pipeline** combining a classical statistical model (**SARIMAX**) and a deep learning model (**Attention-based LSTM**).  
The goal is to model a **complex, high-frequency, non-stationary multivariate time series** and evaluate forecasting performance using multiple metrics and visualizations.

---

## 🧠 Key Objectives
- Simulate a realistic **non-stationary time series dataset**
- Establish a **SARIMAX baseline model**
- Implement an **Attention-based LSTM** using PyTorch
- Perform **hyperparameter optimization** using Optuna
- Conduct **error analysis** and identify worst predictions
- Visualize **model predictions and attention weights**

---

## 📊 Dataset Description
- **Samples:** 12,000 time steps  
- **Input Features:** 5 exogenous variables (`x1` – `x5`)  
- **Target Variable:** `y`  
- Characteristics:
  - Trend component
  - Multiple seasonal patterns
  - Gaussian noise
  - Non-stationary behavior

---

## 🏗️ Model Architecture

### 1️⃣ SARIMAX (Baseline)
- Order: (1, 1, 1)
- Exogenous variables included
- Used as a **classical benchmark** for comparison

### 2️⃣ Attention-Based LSTM
- LSTM layer for temporal modeling
- Attention mechanism to compute **temporal importance weights**
- Fully connected output layer for regression
- Implemented using **PyTorch**

---

## ⚙️ Hyperparameter Optimization
- Tool: **Optuna**
- Tuned Parameters:
  - Hidden layer size
  - Learning rate
- Objective: Minimize Mean Squared Error (MSE)

---

## 📈 Evaluation Metrics
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- Identification of **Top-5 worst prediction indices**

---

## 📉 Visualizations
1. **Actual vs Predicted Values**
   - Comparison of first 200 test samples
   - Demonstrates forecasting accuracy

2. **Average Attention Weights**
   - Visualizes how the model focuses on temporal information
   - Provides interpretability of the Attention mechanism

---

## ▶️ How to Run (Google Colab Recommended)

### Step 1: Install Dependencies
Step 2: Run the Notebook

Copy the full Python code into a single Colab notebook

Run all cells sequentially

Outputs include:

SARIMAX RMSE

Attention-LSTM MAE & RMSE

Worst prediction indices

Two visualizations

📦 Libraries Used

Python

NumPy

Pandas

PyTorch

Scikit-learn

Statsmodels

Optuna

Matplotlib

✅ Conclusion

This project demonstrates how classical statistical models and deep learning approaches can be combined to handle complex time series forecasting tasks.
The Attention-LSTM model improves flexibility and interpretability, while SARIMAX provides a strong baseline for comparison.
```python
!pip install -q statsmodels optuna
