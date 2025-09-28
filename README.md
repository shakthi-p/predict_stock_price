## LSTM Stock Price Prediction
Predicting stock prices using historical data and a **Long Short-Term Memory (LSTM)** network in PyTorch. This project demonstrates time series forecasting with sequence modeling and provides visualizations of predictions versus actual stock prices.

## Project Overview

This mini-project performs the following tasks:
1. **Data Loading & Preprocessing**
   * Load historical stock price data from CSV
   * Drop unnecessary columns and handle missing values
   * Convert `Date` column to datetime format

2. **Feature Engineering**
   * Generate 7-day moving average (`MA7`)
   * Select relevant features: `Open`, `High`, `Low`, `Close`, `MA7`
   * Scale features using `MinMaxScaler`

3. **Sequence Creation for LSTM**
   * Convert data into sequences for supervised learning
   * Split into **train**, **validation**, and **test** sets

4. **LSTM Model Implementation**
   * Custom PyTorch LSTM with 2 layers, hidden size 128, and dropout
   * Trained with **Smooth L1 Loss (Huber loss)** and Adam optimizer
   * Early stopping to prevent overfitting

5. **Evaluation & Visualization**
   * Metrics: **MAE**, **RMSE**, **MAPE**
   * Plots:
     * Training loss curve
     * Actual vs predicted stock prices
     * Prediction error and distribution

## Dataset

* File: `StockData.csv`
* Contains historical stock data with columns such as:
  * `Date` – Trading date
  * `Open` – Opening price
  * `High` – Highest price of the day
  * `Low` – Lowest price of the day
  * `Close` – Closing price
  * `Volume` – Trading volume (optional)

## Output includes:

   * Training loss curve
   * Actual vs predicted stock prices
   * Prediction error plot and histogram
   * Metrics printed in the console

## Output

* **Training Loss Curve:** Shows how the loss decreases over epochs
* **Actual vs Predicted Prices:** Visual comparison of predictions vs real stock prices
* **Prediction Error:** Visualize differences and distribution of errors


## Technologies Used

* Python 3.x
* PyTorch
* NumPy, Pandas
* Matplotlib
* scikit-learn

