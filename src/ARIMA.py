import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def fit_arima_model(df, column_name, order):
    """
    Fit an ARIMA model to the time series.

    Args:
    df (DataFrame): Pandas DataFrame with DateTime index and price data.
    column_name (str): The name of the column that contains the time series data.
    order (tuple): The (p, d, q) order of the model to be fitted.

    Returns:
    ARIMAResults: The fitted ARIMA model.
    """
    # Extract the time series
    time_series = df[column_name]
    
    # Ensure stationarity
    result = adfuller(time_series.dropna())
    if result[1] > 0.05:
        time_series = time_series.diff().dropna()  # Differencing if necessary

    # Fit the model
    model = ARIMA(time_series, order=order)
    fitted_model = model.fit()
    
    return fitted_model

def evaluate_model(fitted_model, steps=5):
    """
    Evaluate and forecast the ARIMA model.

    Args:
    fitted_model (ARIMAResults): The fitted ARIMA model from fit_arima_model.
    steps (int): Number of future steps to forecast.

    Returns:
    forecast (ndarray): Forecasted values.
    """
    # Print model summary
    print(fitted_model.summary())

    # Forecast
    forecast = fitted_model.forecast(steps=steps)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(fitted_model.model.endog, label='Original')
    plt.plot(np.arange(len(fitted_model.model.endog), len(fitted_model.model.endog) + steps), forecast, label='Forecast', color='red')
    plt.title('Forecast versus Actuals')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    return forecast
