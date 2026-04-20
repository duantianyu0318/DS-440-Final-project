import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_model(model_type: str = 'Linear Regression'):
    if model_type == 'Random Forest':
        return RandomForestRegressor(n_estimators=200, random_state=42)
    return LinearRegression()


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def invert_scale(values, scaler):
    arr = np.array(values).reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()


def summarize_results(model, bundle: dict):
    y_pred_scaled = model.predict(bundle['X_test'])
    y_true = invert_scale(bundle['y_test'], bundle['scaler'])
    y_pred = invert_scale(y_pred_scaled, bundle['scaler'])
    next_close = invert_scale(model.predict(bundle['last_window']), bundle['scaler'])[0]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')

    result_df = pd.DataFrame({
        'Date': pd.to_datetime(bundle['test_dates']),
        'Actual Close': y_true,
        'Predicted Close': y_pred,
    })

    return {
        'result_df': result_df,
        'next_close': float(next_close),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'latest_close': float(bundle['last_actual_close'])
    }


def plot_results(result_df: pd.DataFrame, ticker: str, model_type: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(result_df['Date'], result_df['Actual Close'], label='Actual Close', linewidth=2)
    ax.plot(result_df['Date'], result_df['Predicted Close'], label='Predicted Close', linewidth=2)
    ax.set_title(f'{ticker} - Actual vs Predicted ({model_type})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
