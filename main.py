import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Feature Engineering: Create 'Lag' features from past spending

def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    # Aggregate by day and fill missing days with zero
    df = df.groupby('Date')['Amount'].sum().resample('D').sum().fillna(0).reset_index()

    # Create lag features (spending from the last 7 days)
    for i in range(1, 8):
        df[f'lag_{i}'] = df['Amount'].shift(i)
    
    # Add a 7-day rolling average
    df['rolling_mean_7'] = df['Amount'].shift(1).rolling(window=7).mean()
    return df.dropna()


# 2. Scikit-learn Pipeline: Standardize features and train model
def train_forecast_model(df):
    X = df.drop(columns=['Date', 'Amount'])
    y = df['Amount']

    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Normalizes data for better performance
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# 3. Streamlit UI
st.title("💰 Finance Forecaster")
uploaded_file = st.file_uploader("Upload bank transaction CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    processed_data = prepare_data(data)
    model = train_forecast_model(processed_data)

    # Predict spending for the next day using the latest known features
    latest_features = processed_data.iloc[-1:].drop(columns=['Date', 'Amount'])
    prediction = model.predict(latest_features)

    st.metric("Predicted Tomorrow's Spend", f"${prediction[0]:.2f}")
