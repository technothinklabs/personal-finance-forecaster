import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Enhanced Data Preparation
def prepare_time_series(df, freq='D'):
    df['Date'] = pd.to_datetime(df['Date'])
    # Aggregate by chosen frequency (D=Daily, W=Weekly, ME=Month End)
    df_resampled = df.groupby('Date')['Amount'].sum().resample(freq).sum().fillna(0).reset_index()
    
    # Create Lag Features (Last 3 periods)
    for i in range(1, 4):
        df_resampled[f'lag_{i}'] = df_resampled['Amount'].shift(i)
    
    return df_resampled.dropna()

# 2. Forecasting Function
def get_forecast(df, steps=1):
    X = df.drop(columns=['Date', 'Amount'])
    y = df['Amount']
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    
    # Predict the next immediate step
    latest_features = df.iloc[-1:].drop(columns=['Date', 'Amount'])
    return model.predict(latest_features)[0], model

def show_sidebar_insights(model, feature_names):
    with st.sidebar:
        st.header("🧠 Model Insights")
        st.write("This section shows how the AI makes decisions.")
        
        # 1. Extract Feature Importance
        # Access the regressor inside the pipeline
        importances = model.named_steps['regressor'].feature_importances_
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 2. Display as a Horizontal Bar Chart
        st.subheader("Key Spending Drivers")
        fig_importance = px.bar(
            feature_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="What drives the forecast?",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_importance, width="stretch")

        # 3. Model Parameters (Shows professionalism)
        with st.expander("Model Configuration"):
            params = model.named_steps['regressor'].get_params()
            st.json({
                "Algorithm": "Random Forest Regressor",
                "n_estimators": params['n_estimators'],
                "random_state": params['random_state']
            })



# 3. Streamlit UI
st.set_page_config(page_title="Finance AI", layout="wide")
st.title("📈 Advanced Finance Forecast & Visualization")

uploaded_file = st.file_uploader("Upload transactions.csv", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    
    # --- VISUALIZATION SECTION ---
    st.subheader("Historical Spending Trends")
    chart_freq = st.selectbox("View Trend By:", ["Daily", "Weekly", "Monthly"])
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    
    plot_df = raw_data.copy()
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df = plot_df.groupby('Date')['Amount'].sum().resample(freq_map[chart_freq]).sum().reset_index()
    
    fig = px.line(plot_df, x='Date', y='Amount', title=f"{chart_freq} Spending Over Time",
                 line_shape="spline", render_mode="svg")
    st.plotly_chart(fig, width="stretch")

    # --- PREDICTION SECTION ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    # Daily Predict
    daily_df = prepare_time_series(raw_data, 'D')
    next_day, model_daily = get_forecast(daily_df)
    col1.metric("Predicted Tomorrow", f"${next_day:,.2f}")
    
    # Weekly Predict
    weekly_df = prepare_time_series(raw_data, 'W')
    next_week, model_weekly = get_forecast(weekly_df)
    col2.metric("Predicted Next Week", f"${next_week:,.2f}")
    
    # Monthly Predict
    monthly_df = prepare_time_series(raw_data, 'ME')
    next_month, model_monthly = get_forecast(monthly_df)
    col3.metric("Predicted Next Month", f"${next_month:,.2f}")

    # --- CATEGORY BREAKDOWN ---
    st.subheader("Spending by Category")
    cat_df = raw_data.groupby('Category')['Amount'].sum().reset_index()
    fig_pie = px.pie(cat_df, values='Amount', names='Category', hole=0.4)
    st.plotly_chart(fig_pie)

    # Define feature names for the chart
    feature_cols = [col for col in daily_df.columns if col not in ['Date', 'Amount']]
    
    # Call the sidebar function
    show_sidebar_insights(model_daily, feature_cols)
