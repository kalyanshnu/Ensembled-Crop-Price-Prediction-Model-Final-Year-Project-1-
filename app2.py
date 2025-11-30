import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Price Prediction Analytics Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #2E8B57;
        border-bottom: 3px solid #2E8B57;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0;
    }
    .comparison-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.markdown('<h1 class="main-header">🌾 Crop Price Prediction Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Section:",
    ["🏠 Overview", "📈 Model Performance", "🔄 Model Comparison", "🌾 Ensemble vs ARIMA", "📊 Interactive Analytics", "🎯 Predictions"]
)

# Sample data generation (replace with your actual data)
@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Model performance data
    models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Linear Regression', 'Ensemble (Weighted)', 'ARIMA']
    crops = ['Onion', 'Carrot', 'Radish', 'Pumpkin']
    
    performance_data = []
    for crop in crops:
        for model in models:
            if model == 'ARIMA':
                mae = np.random.uniform(15, 35)
                rmse = np.random.uniform(20, 45)
                r2 = np.random.uniform(0.6, 0.85)
                mape = np.random.uniform(8, 25)
            elif 'Ensemble' in model:
                mae = np.random.uniform(8, 18)
                rmse = np.random.uniform(12, 25)
                r2 = np.random.uniform(0.85, 0.95)
                mape = np.random.uniform(3, 12)
            else:
                mae = np.random.uniform(10, 30)
                rmse = np.random.uniform(15, 40)
                r2 = np.random.uniform(0.7, 0.9)
                mape = np.random.uniform(5, 20)
            
            performance_data.append({
                'Crop': crop,
                'Model': model,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Time series data for predictions
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    actual_prices = 100 + np.cumsum(np.random.randn(365) * 2) + 20 * np.sin(np.arange(365) * 2 * np.pi / 365)
    ensemble_pred = actual_prices + np.random.normal(0, 5, 365)
    arima_pred = actual_prices + np.random.normal(0, 8, 365)
    
    time_series_data = pd.DataFrame({
        'Date': dates,
        'Actual': actual_prices,
        'Ensemble_Prediction': ensemble_pred,
        'ARIMA_Prediction': arima_pred
    })
    
    return performance_df, time_series_data

# Load data
performance_df, time_series_data = load_sample_data()

# Page content based on selection
if page == "🏠 Overview":
    st.markdown('<h2 class="section-header">📊 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", "6", "3 Ensemble Methods")
    
    with col2:
        st.metric("Crops Analyzed", "4", "Onion, Carrot, Radish, Pumpkin")
    
    with col3:
        best_r2 = performance_df['R²'].max()
        st.metric("Best R² Score", f"{best_r2:.3f}", "Ensemble Model")
    
    with col4:
        best_mae = performance_df['MAE'].min()
        st.metric("Best MAE", f"{best_mae:.2f}", "Lower is Better")
    
    st.markdown("---")
    
    # Key insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">🔑 Key Insights</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Ensemble methods consistently outperform individual models**
        - **Best overall performance**: Weighted Ensemble approach
        - **ARIMA shows good performance for time-series patterns**
        - **Multi-crop analysis for Onion, Carrot, Radish & Pumpkin**
        - **Performance varies significantly across different vegetable crops**
        """)
    
    with col2:
        st.markdown('<h3 class="section-header">📈 Quick Stats</h3>', unsafe_allow_html=True)
        avg_ensemble_r2 = performance_df[performance_df['Model'].str.contains('Ensemble')]['R²'].mean()
        avg_individual_r2 = performance_df[~performance_df['Model'].str.contains('Ensemble|ARIMA')]['R²'].mean()
        
        st.info(f"🎯 **Ensemble Avg R²**: {avg_ensemble_r2:.3f}")
        st.info(f"📊 **Individual Avg R²**: {avg_individual_r2:.3f}")
        st.success(f"🚀 **Improvement**: {((avg_ensemble_r2/avg_individual_r2 - 1) * 100):.1f}%")

elif page == "📈 Model Performance":
    st.markdown('<h2 class="section-header">📈 Individual Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Performance metrics comparison
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # MAE comparison
        fig_mae = px.box(performance_df, x='Model', y='MAE', color='Model',
                        title="MAE Distribution Across Models")
        fig_mae.update_layout(height=400, showlegend=False)
        fig_mae.update_xaxes(tickangle=45)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        # R² comparison
        fig_r2 = px.box(performance_df, x='Model', y='R²', color='Model',
                       title="R² Score Distribution Across Models")
        fig_r2.update_layout(height=400, showlegend=False)
        fig_r2.update_xaxes(tickangle=45)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Detailed performance table
    st.markdown('<h3 class="section-header">📊 Detailed Performance Metrics</h3>', unsafe_allow_html=True)
    
    # Calculate summary statistics
    summary_stats = performance_df.groupby('Model')[['MAE', 'RMSE', 'R²', 'MAPE']].agg({
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'R²': ['mean', 'std', 'min', 'max'],
        'MAPE': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    st.dataframe(summary_stats, use_container_width=True, height=300)
    
    # Crop-wise performance heatmap
    st.markdown('<h3 class="section-header">🌡️ Performance Heatmap by Crop</h3>', unsafe_allow_html=True)
    
    metric_choice = st.selectbox("Select Metric for Heatmap:", ['MAE', 'RMSE', 'R²', 'MAPE'])
    
    heatmap_data = performance_df.pivot(index='Crop', columns='Model', values=metric_choice)
    
    fig_heatmap = px.imshow(heatmap_data, 
                           labels=dict(x="Model", y="Crop", color=metric_choice),
                           title=f"{metric_choice} Heatmap: Model vs Crop Performance")
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "🔄 Model Comparison":
    st.markdown('<h2 class="section-header">🔄 Comprehensive Model Comparison</h2>', unsafe_allow_html=True)
    
    # Model ranking
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate composite scores
        def calculate_composite_score(row):
            # Normalize metrics (lower is better for MAE, RMSE, MAPE; higher is better for R²)
            mae_norm = 1 / (1 + row['MAE'])
            rmse_norm = 1 / (1 + row['RMSE'])
            mape_norm = 1 / (1 + row['MAPE'])
            r2_norm = max(0, row['R²'])
            
            return (mae_norm + rmse_norm + mape_norm + r2_norm) / 4
        
        model_summary = performance_df.groupby('Model')[['MAE', 'RMSE', 'R²', 'MAPE']].mean()
        model_summary['Composite_Score'] = model_summary.apply(calculate_composite_score, axis=1)
        model_summary = model_summary.sort_values('Composite_Score', ascending=False)
        
        # Ranking chart
        fig_ranking = px.bar(
            x=model_summary.index,
            y=model_summary['Composite_Score'],
            title="🏆 Model Ranking by Composite Performance Score",
            labels={'x': 'Model', 'y': 'Composite Score'},
            color=model_summary['Composite_Score'],
            color_continuous_scale='viridis'
        )
        fig_ranking.update_layout(height=400)
        st.plotly_chart(fig_ranking, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">🥇 Top Performers</h3>', unsafe_allow_html=True)
        
        top_models = model_summary.head(3)
        for i, (model, scores) in enumerate(top_models.iterrows(), 1):
            if i == 1:
                st.success(f"🥇 **{model}**\nScore: {scores['Composite_Score']:.4f}")
            elif i == 2:
                st.info(f"🥈 **{model}**\nScore: {scores['Composite_Score']:.4f}")
            else:
                st.warning(f"🥉 **{model}**\nScore: {scores['Composite_Score']:.4f}")
    
    # Detailed comparison table
    st.markdown('<h3 class="section-header">📋 Detailed Comparison Table</h3>', unsafe_allow_html=True)
    
    comparison_df = model_summary[['MAE', 'RMSE', 'R²', 'MAPE', 'Composite_Score']].round(4)
    comparison_df = comparison_df.reset_index()
    
    st.dataframe(
        comparison_df.style.highlight_max(subset=['R²', 'Composite_Score'], color='lightgreen')
                          .highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen'),
        use_container_width=True
    )
    
    # Radar chart for top 3 models
    st.markdown('<h3 class="section-header">🕸️ Performance Radar Chart</h3>', unsafe_allow_html=True)
    
    top_3_models = model_summary.head(3)
    
    fig_radar = go.Figure()
    
    metrics = ['MAE', 'RMSE', 'R²', 'MAPE']
    
    for model in top_3_models.index:
        values = []
        for metric in metrics:
            if metric in ['MAE', 'RMSE', 'MAPE']:
                # Invert for radar chart (higher is better)
                values.append(1 / (1 + top_3_models.loc[model, metric]))
            else:
                values.append(top_3_models.loc[model, metric])
        
        values += values[:1]  # Complete the polygon
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Top 3 Models Performance Comparison",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

elif page == "🌾 Ensemble vs ARIMA":
    st.markdown('<h2 class="section-header">🌾 Ensemble vs ARIMA Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data for ensemble vs ARIMA
    ensemble_data = performance_df[performance_df['Model'].str.contains('Ensemble')]
    arima_data = performance_df[performance_df['Model'] == 'ARIMA']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series comparison plot
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            x=time_series_data['Date'],
            y=time_series_data['Actual'],
            mode='lines',
            name='Actual Prices',
            line=dict(color='black', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=time_series_data['Date'],
            y=time_series_data['Ensemble_Prediction'],
            mode='lines',
            name='Ensemble Prediction',
            line=dict(color='green', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=time_series_data['Date'],
            y=time_series_data['ARIMA_Prediction'],
            mode='lines',
            name='ARIMA Prediction',
            line=dict(color='blue', width=2)
        ))
        
        fig_ts.update_layout(
            title="🔄 Ensemble vs ARIMA Predictions Over Time",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">📊 Performance Comparison</h3>', unsafe_allow_html=True)
        
        ensemble_avg = ensemble_data[['MAE', 'RMSE', 'R²', 'MAPE']].mean()
        arima_avg = arima_data[['MAE', 'RMSE', 'R²', 'MAPE']].mean()
        
        metrics_comp = ['MAE', 'RMSE', 'R²', 'MAPE']
        
        for metric in metrics_comp:
            ensemble_val = ensemble_avg[metric]
            arima_val = arima_avg[metric]
            
            if metric == 'R²':
                improvement = ((ensemble_val / arima_val - 1) * 100)
                better = "📈" if ensemble_val > arima_val else "📉"
            else:
                improvement = ((arima_val / ensemble_val - 1) * 100)
                better = "📉" if ensemble_val < arima_val else "📈"
            
            st.metric(
                f"{metric}",
                f"E: {ensemble_val:.3f} | A: {arima_val:.3f}",
                f"{better} {improvement:.1f}%"
            )
    
    # Detailed comparison by crop
    st.markdown('<h3 class="section-header">🌾 Crop-wise Ensemble vs ARIMA</h3>', unsafe_allow_html=True)
    
    comparison_metrics = st.selectbox("Select Metric:", ['MAE', 'RMSE', 'R²', 'MAPE'], key='ensemble_arima')
    
    ensemble_vs_arima = performance_df[performance_df['Model'].isin(['Ensemble (Weighted)', 'ARIMA'])]
    
    fig_comparison = px.bar(
        ensemble_vs_arima,
        x='Crop',
        y=comparison_metrics,
        color='Model',
        barmode='group',
        title=f"{comparison_metrics} Comparison: Ensemble vs ARIMA by Crop"
    )
    fig_comparison.update_layout(height=400)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Statistical significance test simulation
    st.markdown('<h3 class="section-header">📈 Statistical Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate improvement percentages
        mae_improvement = ((arima_data['MAE'].mean() - ensemble_data['MAE'].mean()) / arima_data['MAE'].mean()) * 100
        st.success(f"🎯 **MAE Improvement**\n{mae_improvement:.1f}%")
    
    with col2:
        rmse_improvement = ((arima_data['RMSE'].mean() - ensemble_data['RMSE'].mean()) / arima_data['RMSE'].mean()) * 100
        st.success(f"📊 **RMSE Improvement**\n{rmse_improvement:.1f}%")
    
    with col3:
        r2_improvement = ((ensemble_data['R²'].mean() - arima_data['R²'].mean()) / arima_data['R²'].mean()) * 100
        st.success(f"🚀 **R² Improvement**\n{r2_improvement:.1f}%")

elif page == "📊 Interactive Analytics":
    st.markdown('<h2 class="section-header">📊 Interactive Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Interactive filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_crops = st.multiselect(
            "Select Crops:",
            options=performance_df['Crop'].unique(),
            default=performance_df['Crop'].unique()[:3]
        )
    
    with col2:
        selected_models = st.multiselect(
            "Select Models:",
            options=performance_df['Model'].unique(),
            default=performance_df['Model'].unique()
        )
    
    with col3:
        selected_metric = st.selectbox(
            "Primary Metric:",
            options=['MAE', 'RMSE', 'R²', 'MAPE']
        )
    
    # Filter data
    filtered_df = performance_df[
        (performance_df['Crop'].isin(selected_crops)) & 
        (performance_df['Model'].isin(selected_models))
    ]
    
    if not filtered_df.empty:
        # Interactive scatter plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_scatter = px.scatter(
                filtered_df,
                x='MAE',
                y='R²',
                color='Model',
                size='RMSE',
                hover_data=['Crop', 'MAPE'],
                title="🔍 Interactive Model Performance Scatter Plot"
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="section-header">📊 Summary Stats</h3>', unsafe_allow_html=True)
            
            summary = filtered_df.groupby('Model')[selected_metric].agg(['mean', 'std', 'min', 'max']).round(3)
            st.dataframe(summary, use_container_width=True)
        
        # Correlation analysis
        st.markdown('<h3 class="section-header">🔗 Metric Correlations</h3>', unsafe_allow_html=True)
        
        corr_matrix = filtered_df[['MAE', 'RMSE', 'R²', 'MAPE']].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Correlation Matrix of Performance Metrics",
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution analysis
        st.markdown('<h3 class="section-header">📈 Performance Distributions</h3>', unsafe_allow_html=True)
        
        fig_violin = px.violin(
            filtered_df,
            x='Model',
            y=selected_metric,
            color='Model',
            title=f"{selected_metric} Distribution by Model",
            box=True
        )
        fig_violin.update_layout(height=400, showlegend=False)
        fig_violin.update_xaxes(tickangle=45)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")

else:  # Predictions page
    st.markdown('<h2 class="section-header">🎯 Model Predictions & Forecasting</h2>', unsafe_allow_html=True)
    
    # Prediction interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h3 class="section-header">🔧 Prediction Settings</h3>', unsafe_allow_html=True)
        
        pred_crop = st.selectbox("Select Crop:", ['Onion', 'Carrot', 'Radish', 'Pumpkin'])
        pred_model = st.selectbox("Select Model:", performance_df['Model'].unique())
        forecast_days = st.slider("Forecast Days:", 1, 30, 7)
        
        # Mock prediction parameters
        st.markdown("**Input Parameters:**")
        temperature = st.slider("Temperature (°C):", 15, 45, 25)
        rainfall = st.slider("Rainfall (mm):", 0, 200, 50)
        market_demand = st.slider("Market Demand:", 0.5, 2.0, 1.0)
        
        if st.button("🚀 Generate Prediction"):
            # Mock prediction calculation
            base_price = np.random.uniform(80, 150)
            
            # Simulate model-specific adjustments
            if 'Ensemble' in pred_model:
                prediction_accuracy = 0.95
                confidence_interval = 5
            elif pred_model == 'ARIMA':
                prediction_accuracy = 0.88
                confidence_interval = 8
            else:
                prediction_accuracy = 0.82
                confidence_interval = 10
            
            predicted_price = base_price * (1 + (temperature - 25) * 0.01) * market_demand
            
            st.success(f"🎯 **Predicted Price**: ₹{predicted_price:.2f}")
            st.info(f"📊 **Confidence**: {prediction_accuracy:.1%}")
            st.info(f"🔄 **Range**: ±₹{confidence_interval}")
    
    with col2:
        # Forecast visualization
        forecast_dates = pd.date_range(start='2024-01-01', periods=forecast_days, freq='D')
        base_prices = np.random.uniform(90, 120, forecast_days)
        noise = np.random.normal(0, 5, forecast_days)
        forecast_prices = base_prices + noise
        
        # Create confidence bands
        upper_band = forecast_prices + np.random.uniform(5, 15, forecast_days)
        lower_band = forecast_prices - np.random.uniform(5, 15, forecast_days)
        
        fig_forecast = go.Figure()
        
        # Add confidence bands
        fig_forecast.add_trace(go.Scatter(
            x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
            y=np.concatenate([upper_band, lower_band[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band'
        ))
        
        # Add forecast line
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', width=3)
        ))
        
        fig_forecast.update_layout(
            title=f"📈 {forecast_days}-Day Price Forecast for {pred_crop} using {pred_model}",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Recent predictions table
    st.markdown('<h3 class="section-header">📋 Recent Predictions Log</h3>', unsafe_allow_html=True)
    
    # Mock recent predictions data
    recent_predictions = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'Crop': np.random.choice(['Onion', 'Carrot', 'Radish', 'Pumpkin'], 10),
        'Model': np.random.choice(performance_df['Model'].unique(), 10),
        'Predicted_Price': np.random.uniform(80, 150, 10).round(2),
        'Actual_Price': np.random.uniform(75, 155, 10).round(2),
        'Accuracy': np.random.uniform(0.85, 0.98, 10).round(3)
    })
    
    recent_predictions['Error'] = abs(recent_predictions['Predicted_Price'] - recent_predictions['Actual_Price']).round(2)
    
    st.dataframe(
        recent_predictions.style.background_gradient(subset=['Accuracy'], cmap='RdYlGn'),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 Crop Price Prediction Analytics Dashboard | Built with Streamlit</p>
    <p>📊 Real-time Model Performance Monitoring & Analysis</p>
</div>
""", unsafe_allow_html=True)
