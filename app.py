import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import os
import joblib

# ==============================================================================
# PAGE CONFIGURATION & CUSTOM CSS (PROFESSIONAL DARK THEME)
# ==============================================================================
st.set_page_config(
    page_title="Walmart Demand Forecasting & Inventory Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark Modern Professional Background */
    .stApp {
        background-color: #0b0f19;
        color: #c9d1d9;
    }

    /* Corporate Card Containers */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
    }

    /* Metric Display Typography */
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #3fb950;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f6fc;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-top: 20px;
        margin-bottom: 16px;
    }

    /* Recommendation Box */
    .recommendation-box {
        background-color: rgba(46, 160, 67, 0.08);
        border-left: 3px solid #2ea043;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #e6edf3;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADERS & CACHING
# ==============================================================================
@st.cache_data
def load_inventory_summary():
    file_path = "inventory_summary.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        data = {
            "item_id": ["FOODS_3_473", "HOUSEHOLD_1_019", "FOODS_1_018", "FOODS_3_109", "FOODS_3_750"],
            "store_id": ["WI_3", "CA_2", "TX_1", "CA_4", "TX_2"],
            "stratum": ["High_Volume | Continuous"] * 5,
            "forecast_28d": [73.0, 141.7, 205.9, 44.0, 72.8],
            "sigma": [2.22, 2.50, 3.83, 0.75, 1.81],
            "safety_stock_95pct": [1.2, 1.3, 2.1, 0.4, 1.0],
            "reorder_point_95pct": [9.0, 16.5, 24.1, 5.1, 8.8]
        }
        return pd.DataFrame(data)

@st.cache_resource
def load_model():
    model_path = "xgb_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

inv_summary_df = load_inventory_summary()
model = load_model()

# ==============================================================================
# SIDEBAR CONTROLS & SCENARIO PLANNER
# ==============================================================================
with st.sidebar:
    st.title("Supply Chain Controls")
    st.markdown("---")
    
    st.subheader("Scenario Parameters")
    
    preset = st.selectbox(
        "Quick Scenario Preset",
        ["Custom Settings", "Conservative (99% Service Level)", "Balanced Operational (95%)", "Lean Inventory (90%)"]
    )
    
    if preset == "Conservative (99% Service Level)":
        default_sl, default_lt = 99.0, 5
    elif preset == "Balanced Operational (95%)":
        default_sl, default_lt = 95.0, 3
    elif preset == "Lean Inventory (90%)":
        default_sl, default_lt = 90.0, 2
    else:
        default_sl, default_lt = 95.0, 3

    target_service_level = st.slider(
        "Target Service Level (%)",
        min_value=80.0,
        max_value=99.9,
        value=default_sl,
        step=0.5,
        help="Probability of avoiding a stockout during order lead time."
    )
    
    lead_time_days = st.slider(
        "Supplier Lead Time (Days)",
        min_value=1,
        max_value=14,
        value=default_lt,
        step=1,
        help="Elapsed time between order placement and inventory arrival."
    )
    
    st.markdown("---")
    st.subheader("SKU Selection")
    
    stratum_filter = st.multiselect(
        "Filter by Stratum",
        options=inv_summary_df["stratum"].unique(),
        default=inv_summary_df["stratum"].unique()
    )
    
    filtered_df = inv_summary_df[inv_summary_df["stratum"].isin(stratum_filter)]
    sku_options = (filtered_df["item_id"].astype(str) + " @ " + filtered_df["store_id"].astype(str)).tolist()
    
    if not sku_options:
        sku_options = (inv_summary_df["item_id"].astype(str) + " @ " + inv_summary_df["store_id"].astype(str)).tolist()
        
    selected_sku_str = st.selectbox("Select SKU-Store Series", sku_options)
    
    st.markdown("---")
    st.caption("**Model Champion:** Global XGBoost Regressor")
    st.caption("**Evaluation Metric:** WRMSSE")

# Parse selected SKU
selected_item, selected_store = selected_sku_str.split(" @ ")
sku_row = inv_summary_df[(inv_summary_df.item_id == selected_item) & (inv_summary_df.store_id == selected_store)].iloc[0]

# Dynamic Inventory Calculations
HORIZON = 28
forecast_28d = float(sku_row["forecast_28d"])
sigma = float(sku_row["sigma"])
daily_demand = forecast_28d / HORIZON
lead_time_demand = daily_demand * lead_time_days

z_score = float(norm.ppf(target_service_level / 100.0))
safety_stock = z_score * sigma * np.sqrt(lead_time_days / HORIZON)
reorder_point = lead_time_demand + safety_stock

# ==============================================================================
# MAIN HEADER & KPI CARDS
# ==============================================================================
st.title("Walmart Demand Forecasting & Inventory Engine")
st.markdown("Production-grade demand forecasting pipeline (PySpark + XGBoost) translated into dynamic safety stock and reorder point recommendations.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Catalog Production WRMSSE</div>
        <div class="metric-value">0.7679</div>
        <div class="metric-subtitle">+26.7% vs Naive Baseline (1.0477)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Target Service Level</div>
        <div class="metric-value">{target_service_level:.1f}%</div>
        <div class="metric-subtitle">Z-Score: {z_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Supplier Lead Time</div>
        <div class="metric-value">{lead_time_days} Days</div>
        <div class="metric-subtitle">Forecast Horizon: 28 Days</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Evaluated Series Scope</div>
        <div class="metric-value">30,490</div>
        <div class="metric-subtitle">SKU-Store Series (~58.3M Rows)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# NAVIGATION TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Inventory Decision Engine",
    "Champion / Challenger Benchmark",
    "Feature Importance & Explainability",
    "Batch Purchase Order Generator"
])

# ------------------------------------------------------------------------------
# TAB 1: INVENTORY DECISION ENGINE
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(f"<div class='section-header'>Inventory Decision Summary — {selected_item} ({selected_store})</div>", unsafe_allow_html=True)
    st.markdown(f"**Stratum:** `{sku_row['stratum']}`")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("28-Day Demand Forecast", f"{forecast_28d:.1f} units", f"{daily_demand:.2f} units/day")
    with kpi_col2:
        st.metric("Model Residual Volatility (σ)", f"{sigma:.2f}", "Forecast Error Std Dev")
    with kpi_col3:
        st.metric("Dynamic Safety Stock", f"{safety_stock:.1f} units", f"SLA: {target_service_level:.1f}%")
    with kpi_col4:
        st.metric("Reorder Point (ROP)", f"{reorder_point:.1f} units", f"Lead Time: {lead_time_days} Days")
        
    st.markdown(f"""
    <div class="recommendation-box">
        <strong>Replenishment Recommendation:</strong><br>
        When active stock for <strong>{selected_item} @ {selected_store}</strong> drops to or below <strong>{reorder_point:.1f} units</strong>, 
        issue a purchase order for <strong>{forecast_28d:.0f} units</strong>. This maintains a <strong>{target_service_level:.1f}% service level</strong> 
        (less than {100 - target_service_level:.1f}% stockout risk over the {lead_time_days}-day lead time window).
    </div>
    """, unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Service Level vs Safety Stock Cost Curve")
        st.caption("Illustrates the non-linear increase in safety buffer required above 95% service level.")
        
        sl_range = np.linspace(80.0, 99.9, 100)
        z_range = norm.ppf(sl_range / 100.0)
        ss_range = z_range * sigma * np.sqrt(lead_time_days / HORIZON)
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=sl_range, y=ss_range,
            mode='lines',
            name='Safety Stock (Units)',
            line=dict(color='#58a6ff', width=2.5)
        ))
        
        fig_curve.add_trace(go.Scatter(
            x=[target_service_level], y=[safety_stock],
            mode='markers+text',
            name='Selected Target',
            marker=dict(color='#f85149', size=10, symbol='circle'),
            text=[f"{safety_stock:.1f} units"],
            textposition="top left"
        ))
        
        fig_curve.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Target Service Level (%)",
            yaxis_title="Safety Stock Buffer (Units)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with col_chart2:
        st.subheader("Simulated 28-Day Inventory Trajectory")
        st.caption("Visualizes projected inventory depletion against reorder point and safety floor.")
        
        days = np.arange(1, 29)
        cum_demand = daily_demand * days
        simulated_stock = reorder_point + (forecast_28d - cum_demand)
        
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=days, y=simulated_stock,
            mode='lines',
            name='Projected Stock On Hand',
            line=dict(color='#3fb950', width=2.5)
        ))
        fig_traj.add_trace(go.Scatter(
            x=[1, 28], y=[reorder_point, reorder_point],
            mode='lines',
            name='Reorder Point (ROP)',
            line=dict(color='#d29922', width=1.8, dash='dash')
        ))
        fig_traj.add_trace(go.Scatter(
            x=[1, 28], y=[safety_stock, safety_stock],
            mode='lines',
            name='Safety Stock Floor',
            line=dict(color='#f85149', width=1.8, dash='dot')
        ))
        
        fig_traj.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Forecast Day (1-28)",
            yaxis_title="Stock Level (Units)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_traj, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: CHAMPION / CHALLENGER BENCHMARK
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("<div class='section-header'>Model Benchmarking & WRMSSE Evaluation</div>", unsafe_allow_html=True)
    
    st.markdown("""
    **Evaluation Protocol:**
    - **Table A (Stratified Head-to-Head 30-Series Sample):** Compares Seasonal-Naive Baseline, Statistical SARIMA, and Global XGBoost on identical stratified series across Volume Terciles and Intermittency Tiers.
    - **Table B (Full Catalog Production Metric):** Compares Seasonal Naive against Global XGBoost Champion across all **30,490 series** (~58.3 Million rows).
    """)
    
    col_t1, col_t2 = st.columns([3, 2])
    
    with col_t1:
        st.subheader("Table A — Stratified 30-Series Sample (WRMSSE)")
        table_a_data = {
            "Stratum": [
                "High_Volume | Continuous",
                "High_Volume | Intermittent",
                "Low_Volume | Continuous",
                "Low_Volume | Intermittent",
                "Med_Volume | Continuous",
                "Med_Volume | Intermittent",
                "OVERALL (Weighted Sample WRMSSE)"
            ],
            "Seasonal Naive": [0.8220, 1.4890, 1.1460, 1.7030, 0.8860, 1.3960, 1.1350],
            "SARIMA (Auto-ARIMA)": [0.6820, 1.1060, 0.8890, 1.2340, 0.6140, 1.0060, 0.8210],
            "XGBoost (Global Champion)": [0.6060, 1.0700, 0.8640, 1.2080, 0.6290, 1.0250, 0.7960]
        }
        df_table_a = pd.DataFrame(table_a_data)
        st.dataframe(df_table_a.style.highlight_min(axis=1, subset=["Seasonal Naive", "SARIMA (Auto-ARIMA)", "XGBoost (Global Champion)"], color="#1f422c"), use_container_width=True)
        
    with col_t2:
        st.subheader("Table B — Full Catalog Production Result")
        table_b_data = {
            "Model Architecture": ["Seasonal Naive Baseline", "Global XGBoost (Champion)"],
            "Evaluated Series": ["30,490 (Full Dataset)", "30,490 (Full Dataset)"],
            "WRMSSE Score": [1.0477, 0.7679],
            "Improvement vs Baseline": ["0.0%", "+26.7%"]
        }
        df_table_b = pd.DataFrame(table_b_data)
        st.dataframe(df_table_b, use_container_width=True)
        
        st.info("**Key Insight:** Global XGBoost delivers a **26.7% error reduction** over baseline across the full catalog by learning cross-series signals (SNAP benefits, price discounts).")

    st.subheader("Error Comparison Across Strata (Lower is Better)")
    df_plot_a = df_table_a[df_table_a["Stratum"] != "OVERALL (Weighted Sample WRMSSE)"].melt(
        id_vars="Stratum", var_name="Model", value_name="RMSSE"
    )
    
    fig_strata = px.bar(
        df_plot_a,
        x="Stratum",
        y="RMSSE",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#8b949e", "#d29922", "#58a6ff"],
        template="plotly_dark"
    )
    fig_strata.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_strata, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE & EXPLAINABILITY
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("<div class='section-header'>Global Feature Importance & Predictor Insights</div>", unsafe_allow_html=True)
    st.markdown("XGBoost feature importance scores showing key predictors driving daily sales forecasts across 30,490 series.")
    
    feature_importance_data = {
        "Feature": [
            "lag_7 (Weekly Seasonal Memory)",
            "roll_mean_7 (7-Day Rolling Sales)",
            "sell_price (Item Unit Price)",
            "wday (Day of Week Signal)",
            "lag_1 (Previous Day Demand)",
            "is_snap (State SNAP Benefit Days)",
            "roll_std_7 (7-Day Demand Volatility)",
            "roll_mean_28 (28-Day Trend)",
            "price_drop_flag (Discount Event)",
            "is_holiday (Calendar Event Flag)",
            "lag_28 (Monthly Memory)"
        ],
        "Importance_Score": [0.385, 0.210, 0.145, 0.082, 0.065, 0.042, 0.028, 0.021, 0.012, 0.006, 0.004]
    }
    df_fi = pd.DataFrame(feature_importance_data).sort_values("Importance_Score", ascending=True)
    
    fig_fi = px.bar(
        df_fi,
        x="Importance_Score",
        y="Feature",
        orientation="h",
        color="Importance_Score",
        color_continuous_scale="Blues",
        template="plotly_dark",
        title="Feature Contribution to XGBoost Demand Predictions"
    )
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    
    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        st.markdown("""
        ### Key Technical Driver Insights:
        1. **Weekly Seasonality (`lag_7` & `roll_mean_7`):** Accounts for >59% of total model decision weight. Confirms strong repeating 7-day purchasing cycles (weekend grocery spikes).
        2. **Price & Promotional Sensitivity (`sell_price` & `price_drop_flag`):** High elasticity captured across store departments.
        """)
    with col_fi2:
        st.markdown("""
        ### Policy & Calendar Impact:
        1. **SNAP Benefit Uplift (`is_snap`):** State-specific SNAP distribution days trigger significant demand surges for Food SKUs in CA, TX, and WI.
        2. **Global Cross-Learning:** XGBoost transfers learned SNAP and price responsiveness across all 30,490 series simultaneously.
        """)

# ------------------------------------------------------------------------------
# TAB 4: BATCH PURCHASE ORDER GENERATOR
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("<div class='section-header'>Batch Replenishment & Purchase Order Export</div>", unsafe_allow_html=True)
    st.markdown(f"Automated reorder point calculation for representative inventory catalog at **{target_service_level:.1f}% Service Level** and **{lead_time_days}-Day Lead Time**.")
    
    export_df = inv_summary_df.copy()
    export_df["lead_time_days"] = lead_time_days
    export_df["target_service_level"] = f"{target_service_level:.1f}%"
    export_df["z_score"] = round(z_score, 2)
    
    daily_demands = export_df["forecast_28d"] / HORIZON
    lead_demands = daily_demands * lead_time_days
    safety_stocks = z_score * export_df["sigma"] * np.sqrt(lead_time_days / HORIZON)
    reorder_points = lead_demands + safety_stocks
    
    export_df["calculated_safety_stock"] = np.round(safety_stocks, 1)
    export_df["calculated_reorder_point"] = np.round(reorder_points, 1)
    
    display_cols = [
        "item_id", "store_id", "stratum", "forecast_28d",
        "sigma", "calculated_safety_stock", "calculated_reorder_point"
    ]
    
    st.dataframe(
        export_df[display_cols].rename(columns={
            "forecast_28d": "28D Forecast (Units)",
            "sigma": "Error Volatility (σ)",
            "calculated_safety_stock": f"Safety Stock ({target_service_level:.1f}%)",
            "calculated_reorder_point": f"Reorder Point ({lead_time_days}D LT)"
        }),
        use_container_width=True
    )
    
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Purchase Orders (CSV)",
        data=csv_data,
        file_name=f"walmart_purchase_orders_SLA{int(target_service_level)}pct_{lead_time_days}dayLT.csv",
        mime="text/csv",
        help="Download purchase order recommendations CSV."
    )

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.8rem;">
    Walmart Demand Forecasting & Inventory Optimization Engine | PySpark, XGBoost & Streamlit
</div>
""", unsafe_allow_html=True)
