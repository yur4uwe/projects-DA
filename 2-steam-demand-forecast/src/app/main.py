"""Streamlit Frontend Client for Steam Indie Demand Forecasting API."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.server_api import (
    API_URL,
    execute_elasticity_simulation,
    execute_forecast_simulation,
    fetch_catalog,
    fetch_experiments,
    fetch_feature_importances,
    fetch_game_history,
    query_api_health,
)

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Indie Demand & Sales Scheduler",
    layout="wide",
)

POLL_INTERVAL_SECONDS = 5

st.title("Steam Indie Demand & Sales Scheduling Studio")
st.markdown("Powered by the **Hybrid Prophet + LightGBM** architecture.")


health_data = query_api_health()

if not health_data:
    st.error(
        f"### Prediction Service Offline\n\n"
        f"The backend forecasting service is not reachable at **`{API_URL}`**.\n\n"
    )

    st.warning(
        f"Next health check retry in {POLL_INTERVAL_SECONDS} seconds...\n\n"
        "Start the forecasting service in your terminal with:\n"
        "```bash\n"
        "uv run python -m src.api.server\n"
        "```"
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("Retry Connection Now"):
            st.rerun()

    # Sleep briefly and trigger rerun to poll health
    import time

    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()
    st.stop()

catalog = fetch_catalog()

# -----------------------------------------------------------------------------
# Sidebar: Parameter Configuration Studio
# -----------------------------------------------------------------------------
api_tooltip = (
    f"API Online at {API_URL} | Milestone: {health_data.get('production_milestone', 'M4')} | "
    f"Engines: {', '.join(health_data.get('loaded_models', []))}"
)
st.sidebar.markdown(
    f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">'
    f'<span style="font-weight: 600; font-size: 1.05rem;">Configuration</span>'
    f'<span title="{api_tooltip}" style="display: inline-block; width: 10px; height: 10px; background-color: #2e7d32; border-radius: 2px; cursor: help;"></span>'
    f"</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("Game & Commercial Profile")

game_names = [g["name"] for g in catalog]
selected_game_name = st.sidebar.selectbox("Target Indie Title", game_names, index=0)
selected_game = next(g for g in catalog if g["name"] == selected_game_name)
appid = selected_game["appid"]

actual_base_price = float(selected_game.get("base_price", 14.99))
base_price = actual_base_price

st.sidebar.caption(
    f"AppID: `{appid}` | Base Price: `${actual_base_price:.2f}`\n\n"
    f"Live CCU: `{selected_game.get('live_ccu', 0):,}` | "
    f"Co-op: `{'Yes' if selected_game.get('is_coop') else 'No'}`"
)

st.sidebar.markdown("---")
st.sidebar.header("Promotional Campaign Scheduler")

DATA_CUTOFF_DATE = date(2026, 10, 18)

campaign_type = st.sidebar.radio(
    "Promotional Event Type",
    ["Custom Publisher / Weekend Deal", "Major Steam Seasonal Sale"],
    index=0,
)

is_custom = campaign_type == "Custom Publisher / Weekend Deal"

if is_custom:
    custom_title = st.sidebar.text_input(
        "Campaign Name",
        value="Indie Spotlight Weekend",
        help="Purely decorative label for display and chart annotations. Does not affect model forecast calculations.",
    )
    default_start = max(
        date.today() + timedelta(days=7), DATA_CUTOFF_DATE + timedelta(days=1)
    )
    sale_duration = st.sidebar.slider(
        "Sale Duration (Days)", min_value=3, max_value=14, value=4
    )
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=DATA_CUTOFF_DATE + timedelta(days=1),
    )
else:
    custom_title = "Official Steam Seasonal Sale"
    sale_options = [
        "Steam Autumn Sale 2026 (Nov 24 - Dec 01, 2026)",
        "Steam Winter Sale 2026 (Dec 17 - Jan 01, 2027)",
        "Steam Spring Sale 2026 (Mar 12 - Mar 19, 2026)",
        "Steam Summer Sale 2026 (Jun 25 - Jul 09, 2026)",
    ]
    sale_choice = st.sidebar.selectbox("Seasonal Sale Event", sale_options, index=0)
    # Parse approximate default date
    start_date = date(2026, 11, 24) if "Autumn" in sale_choice else date(2026, 12, 17)
    sale_duration = 7 if "Autumn" in sale_choice else 14
    st.sidebar.info(f"Duration: **{sale_duration} days** starting {start_date}")

# Validation: Start date cannot be before or on the historical data cutoff date
if start_date <= DATA_CUTOFF_DATE:
    st.sidebar.error(
        f"Invalid Start Date: The selected date ({start_date}) is on or before historical cutoff ({DATA_CUTOFF_DATE})."
    )
    st.error(
        f"### Invalid Campaign Schedule\n\n"
        f"The selected campaign start date **`{start_date}`** cannot be on or before the historical data cutoff date (**`{DATA_CUTOFF_DATE}`**).\n\n"
        "Forecasting requires projecting into future unobserved time horizons. Please choose a date on or after "
        f"**`{DATA_CUTOFF_DATE + timedelta(days=1)}`**."
    )
    st.stop()

discount_pct = st.sidebar.slider(
    "Planned Discount Depth (%)",
    min_value=10,
    max_value=75,
    value=33,
    step=5,
)
discount_depth = discount_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")

horizon_days = st.sidebar.select_slider(
    "Forecast Horizon (Days)",
    options=[7, 14, 21, 30],
    value=14,
)

model_engine_choice = st.sidebar.selectbox(
    "Forecasting Engine",
    ["Hybrid Prophet + LightGBM (Listing 5)", "LightGBM Global Regressor"],
    index=0,
)
model_engine_key = "hybrid" if "Hybrid" in model_engine_choice else "lgbm"


# -----------------------------------------------------------------------------
# Execute Simulation via HTTP API
# -----------------------------------------------------------------------------
forecast_payload = {
    "appid": appid,
    "horizon_days": horizon_days,
    "start_date": str(start_date) if start_date else None,
    "discount_depth": discount_depth,
    "sale_duration_days": sale_duration,
    "is_custom_event": is_custom,
    "custom_event_name": custom_title,
    "model_engine": model_engine_key,
}

with st.spinner(f"Querying server for {horizon_days}-day multi-step simulation..."):
    sim_data = execute_forecast_simulation(forecast_payload)

forecast_df = pd.DataFrame(sim_data["forecast_series"])
forecast_df["date"] = pd.to_datetime(forecast_df["date"])
baseline_ccu = sim_data["recent_baseline_ccu"]
peak_promo_ccu = sim_data["peak_promo_ccu"]
peak_promo_peak_ccu = sim_data.get("peak_promo_peak_ccu", peak_promo_ccu * 1.45)
game_peak_ratio = sim_data.get("peak_ratio", 1.45)
surge_multiplier = sim_data["peak_surge_multiplier"]
incremental_player_days = sim_data["total_incremental_player_days"]
cooldown_compliant = sim_data["cooldown_compliant"]
cooldown_note = sim_data["cooldown_note"]

# Fetch historical data from API
history_df = fetch_game_history(appid=appid, limit=60)

# -----------------------------------------------------------------------------
# Header KPI Cards
# -----------------------------------------------------------------------------
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric(
        label="Recent Baseline CCU",
        value=f"{baseline_ccu:,.0f}",
        help=f"Average daily concurrent players over the last 14 days (Peak equivalent: ~{baseline_ccu * game_peak_ratio:,.0f}).",
    )

with kpi_cols[1]:
    st.metric(
        label="Forecasted Peak CCU",
        value=f"{peak_promo_ccu:,.0f}",
        delta=f"+{(surge_multiplier - 1.0) * 100:.1f}% Surge (~{peak_promo_peak_ccu:,.0f} Peak)",
        help=f"Average daily CCU during promo peak. Estimated 24h Peak CCU on SteamDB scale is ~{peak_promo_peak_ccu:,.0f} (peak/avg ratio: {game_peak_ratio:.2f}x).",
    )

with kpi_cols[2]:
    total_base_vol = max(baseline_ccu * len(forecast_df), 1.0)
    pct_lift = (incremental_player_days / total_base_vol) * 100
    st.metric(
        label="Incremental Demand Lift",
        value=f"{incremental_player_days:,.0f}",
        delta=f"+{pct_lift:.1f}% Total Lift",
        help="Estimated total incremental player-days above baseline throughout the forecast horizon.",
    )

with kpi_cols[3]:
    if cooldown_compliant:
        st.success(f"**Steam Cooldown**: Compliant\n\n{cooldown_note}")
    else:
        st.warning(f"**Steam Cooldown**: Flagged\n\n{cooldown_note}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Analytical Workspace Tabs
# -----------------------------------------------------------------------------
tab_forecast, tab_elasticity, tab_diagnostics = st.tabs(
    [
        "Forecast & Schedule Visualizer",
        "Discount Elasticity Simulator",
        "Model Lineage & Diagnostics",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Forecast & Schedule Visualizer
# -----------------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"Demand Trajectory & Promotional Window for {selected_game_name}")

    chart_mode = st.radio(
        "Trajectory Display Scale",
        [
            "Average Daily CCU (Model Baseline)",
            "Estimated Peak CCU (SteamDB 24h Peak)",
            "Indexed to Baseline (% of Organic Baseline)",
        ],
        horizontal=True,
        help="SteamDB charts report daily peak CCU, while demand modeling operates on 24-hour average CCU. Select 'Estimated Peak CCU' to compare directly with SteamDB peaks.",
    )
    is_indexed = "Indexed" in chart_mode
    is_peak_mode = "Peak CCU" in chart_mode

    if is_indexed:
        y_norm = (baseline_ccu / 100.0) if (baseline_ccu > 0) else 1.0
        y_label = "Demand (% of Baseline)"
        scale_mult = 1.0
    elif is_peak_mode:
        y_norm = 1.0
        y_label = "Peak Concurrent Players (SteamDB Scale)"
        scale_mult = game_peak_ratio
    else:
        y_norm = 1.0
        y_label = "Average Daily Concurrent Players (CCU)"
        scale_mult = 1.0

    fig = go.Figure()

    # Historical actual curve from API
    if not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["date"],
                y=(history_df["demand"] * scale_mult) / y_norm,
                name="Historical Actual",
                line=dict(color="#2962FF", width=2),
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history_df["date"],
                y=(history_df["sales_rolling_7"] * scale_mult) / y_norm,
                name="Historical 7d Baseline",
                line=dict(color="#90CAF9", width=1.5, dash="dot"),
                mode="lines",
            )
        )

    # Shaded confidence band
    target_pred_col = "predicted_peak_ccu" if is_peak_mode else "predicted_ccu"
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=(forecast_df["upper_bound"] * scale_mult) / y_norm,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=(forecast_df["lower_bound"] * scale_mult) / y_norm,
            fill="tonexty",
            fillcolor="rgba(255, 152, 0, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Forecast Confidence Band",
        )
    )

    # Forecast curve
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df[target_pred_col] / y_norm if not is_indexed else (forecast_df["predicted_ccu"] / y_norm),
            name="Forecasted Demand",
            line=dict(color="#FF6D00", width=3),
            mode="lines+markers",
        )
    )

    # Highlight promotional period
    promo_dates = forecast_df[forecast_df["is_promo_active"]]
    if not promo_dates.empty:
        p_start = promo_dates["date"].iloc[0]
        p_end = promo_dates["date"].iloc[-1]
        fig.add_vrect(
            x0=p_start,
            x1=p_end,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=1,
            line_dash="dash",
            annotation_text=f"-{discount_pct}% Discount Window",
            annotation_position="top left",
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, width="stretch")


# -----------------------------------------------------------------------------
# Tab 2: What-If Discount Elasticity Simulator
# -----------------------------------------------------------------------------
with tab_elasticity:
    st.subheader("Promotional Discount Elasticity & Sensitivity Curve")

    elast_payload = {
        "appid": appid,
        "base_price": base_price,
        "start_date": str(start_date) if start_date else None,
        "sale_duration_days": sale_duration,
        "is_custom_event": is_custom,
        "test_discounts": [0.10, 0.20, 0.33, 0.50, 0.66, 0.75],
        "model_engine": model_engine_key,
    }

    with st.spinner("Calculating elasticity matrix..."):
        raw_elast = execute_elasticity_simulation(elast_payload)

    elast_df = pd.DataFrame(raw_elast)
    elast_df = elast_df.rename(
        columns={
            "discount_percent": "Discount %",
            "effective_price": "Effective Price ($)",
            "peak_ccu": "Peak CCU",
            "surge_multiplier": "Surge Multiplier",
            "incremental_player_days": "Incremental Player-Days",
        }
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        fig_elast = px.line(
            elast_df,
            x="Discount %",
            y="Peak CCU",
            markers=True,
            title=f"Peak CCU vs. Discount Depth for {selected_game_name} (Base: ${base_price:.2f})",
            labels={"Peak CCU": "Forecasted Peak CCU"},
        )
        fig_elast.add_hline(
            y=baseline_ccu,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Baseline CCU ({baseline_ccu:,.0f})",
        )
        fig_elast.update_layout(template="plotly_white")
        st.plotly_chart(fig_elast, width="stretch")

    with c2:
        st.markdown("#### Elasticity Matrix")
        st.dataframe(
            elast_df[
                [
                    "Discount %",
                    "Effective Price ($)",
                    "Surge Multiplier",
                    "Incremental Player-Days",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# Tab 3: Model Lineage & Diagnostics
# -----------------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Model Progression & Experiment Lineage")

    experiments = fetch_experiments()
    if experiments:
        experiments_df = pd.DataFrame(experiments)
        st.dataframe(
            experiments_df[
                [
                    "milestone",
                    "model_name",
                    "mae",
                    "wape",
                    "rmse",
                    "r2",
                    "description",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        fig_comp = px.bar(
            experiments_df,
            x="milestone",
            y="wape",
            color="milestone",
            text="wape",
            title="Weighted Absolute Percentage Error (WAPE) Across Milestones (Lower is Better)",
            labels={"wape": "WAPE (Scale-independent error)", "milestone": "Milestone"},
        )
        fig_comp.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_comp.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_comp, width="stretch")

    importances = fetch_feature_importances()
    if importances:
        st.markdown("#### Global Feature Importances")
        fi_df = pd.DataFrame(importances)
        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 12 Most Influential Predictive Signals",
        )
        fig_fi.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, width="stretch")
