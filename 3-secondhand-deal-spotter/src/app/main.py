"""Interactive Streamlit dashboard for secondary PC marketplace deal detection."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.schema import DEAL_TIERS
from src.models.predict import predict_deal

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pc_listings_processed.parquet"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
IMPORTANCE_PATH = PROJECT_ROOT / "models" / "global_feature_importance.json"

st.set_page_config(
    page_title="Second-Hand Deal Spotter & Market Valuation",
    layout="wide",
)


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importance() -> list[dict[str, Any]]:
    if IMPORTANCE_PATH.exists():
        with IMPORTANCE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


df_data = load_processed_data()
metadata = load_metadata()
feature_importance = load_feature_importance()

st.title("Second-Hand Deal Spotter & Market Classification")
st.markdown(
    "Ensemble classification system detecting pricing attractiveness and market anomalies in secondary PC marketplaces."
)

if df_data.empty:
    st.error("Processed dataset not found. Run `uv run python -m src.features.builder` first.")
    st.stop()

# Header KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_count = len(df_data)
median_price = df_data["asking_price"].median()
great_deals_pct = (df_data["deal_tier"] == "Great Deal").mean() * 100
fair_deals_pct = (df_data["deal_tier"] == "Fair Price").mean() * 100
overpriced_pct = (df_data["deal_tier"] == "Overpriced").mean() * 100

with kpi1:
    st.metric("Total Listings", f"{total_count:,}")
with kpi2:
    st.metric("Median Market Price", f"€{median_price:,.0f}")
with kpi3:
    st.metric("Great Deals", f"{great_deals_pct:.1f}%")
with kpi4:
    st.metric("Fair Price", f"{fair_deals_pct:.1f}%")
with kpi5:
    st.metric("Overpriced", f"{overpriced_pct:.1f}%")

tab_spotter, tab_market, tab_model = st.tabs(
    ["Deal Spotter & Evaluator", "Marketplace Explorer", "Model Diagnostics & Attribution"]
)

# -------------------------------------------------------------
# TAB 1: DEAL SPOTTER & EVALUATOR
# -------------------------------------------------------------
with tab_spotter:
    st.subheader("Listing Valuation & Deal Tier Classifier")
    st.markdown(
        "Configure product hardware specifications and asking price to classify deal attractiveness."
    )

    with st.form("listing_eval_form"):
        col_spec1, col_spec2, col_spec3 = st.columns(3)

        with col_spec1:
            company = st.selectbox(
                "Manufacturer / Brand",
                options=sorted(df_data["Company"].unique()),
                index=sorted(df_data["Company"].unique()).index("Dell")
                if "Dell" in df_data["Company"].unique()
                else 0,
            )
            typename = st.selectbox(
                "Form Factor / Type",
                options=sorted(df_data["TypeName"].unique()),
                index=sorted(df_data["TypeName"].unique()).index("Notebook")
                if "Notebook" in df_data["TypeName"].unique()
                else 0,
            )
            cpu_tier = st.selectbox(
                "CPU Processor Tier",
                options=[
                    "Core i7",
                    "Core i5",
                    "Core i3",
                    "AMD Ryzen",
                    "AMD A-Series",
                    "Entry (Celeron/Pentium)",
                    "Intel Xeon",
                    "Core M",
                ],
                index=0,
            )
            cpu_freq = st.slider("CPU Base Clock (GHz)", 1.0, 4.0, 2.5, 0.1)

        with col_spec2:
            ram_gb = st.select_slider(
                "RAM Memory (GB)",
                options=[2, 4, 8, 12, 16, 24, 32, 64],
                value=8,
            )
            primary_storage_gb = st.select_slider(
                "Storage Capacity (GB)",
                options=[32, 64, 128, 256, 512, 1024, 2048],
                value=256,
            )
            primary_storage_type = st.selectbox(
                "Primary Storage Type",
                options=["SSD", "HDD", "Flash Storage", "Hybrid"],
                index=0,
            )
            gpu_tier = st.selectbox(
                "GPU Graphics Tier",
                options=["Integrated", "Dedicated Gaming", "Dedicated Pro"],
                index=0,
            )

        with col_spec3:
            opsys = st.selectbox(
                "Operating System",
                options=["Windows", "macOS", "Linux", "Other / No OS"],
                index=0,
            )
            inches = st.slider("Screen Size (Inches)", 10.1, 18.4, 15.6, 0.1)
            resolution_preset = st.selectbox(
                "Display Resolution",
                options=[
                    "Full HD (1920x1080)",
                    "HD (1366x768)",
                    "Retina / QHD (2560x1440)",
                    "4K Ultra HD (3840x2160)",
                ],
                index=0,
            )
            is_touchscreen = st.checkbox("Touchscreen Support", value=False)
            is_ips = st.checkbox("IPS Display Panel", value=True)
            weight_kg = st.slider("Weight (kg)", 0.8, 5.0, 1.9, 0.1)

        st.divider()
        asking_price = st.number_input(
            "Listing Asking Price (€)",
            min_value=50.0,
            max_value=7000.0,
            value=650.0,
            step=25.0,
        )

        submitted = st.form_submit_button("Evaluate Listing Attractiveness")

    if submitted:
        res_map = {
            "Full HD (1920x1080)": (1920, 1080),
            "HD (1366x768)": (1366, 768),
            "Retina / QHD (2560x1440)": (2560, 1440),
            "4K Ultra HD (3840x2160)": (3840, 2160),
        }
        width, height = res_map[resolution_preset]

        gpu_brand = "Intel"
        if gpu_tier in ["Dedicated Gaming", "Dedicated Pro"]:
            gpu_brand = "NVIDIA"

        cpu_brand = "AMD" if "AMD" in cpu_tier else "Intel"

        input_record = {
            "Company": company,
            "TypeName": typename,
            "Inches": inches,
            "Ram_GB": ram_gb,
            "Weight_kg": weight_kg,
            "Cpu_Brand": cpu_brand,
            "Cpu_Tier": cpu_tier,
            "Cpu_Freq_GHz": cpu_freq,
            "Primary_Storage_GB": primary_storage_gb,
            "Primary_Storage_Type": primary_storage_type,
            "Total_Storage_GB": primary_storage_gb,
            "Gpu_Brand": gpu_brand,
            "Gpu_Tier": gpu_tier,
            "OpSys": opsys,
            "Screen_Width": width,
            "Screen_Height": height,
            "Is_Touchscreen": is_touchscreen,
            "Is_IPS": is_ips,
            "asking_price": float(asking_price),
        }

        eval_result = predict_deal(input_record)

        tier = eval_result["deal_tier"]
        conf = eval_result["confidence"]
        baseline_price = eval_result["market_baseline_price"]
        delta_eur = eval_result["price_delta_euros"]
        savings_pct = eval_result["relative_savings_pct"]

        st.subheader("Classification Outcome")
        verdict_col1, verdict_col2 = st.columns([1, 1])

        with verdict_col1:
            with st.container(border=True):
                if tier == "Great Deal":
                    st.success(f"Classification: **{tier}**")
                    st.markdown(
                        f"This listing is priced at a significant discount (**{abs(savings_pct):.1f}% below fair market value**)."
                    )
                elif tier == "Fair Price":
                    st.info(f"Classification: **{tier}**")
                    st.markdown(
                        "This listing aligns closely with prevailing secondary market clearing prices."
                    )
                else:
                    st.warning(f"Classification: **{tier}**")
                    st.markdown(
                        f"This listing carries a price premium of **{abs(savings_pct):.1f}% above fair market value**."
                    )

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Asking Price", f"€{asking_price:,.2f}")
                    st.metric("Model Confidence", f"{conf * 100:.1f}%")
                with c2:
                    st.metric("Market Baseline", f"€{baseline_price:,.2f}")
                    delta_label = "Net Discount" if delta_eur <= 0 else "Price Premium"
                    st.metric(delta_label, f"€{abs(delta_eur):,.2f}")

        with verdict_col2:
            with st.container(border=True):
                st.markdown("**Calibrated Class Probabilities**")
                prob_data = eval_result["probabilities"]
                prob_df = pd.DataFrame(
                    list(prob_data.items()), columns=["Tier", "Probability"]
                )

                color_discrete_map = {
                    "Great Deal": "#2ecc71",
                    "Fair Price": "#3498db",
                    "Overpriced": "#e74c3c",
                }

                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Tier",
                    orientation="h",
                    color="Tier",
                    color_discrete_map=color_discrete_map,
                    text_auto=".1%",
                    range_x=[0.0, 1.0],
                )
                fig_prob.update_layout(
                    showlegend=False,
                    height=240,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Confidence Probability",
                    yaxis_title="",
                )
                st.plotly_chart(fig_prob)

# -------------------------------------------------------------
# TAB 2: MARKETPLACE EXPLORER
# -------------------------------------------------------------
with tab_market:
    st.subheader("Marketplace Distribution & Deal Analysis")

    filter_c1, filter_c2, filter_c3 = st.columns(3)
    with filter_c1:
        sel_tier = st.multiselect(
            "Filter by Deal Tier",
            options=DEAL_TIERS,
            default=DEAL_TIERS,
        )
    with filter_c2:
        sel_type = st.multiselect(
            "Filter by Form Factor",
            options=sorted(df_data["TypeName"].unique()),
            default=sorted(df_data["TypeName"].unique()),
        )
    with filter_c3:
        sel_brand = st.multiselect(
            "Filter by Brand",
            options=sorted(df_data["Company"].unique()),
            default=sorted(df_data["Company"].unique())[:6],
        )

    filtered_df = df_data[
        (df_data["deal_tier"].isin(sel_tier))
        & (df_data["TypeName"].isin(sel_type))
        & (df_data["Company"].isin(sel_brand))
    ]

    st.markdown(f"Displaying **{len(filtered_df)}** matching marketplace records.")

    scatter_c1, scatter_c2 = st.columns([3, 2])

    with scatter_c1:
        color_map = {
            "Great Deal": "#2ecc71",
            "Fair Price": "#3498db",
            "Overpriced": "#e74c3c",
        }
        fig_scatter = px.scatter(
            filtered_df,
            x="market_baseline_price",
            y="asking_price",
            color="deal_tier",
            color_discrete_map=color_map,
            hover_data=["Company", "Product", "Ram_GB", "Cpu_Tier", "price_ratio"],
            title="Asking Price vs Spec Baseline Valuation",
            labels={
                "market_baseline_price": "Fair Market Baseline (€)",
                "asking_price": "Listed Asking Price (€)",
                "deal_tier": "Deal Tier",
            },
        )
        # 1:1 parity line
        max_val = max(
            filtered_df["market_baseline_price"].max(),
            filtered_df["asking_price"].max(),
            1000,
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Parity Baseline (1.0x)",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig_scatter.update_layout(height=460)
        st.plotly_chart(fig_scatter)

    with scatter_c2:
        fig_box = px.box(
            filtered_df,
            x="TypeName",
            y="asking_price",
            color="deal_tier",
            color_discrete_map=color_map,
            title="Price Distribution Across Form Factors",
            labels={"TypeName": "Form Factor", "asking_price": "Asking Price (€)"},
        )
        fig_box.update_layout(height=460, xaxis_tickangle=-30)
        st.plotly_chart(fig_box)

    st.subheader("Filterable Marketplace Listings")
    table_cols = [
        "Company",
        "Product",
        "TypeName",
        "Cpu_Tier",
        "Ram_GB",
        "Primary_Storage_Type",
        "asking_price",
        "market_baseline_price",
        "price_ratio",
        "deal_tier",
    ]
    st.dataframe(
        filtered_df[table_cols].sort_values(by="price_ratio", ascending=True),
        column_config={
            "asking_price": st.column_config.NumberColumn(
                "Asking Price", format="€%.2f"
            ),
            "market_baseline_price": st.column_config.NumberColumn(
                "Baseline Value", format="€%.2f"
            ),
            "price_ratio": st.column_config.NumberColumn(
                "Price Ratio", format="%.2fx"
            ),
        },
    )

# -------------------------------------------------------------
# TAB 3: MODEL DIAGNOSTICS & ATTRIBUTION
# -------------------------------------------------------------
with tab_model:
    st.subheader("Model Benchmark & Evaluation")

    benchmarks = metadata.get("benchmarks", {})
    if benchmarks:
        bench_rows = []
        for model_name, m in benchmarks.items():
            bench_rows.append(
                {
                    "Model Architecture": model_name,
                    "Accuracy": f"{m.get('accuracy', 0):.4f}",
                    "Balanced Accuracy": f"{m.get('balanced_accuracy', 0):.4f}",
                    "Macro F1": f"{m.get('macro_f1', 0):.4f}",
                    "Weighted F1": f"{m.get('weighted_f1', 0):.4f}",
                    "Log Loss": f"{m.get('log_loss', 0):.4f}"
                    if m.get("log_loss") is not None
                    else "N/A",
                }
            )
        st.dataframe(pd.DataFrame(bench_rows))

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("**Confusion Matrix (Normalized)**")
        primary_metrics = metadata.get("primary_metrics", {})
        cm_data = primary_metrics.get("confusion_matrix", {})
        if cm_data and "normalized" in cm_data:
            labels = cm_data["labels"]
            norm_matrix = np.array(cm_data["normalized"])

            fig_cm = px.imshow(
                norm_matrix,
                labels=dict(x="Predicted Deal Tier", y="Ground Truth Deal Tier"),
                x=labels,
                y=labels,
                color_continuous_scale="Blues",
                text_auto=".2%",
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm)

    with diag_c2:
        st.markdown("**Top Feature Attribution (Gini Importance)**")
        if feature_importance:
            top_feats = pd.DataFrame(feature_importance).head(12)
            fig_imp = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                text_auto=".3f",
                labels={"importance": "Relative Importance", "feature": "Feature"},
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_imp)
