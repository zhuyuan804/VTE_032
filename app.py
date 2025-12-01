import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ------------------ Page config ------------------
st.set_page_config(page_title="VTE Risk Digital Twin Simulator", layout="wide")

PLOTLY_TEMPLATE = "plotly_white"
PRIMARY = "#2ECC71"
GRAY = "#BDC3C7"
DARK = "#2C3E50"
RED = "#E74C3C"
BLUE = "#3498DB"

# ------------------ Global CSS (clean dashboard style) ------------------
st.markdown(
    """
    <style>
      /* Overall app typography + spacing */
      html, body, [class*="css"]  {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
          color: #1f2937;
      }
      .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2.5rem;
      }
      /* Sidebar */
      section[data-testid="stSidebar"] {
          background: #F8FAFC;
          border-right: 1px solid #E5E7EB;
      }

      /* KPI cards */
      div[data-testid="metric-container"] {
          background: white;
          border: 1px solid #EEF2F7;
          padding: 14px 14px 10px 14px;
          border-radius: 14px;
          box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
      }

      /* Tabs style */
      button[data-baseweb="tab"] {
          font-weight: 600;
          padding: 10px 14px;
      }

      /* Plotly charts container */
      .stPlotlyChart {
          background: white;
          border: 1px solid #EEF2F7;
          border-radius: 14px;
          padding: 8px;
          box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
      }

      /* Buttons */
      .stButton > button {
          border-radius: 10px;
          font-weight: 600;
          border: 1px solid #E5E7EB;
          padding: 0.45rem 0.9rem;
      }
      .stButton > button:hover {
          border-color: #CBD5E1;
          background: #F8FAFC;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ 1. Load system ------------------
@st.cache_resource
def load_system():
    return joblib.load("vte_system.pkl")

system = load_system()
risk_model = system["risk_model"]
coef_matrix = system["coef_matrix"]   # proteins x behaviors
proteins = system["proteins"]
behaviors_map = system["behaviors"]
behavior_std = system["behavior_std"]
behavior_names = list(behaviors_map.keys())

# ------------------ 2. Helpers ------------------
def init_state():
    """Initialize session_state for sliders if missing."""
    for p in proteins:
        st.session_state.setdefault(f"base_{p}", 0.0)
    for b in behavior_names:
        st.session_state.setdefault(f"delta_{b}", 0.0)

def reset_all():
    """Reset baseline & delta sliders to 0."""
    for p in proteins:
        st.session_state[f"base_{p}"] = 0.0
    for b in behavior_names:
        st.session_state[f"delta_{b}"] = 0.0

init_state()

# ------------------ 3. Title & intro ------------------
st.title("🧬 VTE Risk Digital Twin Simulator")
st.markdown(
    """
Adjust lifestyle behaviors in the sidebar to simulate **plasma protein expression**
changes and estimate the impact on **venous thromboembolism (VTE) risk**.
"""
)

# ------------------ 4. Sidebar ------------------
st.sidebar.header("1. Baseline Protein Profile")
st.sidebar.caption("All proteins default to population mean (Z-score = 0).")

# Reset button
st.sidebar.button("🔄 Reset all sliders", on_click=reset_all)

baseline_vals = {}
with st.sidebar.expander("Advanced: Tune baseline protein levels", expanded=False):
    for p in proteins:
        baseline_vals[p] = st.slider(
            f"{p} (Z-score)",
            -3.0, 3.0,
            key=f"base_{p}",
            value=st.session_state[f"base_{p}"],
            step=0.1
        )

st.sidebar.header("2. Behavior Interventions")
intervention_deltas = {}
for b_name in behavior_names:
    std = float(behavior_std[b_name])
    min_val = float(-2.0 * std)
    max_val = float(2.0 * std)
    step = float(std / 10.0) if std > 0 else 0.1

    intervention_deltas[b_name] = st.sidebar.slider(
        f"Δ {b_name}",
        min_val, max_val,
        key=f"delta_{b_name}",
        value=st.session_state[f"delta_{b_name}"],
        step=step
    )

# ------------------ 5. Core calculation ------------------
df_base = pd.DataFrame([list(baseline_vals.values())], columns=proteins)
base_prob = risk_model.predict_proba(df_base)[0, 1]

drift = np.zeros(len(proteins))
for b_name, delta in intervention_deltas.items():
    if delta != 0:
        betas = coef_matrix.loc[proteins, b_name].values
        drift += betas * delta

new_vals = np.clip(df_base.values[0] + drift, -5, 5)
df_new = pd.DataFrame([new_vals], columns=proteins)
new_prob = risk_model.predict_proba(df_new)[0, 1]

# 统一用“变化”语义（new - base）
risk_change_abs = new_prob - base_prob
risk_change_rel = risk_change_abs / base_prob if base_prob > 0 else 0.0

# ------------------ 6. KPI row ------------------
st.divider()
k1, k2, k3 = st.columns(3)
k1.metric("Baseline VTE Risk", f"{base_prob:.1%}")
k2.metric(
    "Post-intervention Risk",
    f"{new_prob:.1%}",
    delta=f"{risk_change_abs:.1%}",
    delta_color="inverse",  # 风险下降为绿色，风险上升为红色
)
k3.metric(
    "Relative Risk Change",
    f"{risk_change_rel:+.1%}",
)

# ================== Tabs: Main / Curve / Heatmap ==================
tab_main, tab_curve, tab_heat = st.tabs(
    ["📌 Main View", "📈 Continuous Risk Curve", "🔥 Sensitivity Heatmap"]
)

# ================== TAB 1: Main View ==================
with tab_main:
    st.subheader("🧬 Protein-level Changes")
    prot_col1, prot_col2 = st.columns([1.25, 1])

    baseline_vals_arr = df_base.values[0]
    intervention_vals_arr = new_vals
    delta_arr = intervention_vals_arr - baseline_vals_arr

    # --- Protein Expression Shift ---
    with prot_col1:
        df_plot = pd.DataFrame({
            "Protein": proteins,
            "Baseline": baseline_vals_arr,
            "Intervention": intervention_vals_arr,
            "Δ": delta_arr
        })

        # 按 |Δ| 排序，让变化最大的蛋白排在前面
        order = (
            df_plot.assign(abs_delta=lambda d: d["Δ"].abs())
            .sort_values("abs_delta", ascending=False)["Protein"]
            .tolist()
        )

        df_long = df_plot.melt(
            id_vars=["Protein", "Δ"],
            value_vars=["Baseline", "Intervention"],
            var_name="State",
            value_name="Z"
        )

        fig_prot = px.bar(
            df_long, x="Protein", y="Z", color="State",
            barmode="group", template=PLOTLY_TEMPLATE,
            color_discrete_map={"Baseline": GRAY, "Intervention": PRIMARY},
            hover_data={"Δ":":+.3f", "Z":":.3f"},
        )

        # annotate noticeable changes
        for _, row in df_plot.iterrows():
            if abs(row["Δ"]) > 0.05:
                fig_prot.add_annotation(
                    x=row["Protein"], y=row["Intervention"],
                    text=f"{row['Δ']:+.2f}",
                    showarrow=False, yshift=10,
                    font=dict(size=11, color=DARK)
                )

        fig_prot.update_layout(
            height=420,
            legend_title_text="State",
            yaxis_title="Expression (Z-score)",
            xaxis_title="Proteins",
            title=dict(text="Baseline vs Intervention", x=0.01),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig_prot.update_xaxes(
            categoryorder="array",
            categoryarray=order,
            tickangle=-20,
            tickfont=dict(size=11),
            automargin=True
        )

        st.plotly_chart(fig_prot, width="stretch")

    # --- Key Drift Proteins ---
    with prot_col2:
        drift_series = pd.Series(drift, index=proteins)
        threshold = 0.01
        sig_drift = drift_series[drift_series.abs() > threshold]

        if not sig_drift.empty:
            sig_drift = sig_drift.reindex(
                sig_drift.abs().sort_values(ascending=False).index
            ).iloc[:10]

            df_sig = sig_drift.reset_index()
            df_sig.columns = ["Protein", "Drift"]
            df_sig = df_sig.sort_values("Drift")

            fig_sig = px.bar(
                df_sig, x="Drift", y="Protein", orientation="h",
                template=PLOTLY_TEMPLATE, color="Drift",
                color_continuous_scale=[(0.0, RED), (0.5, GRAY), (1.0, BLUE)],
                hover_data={"Drift":":+.4f"}
            )
            fig_sig.update_layout(
                height=420,
                coloraxis_showscale=False,
                xaxis_title="Protein Drift (Δ Z-score)",
                yaxis_title="",
                title=dict(text="Top Drift Proteins", x=0.01),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig_sig.add_vline(x=0, line_width=1, line_dash="dash", line_color=DARK)
            st.plotly_chart(fig_sig, width="stretch")
        else:
            st.info("No protein shows a substantial shift yet. Try stronger interventions.")

    st.subheader("📈 Behavior & Risk")
    beh_col1, beh_col2 = st.columns([1.25, 1])

    with beh_col1:
        active_interventions = {k: v for k, v in intervention_deltas.items() if v != 0}
        if active_interventions:
            cols = st.columns(2)
            for i, (k, v) in enumerate(active_interventions.items()):
                cols[i % 2].metric(label=k, value=f"{v:+.2f}")
        else:
            st.warning("Adjust behavior sliders in the sidebar to see their effects.")

        sd_units = []
        for b_name in behavior_names:
            sd = float(behavior_std[b_name])
            sd_units.append(intervention_deltas[b_name] / sd if sd > 0 else 0.0)

        df_beh = pd.DataFrame({"Behavior": behavior_names, "Change_SD": sd_units})

        if np.isclose(df_beh["Change_SD"].abs().sum(), 0):
            st.write("_No behavior changes yet._")
        else:
            # 按 |Change_SD| 排序，突出主要干预
            df_beh = df_beh.assign(abs_change=lambda d: d["Change_SD"].abs())
            df_beh = df_beh.sort_values("abs_change", ascending=False)

            fig_beh = px.bar(
                df_beh, x="Behavior", y="Change_SD",
                template=PLOTLY_TEMPLATE, color="Change_SD",
                color_continuous_scale=[(0.0, RED), (0.5, GRAY), (1.0, BLUE)],
                hover_data={"Change_SD":":+.2f"}
            )
            fig_beh.update_layout(
                height=360,
                coloraxis_showscale=False,
                yaxis_title="Change (SD units)",
                title=dict(text="Magnitude of Behavior Interventions", x=0.01),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig_beh.add_hline(y=0, line_width=1, line_dash="dash", line_color=DARK)
            fig_beh.update_xaxes(tickangle=20, automargin=True)
            st.plotly_chart(fig_beh, width="stretch")

    # ---- Overall Risk Change 图（美化版） ----
    with beh_col2:
        # 根据方向设置颜色：下降=绿色，上升=红色
        post_color = RED if risk_change_abs > 0 else PRIMARY

        df_risk = pd.DataFrame({
            "State": ["Baseline", "Post-intervention"],
            "Risk": [base_prob, new_prob]
        })

        fig_risk = px.bar(
            df_risk, x="State", y="Risk",
            template=PLOTLY_TEMPLATE, color="State",
            color_discrete_map={
                "Baseline": GRAY,
                "Post-intervention": post_color
            },
            text=df_risk["Risk"].map(lambda x: f"{x:.3f}")
        )
        fig_risk.update_traces(textposition="outside")

        direction = "↑ Risk increase" if risk_change_abs > 0 else "↓ Risk reduction"

        # 统一 y 轴上限
        y_top = max(df_risk["Risk"]) * 1.25 + 0.02
        fig_risk.update_layout(
            height=360,
            showlegend=False,
            yaxis_title="Predicted Risk (probability)",
            title=dict(text="Overall Risk Change", x=0.01),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        fig_risk.update_yaxes(range=[0, y_top])

        # baseline 虚线，方便比较
        fig_risk.add_hline(
            y=base_prob,
            line_width=1,
            line_dash="dash",
            line_color=DARK
        )

        fig_risk.add_annotation(
            x=1,
            y=y_top * 0.96,
            text=f"{direction}: {risk_change_rel:+.1%}",
            showarrow=False,
            font=dict(size=12, color=DARK)
        )

        st.plotly_chart(fig_risk, width="stretch")


# ================== TAB 2: Continuous Risk Curve ==================
with tab_curve:
    st.subheader("📈 Continuous Intervention → Risk Curve")
    st.markdown(
        "Select a behavior. The simulator sweeps intervention strength (default ±2SD) "
        "and shows a continuous risk-response curve."
    )

    beh_choice = st.selectbox("Choose behavior", behavior_names, index=0)
    sd = float(behavior_std[beh_choice])

    sweep_range = st.slider("Sweep range (in SD units)", 0.5, 4.0, 2.0, 0.5)
    num_points = st.slider("Number of sweep points", 20, 200, 60, 10)

    deltas = np.linspace(-sweep_range * sd, sweep_range * sd, num_points)

    risks = []
    for d in deltas:
        betas = coef_matrix.loc[proteins, beh_choice].values
        drift_tmp = betas * d

        vals_tmp = np.clip(df_base.values[0] + drift_tmp, -5, 5)
        df_tmp = pd.DataFrame([vals_tmp], columns=proteins)
        r = risk_model.predict_proba(df_tmp)[0, 1]
        risks.append(r)

    df_curve = pd.DataFrame({
        "Delta_raw": deltas,
        "Delta_SD": deltas / sd if sd > 0 else deltas,
        "Risk": risks
    })

    fig_curve = px.line(
        df_curve, x="Delta_SD", y="Risk",
        template=PLOTLY_TEMPLATE, markers=True,
        hover_data={"Delta_raw":":+.2f", "Risk":":.3f"}
    )
    fig_curve.update_layout(
        height=480,
        xaxis_title=f"Δ {beh_choice} (SD units)",
        yaxis_title="Predicted VTE Risk",
        title=dict(text=f"Risk curve for {beh_choice}", x=0.01),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # baseline risk 虚线
    fig_curve.add_hline(y=base_prob, line_width=1, line_dash="dash", line_color=DARK)
    fig_curve.add_annotation(
        x=df_curve["Delta_SD"].min(), y=base_prob,
        text="Baseline risk", showarrow=False, yshift=10,
        font=dict(size=11, color=DARK)
    )

    # 当前干预位置（如果这个行为有调整）
    current_delta_raw = intervention_deltas[beh_choice]
    current_delta_sd = current_delta_raw / sd if sd > 0 else 0.0
    if current_delta_raw != 0:
        fig_curve.add_vline(
            x=current_delta_sd,
            line_width=1,
            line_dash="dot",
            line_color=BLUE
        )
        fig_curve.add_annotation(
            x=current_delta_sd,
            y=max(risks),
            text="Current intervention",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=11, color=DARK)
        )

    st.plotly_chart(fig_curve, width="stretch")

# ================== TAB 3: Sensitivity Heatmap ==================
with tab_heat:
    st.subheader("🔥 Single-behavior Sensitivity Heatmap")
    st.markdown(
        "Sensitivity analysis for one behavior: sweep intervention strength (±2SD), "
        "compute protein drifts, and render a protein-by-delta heatmap."
    )

    beh_choice2 = st.selectbox("Choose behavior for heatmap", behavior_names, index=2)
    sd2 = float(behavior_std[beh_choice2])

    sweep_range2 = st.slider(
        "Heatmap sweep range (SD units)", 0.5, 4.0, 2.0, 0.5, key="hm_range"
    )
    num_points2 = st.slider(
        "Heatmap sweep points", 10, 80, 30, 5, key="hm_points"
    )

    deltas2 = np.linspace(-sweep_range2 * sd2, sweep_range2 * sd2, num_points2)

    heat = []
    for d in deltas2:
        betas = coef_matrix.loc[proteins, beh_choice2].values
        drift_tmp = betas * d
        heat.append(drift_tmp)

    heat = np.array(heat).T  # proteins x points

    df_heat = pd.DataFrame(
        heat,
        index=proteins,
        columns=[f"{x/sd2:+.2f}SD" if sd2 > 0 else f"{x:+.2f}" for x in deltas2]
    )

    fig_heat = px.imshow(
        df_heat,
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale=[[0, RED], [0.5, GRAY], [1, BLUE]],
        labels=dict(color="Protein Drift (Δ Z-score)")
    )
    # 颜色中心设为 0（替代 zmid 参数）
    fig_heat.update_coloraxes(cmid=0.0)

    fig_heat.update_layout(
        height=520,
        title=dict(text=f"Sensitivity heatmap for {beh_choice2}", x=0.01),
        xaxis_title=f"Δ {beh_choice2} (SD units)",
        yaxis_title="Proteins",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_heat, width="stretch")

    with st.expander("Heatmap table (ΔZ values)"):
        st.dataframe(df_heat.style.format("{:+.4f}"))
