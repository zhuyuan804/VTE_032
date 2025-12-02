import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------ Page config ------------------
st.set_page_config(page_title="VTE Risk Digital Twin Simulator", layout="wide")

# ------------------ Constants ------------------
PLOTLY_TEMPLATE = "plotly_white"
PRIMARY = "#2ECC71"
GRAY = "#BDC3C7"
DARK = "#2C3E50"
RED = "#E74C3C"
BLUE = "#3498DB"

# Unified figure settings for journal-style plots
FIG_FONT_FAMILY = "Times New Roman"  # Commonly accepted by many journals
FIG_BASE_FONT_SIZE = 14
FIG_TITLE_SIZE = 16
FIG_TICK_FONT_SIZE = 12
FIG_WIDTH_PX = 900
FIG_HEIGHT_PX = 600
FIG_LINE_WIDTH = 2.0

# ------------------ Export config for paper figures ------------------
# 使用相对路径，导出到当前目录
EXPORT_DIR = "."


def apply_paper_style(fig, width=FIG_WIDTH_PX, height=FIG_HEIGHT_PX):
    """Apply journal-like styling to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=width,
        height=height,
        font=dict(
            family=FIG_FONT_FAMILY,
            size=FIG_BASE_FONT_SIZE,
        ),
        title=dict(
            font=dict(
                family=FIG_FONT_FAMILY,
                size=FIG_TITLE_SIZE,
            )
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=FIG_BASE_FONT_SIZE - 2),
        ),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=FIG_LINE_WIDTH,
        ticks="outside",
        tickwidth=FIG_LINE_WIDTH,
        tickfont=dict(size=FIG_TICK_FONT_SIZE, family=FIG_FONT_FAMILY),
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=FIG_LINE_WIDTH,
        ticks="outside",
        tickwidth=FIG_LINE_WIDTH,
        tickfont=dict(size=FIG_TICK_FONT_SIZE, family=FIG_FONT_FAMILY),
        mirror=True,
    )
    return fig


def export_figure_for_paper(
    fig,
    filename_base,
    width=FIG_WIDTH_PX,
    height=FIG_HEIGHT_PX,
    scale=3,
):
    """
    Export a Plotly figure as PNG + PDF into EXPORT_DIR.
    Requires kaleido (compatible version).
    """
    png_path = os.path.join(EXPORT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(EXPORT_DIR, f"{filename_base}.pdf")

    fig.write_image(png_path, width=width, height=height, scale=scale)
    fig.write_image(pdf_path, width=width, height=height, scale=scale)

    return png_path, pdf_path


def export_and_download_for_paper(
    fig,
    filename_base: str,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: int = 3,
    preview: bool = True,
    caption: str | None = None,
    key_prefix: str | None = None,
):
    """
    统一的导出 + 浏览器下载助手：
    1. 使用 export_figure_for_paper 导出 PNG & PDF；
    2. 在页面中可选预览 PNG；
    3. 提供两个 download_button 供下载。
    """
    export_kwargs = {"scale": scale}
    if width is not None:
        export_kwargs["width"] = width
    if height is not None:
        export_kwargs["height"] = height

    png_path, pdf_path = export_figure_for_paper(fig, filename_base, **export_kwargs)

    # 读取二进制数据
    with open(png_path, "rb") as f:
        png_bytes = f.read()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    basename_png = os.path.basename(png_path)
    basename_pdf = os.path.basename(pdf_path)

    st.success(
        f"✅ Exported figure files:\n- {basename_png}\n- {basename_pdf}"
    )

    if preview:
        st.image(
            png_bytes,
            caption=caption or filename_base,
            use_column_width=True,
        )

    key_prefix = key_prefix or filename_base

    st.download_button(
        label="⬇️ Download PNG",
        data=png_bytes,
        file_name=basename_png,
        mime="image/png",
        key=f"{key_prefix}_png",
    )
    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name=basename_pdf,
        mime="application/pdf",
        key=f"{key_prefix}_pdf",
    )


# ------------------ Global CSS (clean dashboard style) ------------------
st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                       Roboto, "Helvetica Neue", Arial;
          color: #1f2937;
      }
      .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2.5rem;
      }
      section[data-testid="stSidebar"] {
          background: #F8FAFC;
          border-right: 1px solid #E5E7EB;
      }
      div[data-testid="metric-container"] {
          background: white;
          border: 1px solid #EEF2F7;
          padding: 14px 14px 10px 14px;
          border-radius: 14px;
          box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
      }
      button[data-baseweb="tab"] {
          font-weight: 600;
          padding: 10px 14px;
      }
      .stPlotlyChart {
          background: white;
          border: 1px solid #EEF2F7;
          border-radius: 14px;
          padding: 8px;
          box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
      }
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
    """Initialize session_state for all sliders if missing."""
    for p in proteins:
        st.session_state.setdefault(f"base_{p}", 0.0)
    for b in behavior_names:
        st.session_state.setdefault(f"delta_{b}", 0.0)


def reset_all():
    """Reset all baseline & intervention sliders to 0."""
    for p in proteins:
        st.session_state[f"base_{p}"] = 0.0
    for b in behavior_names:
        st.session_state[f"delta_{b}"] = 0.0


init_state()

# ------------------ 3. Title & intro ------------------
st.title("🧬 VTE Risk Digital Twin Simulator")
st.markdown(
    """
Use the sidebar to adjust lifestyle behaviors, simulate **plasma protein expression**
changes, and estimate the impact on **venous thromboembolism (VTE) risk**.
"""
)

# ------------------ 4. Sidebar ------------------
st.sidebar.header("1. Baseline Protein Profile")
st.sidebar.caption("All proteins default to the population mean (Z-score = 0).")

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
    delta_color="inverse",  # decrease = green, increase = red
)
k3.metric("Relative Risk Change", f"{risk_change_rel:+.1%}")

# ================== Tabs: Main / Curve / Heatmap ==================
tab_main, tab_curve, tab_heat = st.tabs(
    ["📌 Main View", "📈 Continuous Risk Curve", "🔥 Sensitivity Heatmap"]
)

# ================== TAB 1: Main View ==================
with tab_main:
    st.subheader("🧬 Protein-level Changes")
    prot_col1, prot_col2 = st.columns([1.25, 1])

    # store per-panel figures for later combination
    fig_prot = None
    fig_sig = None
    fig_beh = None
    fig_risk = None

    baseline_vals_arr = df_base.values[0]
    intervention_vals_arr = new_vals
    delta_arr = intervention_vals_arr - baseline_vals_arr

    # ---------- Panel A: Protein Expression Shift ----------
    with prot_col1:
        # 根据绝对变化排序
        order_idx = np.argsort(-np.abs(delta_arr))
        ordered_proteins = [proteins[i] for i in order_idx]
        baseline_ord = baseline_vals_arr[order_idx]
        intervention_ord = intervention_vals_arr[order_idx]

        fig_prot = go.Figure()

        # Baseline bars
        fig_prot.add_trace(
            go.Bar(
                x=ordered_proteins,
                y=baseline_ord,
                name="Baseline",
                marker_color=[GRAY] * len(ordered_proteins),
                text=[f"{v:.3f}" if abs(v) > 1e-6 else "" for v in baseline_ord],
                texttemplate="%{text}",
                textposition="outside",
                marker_line=dict(width=FIG_LINE_WIDTH, color="black"),
            )
        )

        # Intervention bars
        fig_prot.add_trace(
            go.Bar(
                x=ordered_proteins,
                y=intervention_ord,
                name="Intervention",
                marker_color=[PRIMARY] * len(ordered_proteins),
                text=[f"{v:.3f}" if abs(v) > 1e-6 else "" for v in intervention_ord],
                texttemplate="%{text}",
                textposition="outside",
                marker_line=dict(width=FIG_LINE_WIDTH, color="black"),
            )
        )

        fig_prot.update_layout(
            barmode="group",
            height=420,
            legend_title_text="State",
            yaxis_title="Expression (Z-score)",
            xaxis_title="Proteins",
            title=dict(text="Protein Expression (Baseline vs Intervention)",
                       x=0.02, xanchor="left"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig_prot.update_xaxes(tickangle=-20, automargin=True)

        # 适当扩展 y 范围，防止数值被裁掉
        max_y = max(baseline_ord.max(), intervention_ord.max(), 0)
        min_y = min(baseline_ord.min(), intervention_ord.min(), 0)
        padding = (max_y - min_y) * 0.25 if max_y != min_y else 0.2
        fig_prot.update_yaxes(range=[min_y - padding, max_y + padding])

        fig_prot = apply_paper_style(fig_prot, width=FIG_WIDTH_PX, height=420)
        st.plotly_chart(fig_prot, use_container_width=True)

    # ---------- Panel B: Top Drift Proteins ----------
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

            colors_sig = [
                RED if v < 0 else PRIMARY
                for v in df_sig["Drift"]
            ]

            fig_sig = go.Figure()
            fig_sig.add_trace(
                go.Bar(
                    x=df_sig["Drift"],
                    y=df_sig["Protein"],
                    orientation="h",
                    marker_color=colors_sig,
                    text=[f"{v:.3f}" if abs(v) > 1e-6 else "" for v in df_sig["Drift"]],
                    texttemplate="%{text}",
                    # 关键修改：使用 auto，把文字放在柱子内部，避免相互重叠
                    textposition="auto",
                    insidetextanchor="middle",
                    marker_line=dict(width=FIG_LINE_WIDTH, color="black"),
                    showlegend=False,
                )
            )

            fig_sig.update_layout(
                height=420,
                xaxis_title="Protein Drift (Δ Z-score)",
                yaxis_title="Proteins",
                title=dict(text="Top Drift Proteins", x=0.02, xanchor="left"),
                margin=dict(l=70, r=20, t=50, b=40),
            )
            fig_sig.add_vline(
                x=0, line_width=1.5, line_dash="dash", line_color=DARK
            )
            max_x = df_sig["Drift"].max()
            min_x = df_sig["Drift"].min()
            padding = (max_x - min_x) * 0.4 if max_x != min_x else 0.05
            fig_sig.update_xaxes(range=[min_x - padding, max_x + padding])

            fig_sig = apply_paper_style(fig_sig, width=FIG_WIDTH_PX, height=420)
            st.plotly_chart(fig_sig, use_container_width=True)
        else:
            st.info("No protein shows a substantial shift yet. Try stronger interventions.")

    st.subheader("📈 Behavior & Risk")
    beh_col1, beh_col2 = st.columns([1.25, 1])

    # ---------- Panel C: Magnitude of Behavior Interventions ----------
    with beh_col1:
        sd_units = []
        for b_name in behavior_names:
            sd = float(behavior_std[b_name])
            sd_units.append(intervention_deltas[b_name] / sd if sd > 0 else 0.0)

        df_beh = pd.DataFrame({"Behavior": behavior_names, "SD": sd_units})
        df_beh = df_beh.assign(abs_change=lambda d: d["SD"].abs()).sort_values("abs_change", ascending=False)

        bar_colors = [BLUE if v > 0 else RED for v in df_beh["SD"]]
        figC = go.Figure()
        figC.add_trace(
            go.Bar(
                x=df_beh["Behavior"],
                y=df_beh["SD"],
                marker_color=bar_colors,
                marker_line=dict(width=FIG_LINE_WIDTH, color="black"),
                text=[f"{v:.3f}" if abs(v) > 1e-6 else "" for v in df_beh["SD"]],
                texttemplate="%{text}",
                textposition="outside",
                showlegend=False,
            )
        )

        # 扩展 Y 轴高度，避免数值被裁掉
        ymax = df_beh["SD"].max()
        ymin = df_beh["SD"].min()
        padding = (ymax - ymin) * 0.4 if ymax != ymin else 0.4
        figC.update_yaxes(range=[ymin - padding, ymax + padding])

        figC.update_layout(
            height=360,
            yaxis_title="Change (SD units)",
            title=dict(text="Magnitude of Behavior Interventions", x=0.02),
            margin=dict(l=40, r=20, t=50, b=40),
        )

        figC.add_hline(y=0, line_dash="dash", line_color=DARK)
        figC.update_xaxes(tickangle=20)

        figC = apply_paper_style(figC, FIG_WIDTH_PX, 360)
        fig_beh = figC
        st.plotly_chart(figC, use_container_width=True)

    # ---------- Panel D: Overall Risk Change ----------
    with beh_col2:
        post_color = RED if risk_change_abs > 0 else PRIMARY

        baseline_val = base_prob
        post_val = new_prob

        fig_risk = go.Figure()
        fig_risk.add_trace(
            go.Bar(
                x=["Baseline", "Post-intervention"],
                y=[baseline_val, post_val],
                marker_color=[GRAY, post_color],
                text=[
                    f"{baseline_val:.3f}" if abs(baseline_val) > 1e-6 else "",
                    f"{post_val:.3f}" if abs(post_val) > 1e-6 else "",
                ],
                texttemplate="%{text}",
                textposition="outside",
                marker_line=dict(width=FIG_LINE_WIDTH, color="black"),
                showlegend=False,
            )
        )

        direction = "Risk increase" if risk_change_abs > 0 else "Risk reduction"
        y_top = max(baseline_val, post_val) * 1.25 + 0.02

        fig_risk.update_layout(
            height=420,
            yaxis_title="Predicted Risk (probability)",
            xaxis_title="State",
            title=dict(text="Overall Risk Change", x=0.02, xanchor="left"),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        fig_risk.update_yaxes(range=[0, y_top])

        fig_risk.add_hline(
            y=baseline_val,
            line_width=1.5,
            line_dash="dash",
            line_color=DARK,
        )

        fig_risk.add_annotation(
            x=1,
            y=y_top * 0.96,
            text=f"{direction}: {risk_change_rel:+.1%}",
            showarrow=False,
            font=dict(size=FIG_BASE_FONT_SIZE, family=FIG_FONT_FAMILY, color=DARK),
        )

        fig_risk = apply_paper_style(fig_risk, width=FIG_WIDTH_PX, height=420)
        st.plotly_chart(fig_risk, use_container_width=True)

    # ---------- Export combined Main View (A–D) ----------
    st.markdown("---")
    st.markdown("### Export combined Main View (panels A–D)")

    if st.button("💾 Export combined Main View (PNG + PDF)", key="export_main_view"):
        if fig_prot is None or fig_risk is None:
            st.error("Protein or Overall Risk plots are missing.")
        elif fig_sig is None:
            st.error("Top Drift Proteins plot is missing (no significant drifts).")
        elif fig_beh is None:
            st.error("Magnitude of Behavior Interventions plot is missing.")
        else:
            combined = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Protein Expression (Baseline vs Intervention)",
                    "Top Drift Proteins",
                    "Magnitude of Behavior Interventions",
                    "Overall Risk Change",
                ),
                horizontal_spacing=0.18,
                vertical_spacing=0.20,
            )

            # Panel A traces
            for trace in fig_prot.data:
                combined.add_trace(trace, row=1, col=1)

            # Panel B traces
            for trace in fig_sig.data:
                combined.add_trace(trace, row=1, col=2)

            # Panel C traces
            for trace in fig_beh.data:
                combined.add_trace(trace, row=2, col=1)

            # Panel D trace（重新构建，确保数值标签）
            combined.add_trace(
                go.Bar(
                    x=["Baseline", "Post-intervention"],
                    y=[baseline_val, post_val],
                    marker_color=[GRAY, PRIMARY],
                    marker_line=dict(width=FIG_LINE_WIDTH, color="black"),
                    showlegend=False,
                    text=[
                        f"{baseline_val:.3f}" if abs(baseline_val) > 1e-6 else "",
                        f"{post_val:.3f}" if abs(post_val) > 1e-6 else "",
                    ],
                    texttemplate="%{text}",
                    textposition="outside",
                ),
                row=2,
                col=2,
            )

            # baseline hline in panel D
            combined.add_hline(
                y=baseline_val,
                line_dash="dash",
                line_width=1.5,
                line_color=DARK,
                row=2,
                col=2,
            )

            # risk reduction annotation in panel D
            combined.add_annotation(
                x="Post-intervention",
                y=baseline_val * 1.05,
                xref="x4",
                yref="y4",
                text=f"Risk reduction: {risk_change_rel:+.1%}",
                showarrow=False,
                font=dict(size=FIG_TICK_FONT_SIZE, family=FIG_FONT_FAMILY, color=DARK),
            )

            # Axis titles
            combined.update_xaxes(title_text="Proteins", row=1, col=1)
            combined.update_yaxes(title_text="Expression (Z-score)", row=1, col=1)

            combined.update_xaxes(title_text="Protein Drift (Δ Z-score)", row=1, col=2)
            combined.update_yaxes(title_text="Proteins", row=1, col=2)

            combined.update_xaxes(title_text="Behavior", row=2, col=1)
            combined.update_yaxes(title_text="Change (SD units)", row=2, col=1)

            combined.update_xaxes(title_text="State", row=2, col=2)
            combined.update_yaxes(
                title_text="Predicted Risk (probability)", row=2, col=2
            )

            combined = apply_paper_style(combined, width=1200, height=950)
            combined.update_layout(showlegend=False)

            # remove any colorbar if exists
            for attr in list(combined.layout):
                if attr.startswith("coloraxis"):
                    getattr(combined.layout, attr).showscale = False

            # Panel labels A/B/C/D
            label_font = dict(family=FIG_FONT_FAMILY, size=18, color="black")

            combined.add_annotation(
                text="A",
                xref="x domain",
                yref="y domain",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=label_font,
                row=1,
                col=1,
            )
            combined.add_annotation(
                text="B",
                xref="x domain",
                yref="y domain",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=label_font,
                row=1,
                col=2,
            )
            combined.add_annotation(
                text="C",
                xref="x domain",
                yref="y domain",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=label_font,
                row=2,
                col=1,
            )
            combined.add_annotation(
                text="D",
                xref="x domain",
                yref="y domain",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=label_font,
                row=2,
                col=2,
            )

            # 统一导出 + 浏览器下载
            export_and_download_for_paper(
                combined,
                filename_base="main_view_panel",
                width=1200,
                height=950,
                scale=3,
                preview=True,
                caption="Main View (panels A–D)",
                key_prefix="main_view_panel",
            )

# ================== TAB 2: Continuous Risk Curve ==================
with tab_curve:
    st.subheader("📈 Continuous Intervention → Risk Curve")
    st.markdown(
        "Select one behavior, sweep intervention strength (default ±2 SD), "
        "and visualize the continuous risk–response curve."
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

    df_curve = pd.DataFrame(
        {
            "Delta_raw": deltas,
            "Delta_SD": deltas / sd if sd > 0 else deltas,
            "Risk": risks,
        }
    )

    fig_curve = px.line(
        df_curve,
        x="Delta_SD",
        y="Risk",
        markers=True,
        hover_data={"Delta_raw": ":+.2f", "Risk": ":.3f"},
    )
    fig_curve.update_traces(
        line=dict(width=FIG_LINE_WIDTH),
        marker=dict(size=6, line=dict(width=FIG_LINE_WIDTH / 1.5, color="black")),
    )

    fig_curve.update_layout(
        height=480,
        xaxis_title=f"Δ {beh_choice} (SD units)",
        yaxis_title="Predicted VTE Risk",
        title=dict(text=f"Risk curve for {beh_choice}", x=0.02, xanchor="left"),
        margin=dict(l=60, r=20, t=60, b=50),
    )

    fig_curve.add_hline(
        y=base_prob,
        line_width=1.5,
        line_dash="dash",
        line_color=DARK,
    )
    fig_curve.add_annotation(
        x=df_curve["Delta_SD"].min(),
        y=base_prob,
        text="Baseline risk",
        showarrow=False,
        yshift=10,
        font=dict(size=FIG_BASE_FONT_SIZE - 2, family=FIG_FONT_FAMILY, color=DARK),
    )

    current_delta_raw = intervention_deltas[beh_choice]
    current_delta_sd = current_delta_raw / sd if sd > 0 else 0.0
    if current_delta_raw != 0:
        fig_curve.add_vline(
            x=current_delta_sd,
            line_width=1.5,
            line_dash="dot",
            line_color=BLUE,
        )
        fig_curve.add_annotation(
            x=current_delta_sd,
            y=max(risks),
            text="Current intervention",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=FIG_BASE_FONT_SIZE - 2, family=FIG_FONT_FAMILY, color=DARK),
        )

    fig_curve = apply_paper_style(fig_curve, width=FIG_WIDTH_PX, height=480)
    st.plotly_chart(fig_curve, use_container_width=True)

    export_name = f"risk_curve_{beh_choice.replace(' ', '_')}"
    if st.button("💾 Export Risk Curve (PNG + PDF)", key="export_curve"):
        export_and_download_for_paper(
            fig_curve,
            filename_base=export_name,
            preview=True,
            caption=f"Risk curve for {beh_choice}",
            key_prefix=f"risk_curve_{beh_choice}",
        )

# ================== TAB 3: Sensitivity Heatmap ==================
with tab_heat:
    st.subheader("🔥 Single-behavior Sensitivity Heatmap")
    st.markdown(
        "One-way sensitivity analysis for a single behavior: sweep intervention "
        "strength (±2 SD), compute protein drifts, and visualize them as a "
        "protein-by-intervention heatmap."
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
        columns=[f"{x / sd2:+.2f}SD" if sd2 > 0 else f"{x:+.2f}" for x in deltas2],
    )

    fig_heat = px.imshow(
        df_heat,
        aspect="auto",
        color_continuous_scale=[[0, RED], [0.5, GRAY], [1, BLUE]],
        labels=dict(color="Protein Drift (Δ Z-score)"),
    )
    fig_heat.update_coloraxes(cmid=0.0)

    fig_heat.update_layout(
        height=520,
        title=dict(text=f"Sensitivity heatmap for {beh_choice2}", x=0.02, xanchor="left"),
        xaxis_title=f"Δ {beh_choice2} (SD units)",
        yaxis_title="Proteins",
        margin=dict(l=80, r=40, t=60, b=80),
    )

    fig_heat = apply_paper_style(fig_heat, width=FIG_WIDTH_PX, height=520)
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Heatmap table (ΔZ values)"):
        st.dataframe(df_heat.style.format("{:+.4f}"))

    if st.button("💾 Export Heatmap (PNG + PDF)", key="export_heatmap"):
        export_and_download_for_paper(
            fig_heat,
            filename_base=f"heatmap_{beh_choice2.replace(' ', '_')}",
            preview=True,
            caption=f"Sensitivity heatmap for {beh_choice2}",
            key_prefix=f"heatmap_{beh_choice2}",
        )
