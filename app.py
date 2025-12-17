import os
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------ Page config ------------------
st.set_page_config(
    page_title="VTE Risk Simulation System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Constants & Style ------------------
PLOTLY_TEMPLATE = "plotly_white"

PRIMARY_COLOR = "#2ECC71"  # Green (lower / healthier)
RISK_COLOR = "#E74C3C"     # Red (higher / risky)
NEUTRAL_COLOR = "#95A5A6"  # Gray

# Dashboard font (on-screen)
DASH_FONT_FAMILY = "Arial"

# Paper/report export style (PNG/PDF)
FIG_FONT_FAMILY = "Times New Roman"  # If unavailable, switch to "Arial" or "DejaVu Sans"
FIG_BASE_FONT_SIZE = 14
FIG_TITLE_SIZE = 16
FIG_TICK_FONT_SIZE = 12
FIG_LINE_WIDTH = 2.0

# Export directory
EXPORT_DIR = "./export_figures"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Default export size (pixels)
EXPORT_WIDTH_PX = 1200
EXPORT_HEIGHT_PX = 700


# ------------------ Export helpers ------------------
def apply_paper_style(fig, width=EXPORT_WIDTH_PX, height=EXPORT_HEIGHT_PX):
    """Apply journal/report-like styling to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=width,
        height=height,
        font=dict(family=FIG_FONT_FAMILY, size=FIG_BASE_FONT_SIZE),
        title=dict(font=dict(family=FIG_FONT_FAMILY, size=FIG_TITLE_SIZE)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=FIG_BASE_FONT_SIZE - 2),
        ),
        margin=dict(l=60, r=30, t=70, b=60),
    )
    # Note: Indicator(gauge) has no axes; update_xaxes/update_yaxes won't affect it.
    fig.update_xaxes(
        showline=True, linecolor="black", linewidth=FIG_LINE_WIDTH,
        ticks="outside", tickwidth=FIG_LINE_WIDTH,
        tickfont=dict(size=FIG_TICK_FONT_SIZE, family=FIG_FONT_FAMILY),
        mirror=True,
    )
    fig.update_yaxes(
        showline=True, linecolor="black", linewidth=FIG_LINE_WIDTH,
        ticks="outside", tickwidth=FIG_LINE_WIDTH,
        tickfont=dict(size=FIG_TICK_FONT_SIZE, family=FIG_FONT_FAMILY),
        mirror=True,
    )
    return fig


def export_figure_png_pdf(fig, filename_base, *, width=EXPORT_WIDTH_PX, height=EXPORT_HEIGHT_PX, scale=3):
    """
    Export a Plotly figure as PNG + PDF into EXPORT_DIR.
    Requires kaleido:
      pip install -U kaleido
    """
    png_path = os.path.join(EXPORT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(EXPORT_DIR, f"{filename_base}.pdf")

    fig.write_image(png_path, width=width, height=height, scale=scale)
    fig.write_image(pdf_path, width=width, height=height, scale=scale)
    return png_path, pdf_path


def export_and_download(fig, filename_base, *, width=None, height=None, scale=3,
                        preview=True, caption=None, key_prefix=None):
    """Export + preview + download buttons (PNG & PDF)."""
    w = width if width is not None else EXPORT_WIDTH_PX
    h = height if height is not None else EXPORT_HEIGHT_PX

    png_path, pdf_path = export_figure_png_pdf(fig, filename_base, width=w, height=h, scale=scale)

    with open(png_path, "rb") as f:
        png_bytes = f.read()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.success(f"✅ Exported: {os.path.basename(png_path)} / {os.path.basename(pdf_path)}")

    if preview:
        st.image(png_bytes, caption=caption or filename_base, use_column_width=True)

    kp = key_prefix or filename_base
    st.download_button(
        "⬇️ Download PNG",
        data=png_bytes,
        file_name=os.path.basename(png_path),
        mime="image/png",
        key=f"{kp}_png",
    )
    st.download_button(
        "⬇️ Download PDF",
        data=pdf_bytes,
        file_name=os.path.basename(pdf_path),
        mime="application/pdf",
        key=f"{kp}_pdf",
    )


# ------------------ System loading ------------------
@st.cache_resource
def load_system():
    # Relative path: vte_system.pkl located in the same directory as this script.
    script_dir = Path(__file__).resolve().parent
    pkl_path = script_dir / "vte_system.pkl"

    if not pkl_path.exists():
        return None, str(pkl_path)

    return joblib.load(str(pkl_path)), str(pkl_path)


system, model_path = load_system()
if system is None:
    st.error(f"⚠️ Model file not found: {model_path}")
    st.stop()

# ------------------ Helpers ------------------
def sanitize_key(s):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(s)).strip("_")

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    return 0.5

def apply_chart_style(fig, height=400):
    """Dashboard display style (screen)."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(family=DASH_FONT_FAMILY, size=13),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ------------------ Unpack system ------------------
risk_model = system["risk_model"]
coef_matrix = system["coef_matrix"]
proteins = system["proteins"]
behavior_meta = system["behavior_meta"]
behavior_names = list(behavior_meta.keys())

# std dict: for "unit consistency" checks (kept as in your original code)
behavior_std_dict = system.get("behavior_std", {})


# ------------------ Sidebar ------------------
st.sidebar.title("🕹️ Simulation Console")
st.sidebar.markdown("Adjust **daily averages** of lifestyle factors to predict changes in VTE risk.")

st.sidebar.subheader("1. Set lifestyle targets")

intervention_deltas = {}

# Reset mechanism: avoid widget state collision
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0
if st.sidebar.button("🔄 Reset all settings", use_container_width=True):
    st.session_state.reset_trigger += 1

for b in behavior_names:
    meta = behavior_meta[b]
    unit = meta.get("unit", "")
    display_name = b.split("(")[0].strip()

    k_cur = f"cur_{sanitize_key(b)}_{st.session_state.reset_trigger}"
    k_tar = f"tar_{sanitize_key(b)}_{st.session_state.reset_trigger}"

    default_val = float(meta.get("default_current", 0.0))
    min_v = float(meta.get("min", 0.0))
    max_v = float(meta.get("max", 9999.0))
    step_v = float(meta.get("step", 1.0))

    with st.sidebar.expander(f"{display_name} ({unit})", expanded=False):
        c1, c2 = st.columns(2)
        cur_val = c1.number_input("Current", value=default_val, min_value=min_v, max_value=max_v, step=step_v, key=k_cur)
        tar_val = c2.number_input("Target", value=cur_val, min_value=min_v, max_value=max_v, step=step_v, key=k_tar)

        slider_val = st.slider(
            "Quick adjust target",
            min_value=min_v,
            max_value=max_v,
            value=tar_val,
            step=step_v,
            key=f"sl_{k_tar}",
        )

        final_tar = slider_val
        delta = final_tar - cur_val
        intervention_deltas[b] = delta

        if abs(delta) > 1e-4:
            color = "green" if delta > 0 else "red"
            st.markdown(f"**Change:** <span style='color:{color}'>{delta:+.2f} {unit}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced: baseline protein levels (Z-score)"):
    st.info("Adjust the participant's baseline physiological status (default is 0 for all, i.e., population mean).")
    base_protein_vals = []
    for p in proteins:
        val = st.slider(f"{p}", -3.0, 3.0, 0.0, 0.1, key=f"base_{sanitize_key(p)}_{st.session_state.reset_trigger}")
        base_protein_vals.append(val)
    base_protein_arr = np.array(base_protein_vals)


# ------------------ Calculation ------------------
# 1) baseline risk
df_base = pd.DataFrame([base_protein_arr], columns=proteins)
if hasattr(risk_model, "feature_names_in_"):
    df_base = df_base[risk_model.feature_names_in_]
base_prob = safe_predict_proba(risk_model, df_base)

# 2) protein drift
drift = np.zeros(len(proteins))

for b, d_raw in intervention_deltas.items():
    if abs(d_raw) < 1e-6:
        continue
    if b not in coef_matrix.columns:
        continue

    betas = coef_matrix.loc[proteins, b].values

    # Assumption: coef_matrix is "protein Z change per raw unit" of behavior change.
    # If coef_matrix is per-SD instead, replace with:
    # d_sd = d_raw / max(behavior_std_dict.get(b, 1.0), 1e-12)
    # drift += betas * d_sd
    drift += betas * d_raw

new_vals = base_protein_arr + drift
df_new = pd.DataFrame([new_vals], columns=proteins)
if hasattr(risk_model, "feature_names_in_"):
    df_new = df_new[risk_model.feature_names_in_]
new_prob = safe_predict_proba(risk_model, df_new)

risk_change = new_prob - base_prob


# ------------------ Main Dashboard ------------------
st.title("🧬 VTE Risk Simulation System")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Current baseline risk", f"{base_prob:.1%}", help="Predicted risk based on current protein levels")
with col_m2:
    delta_color = "normal" if risk_change == 0 else ("inverse" if risk_change < 0 else "off")
    st.metric("Post-intervention risk", f"{new_prob:.1%}", f"{risk_change:+.1%}", delta_color=delta_color)
with col_m3:
    status_text = "No change"
    if risk_change < -0.01:
        status_text = "📉 Risk reduced (notable)"
    elif risk_change > 0.01:
        status_text = "📈 Risk increased"
    st.markdown(f"### {status_text}")

st.markdown("---")

# ------------------ Visuals ------------------
col_viz_left, col_viz_right = st.columns([1, 1.2])

# (A) Gauge
with col_viz_left:
    st.subheader("📊 Overall risk assessment")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=new_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Predicted VTE Probability (%)"},
            delta={
                "reference": base_prob * 100,
                "increasing": {"color": RISK_COLOR},
                "decreasing": {"color": PRIMARY_COLOR},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": PRIMARY_COLOR if risk_change <= 0 else RISK_COLOR},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 20], "color": "rgba(46, 204, 113, 0.3)"},
                    {"range": [20, 50], "color": "rgba(241, 196, 15, 0.3)"},
                    {"range": [50, 100], "color": "rgba(231, 76, 60, 0.3)"},
                ],
            },
        )
    )
    apply_chart_style(fig_gauge, height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("Note: This probability is the model prediction and reflects relative risk. Green indicates a low-risk range.")

    st.markdown("#### Export figure: overall risk gauge")
    if st.button("💾 Export risk gauge (PNG + PDF)", key="export_gauge"):
        fig_out = apply_paper_style(fig_gauge, width=1000, height=600)
        export_and_download(
            fig_out,
            filename_base="vte_risk_gauge",
            width=1000,
            height=600,
            scale=3,
            preview=True,
            caption="VTE Risk Gauge",
            key_prefix="vte_risk_gauge",
        )

# (B) Top protein bars
with col_viz_right:
    st.subheader("🧬 Key protein response mechanism")

    delta_abs = np.abs(drift)
    top_idx = np.argsort(delta_abs)[::-1][:10]  # change to [:5] for Top5

    top_proteins = [proteins[i] for i in top_idx]
    top_base = base_protein_arr[top_idx]
    top_new = new_vals[top_idx]

    base_text = [f"{v:+.3f}" for v in top_base]
    new_text  = [f"{v:+.3f}" for v in top_new]

    fig_prot = go.Figure()

    # Baseline
    fig_prot.add_trace(
        go.Bar(
            y=top_proteins,
            x=top_base,
            name="Baseline",
            orientation="h",
            marker=dict(color=NEUTRAL_COLOR, opacity=0.6),
            text=base_text,
            texttemplate="%{text}",
            textposition="auto",
            cliponaxis=False,
        )
    )

    # Predicted
    pred_color = PRIMARY_COLOR if risk_change <= 0 else RISK_COLOR
    fig_prot.add_trace(
        go.Bar(
            y=top_proteins,
            x=top_new,
            name="Predicted",
            orientation="h",
            marker=dict(color=pred_color),
            text=new_text,
            texttemplate="%{text}",
            textposition="auto",
            cliponaxis=False,
        )
    )

    fig_prot.update_layout(
        barmode="group",
        xaxis_title="Protein expression (Z-score)",
        yaxis=dict(autorange="reversed"),
    )

    # Add x-axis padding so labels don't get cramped near edges
    x_min = float(min(np.min(top_base), np.min(top_new), 0))
    x_max = float(max(np.max(top_base), np.max(top_new), 0))
    pad = (x_max - x_min) * 0.25 if x_max != x_min else 0.2
    fig_prot.update_xaxes(range=[x_min - pad, x_max + pad])

    apply_chart_style(fig_prot, height=400)
    st.plotly_chart(fig_prot, use_container_width=True)
    st.caption("Shows protein biomarkers most affected by lifestyle changes (values displayed on the bars).")

# (C) Optional: combined export (Gauge + Top5)
st.markdown("---")
st.subheader("📦 One-click export (Gauge + Top5)")

if st.button("💾 Export combined figure (PNG + PDF)", key="export_combo"):
    combo = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "xy"}]],
        subplot_titles=("Overall Risk Gauge", "Top5 Protein Response"),
        horizontal_spacing=0.15,
    )

    for tr in fig_gauge.data:
        combo.add_trace(tr, row=1, col=1)
    for tr in fig_prot.data:
        combo.add_trace(tr, row=1, col=2)

    combo.update_layout(showlegend=True)

    combo = apply_paper_style(combo, width=1600, height=650)
    export_and_download(
        combo,
        filename_base="vte_report_combo",
        width=1600,
        height=650,
        scale=3,
        preview=True,
        caption="VTE Report (Gauge + Top5)",
        key_prefix="vte_report_combo",
    )

# ------------------ Details table ------------------
with st.expander("🔎 View detailed changes"):
    res_df = pd.DataFrame({
        "Protein": proteins,
        "Baseline (Z)": base_protein_arr,
        "Predicted (Z)": new_vals,
        "Change": drift,
    }).sort_values(by="Change", key=lambda s: s.abs(), ascending=False)

    st.dataframe(
        res_df.style.background_gradient(subset=["Change"], cmap="RdBu_r"),
        use_container_width=True,
    )
