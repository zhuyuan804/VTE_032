import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl  # 用于获取 matplotlib 版本

# ------------------ Page config ------------------
st.set_page_config(
    page_title="VTE Risk Digital Twin Simulator",
    layout="wide",
)

# Use a clean matplotlib style
plt.style.use("ggplot")


# ------------------ 1. Load system ------------------
@st.cache_resource
def load_system():
    # Path to the exported system from your offline script
    return joblib.load("vte_system.pkl")


system = load_system()
risk_model = system["risk_model"]
coef_matrix = system["coef_matrix"]
proteins = system["proteins"]
behaviors_map = system["behaviors"]      # Name -> column
behavior_std = system["behavior_std"]    # behavior SD
behavior_names = list(behaviors_map.keys())


# ------------------ 2. Title & intro ------------------
st.title("🧬 VTE Risk Digital Twin Simulator")

st.markdown(
    """
Adjust lifestyle behaviors in the sidebar to simulate **plasma protein expression**
changes and estimate the impact on **venous thromboembolism (VTE) risk**.
"""
)


# ------------------ 3. Sidebar: baseline & interventions ------------------
st.sidebar.header("1. Baseline Protein Profile")
st.sidebar.info("By default, all proteins are set to the population mean (Z-score = 0).")

baseline_vals = {}
with st.sidebar.expander("Advanced: Tune baseline protein levels", expanded=False):
    for p in proteins:
        baseline_vals[p] = st.slider(f"{p} (Z-score)", -3.0, 3.0, 0.0, 0.1)

st.sidebar.header("2. Behavior Interventions")
intervention_deltas = {}

for b_name in behavior_names:
    # Use SD to define a reasonable slider range
    std = float(behavior_std[b_name])
    min_val = float(-2.0 * std)
    max_val = float(2.0 * std)
    step = float(std / 10.0) if std > 0 else 0.1

    val = st.sidebar.slider(f"Δ {b_name}", min_val, max_val, 0.0, step)
    intervention_deltas[b_name] = val


# ------------------ 4. Core calculation ------------------
# A. Baseline proteins & risk
df_base = pd.DataFrame([list(baseline_vals.values())], columns=proteins)
base_prob = risk_model.predict_proba(df_base)[0, 1]

# B. Protein drift from behaviors (linear mapping)
drift = np.zeros(len(proteins))
for b_name, delta in intervention_deltas.items():
    if delta != 0:
        betas = coef_matrix.loc[proteins, b_name].values  # protein x behavior beta
        drift += betas * delta

# C. New protein state & risk
new_vals = df_base.values[0] + drift
new_vals = np.clip(new_vals, -5, 5)  # safety clip
df_new = pd.DataFrame([new_vals], columns=proteins)
new_prob = risk_model.predict_proba(df_new)[0, 1]

# D. Risk changes
risk_reduction_abs = base_prob - new_prob
risk_reduction_rel = (base_prob - new_prob) / base_prob if base_prob > 0 else 0.0


# ------------------ 5. KPI row ------------------
st.divider()
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Baseline VTE Risk", f"{base_prob:.1%}")
kpi_col2.metric(
    "Post-intervention Risk",
    f"{new_prob:.1%}",
    delta=f"{-risk_reduction_abs:.1%}",
    delta_color="inverse",
)
kpi_col3.metric("Relative Risk Change", f"{risk_reduction_rel:.1%}")


# ================== 6. Proteins block ==================
st.divider()
st.subheader("🧬 Protein-level Changes")

prot_col1, prot_col2 = st.columns(2)

# 6.1 Protein expression shift (left)
with prot_col1:
    st.markdown("**Protein Expression Shift**")

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(proteins))
    width = 0.35

    baseline_vals_arr = df_base.values[0]
    intervention_vals_arr = new_vals

    ax.bar(
        x - width / 2,
        baseline_vals_arr,
        width,
        label="Baseline",
        color="#BDC3C7",
        edgecolor="#7F8C8D",
    )
    ax.bar(
        x + width / 2,
        intervention_vals_arr,
        width,
        label="Intervention",
        color="#2ECC71",
        edgecolor="#27AE60",
    )

    ax.axhline(0, color="#7F8C8D", linewidth=1, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(proteins)
    ax.set_ylabel("Expression (Z-score)")
    ax.set_xlabel("Proteins")
    ax.set_title("Baseline vs Intervention")

    # Annotate only noticeable changes
    for i, (b, n) in enumerate(zip(baseline_vals_arr, intervention_vals_arr)):
        delta = n - b
        if abs(delta) > 0.05:
            ax.text(
                i + width / 2,
                n if n >= 0 else n,
                f"{delta:+.2f}",
                ha="center",
                va="bottom" if n >= 0 else "top",
                fontsize=8,
            )

    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    st.pyplot(fig)

# 6.2 Key proteins with significant drift (right)
with prot_col2:
    st.markdown("**Key Proteins With Significant Drift**")

    drift_series = pd.Series(drift, index=proteins)
    threshold = 0.01
    top_n = 10
    sig_drift = drift_series[drift_series.abs() > threshold]

    if not sig_drift.empty:
        sig_drift = sig_drift.reindex(
            sig_drift.abs().sort_values(ascending=False).index
        ).iloc[:top_n]

        sig_drift_sorted = sig_drift.sort_values()

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        y = np.arange(len(sig_drift_sorted))
        values = sig_drift_sorted.values

        bars = ax2.barh(
            y,
            values,
            color=["#E74C3C" if v < 0 else "#3498DB" for v in values],
            edgecolor="#2C3E50",
        )

        ax2.axvline(0, color="#7F8C8D", linewidth=1, linestyle="--")

        ax2.set_yticks(y)
        ax2.set_yticklabels(sig_drift_sorted.index)
        ax2.set_xlabel("Protein Drift (Δ Z-score)")

        # robust x-limits
        v_min = float(values.min())
        v_max = float(values.max())
        if v_min == v_max:
            v_min -= 0.01
            v_max += 0.01
        data_range = v_max - v_min
        pad = max(0.1 * data_range, 0.01)
        ax2.set_xlim(v_min - pad, v_max + pad)

        # annotate just outside bar ends
        for bar, v in zip(bars, values):
            bar_y = bar.get_y() + bar.get_height() / 2
            offset = 0.03 * data_range if data_range > 0 else 0.01
            if v >= 0:
                text_x = v + offset
                ha = "left"
            else:
                text_x = v - offset
                ha = "right"

            ax2.text(
                text_x,
                bar_y,
                f"{v:+.3f}",
                va="center",
                ha=ha,
                fontsize=9,
            )

        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        st.write("_No protein shows a substantial shift yet. Try stronger interventions._")


# ================== 7. Behavior & risk block ==================
st.divider()
st.subheader("📈 Behavior & Risk")

beh_col1, beh_col2 = st.columns([1.2, 1])

# 7.1 Intervention summary + behavior magnitude (left)
with beh_col1:
    st.markdown("**Intervention Summary**")
    active_interventions = {k: v for k, v in intervention_deltas.items() if v != 0}
    if active_interventions:
        st.write("Current behavior changes:")
        for k, v in active_interventions.items():
            st.info(f"**{k}**: {v:+.2f} unit")
    else:
        st.warning("Adjust behavior sliders in the sidebar to see their effects.")

    st.markdown("**Behavior Changes (expressed in SD units)**")

    sd_units = []
    beh_labels = []
    for b_name in behavior_names:
        sd = float(behavior_std[b_name])
        if sd > 0:
            sd_units.append(intervention_deltas[b_name] / sd)
        else:
            sd_units.append(0.0)
        beh_labels.append(b_name)

    sd_series = pd.Series(sd_units, index=beh_labels)

    if sd_series.abs().sum() == 0:
        st.write("_No behavior changes yet. Move sliders in the sidebar._")
    else:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        x = np.arange(len(sd_series))
        bars4 = ax4.bar(
            x,
            sd_series.values,
            color=["#E74C3C" if v < 0 else "#3498DB" for v in sd_series.values],
            edgecolor="#2C3E50",
        )
        ax4.axhline(0, color="#7F8C8D", linewidth=1, linestyle="--")
        ax4.set_xticks(x)
        ax4.set_xticklabels(sd_series.index, rotation=20, ha="right")
        ax4.set_ylabel("Change (SD units)")
        ax4.set_title("Magnitude of Behavior Interventions")

        v_min2 = float(sd_series.min())
        v_max2 = float(sd_series.max())
        if v_min2 == v_max2:
            v_min2 -= 0.1
            v_max2 += 0.1
        rng2 = v_max2 - v_min2
        pad2 = max(0.1 * rng2, 0.05)
        ax4.set_ylim(v_min2 - pad2, v_max2 + pad2)

        for bar, v in zip(bars4, sd_series.values):
            y = bar.get_height()
            offset = 0.03 * rng2 if rng2 > 0 else 0.05
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                y + (offset if y >= 0 else -offset),
                f"{v:+.2f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

        fig4.tight_layout()
        st.pyplot(fig4)

# 7.2 Overall risk change (right)
with beh_col2:
    st.markdown("**Baseline vs Post-intervention VTE Risk**")

    fig3, ax3 = plt.subplots(figsize=(4, 4))

    labels = ["Baseline", "Post-intervention"]
    risks = [base_prob, new_prob]
    x_pos = np.arange(len(labels))

    bars3 = ax3.bar(x_pos, risks, color=["#BDC3C7", "#2ECC71"], edgecolor="#7F8C8D")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Predicted Risk (probability)")
    ax3.set_ylim(0, max(risks) * 1.2 if max(risks) > 0 else 0.1)

    for bar, r in zip(bars3, risks):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{r:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax3.set_title("Overall Risk Change")
    fig3.tight_layout()
    st.pyplot(fig3)

