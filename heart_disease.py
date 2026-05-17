# Heart Disease Risk Analytics Dashboard
#This Streamlit app provides an interactive dashboard for analyzing heart disease risk based on a clinical dataset.

#importing necessary libraries
import base64 # For encoding background image to embed in CSS
from pathlib import Path # File path handling
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

#page configuration
st.set_page_config(
    page_title="Heart Disease - Risk Analytics",
    page_icon="assets/heart_icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

#image and dataset path
ROOT = Path(__file__).parent 
CSV_PATH = ROOT / "heart_disease.csv"
BG_PATH = ROOT / "assets" / "dashboard-bg.jpg"

#color palette
C = {
    "bg":        "#0e1a2b",
    "card":      "rgba(22, 32, 48, 0.80)",
    "border":    "rgba(120, 150, 180, 0.25)",
    "text":      "#f1f5f9",
    "muted":     "#9aa6b8",
    "risk":      "#ef6a52",   
    "safe":      "#5fc4cf",   
    "amber":     "#f0b84a",
    "violet":    "#9c7ce8",
    "mint":      "#5fd0a8",
    "male":      "#6aa6e0",
    "female":    "#e08ac0",
}
SERIES = [C["risk"], C["safe"], C["amber"], C["violet"], C["mint"]]

#global css and background image injection (hides Streamlit's default background and adds our own with a dark overlay)
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def inject_css():
    bg_b64 = _b64(BG_PATH) if BG_PATH.exists() else ""

    st.markdown("""
    <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
<style>
/* ---- global background image + dark overlay ---- */
[data-testid="stAppViewContainer"] {{
    background:
      linear-gradient(135deg, rgba(10,18,32,0.90), rgba(12,20,36,0.92)),
      url("data:image/jpeg;base64,{bg_b64}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: {C["text"]};
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1600px; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] > div {{
    background: rgba(15, 23, 38, 0.88);
    backdrop-filter: blur(12px);
    border-right: 1px solid {C["border"]};
}}
[data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}

#icons and badges
.icon {{margin-right: 8px;font-size: 14px;}}

.icon-lg {{font-size: 20px;}}

/* ---- HEADER ---- */
.app-header {{
    background: rgba(18, 26, 42, 0.72);
    backdrop-filter: blur(14px);
    border: 1px solid {C["border"]};
    border-radius: 18px;
    padding: 16px 22px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}}
.app-header .brand {{ display:flex; align-items:center; gap:14px; }}
.app-header .logo {{
    height:46px; width:46px; border-radius:14px;
    background: linear-gradient(135deg,#5fc4cf,#3a6fc4);
    display:flex; align-items:center; justify-content:center;
    font-size:22px; box-shadow: 0 6px 16px rgba(95,196,207,0.35);
}}
.app-header h1 {{ margin:0; font-size:20px; font-weight:700; letter-spacing:-0.3px; }}
.app-header p  {{ margin:0; font-size:12px; color:{C["muted"]}; }}
.badge-pill {{
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 12px; border-radius:999px; font-size:12px; font-weight:600;
}}
.badge-muted  {{ background: rgba(255,255,255,0.08); color:{C["text"]}; border:1px solid {C["border"]}; }}
.badge-risk   {{ background:{C["risk"]}; color:white; }}

/* ---- KPI cards ---- */
.kpi-row {{ display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; margin-bottom:18px; }}
@media (max-width: 900px) {{ .kpi-row {{ grid-template-columns: repeat(2,1fr); }} }}
.kpi {{
    background: {C["card"]};
    backdrop-filter: blur(12px);
    border:1px solid {C["border"]};
    border-radius: 16px; padding: 14px 16px;
}}
.kpi.accent {{
    background: linear-gradient(135deg,#ef6a52,#c84a36);
    border:0; color:white;
}}
.kpi .lbl {{ font-size:12px; color:{C["muted"]}; display:flex; gap:6px; align-items:center; }}
.kpi.accent .lbl {{ color: rgba(255,255,255,0.85); }}
.kpi .val {{ font-size:26px; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums; }}

/* ---- chart cards ---- */
.chart-card {{
    background: {C["card"]};
    backdrop-filter: blur(12px);
    border:1px solid {C["border"]};
    border-radius: 18px;
    padding: 16px 18px 8px 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}}
.chart-card h3 {{ margin:0; font-size:14px; font-weight:700; color:{C["text"]}; }}
.chart-card .sub {{ font-size:11.5px; color:{C["muted"]}; margin-top:2px; }}

/* ---- insight cards ---- */
.insight {{
    border-radius:14px; padding:12px 14px;
    background: rgba(20,28,44,0.55);
    border:1px solid {C["border"]};
    margin-bottom:10px;
}}
.insight .title {{ font-weight:700; font-size:13.5px; }}
.insight .body  {{ font-size:13px; color:#dbe3ee; margin-top:6px; line-height:1.45; }}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["card"]};
    border:1px solid {C["border"]};
    border-radius: 14px; padding: 4px; gap: 4px;
    backdrop-filter: blur(10px);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:10px !important; padding: 8px 14px !important;
    color: {C["muted"]} !important; font-weight:600;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(95,196,207,0.18) !important;
    color: {C["text"]} !important;
}}

/* ---- dataframes ---- */
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

/* hide default streamlit chrome */
#MainMenu, footer {{ visibility: hidden; }}

</style>
""",
        unsafe_allow_html=True,
    )

inject_css()

#loading dataset
@st.cache_data
def load_data() -> pd.DataFrame:
    # Prefer the cleaned CSV if present, otherwise fall back to bundled JSON
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        if "gender" in df.columns and "male" not in df.columns:
            df = df.rename(columns={"gender": "male"})
        df = df.dropna()
        for c in ["age","currentSmoker","cigsPerDay","BPMeds","diabetes",
                  "totChol","heartRate","glucose","Risk","male"]:
            if c in df.columns:
                df[c] = df[c].astype(int)
        return df.reset_index(drop=True)

df = load_data()

# plotly theme configuration
def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], size=12, family="Inter, system-ui, sans-serif"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11, color=C["text"])),
        xaxis=dict(gridcolor="rgba(150,170,200,0.18)", linecolor="rgba(150,170,200,0.25)", tickfont=dict(color=C["muted"])),
        yaxis=dict(gridcolor="rgba(150,170,200,0.18)", linecolor="rgba(150,170,200,0.25)", tickfont=dict(color=C["muted"])),
        hoverlabel=dict(bgcolor="#162033", bordercolor=C["border"], font_color="white"),
    )
    return fig

def card_open(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="chart-card"><h3>{title}</h3>'
        + (f'<div class="sub">{subtitle}</div>' if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# dynamic sidebar filters for dataset
with st.sidebar:
    st.markdown("""<h3><i class="bi bi-funnel-fill"></i>   Clinical Filters</h3>""", unsafe_allow_html=True)
    sex_f = st.selectbox("Gender", ["All", "Male", "Female"], index=0)
    age_range = st.slider("Age range", 32, 70, (32, 70), 1)
    smoker_f = st.radio("Smoker", ["All", "Yes", "No"], horizontal=True, index=0)
    diabetes_f = st.radio("Diabetes", ["All", "Yes", "No"], horizontal=True, index=0)
    bp_f = st.radio("BP Meds", ["All", "Yes", "No"], horizontal=True, index=0)
    st.markdown("---")
    st.caption("CardioVascularInsight · 10-year coronary heart disease risk")

# function to apply sidebar filters to the dataset
def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if sex_f == "Male":   out = out[out.male == 1]
    if sex_f == "Female": out = out[out.male == 0]
    out = out[(out.age >= age_range[0]) & (out.age <= age_range[1])]
    if smoker_f   == "Yes": out = out[out.currentSmoker == 1]
    if smoker_f   == "No":  out = out[out.currentSmoker == 0]
    if diabetes_f == "Yes": out = out[out.diabetes == 1]
    if diabetes_f == "No":  out = out[out.diabetes == 0]
    if bp_f       == "Yes": out = out[out.BPMeds == 1]
    if bp_f       == "No":  out = out[out.BPMeds == 0]
    return out

# apply filters and compute KPIs
data = apply_filters(df)
n = max(len(data), 1)
at_risk = int((data.Risk == 1).sum())
risk_pct = at_risk / n * 100
avg_age  = data.age.mean() if len(data) else 0
smokers_pct = (data.currentSmoker.sum() / n * 100) if len(data) else 0
avg_bp = data.sysBP.mean() if len(data) else 0

#header with dynamic patient count and risk percentage
st.markdown(
    f"""
<div class="app-header">
  <div class="brand">
    <div class="logo"><i class="bi bi-heart-pulse-fill icon-lg"></i></div>
    <div>
      <h1>Heart Disease - Risk Analytics</h1>
      <p>10-year coronary heart disease risk </p>
    </div>
  </div>
  <div style="display:flex; gap:8px;">
    <span class="badge-pill badge-muted"><i class="bi bi-people-fill"></i> {len(data):,} patients</span>
    <span class="badge-pill badge-risk"><i class="bi bi-heart-pulse-fill"></i> {risk_pct:.1f}% at risk</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

#KPI row with key metrics that update based on filters
st.markdown(
    f"""
<div class="kpi-row">
  <div class="kpi"><div class="lbl"><i class="bi bi-people-fill"></i> Patients</div><div class="val">{len(data):,}</div></div>
  <div class="kpi accent"><div class="lbl"><i class="bi bi-heart-pulse-fill"></i> At-risk %</div><div class="val">{risk_pct:.1f}%</div></div>
  <div class="kpi"><div class="lbl"><i class="bi bi-graph-up"></i> Avg Age</div><div class="val">{avg_age:.1f}</div></div>
  <div class="kpi"><div class="lbl"><i class="bi bi-lungs-fill"></i> Smokers</div><div class="val">{smokers_pct:.1f}%</div></div>
  <div class="kpi"><div class="lbl"><i class="bi bi-droplet"></i> Avg sysBP</div><div class="val">{avg_bp:.0f}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

#tiny helper function to compute percentages for various filters and groupings
def pct(d, mask):
    sub = d[mask]
    return 0 if len(sub) == 0 else (sub.Risk == 1).mean() * 100

def age_group(a):
    return "<40" if a < 40 else "40–49" if a < 50 else "50–59" if a < 60 else "60–69" if a < 70 else "70+"

def bmi_band(b):
    return "Underweight" if b < 18.5 else "Normal" if b < 25 else "Overweight" if b < 30 else "Obese"

#tabs section
tabs = st.tabs(["Overview", "Demographics", "Clinical Vitals", "Risk Drivers", "Patient Explorer", "Findings Report"])

#overview section
with tabs[0]:
    col1, col2, col3 = st.columns(3)
#risk distribution pie chart
    with col1:
        card_open("Risk Distribution", "10-year Heart Disease outcome")
        pie_df = pd.DataFrame({"name": ["At Risk", "Not at Risk"], "value": [at_risk, len(data) - at_risk]})
        fig = px.pie(pie_df, names="name", values="value", hole=0.55, color="name",
                     color_discrete_map={"At Risk": C["risk"], "Not at Risk": C["safe"]})
        fig.update_traces(textfont_color="white")
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
#risk by gender
    with col2:
        card_open("Risk by Gender", "% of patients flagged at-risk in each sex")
        gd = pd.DataFrame([
            {"sex": "Female", "Risk %": pct(data, data.male == 0), "n": int((data.male == 0).sum())},
            {"sex": "Male",   "Risk %": pct(data, data.male == 1), "n": int((data.male == 1).sum())},
        ])
        fig = px.bar(gd, x="sex", y="Risk %", color="sex", text=gd["Risk %"].round(1),
                     color_discrete_map={"Male": C["male"], "Female": C["female"]})
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(showlegend=False, yaxis_ticksuffix="%")
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
#risk by age group
    with col3:
        card_open("Risk Rate by Age", "% of age group classified at-risk")
        groups = ["<40", "40–49", "50–59", "60–69", "70+"]
        adata = pd.DataFrame([{"group": g, "pct": pct(data, data.age.apply(age_group) == g)} for g in groups])
        fig = px.area(adata, x="group", y="pct", markers=True, color_discrete_sequence=[C["risk"]])
        fig.update_traces(line=dict(width=2.5), fillcolor="rgba(239,106,82,0.25)")
        fig.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
#risk by age group stacked counts and risk by clinical factor
    col4, col5 = st.columns([1, 2])
    with col4:
        card_open("Risk by Age Group", "Stacked patient counts")
        groups = ["<40", "40–49", "50–59", "60–69", "70+"]
        rows = []
        for g in groups:
            sub = data[data.age.apply(age_group) == g]
            rows.append({"group": g, "At Risk": int((sub.Risk == 1).sum()), "Not at Risk": int((sub.Risk == 0).sum())})
        sd = pd.DataFrame(rows).melt(id_vars="group", var_name="status", value_name="count")
        fig = px.bar(sd, x="group", y="count", color="status", barmode="stack",
                     color_discrete_map={"At Risk": C["risk"], "Not at Risk": C["safe"]})
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    with col5:
        card_open("Risk % by Clinical Factor", "Among patients with each condition")
        factors = pd.DataFrame([
            {"name": "Smokers",      "risk": pct(data, data.currentSmoker == 1)},
            {"name": "Diabetes",     "risk": pct(data, data.diabetes == 1)},
            {"name": "BP Meds",      "risk": pct(data, data.BPMeds == 1)},
            {"name": "Hypertensive", "risk": pct(data, data.sysBP >= 140)},
            {"name": "Obese",        "risk": pct(data, data.BMI >= 30)},
            {"name": "High Chol",    "risk": pct(data, data.totChol >= 240)},
        ]).sort_values("risk")
        fig = px.bar(factors, x="risk", y="name", orientation="h",
                     color_discrete_sequence=[C["amber"]], text=factors["risk"].round(1))
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_ticksuffix="%")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    #top significant variables
    card_open("Most Significant Risk Variables", "Strength of statistical link to heart disease risk")
    features = [("age","Age"),("sysBP","Systolic BP"),("diaBP","Diastolic BP"),
                ("totChol","Cholesterol"),("BMI","BMI"),("glucose","Glucose"),
                ("cigsPerDay","Cigs/day"),("heartRate","Heart rate")]
    corrs = []
    for k, lbl in features:
        if data[k].std() == 0: r = 0
        else: r = float(np.corrcoef(data[k], data.Risk)[0,1])
        corrs.append({"feature": lbl, "r": r, "abs": abs(r)})
    top = pd.DataFrame(corrs).sort_values("abs", ascending=False).head(5)
    cols = st.columns(5)
    for i,(_, row) in enumerate(top.iterrows()):
        with cols[i]:
            direction = "↑ raises risk" if row.r >= 0 else "↓ lowers risk"
            st.markdown(
                f"""<div class="insight">
                <div style="display:flex;justify-content:space-between;font-size:11px;color:{C["muted"]}">
                <span>#{i+1}</span><span>{direction}</span></div>
                <div class="title" style="margin-top:4px">{row.feature}</div>
                <div style="font-size:11px;color:{C["muted"]}">corr = {row.r:.3f}</div>
                <div style="height:6px;background:rgba(255,255,255,0.1);border-radius:3px;margin-top:8px;overflow:hidden">
                  <div style="height:100%;width:{min(100, abs(row.r)*400):.0f}%;background:{SERIES[i%5]}"></div>
                </div></div>""", unsafe_allow_html=True)


#demographics section
with tabs[1]:
    c1, c2, c3 = st.columns(3)
#age distribution by gender
    with c1:
        card_open("Age Distribution by Sex", "Distribution of patients by age and sex")
        bins = list(range(30, 80, 5))
        d = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            d.append({"bin": f"{lo}-{hi-1}",
                      "Male":   int(((data.age>=lo)&(data.age<hi)&(data.male==1)).sum()),
                      "Female": int(((data.age>=lo)&(data.age<hi)&(data.male==0)).sum())})
        d = pd.DataFrame(d).melt(id_vars="bin", var_name="sex", value_name="count")
        fig = px.bar(d, x="bin", y="count", color="sex", barmode="group",
                     color_discrete_map={"Male": C["male"], "Female": C["female"]})
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
#gender distribution 
    with c2:
        card_open("Gender Composition", "Overall distribution of patients by gender")
        sd = pd.DataFrame({"name": ["Male", "Female"],
                           "value": [int((data.male==1).sum()), int((data.male==0).sum())]})
        fig = px.pie(sd, names="name", values="value", color="name",
                     color_discrete_map={"Male": C["male"], "Female": C["female"]})
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
#cigarettes per day by risk status
    with c3:
        card_open("Avg Cigarettes/Day", "Per risk group")
        cd = pd.DataFrame({
            "group": ["Not at Risk", "At Risk"],
            "avg":   [data[data.Risk==0].cigsPerDay.mean(), data[data.Risk==1].cigsPerDay.mean()],
        })
        fig = px.bar(cd, x="group", y="avg", color_discrete_sequence=[C["amber"]],
                     text=cd["avg"].round(2))
        fig.update_traces(textposition="outside")
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
#risk by BMI band and heart rate distribution by risk status
    c4, c5 = st.columns([1, 2])
#risk by BMI band
    with c4:
        card_open("Risk by BMI band", "% of patients flagged at-risk in each BMI band")
        bands = ["Underweight", "Normal", "Overweight", "Obese"]
        rows = []
        for b in bands:
            sub = data[data.BMI.apply(bmi_band) == b]
            rows.append({"band": b, "At Risk": int((sub.Risk==1).sum()), "Not at Risk": int((sub.Risk==0).sum())})
        bd = pd.DataFrame(rows).melt(id_vars="band", var_name="status", value_name="count")
        fig = px.bar(bd, x="band", y="count", color="status", barmode="stack",
                     color_discrete_map={"At Risk": C["risk"], "Not at Risk": C["safe"]})
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
#heart rate distribution by risk status
    with c5:
        card_open("Heart Rate Distribution", "By risk status")
        buckets = [50, 60, 70, 80, 90, 100, 110, 130]
        rows = []
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            rows.append({
                "bin": f"{lo}-{hi}",
                "At Risk":     int(((data.heartRate>=lo)&(data.heartRate<hi)&(data.Risk==1)).sum()),
                "Not at Risk": int(((data.heartRate>=lo)&(data.heartRate<hi)&(data.Risk==0)).sum()),
            })
        hd = pd.DataFrame(rows).melt(id_vars="bin", var_name="status", value_name="count")
        fig = px.line(hd, x="bin", y="count", color="status", markers=True,
                      color_discrete_map={"At Risk": C["risk"], "Not at Risk": C["safe"]})
        fig.update_traces(line=dict(width=2.5))
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

#clinical vital section
with tabs[2]:
    v1, v2 = st.columns(2)
#blood pressure scatter plot
    with v1:
        card_open("Blood Pressure (sys vs dia)", "sample of up to 800 patients")
        sample = data.sample(min(800, len(data)), random_state=1) if len(data) else data
        sample = sample.assign(risk_lbl=sample.Risk.map({0: "Not at Risk", 1: "At Risk"}))
        fig = px.scatter(sample, x="sysBP", y="diaBP", color="risk_lbl",
                         color_discrete_map={"At Risk": C["risk"], "Not at Risk": C["safe"]},
                         opacity=0.65)
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
#cholesterol bands and risk bar chart
    with v2:
        card_open("Cholesterol Band", "Risk % by cholesterol bands")
        bands = [("<200",0,200),("200-239",200,240),("240-279",240,280),("280+",280,1e9)]
        cd = pd.DataFrame([{"band":b,"risk":pct(data,(data.totChol>=lo)&(data.totChol<hi))} for b,lo,hi in bands])
        fig = px.bar(cd, x="band", y="risk", color_discrete_sequence=[C["amber"]],
                     text=cd["risk"].round(1))
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    v3, v4 = st.columns(2)
#average vitals bar chart and glucose vs age scatter plot
    with v3:
        card_open("Average Vitals", "Comparison of key vitals by risk status")
        metrics = ["sysBP", "diaBP", "totChol", "glucose", "heartRate", "BMI"]
        rows = []
        for m in metrics:
            rows.append({"metric": m,
                         "At Risk":     data[data.Risk==1][m].mean(),
                         "Not at Risk": data[data.Risk==0][m].mean()})
        md = pd.DataFrame(rows).melt(id_vars="metric", var_name="status", value_name="value")
        fig = px.bar(md, x="metric", y="value", color="status", barmode="group",
                     color_discrete_map={"At Risk": C["risk"], "Not at Risk": C["safe"]})
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
#glucose vs age scatter plot
    with v4:
        card_open("Glucose vs Age", "Diabetes overlay")
        sample = data.sample(min(700, len(data)), random_state=2) if len(data) else data
        sample = sample.assign(d_lbl=sample.diabetes.map({0: "Non-Diabetic", 1: "Diabetic"}))
        fig = px.scatter(sample, x="age", y="glucose", color="d_lbl",
                         color_discrete_map={"Diabetic": C["risk"], "Non-Diabetic": C["safe"]},
                         opacity=0.65)
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

#risk driver section
with tabs[3]:
    features = ["age","sysBP","diaBP","totChol","BMI","heartRate","glucose","cigsPerDay"]
    corrs = []
    for f in features:
        r = 0 if data[f].std() == 0 else float(np.corrcoef(data[f], data.Risk)[0,1])
        corrs.append({"feature": f, "value": round(r,3)})
    cdf = pd.DataFrame(corrs).sort_values("value", key=lambda s: s.abs(), ascending=True)
#feature correlation bar chart
    card_open("Feature Risk Correlation", "Correlation coefficient between each feature and heart disease risk")
    fig = px.bar(cdf, x="value", y="feature", orientation="h",
                 color=cdf["value"].apply(lambda v: "+" if v >= 0 else "-"),
                 color_discrete_map={"+": C["risk"], "-": C["safe"]},
                 text=cdf["value"])
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis=dict(range=[-0.3, 0.3]))
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

#this identifies the top 4 features most strongly correlated with risk and 
#creates area charts showing how risk percentage changes across the distribution of each feature,
#binned into quintiles. It provides a visual representation of how risk varies with different levels of each key feature.
    top4 = cdf.reindex(cdf["value"].abs().sort_values(ascending=False).index).head(4)
    cols = st.columns(2)
    for i, (_, row) in enumerate(top4.iterrows()):
        f = row.feature
        vals = np.sort(data[f].values)
        if len(vals) < 5: continue
        cuts = [vals[0], vals[int(len(vals)*0.2)], vals[int(len(vals)*0.4)],
                vals[int(len(vals)*0.6)], vals[int(len(vals)*0.8)], vals[-1]]
        series = []
        for j in range(5):
            lo, hi = cuts[j], cuts[j+1]
            mask = (data[f] >= lo) & (data[f] < hi + (1 if j==4 else 0))
            sub = data[mask]
            series.append({"bin": f"{int(lo)}-{int(hi)}", "pct": 0 if len(sub)==0 else (sub.Risk==1).mean()*100})
        sdf = pd.DataFrame(series)
        with cols[i % 2]:
            card_open(f"Risk % by {f}", "Quintile bins")
            fig = px.area(sdf, x="bin", y="pct", markers=True,
                          color_discrete_sequence=[SERIES[i % 5]])
            fig.update_traces(line=dict(width=2.5))
            fig.update_layout(yaxis_ticksuffix="%")
            st.plotly_chart(style_fig(fig, 280), use_container_width=True)

#patient explore section
with tabs[4]:
    card_open("Patient Explorer", f"{len(data):,} patients matching current filters")
    r_filter = st.radio("Filter Displays", ["All", "At Risk only", "Not at Risk only"], horizontal=True)
    view = data.copy()
    if r_filter == "At Risk only":     view = view[view.Risk == 1]
    if r_filter == "Not at Risk only": view = view[view.Risk == 0]
    view = view.rename(columns={"male": "Gender", "Risk": "At Risk"})
    st.dataframe(view, use_container_width=True, height=420)
    csv = view.to_csv(index=False).encode()
    st.download_button("Export CSV", csv, "heart_patients.csv", "text/csv")

    with st.expander("Clinical Variable Glossary"):
        glossary = [
            ("Gender", "Biological sex (Male / Female)."),
            ("Age", "Age in years. Strongest non-modifiable risk factor."),
            ("currentSmoker", "Whether the patient currently smokes (Yes/No)."),
            ("cigsPerDay", "Cigarettes smoked per day."),
            ("BPMeds", "On blood-pressure medication (Yes/No)."),
            ("totChol", "Total cholesterol (mg/dL). ≥240 is high."),
            ("sysBP", "Systolic blood pressure (mmHg). ≥140 = hypertension."),
            ("diaBP", "Diastolic blood pressure (mmHg). ≥90 = hypertension."),
            ("BMI", "Body Mass Index. ≥30 = obese."),
            ("heartRate", "Resting heart rate (bpm)."),
            ("diabetes", "Diagnosed with diabetes."),
            ("glucose", "Fasting blood glucose (mg/dL)."),
            ("Risk", "TARGET 1 = predicted heart disease within 10 years."),
        ]
        for k, v in glossary:
            st.markdown(f"**{k}** : {v}")

#report section
with tabs[5]:
    st.markdown("Conclusive Findings Report")
    st.caption("Summary of key insights derived from the data, highlighting the most significant risk factors and their implications for patient care.")

    findings = [
        ('<i class="bi bi-person-fill"></i> Age is the dominant driver',
         f"Risk grows almost linearly with age. Patients under 40 sit around {pct(df, df.age<40):.0f}%, while those 70+ jump to about {pct(df, df.age>=70):.0f}%. Age alone explains a large share of the variance."),
        ('<i class="bi bi-heart-pulse-fill"></i> Blood pressure separates the cohort',
         f"Hypertensive patients (sysBP ≥ 140) carry {pct(df, df.sysBP>=140):.0f}% risk versus {pct(df, df.sysBP<140):.0f}% for the rest nearly a 2× gap."),
        ('<i class="bi bi-gender-male"></i> Men are more affected than women',
         f"In this cohort {pct(df, df.male==1):.0f}% of men develop CHD vs {pct(df, df.male==0):.0f}% of women. Sex is non-modifiable but useful for triage."),
        ('<i class="bi bi-lungs-fill"></i> Smoking compounds every other factor',
         f"{pct(df, df.currentSmoker==1):.0f}% of smokers are flagged at risk. Heavy smokers (≥20 cigs/day) show the steepest jump."),
        ('<i class="bi bi-droplet-half"></i> Diabetes & glucose are amplifiers',
         f"Diabetic patients reach {pct(df, df.diabetes==1):.0f}% risk. Elevated glucose tracks with diabetes and reinforces the signal."),
        ('<i class="bi bi-person-bounding-box"></i> Weight matters, but less than BP and age',
         f"Obese patients show {pct(df, df.BMI>=30):.0f}% risk vs {pct(df, df.BMI<25):.0f}% for normal weight meaningful, but secondary to BP/age."),
    ]
    cols = st.columns(2)
    for i,(title, body) in enumerate(findings):
        with cols[i % 2]:
            st.markdown(f'<div class="insight"><div class="title">{title}</div><div class="body">{body}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"""
<div class="chart-card" style="background:linear-gradient(135deg,rgba(239,106,82,0.18),rgba(95,196,207,0.12));">
<h3><i class="bi bi-bullseye"></i> Clinical Summary</h3>
<p style="font-size:14px; line-height:1.55; margin-top:10px; color:#e6ecf5;">
-The three biggest levers for a 10-year coronary heart disease risk in this dataset are
<b>age</b>, <b>systolic blood pressure</b> and <b>sex</b>. Smoking, diabetes and obesity
amplify the risk further.<br>
-A patient who is older, male, hypertensive, smokes and is diabetic
sits in the <b>highest-risk quadrant</b>, while a young, non-smoking, non-hypertensive
woman with healthy BMI is in the <b>safest quadrant</b>.<br>
<div style="font-size:12px; color:#a0aec0; justify-content:center; display:flex; gap:6px; margin-top:8px;">
The dashboard above lets you filter and explore exactly where any sub-population falls on this spectrum.
</div>
</p>
</div>
""",
        unsafe_allow_html=True,
    )
