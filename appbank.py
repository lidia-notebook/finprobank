# app.py — Bank Marketing Prediction & Segmentation (static thresholds, 6 job groups, 7 clusters)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

try:
    import joblib
except Exception:
    joblib = None

# ---------------- Page ----------------
st.set_page_config(page_title="Bank Marketing Prediction & Segmentation", page_icon="📈", layout="wide")
st.title("📈 Bank Marketing Prediction & Segmentation")
st.caption("Uses your artifacts if present in `artifacts/`; otherwise runs a lightweight demo model.")

# --------- Static Thresholds (hidden UI) ---------
YES_THR = 0.60
NO_THR  = 0.40

# --------- Schema & Job Grouping ----------
REQUIRED_COLS = ["job_group", "house_loan", "loan", "age", "pdays"]

#  Expanded job groups
JOB_GROUP_OPTIONS = [
    "White-Collar Professionals",
    "Skilled Workers",
    "Business Owners",
    "Retirees",
    "Student",
    "Unemployed/Unknown",
]
YN_OPTIONS = ["Yes", "No"]

# Raw job → grouped job mapping
RAW_JOB_TO_GROUP = {
    # Skilled Workers
    "blue-collar": "Skilled Workers",
    "technician": "Skilled Workers",
    "housemaid": "Skilled Workers",
    # White-Collar
    "admin.": "White-Collar Professionals",
    "services": "White-Collar Professionals",
    "management": "White-Collar Professionals",
    # Business Owners
    "entrepreneur": "Business Owners",
    "self-employed": "Business Owners",
    # Retirees / Student / Unemployed
    "retired": "Retirees",
    "student": "Student",
    "unemployed": "Unemployed/Unknown",
    "unknown": "Unemployed/Unknown",
    "other": "Unemployed/Unknown",
}
def map_job_to_group(s: pd.Series) -> pd.Series:
    s_lower = s.astype(str).str.strip().str.lower()
    mapped = s_lower.map(RAW_JOB_TO_GROUP)
    return mapped.fillna("Unemployed/Unknown")

def sample_dataframe(n=200, seed=7):
    rng = np.random.default_rng(seed)
    jobs = rng.choice(JOB_GROUP_OPTIONS, size=n, p=[0.35, 0.25, 0.08, 0.07, 0.15, 0.10])
    house = rng.choice(YN_OPTIONS, size=n, p=[0.35, 0.65])
    loan  = rng.choice(YN_OPTIONS, size=n, p=[0.25, 0.75])
    age   = rng.integers(18, 85, size=n)
    pdays = rng.integers(-1, 400, size=n)  # -1 = never contacted
    return pd.DataFrame({"job_group": jobs, "house_loan": house, "loan": loan, "age": age, "pdays": pdays})

# --------- Segment Cards (7 clusters) ----------
segment_cards = pd.DataFrame({
    "Cluster": [0, 1, 2, 3, 4, 5, 6],
    "Segment_Name": [
        "Balanced Borrowers", "Active Responders", "High-Value Professionals",
        "Skeptical Traditionalists", "Steady Planners", "Risk-Averse Families",
        "Emerging Young Savers"
    ],
    "Key_Characteristics": [
        "Middle-aged, stable employment, moderate income, consistent banking activity.",
        "Frequent marketing engagement, responsive to calls and follow-ups, open to offers.",
        "Higher education, strong financial literacy, high employment and deposit potential.",
        "Older demographic, prefers traditional contact (branch/phone), low responsiveness to new offers.",
        "Financially disciplined, low campaign fatigue, long-term mindset.",
        "Family-oriented, cautious decision-makers, high housing-loan presence, risk-averse.",
        "Young, digitally connected, growing financial interest, early career stage."
    ],
    "Value_Proposition": [
        "Reinforce loyalty with exclusive interest rates and reward programs.",
        "Offer time-limited deposit promotions and personalized benefits.",
        "Provide premium deposit plans, investment bundles, and advisory support.",
        "Focus on reliability and security of deposits through clear education.",
        "Promote medium-term savings products with predictable returns.",
        "Emphasize family security, children’s education savings, and long-term safety.",
        "Highlight ease, flexibility, and mobile-first deposit solutions."
    ],
    "Communication_Strategy": [
        "Consistent relationship building; leverage past positive interactions.",
        "Use persuasive, time-sensitive messaging with personalized follow-ups.",
        "Data-driven personalization with professional tone and trust emphasis.",
        "Educational, slow-paced communication via familiar channels.",
        "Informative, goal-oriented communication focusing on future stability.",
        "Warm, emotionally resonant storytelling about family security.",
        "Energetic, reward-driven, digital storytelling focused on convenience."
    ],
    "Recommended_Channels": [
        "Email, SMS, periodic relationship calls.",
        "Phone calls, WhatsApp, email reminders.",
        "Advisory sessions, webinars, online banking notifications.",
        "Branch visits, phone, physical brochures.",
        "Email newsletters, SMS notifications, website banners.",
        "Direct mail, family-focused campaigns, in-branch posters.",
        "Mobile app notifications, social media, influencer-style content."
    ],
    "Campaign_Cadence": [
        "Monthly nurturing with loyalty focus.",
        "Bi-weekly promotional pushes with follow-ups.",
        "Quarterly investment advisory campaigns.",
        "Quarterly educational outreach; low frequency.",
        "Monthly steady updates and reminders.",
        "Seasonal campaigns linked to family milestones.",
        "Frequent digital touchpoints (weekly–biweekly)."
    ],
    "Business_Priority": ["Medium", "High", "High", "Low", "Medium", "Medium", "High"]
})
SEGMENT_MAP = segment_cards.set_index("Cluster").to_dict(orient="index")
N_SEGMENTS = 7

# --------- Artifacts (silent) ----------
ART_DIR = Path("artifacts"); ART_DIR.mkdir(parents=True, exist_ok=True)
CLF_PREPROC_PATH = ART_DIR / "clf_preprocessor.joblib"
CLF_MODEL_PATH   = ART_DIR / "clf_model.joblib"
SEG_PREPROC_PATH = ART_DIR / "seg_preprocessor.joblib"
SEG_KMEANS_PATH  = ART_DIR / "seg_kmeans.joblib"

def load_joblib_or_none(path: Path):
    if joblib is None: return None
    if path.exists():
        try: return joblib.load(path)
        except Exception: return None
    return None

# --------- Demo pipelines (fallback) ----------
from sklearn.compose import ColumnTransformer
def build_demo_clf_pipeline(train_df=None, seed=42):
    if train_df is None: train_df = sample_dataframe(600, seed)
    X = train_df.copy()
    score = (
        (X["age"] > 35).astype(int)
        + (X["house_loan"] == "Yes").astype(int)
        + ((X["pdays"] > 0) & (X["pdays"] < 150)).astype(int)
        - (X["loan"] == "Yes").astype(int)
        + (X["job_group"].isin(["White-Collar Professionals","Business Owners"])).astype(int)
    )
    p = 1/(1+np.exp(-(score-1)))
    y = (p >= 0.55).astype(int)

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["job_group","house_loan","loan"]),
        ("num", StandardScaler(), ["age","pdays"]),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("preprocess", preproc), ("model", clf)])
    pipe.fit(X, y)
    return pipe.named_steps["preprocess"], pipe.named_steps["model"]

def build_demo_seg_pipeline(train_df=None, seed=24, n_clusters=N_SEGMENTS):
    if train_df is None: train_df = sample_dataframe(800, seed)
    seg_preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["job_group","house_loan","loan"]),
        ("num", StandardScaler(), ["age","pdays"]),
    ])
    Z = seg_preproc.fit_transform(train_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    kmeans.fit(Z)
    return seg_preproc, kmeans

# Init (silent)
clf_preproc = load_joblib_or_none(CLF_PREPROC_PATH); clf_model = load_joblib_or_none(CLF_MODEL_PATH)
if clf_preproc is None or clf_model is None: clf_preproc, clf_model = build_demo_clf_pipeline()
seg_preproc = load_joblib_or_none(SEG_PREPROC_PATH); seg_kmeans = load_joblib_or_none(SEG_KMEANS_PATH)
if seg_preproc is None or seg_kmeans is None: seg_preproc, seg_kmeans = build_demo_seg_pipeline()

# --------- Helpers ----------
def predict_label(prob):
    if prob >= YES_THR: return "Yes"
    if prob <= NO_THR:  return "No"
    return "Not Sure"

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={c: c.strip().lower() for c in df.columns}, inplace=True)
    if "job_group" not in df.columns and "job" in df.columns:
        df["job_group"] = map_job_to_group(df["job"])
    for a, t in {"housing":"house_loan","house":"house_loan","home_loan":"house_loan",
                 "personal_loan":"loan","days_since_contact":"pdays"}.items():
        if a in df.columns and t not in df.columns: df[t] = df[a]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}. Expected {REQUIRED_COLS}")
    # Normalize values
    df["job_group"] = df["job_group"].astype(str).str.strip()
    df["job_group"] = np.where(df["job_group"].isin(JOB_GROUP_OPTIONS),
                               df["job_group"], map_job_to_group(df["job_group"]))
    df["house_loan"] = df["house_loan"].astype(str).str.strip().str.title().replace(
        {"True":"Yes","False":"No","Y":"Yes","N":"No","1":"Yes","0":"No"})
    df["loan"] = df["loan"].astype(str).str.strip().str.title().replace(
        {"True":"Yes","False":"No","Y":"Yes","N":"No","1":"Yes","0":"No"})
    df["age"]   = pd.to_numeric(df["age"], errors="coerce").clip(17, 99)
    df["pdays"] = pd.to_numeric(df["pdays"], errors="coerce").fillna(-1).clip(-1, 9999)
    return df[REQUIRED_COLS]

def run_prediction(df: pd.DataFrame):
    pipe = Pipeline([("preprocess", clf_preproc), ("model", clf_model)])
    probs = pipe.predict_proba(df)[:, 1]
    labels = [predict_label(p) for p in probs]
    out = df.copy()
    out["proba_yes"] = probs
    out["proba_yes_pct"] = (probs * 100).round(1)
    out["prediction"] = labels
    return out

def run_clustering(df: pd.DataFrame):
    Z = seg_preproc.transform(df)
    clusters = seg_kmeans.predict(Z)
    out = df.copy(); out["cluster"] = clusters
    return out

def enrich_with_profiles(df_with_cluster: pd.DataFrame):
    rows = []
    for _, r in df_with_cluster.iterrows():
        c = int(r["cluster"]); prof = SEGMENT_MAP.get(c, None)
        if prof is None:
            prof = {"Segment_Name":"Unknown Segment","Key_Characteristics":"","Value_Proposition":"",
                    "Communication_Strategy":"","Recommended_Channels":"","Campaign_Cadence":"","Business_Priority":""}
        rows.append(prof)
    return pd.concat([df_with_cluster.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def pct(n, d): return 0.0 if d == 0 else round(n/d*100, 1)

# ---------------- Tabs ----------------
tab_pred, tab_cluster, tab_seg = st.tabs(["🔮 Prediction", "🧩 Clustering", "📦 Segmentation"])

# ---- Prediction (narrative) ----
with tab_pred:
    st.subheader("Single Prediction")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        job_group = st.selectbox("Job Group", JOB_GROUP_OPTIONS, index=0)
        age = st.number_input("Age", 17, 99, 33, 1)
    with c2:
        house_loan = st.selectbox("House loan", YN_OPTIONS, 1)
        pdays = st.number_input("Days since last contact (pdays, -1 if never)", -1, 9999, -1, 1)
    with c3:
        loan = st.selectbox("Personal loan", YN_OPTIONS, 1)
        st.write(""); go_btn = st.button("Predict", use_container_width=True)

    if go_btn:
        df_in = ensure_schema(pd.DataFrame([{
            "job_group": job_group, "house_loan": house_loan, "loan": loan, "age": age, "pdays": pdays
        }]))
        out = run_prediction(df_in)
        lbl, proba_pct = out.loc[0, "prediction"], out.loc[0, "proba_yes_pct"]
        st.success(f"Prediction: **{lbl}**")
        st.metric("Probability of 'Yes'", f"{proba_pct:.1f}%")
        st.markdown(
            f"- **Profile**: {job_group}, Age {age}, House loan **{house_loan}**, Personal loan **{loan}**, pdays **{pdays}**.\n"
            f"- **Rule**: ≥{int(YES_THR*100)}% → **Yes**, ≤{int(NO_THR*100)}% → **No**, else **Not Sure**.\n"
            f"- **Interpretation**: Likelihood to subscribe is **{proba_pct:.1f}%**, so label is **{lbl}**."
        )

# ---- Clustering (single, narrative) ----
with tab_cluster:
    st.subheader("Single Clustering & Persona")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        job_group_s = st.selectbox("(Clus) Job Group", JOB_GROUP_OPTIONS, 0)
        age_s = st.number_input("(Clus) Age", 17, 99, 33, 1, key="age_seg_single")
    with c2:
        house_loan_s = st.selectbox("(Clus) House loan", YN_OPTIONS, 1, key="house_seg_single")
        pdays_s = st.number_input("(Clus) pdays (-1 if never)", -1, 9999, -1, 1, key="pdays_seg_single")
    with c3:
        loan_s = st.selectbox("(Clus) Personal loan", YN_OPTIONS, 1, key="loan_seg_single")
        st.write(""); seg_btn = st.button("Assign Cluster", use_container_width=True)

    if seg_btn:
        df_in_s = ensure_schema(pd.DataFrame([{
            "job_group": job_group_s, "house_loan": house_loan_s, "loan": loan_s, "age": age_s, "pdays": pdays_s
        }]))
        seg_out = enrich_with_profiles(run_clustering(df_in_s))
        c = int(seg_out.loc[0,"cluster"]); info = SEGMENT_MAP.get(c,{})
        st.success(f"This client belongs to **Cluster {c} — {info.get('Segment_Name','')}**  •  **Priority: {info.get('Business_Priority','')}**")
        st.markdown(
            f"**Who they are**: {info.get('Key_Characteristics','')}\n\n"
            f"**What to offer**: {info.get('Value_Proposition','')}\n\n"
            f"**How to speak**: {info.get('Communication_Strategy','')}\n\n"
            f"**Best channels**: {info.get('Recommended_Channels','')}\n\n"
            f"**Cadence**: {info.get('Campaign_Cadence','')}"
        )

# ---- Segmentation (batch) ----
with tab_seg:
    st.subheader("Segmentation — Upload or Use Dummy Data")
    src = st.radio("Choose data source:", ["Upload CSV", "Use Dummy Data (200 rows)"], horizontal=True)
    df_seg_batch = None
    if src == "Upload CSV":
        up = st.file_uploader("Upload CSV for Segmentation", type=["csv"])
        if up is not None:
            try:
                df_seg_batch = ensure_schema(pd.read_csv(up))
            except Exception as e:
                st.error(f"Upload error: {e}"); df_seg_batch=None
    else:
        if st.button("Generate Dummy Data", use_container_width=True):
            st.session_state["seg_df_batch"] = sample_dataframe(200)
        df_seg_batch = st.session_state.get("seg_df_batch")

    if df_seg_batch is not None and len(df_seg_batch)>0:
        pred_out = run_prediction(df_seg_batch)
        clus_out = run_clustering(df_seg_batch)
        merged = enrich_with_profiles(pred_out.join(clus_out["cluster"]))

        # ----- Overview
        total = len(merged)
        n_yes = (merged["prediction"]=="Yes").sum()
        n_no = (merged["prediction"]=="No").sum()
        n_maybe = (merged["prediction"]=="Not Sure").sum()
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Total Clients", f"{total}")
        with c2: st.metric("Yes", f"{pct(n_yes,total)}%")
        with c3: st.metric("No", f"{pct(n_no,total)}%")
        with c4: st.metric("Not Sure", f"{pct(n_maybe,total)}%")

        # Pie
        try:
            import plotly.express as px
            pie_df = pd.DataFrame({"Outcome":["Yes","No","Not Sure"],"Count":[n_yes,n_no,n_maybe]})
            st.plotly_chart(px.pie(pie_df, names="Outcome", values="Count", hole=0.35), use_container_width=True)
        except Exception:
            pass

        st.divider()
        st.markdown("### Client Drilldown by Cluster")

        # Build full 7-cluster summary (even if a cluster currently has 0 members)
        def cluster_summary(c):
            sub = merged[merged["cluster"]==c]
            total_c = len(sub)
            yes_c = (sub["prediction"]=="Yes").sum()
            no_c = (sub["prediction"]=="No").sum()
            maybe_c = (sub["prediction"]=="Not Sure").sum()
            info = SEGMENT_MAP.get(c, {})
            return {
                "cluster": c,
                "name": info.get("Segment_Name", f"Cluster {c}"),
                "priority": info.get("Business_Priority", ""),
                "n_clients": total_c,
                "yes_pct": pct(yes_c, total_c) if total_c>0 else 0.0,
                "no_pct": pct(no_c, total_c) if total_c>0 else 0.0,
                "maybe_pct": pct(maybe_c, total_c) if total_c>0 else 0.0,
            }

        summary_df = pd.DataFrame([cluster_summary(c) for c in range(N_SEGMENTS)])

        # Selector lists all 7 clusters
        cluster_choices = [f"{row.cluster} — {row.name} (Priority: {row.priority})" for _, row in summary_df.iterrows()]
        sel = st.selectbox("Select a cluster to inspect:", cluster_choices)
        sel_cluster = int(sel.split(" — ")[0])

        row = summary_df[summary_df["cluster"]==sel_cluster].iloc[0]
        cA,cB,cC,cD = st.columns(4)
        with cA: st.metric("Cluster", f"{row.cluster}")
        with cB: st.metric("Yes", f"{row.yes_pct:.1f}%")
        with cC: st.metric("No", f"{row.no_pct:.1f}%")
        with cD: st.metric("Not Sure", f"{row.maybe_pct:.1f}%")

        info = SEGMENT_MAP.get(sel_cluster, {})
        st.markdown(
            f"**Name:** {info.get('Segment_Name','')}\n\n"
            f"**Business Priority:** {info.get('Business_Priority','')}\n\n"
            f"**Key Characteristics:** {info.get('Key_Characteristics','')}\n\n"
            f"**Value Proposition:** {info.get('Value_Proposition','')}\n\n"
            f"**Communication Strategy:** {info.get('Communication_Strategy','')}\n\n"
            f"**Recommended Channels:** {info.get('Recommended_Channels','')}\n\n"
            f"**Campaign Cadence:** {info.get('Campaign_Cadence','')}"
        )
    else:
        st.info("Upload a CSV or generate dummy data to view segmentation.")
