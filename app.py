import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bucknell Lending Club · Loan Evaluator",
    page_icon="🦬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --bucknell-orange: #E87722;
    --bucknell-blue:   #003865;
    --bucknell-light:  #F5F0EB;
    --border:          #E0D9D0;
}
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--bucknell-light);
    color: #1A1A2E;
}
.hero {
    background: linear-gradient(135deg, var(--bucknell-blue) 60%, #005599 100%);
    border-radius: 16px; padding: 2.5rem 3rem;
    margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::after {
    content: "🦬"; position: absolute; right: 3rem; top: 50%;
    transform: translateY(-50%); font-size: 5rem; opacity: 0.15;
}
.hero h1 { font-family: 'Playfair Display', serif; color: white; font-size: 2.4rem; margin: 0 0 0.3rem 0; }
.hero p  { color: rgba(255,255,255,0.75); font-size: 1.05rem; margin: 0; font-weight: 300; }
.hero .badge {
    display: inline-block; background: var(--bucknell-orange);
    color: white; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.2rem 0.7rem; border-radius: 20px; margin-bottom: 0.8rem;
}
.section-header {
    font-family: 'Playfair Display', serif; color: var(--bucknell-blue);
    font-size: 1.25rem; font-weight: 700;
    border-left: 4px solid var(--bucknell-orange);
    padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}
.result-card { border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1rem; border: 1px solid var(--border); }
.result-card.approve { background: linear-gradient(135deg,#E8F5E9,#F1F8F1); border-color:#A5D6A7; }
.result-card.review  { background: linear-gradient(135deg,#FFF8E1,#FFFDF5); border-color:#FFE082; }
.result-card.decline { background: linear-gradient(135deg,#FFEBEE,#FFF5F5); border-color:#FFCDD2; }
.result-card h2 { font-family: 'Playfair Display', serif; font-size: 1.6rem; margin: 0 0 0.3rem 0; }
.orange-line { height: 3px; background: linear-gradient(90deg, var(--bucknell-orange), transparent); border: none; margin: 1.5rem 0; }
.footer { text-align: center; color: #999; font-size: 0.78rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.summary-box {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    border: 1px solid var(--border); text-align: center;
}
.summary-box .big { font-size: 2rem; font-weight: 700; font-family: 'Playfair Display', serif; }
.summary-box .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #666; }
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("classifier.pkl",  "rb") as f: clf        = pickle.load(f)
    with open("scaler_clf.pkl",  "rb") as f: scaler_clf = pickle.load(f)
    with open("regressor.pkl",   "rb") as f: reg        = pickle.load(f)
    with open("scaler_reg.pkl",  "rb") as f: scaler_reg = pickle.load(f)
    return clf, scaler_clf, reg, scaler_reg

clf, scaler_clf, reg, scaler_reg = load_models()

CLF_NUM_COLS = ['loan_amnt','annual_inc','dti','delinq_2yrs','open_acc',
                'pub_rec','revol_bal','revol_util','term_num','fico_score',
                'grade_numeric','risk_loan_amnt_tradeoff']
CLF_ALL_COLS = list(clf.feature_names_in_)   # 28 columns
REG_ALL_COLS = list(reg.feature_names_in_)   # 434 columns

CURRENT_DATA_URL = (
    "https://raw.githubusercontent.com/MattDBailey/ANOP330/"
    "refs/heads/main/Data/BucknellLendingClubCurrentData.csv"
)


# ── Shared preprocessing (single row dict) ─────────────────────────────────────
def preprocess_base(inp):
    df = pd.DataFrame([inp])

    df["fico_score"]   = (df["fico_range_high"] + df["fico_range_low"]) / 2
    df = df.drop(columns=["fico_range_high","fico_range_low"])
    df["grade_numeric"] = df["grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7})
    df["risk_loan_amnt_tradeoff"] = df["loan_amnt"] * df["grade_numeric"]
    df["annual_inc"]   = np.log1p(df["annual_inc"])
    df["revol_util"]   = df["revol_util"].clip(upper=100)
    df["home_ownership"] = df["home_ownership"].apply(
        lambda x: x if x in ["MORTGAGE","RENT"] else "OTHER")
    df["purpose"] = df["purpose"].apply(
        lambda x: x if x in ["debt_consolidation","credit_card"] else "OTHER")
    df = df.drop(columns=["grade","int_rate","installment","funded_amnt",
                           "id","issue_d","earliest_cr_line","term"], errors="ignore")
    df = pd.get_dummies(df, columns=["emp_length","home_ownership",
                                      "verification_status","purpose"], drop_first=True)
    return df


# ── Batch preprocessing (full DataFrame) ──────────────────────────────────────
def preprocess_batch(df_raw):
    df = df_raw.copy()

    # Convert "term" → term_num if needed
    if "term" in df.columns and "term_num" not in df.columns:
        df["term_num"] = df["term"].astype(str).str.extract(r'(\d+)').astype(float)

    df["fico_score"]   = (df["fico_range_high"] + df["fico_range_low"]) / 2
    df = df.drop(columns=["fico_range_high","fico_range_low"], errors="ignore")

    df["grade_numeric"] = df["grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7})
    df["risk_loan_amnt_tradeoff"] = df["loan_amnt"] * df["grade_numeric"]

    df["annual_inc"] = np.log1p(pd.to_numeric(df["annual_inc"], errors="coerce").fillna(0))
    df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce").fillna(0).clip(upper=100)

    df["home_ownership"] = df["home_ownership"].apply(
        lambda x: x if x in ["MORTGAGE","RENT"] else "OTHER")
    df["purpose"] = df["purpose"].apply(
        lambda x: x if x in ["debt_consolidation","credit_card"] else "OTHER")

    df = df.drop(columns=["grade","int_rate","installment","funded_amnt",
                           "id","issue_d","earliest_cr_line","term",
                           "loan_status","sub_grade"], errors="ignore")

    df = pd.get_dummies(df, columns=["emp_length","home_ownership",
                                      "verification_status","purpose"], drop_first=True)
    return df


# ── Classifier input ───────────────────────────────────────────────────────────
def build_clf_input(df):
    df = df.copy()
    num = [c for c in CLF_NUM_COLS if c in df.columns]
    df[num] = scaler_clf.transform(df[num])
    for col in CLF_ALL_COLS:
        if col not in df.columns: df[col] = 0
    return df[CLF_ALL_COLS]


# ── Regressor input ────────────────────────────────────────────────────────────
def build_reg_input(df):
    df = df.copy()
    for col in CLF_ALL_COLS:
        if col not in df.columns: df[col] = 0
    df_28 = df[CLF_ALL_COLS]

    poly      = PolynomialFeatures(degree=2, include_bias=False)
    arr       = poly.fit_transform(df_28)
    poly_cols = poly.get_feature_names_out(CLF_ALL_COLS)
    df_poly   = pd.DataFrame(arr, columns=poly_cols)

    for col in REG_ALL_COLS:
        if col not in df_poly.columns: df_poly[col] = 0
    df_poly = df_poly[REG_ALL_COLS]

    df_poly[REG_ALL_COLS] = scaler_reg.transform(df_poly[REG_ALL_COLS])
    return df_poly


# ── Recommendation helper ──────────────────────────────────────────────────────
def recommend(prob_fully_paid, pred_return):
    if prob_fully_paid >= 0.75 and pred_return >= 0:
        return "✅ Approve"
    elif prob_fully_paid >= 0.55 and pred_return >= -2:
        return "⚠️ Review"
    else:
        return "❌ Decline"


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">Bucknell Lending Club</div>
    <h1>Loan Application Evaluator</h1>
    <p>AI-powered risk assessment · Logistic Regression + LASSO Regressor</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Single Application", "📊 Batch Scoring — Current Data"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Application (original form, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📋 Loan Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=10000, step=500)
        term_num  = st.selectbox("Loan Term", options=[36,60], format_func=lambda x: f"{x} months")
        int_rate  = st.slider("Interest Rate (%)", min_value=5.0, max_value=31.0, value=12.0, step=0.1)
        grade     = st.selectbox("Loan Grade", options=["A","B","C","D","E","F","G"])

    with col2:
        annual_inc          = st.number_input("Annual Income ($)", min_value=1000, max_value=6_100_000, value=65000, step=1000)
        emp_length          = st.selectbox("Employment Length", options=[
            "< 1 year","1 year","2 years","3 years","4 years",
            "5 years","6 years","7 years","8 years","9 years","10+ years"], index=10)
        home_ownership      = st.selectbox("Home Ownership", options=["MORTGAGE","RENT","OWN","OTHER"])
        verification_status = st.selectbox("Income Verification", options=["Verified","Source Verified","Not Verified"])

    with col3:
        purpose  = st.selectbox("Loan Purpose", options=[
            "debt_consolidation","credit_card","home_improvement","other",
            "major_purchase","medical","small_business","car","moving",
            "vacation","house","wedding","renewable_energy","educational"])
        dti      = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=60.0, value=15.0, step=0.1)
        open_acc = st.number_input("Open Credit Lines", min_value=1, max_value=68, value=10)
        pub_rec  = st.number_input("Public Derogatory Records", min_value=0, max_value=21, value=0)

    st.markdown('<hr class="orange-line">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Credit Profile</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
        fico_range_low  = st.number_input("FICO Score (Low)",  min_value=660, max_value=845, value=700)
    with col5:
        fico_range_high = st.number_input("FICO Score (High)", min_value=664, max_value=850, value=704)
    with col6:
        delinq_2yrs = st.number_input("Delinquencies (past 2 yrs)", min_value=0, max_value=26, value=0)

    col7, col8 = st.columns(2)
    with col7:
        revol_bal  = st.number_input("Revolving Balance ($)", min_value=0, max_value=500_000, value=8000, step=100)
    with col8:
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)

    st.markdown('<hr class="orange-line">', unsafe_allow_html=True)

    if st.button("🔍  Evaluate Loan Application", use_container_width=True, type="primary", key="single_eval"):
        inp = {
            "loan_amnt": loan_amnt, "term_num": term_num, "int_rate": int_rate,
            "grade": grade, "emp_length": emp_length, "home_ownership": home_ownership,
            "annual_inc": annual_inc, "verification_status": verification_status,
            "purpose": purpose, "dti": dti, "delinq_2yrs": int(delinq_2yrs),
            "open_acc": int(open_acc), "pub_rec": int(pub_rec),
            "fico_range_high": fico_range_high, "fico_range_low": fico_range_low,
            "revol_bal": revol_bal, "revol_util": revol_util,
        }

        with st.spinner("Running models..."):
            df_base = preprocess_base(inp)
            X_clf           = build_clf_input(df_base)
            proba           = clf.predict_proba(X_clf)[0]
            prob_fully_paid = proba[1]
            prob_default    = proba[0]
            X_reg       = build_reg_input(df_base)
            pred_return = reg.predict(X_reg)[0]

        if prob_fully_paid >= 0.75 and pred_return >= 0:
            rec, rec_label = "approve", "✅ APPROVE"
            rec_detail = "Strong repayment probability and positive expected return. Recommend funding."
        elif prob_fully_paid >= 0.55 and pred_return >= -2:
            rec, rec_label = "review", "⚠️ MANUAL REVIEW"
            rec_detail = "Moderate risk profile. A loan officer should review before proceeding."
        else:
            rec, rec_label = "decline", "❌ DECLINE"
            rec_detail = "High default probability or negative expected return. Not recommended."

        st.markdown(f"""
        <div class="result-card {rec}">
            <div style="font-size:0.8rem;font-weight:600;letter-spacing:0.08em;
                        text-transform:uppercase;opacity:0.6;margin-bottom:0.2rem;">Recommendation</div>
            <h2>{rec_label}</h2>
            <p style="margin:0;opacity:0.8;">{rec_detail}</p>
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Probability of Full Repayment", f"{prob_fully_paid*100:.1f}%",
                      delta=f"Default risk: {prob_default*100:.1f}%", delta_color="inverse")
        with r2:
            st.metric("Predicted Pessimistic Return", f"{pred_return:.2f}%",
                      delta="annualized (ret_PESS)", delta_color="off")
        with r3:
            st.metric("Loan Grade", grade,
                      delta=f"{int_rate:.1f}% interest rate", delta_color="off")

        st.markdown('<div class="section-header">Repayment Confidence</div>', unsafe_allow_html=True)
        st.progress(prob_fully_paid,
                    text=f"{prob_fully_paid*100:.1f}% probability of full repayment")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Scoring (Current Data)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Batch Scoring — Current Loan Applications</div>', unsafe_allow_html=True)
    st.markdown(
        "This tab automatically loads the **current (live) applicant dataset** from GitHub "
        "and scores every row using both models — no manual entry required.",
        unsafe_allow_html=False,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_current_data():
        return pd.read_csv(CURRENT_DATA_URL)

    load_col, reload_col = st.columns([4, 1])
    with reload_col:
        if st.button("🔄 Reload CSV", key="reload_csv"):
            st.cache_data.clear()

    with st.spinner("Loading current applicant data from GitHub..."):
        try:
            df_current = load_current_data()
            load_ok = True
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
            load_ok = False

    if load_ok:
        with load_col:
            st.success(f"Loaded **{len(df_current):,} applicants** from current dataset.")

        with st.expander("👁️ Preview raw data (first 5 rows)"):
            st.dataframe(df_current.head(), use_container_width=True)

        st.markdown('<hr class="orange-line">', unsafe_allow_html=True)

        if st.button("🚀  Run Batch Scoring on All Applicants", use_container_width=True, type="primary", key="batch_run"):

            progress_bar = st.progress(0, text="Preprocessing data...")

            # ── Preprocessing ──────────────────────────────────────────────────
            # Keep display columns before dropping them
            display_cols = []
            for c in ["id","loan_amnt","grade","term","annual_inc","purpose","dti"]:
                if c in df_current.columns:
                    display_cols.append(c)
            df_display_info = df_current[display_cols].copy() if display_cols else pd.DataFrame()

            df_proc = preprocess_batch(df_current)
            progress_bar.progress(30, text="Building classifier input...")

            # ── Classifier ─────────────────────────────────────────────────────
            X_clf_batch = build_clf_input(df_proc)
            progress_bar.progress(50, text="Running classifier on all rows...")
            proba_batch = clf.predict_proba(X_clf_batch)
            prob_paid_batch    = proba_batch[:, 1]
            prob_default_batch = proba_batch[:, 0]

            # ── Regressor ──────────────────────────────────────────────────────
            progress_bar.progress(65, text="Building regressor input (polynomial features)...")
            X_reg_batch = build_reg_input(df_proc)
            progress_bar.progress(85, text="Running regressor on all rows...")
            pred_return_batch = reg.predict(X_reg_batch)

            progress_bar.progress(95, text="Assembling results table...")

            # ── Build results DataFrame ────────────────────────────────────────
            results = pd.DataFrame({
                "Repay Prob (%)":   (prob_paid_batch * 100).round(1),
                "Default Prob (%)": (prob_default_batch * 100).round(1),
                "Pred Return (%)":  pred_return_batch.round(2),
                "Recommendation":   [recommend(p, r) for p, r in
                                     zip(prob_paid_batch, pred_return_batch)],
            })

            # Prepend any handy display columns from the raw data
            if not df_display_info.empty:
                results = pd.concat(
                    [df_display_info.reset_index(drop=True), results.reset_index(drop=True)],
                    axis=1,
                )

            progress_bar.progress(100, text="Done!")

            # ── Summary metrics ────────────────────────────────────────────────
            n_total   = len(results)
            n_approve = (results["Recommendation"] == "✅ Approve").sum()
            n_review  = (results["Recommendation"] == "⚠️ Review").sum()
            n_decline = (results["Recommendation"] == "❌ Decline").sum()
            avg_repay = results["Repay Prob (%)"].mean()
            avg_ret   = results["Pred Return (%)"].mean()

            st.markdown('<div class="section-header">Portfolio Summary</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""<div class="summary-box">
                    <div class="big">{n_total:,}</div>
                    <div class="label">Total Applicants</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="summary-box" style="border-color:#A5D6A7;">
                    <div class="big" style="color:#2E7D32;">{n_approve:,}</div>
                    <div class="label">Approve</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="summary-box" style="border-color:#FFE082;">
                    <div class="big" style="color:#F57F17;">{n_review:,}</div>
                    <div class="label">Manual Review</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="summary-box" style="border-color:#FFCDD2;">
                    <div class="big" style="color:#C62828;">{n_decline:,}</div>
                    <div class="label">Decline</div></div>""", unsafe_allow_html=True)
            with m5:
                st.markdown(f"""<div class="summary-box">
                    <div class="big">{avg_repay:.1f}%</div>
                    <div class="label">Avg Repay Prob</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts ─────────────────────────────────────────────────────────
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown('<div class="section-header">Recommendation Breakdown</div>', unsafe_allow_html=True)
                rec_counts = pd.DataFrame({
                    "Decision": ["✅ Approve", "⚠️ Review", "❌ Decline"],
                    "Count":    [n_approve, n_review, n_decline],
                })
                st.bar_chart(rec_counts.set_index("Decision"), color="#E87722", use_container_width=True)

            with chart_col2:
                st.markdown('<div class="section-header">Repayment Probability Distribution</div>', unsafe_allow_html=True)
                hist_df = pd.DataFrame({"Repay Prob (%)": results["Repay Prob (%)"]})
                st.bar_chart(
                    hist_df["Repay Prob (%)"].value_counts(bins=20, sort=False).sort_index(),
                    color="#003865",
                    use_container_width=True,
                )

            st.markdown('<hr class="orange-line">', unsafe_allow_html=True)

            # ── Full results table ─────────────────────────────────────────────
            st.markdown('<div class="section-header">All Applicant Predictions</div>', unsafe_allow_html=True)

            # Colour-code the Recommendation column
            def colour_rec(val):
                if "Approve" in str(val):   return "background-color: #E8F5E9; color: #2E7D32; font-weight:600;"
                if "Review"  in str(val):   return "background-color: #FFF8E1; color: #F57F17; font-weight:600;"
                if "Decline" in str(val):   return "background-color: #FFEBEE; color: #C62828; font-weight:600;"
                return ""

            styled = results.style.applymap(colour_rec, subset=["Recommendation"])
            st.dataframe(styled, use_container_width=True, height=420)

            # ── Download ───────────────────────────────────────────────────────
            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Results as CSV",
                data=csv_bytes,
                file_name="bucknell_lending_club_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Bucknell Lending Club · ANOP 330 Final Project<br>
    For educational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)