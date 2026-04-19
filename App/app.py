import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'Models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_DIR   = os.path.join(BASE_DIR, '..', 'Data')

# ── Load saved model, scaler, feature columns ────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(MODELS_DIR, 'lr_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()

# ── Build input from user values ──────────────────────────────────────────────
# Replicates exactly what Notebook 2 and Notebook 3 did:
# feature engineering → one-hot encoding → column alignment → scaling
# No dependency on preprocessing.py
def build_input(age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal):

    df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal
    }])

    # Feature engineering — same as Notebook 2
    df['age_group']       = int(pd.cut([age], bins=[0,45,60,100], labels=[0,1,2])[0])
    df['thalach_pct_max'] = round(thalach / (220 - age), 4)
    df['chol_age_ratio']  = round(chol / age, 4)
    df['oldpeak_slope']   = round(oldpeak * (slope + 1), 4)
    df['high_risk_flag']  = int(exang == 1 and oldpeak > 1.0)

    # One-hot encoding — same as Notebook 3
    df = pd.get_dummies(df, columns=['cp','restecg','slope','thal'], drop_first=True)

    # Align to training feature columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    # Scale — using the saved scaler from Notebook 3
    num_cols   = ['age','trestbps','chol','thalach','oldpeak',
                  'thalach_pct_max','chol_age_ratio','oldpeak_slope']
    scale_cols = [c for c in num_cols if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Convert to float — required by XGBoost SHAP
    df = df.astype(float)

    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## CardioAI")
st.sidebar.markdown("Heart Disease Risk Prediction  \nPowered by Logistic Regression + SHAP")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Patient Predictor", "📊 Model Performance", "🔍 SHAP Explorer", "📁 Dataset Overview"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Logistic Regression")
st.sidebar.markdown("**Dataset:** Cleveland Heart Disease (UCI)")
st.sidebar.markdown("**Validation:** Stratified 5-Fold CV")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Built by Arya <br>  Logistic Regression + SHAP + Streamlit</small>",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PATIENT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Patient Predictor":
    st.title("🔮 Patient Risk Predictor")
    st.markdown("Enter patient clinical data to get a real-time prediction with SHAP explanation.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age      = st.slider("Age", 20, 80, 55)
        sex      = st.selectbox("Sex", [1, 0],
                                format_func=lambda x: "Male" if x == 1 else "Female")
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.subheader("Cardiac Measurements")
        cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                                format_func=lambda x: ["Typical Angina","Atypical Angina",
                                                        "Non-anginal Pain","Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)

    with col3:
        st.subheader("Test Results")
        restecg  = st.selectbox("Resting ECG Results", [0, 1, 2],
                                format_func=lambda x: ["Normal","ST-T Abnormality",
                                                        "LV Hypertrophy"][x])
        exang    = st.selectbox("Exercise Induced Angina", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope    = st.selectbox("Slope of ST Segment", [0, 1, 2],
                                format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
        ca       = st.selectbox("Major Vessels Coloured (ca)", [0, 1, 2, 3])
        thal     = st.selectbox("Thalassemia (thal)", [0, 1, 2],
                                format_func=lambda x: ["Normal","Fixed Defect",
                                                        "Reversible Defect"][x])

    st.markdown("---")

    if st.button("🔮 Run Prediction", use_container_width=True):
        X_input = build_input(age, sex, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak, slope, ca, thal)

        prob = model.predict_proba(X_input)[0][1]
        pred = model.predict(X_input)[0]

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.error("### ⚠️ HIGH RISK — Heart Disease Detected")
            else:
                st.success("### ✅ LOW RISK — No Heart Disease Detected")
            st.metric("Risk Probability", f"{prob:.1%}")
            risk_level = ("Critical" if prob > 0.75 else
                          "High"     if prob > 0.5  else
                          "Moderate" if prob > 0.3  else "Low")
            st.info(f"**Risk Level:** {risk_level}")

        with res_col2:
            st.markdown("#### 🔍 SHAP Waterfall — Why this prediction?")
            try:
                # Load background data for proper SHAP calculation
                clean_path = os.path.join(DATA_DIR, 'heart_cleaned.csv')
                df_bg      = pd.read_csv(clean_path)
                ohe_cols   = ['cp', 'restecg', 'slope', 'thal']
                X_bg       = pd.get_dummies(
                                df_bg.drop(columns=['condition']),
                                columns=ohe_cols, drop_first=True
                            )
                for col in feature_cols:
                    if col not in X_bg.columns:
                        X_bg[col] = 0
                X_bg       = X_bg[feature_cols].astype(float)
                num_cols   = ['age','trestbps','chol','thalach','oldpeak',
                                'thalach_pct_max','chol_age_ratio','oldpeak_slope']
                scale_cols = [c for c in num_cols if c in X_bg.columns]
                X_bg[scale_cols] = scaler.transform(X_bg[scale_cols])

                # Compute real SHAP values using full background
                lr_explainer  = shap.LinearExplainer(model, X_bg)
                lr_shap_vals  = lr_explainer.shap_values(X_input.astype(float))
                base_value    = lr_explainer.expected_value
                shap_vals_row = lr_shap_vals[0]
                final_value   = base_value + shap_vals_row.sum()

                # Build series sorted by absolute SHAP value — top 12 only
                shap_series = pd.Series(shap_vals_row, index=feature_cols)
                shap_series = shap_series.reindex(
                    shap_series.abs().sort_values(ascending=False).index
                ).head(12).iloc[::-1]   # reverse so biggest is at top

                # ── Draw proper waterfall ──────────────────────────────────────
                fig, ax = plt.subplots(figsize=(9, 6))

                running = base_value
                lefts   = []
                widths  = []
                colors  = []

                for val in shap_series.values:
                    lefts.append(running if val >= 0 else running + val)
                    widths.append(abs(val))
                    colors.append('#e84040' if val >= 0 else '#378add')
                    running += val

                bars = ax.barh(
                    range(len(shap_series)),
                    widths,
                    left=lefts,
                    color=colors,
                    edgecolor='white',
                    linewidth=0.3,
                    height=0.55
                )

                # Base value line
                ax.axvline(x=base_value, color='#aaaaaa',
                            linewidth=1.2, linestyle='--', alpha=0.7,
                            label=f'Base value: {base_value:.3f}')

                # Final value line
                ax.axvline(x=final_value, color='white',
                            linewidth=1.5, linestyle='-', alpha=0.9,
                            label=f'Prediction: {final_value:.3f}')

                # Value labels
                for i, (val, left) in enumerate(zip(shap_series.values, lefts)):
                    center = left + abs(val) / 2
                    ax.text(center, i, f'{val:+.3f}',
                            va='center', ha='center',
                            fontsize=7.5, color='white', fontweight='bold')

                ax.set_yticks(range(len(shap_series)))
                ax.set_yticklabels(shap_series.index, fontsize=9, color='white')
                ax.set_xlabel('Model Output (log-odds)', color='white', fontsize=9)
                ax.set_title(
                    f'SHAP Waterfall — Base: {base_value:.3f}  →  '
                    f'Prediction: {final_value:.3f}  ({prob:.1%} risk)',
                    fontsize=9, color='white', pad=8
                )
                ax.legend(fontsize=8, labelcolor='white',
                            facecolor='#1E2D40', edgecolor='#3a3f52')
                ax.tick_params(axis='x', colors='white')
                ax.set_facecolor('#1E2D40')
                fig.patch.set_facecolor('#1E2D40')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#3a3f52')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"SHAP plot could not be generated: {e}")
        with st.expander("View processed feature values"):
            st.dataframe(X_input.T.rename(columns={0: "Value"}).round(4))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comparison of all three models — Stratified 5-Fold Cross Validation.")
    st.markdown("---")

    cv_path = os.path.join(MODELS_DIR, 'cv_results.csv')
    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path, index_col=0)
        st.subheader("Cross-Validation Results (5-Fold Stratified)")
        numeric_cols = cv_df.select_dtypes(include='number').columns
        st.dataframe(
            cv_df[numeric_cols].style.highlight_max(axis=0, color='#1B6CA8').format("{:.4f}"),
            use_container_width=True
        )
    else:
        st.warning("cv_results.csv not found. Run Notebook 3 first.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Comparison")
        img = os.path.join(STATIC_DIR, 'plot_model_comparison.png')
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.info("Run Notebook 3 to generate this plot.")

    with col2:
        st.subheader("ROC Curves")
        img = os.path.join(STATIC_DIR, 'plot_roc_curves.png')
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.info("Run Notebook 3 to generate this plot.")

    st.subheader("Confusion Matrix — Logistic Regression")
    img = os.path.join(STATIC_DIR, 'plot_confusion_matrix.png')
    if os.path.exists(img):
        st.image(img, use_container_width=True)
    else:
        st.info("Run Notebook 3 to generate this plot.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explorer":
    st.title("🔍 SHAP Explorer — Global Explainability")
    st.markdown("These plots explain what the Logistic Regression model learned from the entire dataset.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary (Beeswarm)", "Bar Chart", "Waterfall", "Force Plot", "Dependence"
    ])

    plots = [
        (tab1, 'plot_shap_summary.png',
         "Each dot = one patient. Red = high feature value, Blue = low. Top = most impactful."),
        (tab2, 'plot_shap_bar.png',
         "Mean absolute SHAP values — average impact of each feature across all patients."),
        (tab3, 'plot_shap_waterfall.png',
         "Single high-risk patient — how each feature pushed the prediction up or down."),
        (tab4, 'plot_shap_force.png',
         "Tug-of-war view — red features push toward disease, blue push away."),
        (tab5, 'plot_shap_dependence.png',
         "How thalach SHAP value changes with its actual value, coloured by ca (vessels)."),
    ]

    for tab, filename, description in plots:
        with tab:
            st.markdown(f"*{description}*")
            img = os.path.join(STATIC_DIR, filename)
            if os.path.exists(img):
                st.image(img,use_container_width=True)
            else:
                st.info("Run Notebook 4 to generate this plot.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Dataset Overview":
    st.title("📁 Dataset Overview")
    st.markdown("**Source:** UCI Machine Learning Repository (Cleveland Heart Disease)  \n"
                "**Rows:** 297 patients · **Features:** 13 + 1 target · **Missing values:** None")
    st.markdown("---")

    clean_path = os.path.join(DATA_DIR, 'heart_cleaned.csv')
    raw_path   = os.path.join(DATA_DIR, 'heart_cleveland_upload.csv')
    data_path  = clean_path if os.path.exists(clean_path) else raw_path

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients", len(df))
        c2.metric("Features", df.shape[1] - 1)
        c3.metric("Disease Cases", int(df['condition'].sum()))
        c4.metric("No Disease", int((df['condition'] == 0).sum()))

        st.markdown("---")
        st.subheader("Sample Data (first 10 rows)")

        # Column name dictionary — full medical meaning
        col_descriptions = {
            'age'      : 'age — Age in years',
            'sex'      : 'sex — 1=Male, 0=Female',
            'cp'       : 'cp — Chest Pain Type (0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)',
            'trestbps' : 'trestbps — Resting Blood Pressure (mmHg)',
            'chol'     : 'chol — Serum Cholesterol (mg/dl)',
            'fbs'      : 'fbs — Fasting Blood Sugar >120mg/dl (1=True, 0=False)',
            'restecg'  : 'restecg — Resting ECG (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)',
            'thalach'  : 'thalach — Maximum Heart Rate Achieved',
            'exang'    : 'exang — Exercise Induced Angina (1=Yes, 0=No)',
            'oldpeak'  : 'oldpeak — ST Depression Induced by Exercise',
            'slope'    : 'slope — Slope of Peak Exercise ST Segment (0=Upsloping, 1=Flat, 2=Downsloping)',
            'ca'       : 'ca — Number of Major Vessels Coloured by Fluoroscopy (0-3)',
            'thal'     : 'thal — Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)',
            'condition': 'condition — TARGET: 0=No Heart Disease, 1=Heart Disease'
        }

        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("📖 Column Name Reference Guide — click to expand"):
             for col, desc in col_descriptions.items():
                if col in df.columns:
                    st.markdown(f"- **{desc}**")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class Balance")
            img = os.path.join(STATIC_DIR, 'plot_class_balance.png')
            if os.path.exists(img):
                st.image(img, use_container_width=True)
        with col2:
            st.subheader("Correlation Heatmap")
            img = os.path.join(STATIC_DIR, 'plot_correlation.png')
            if os.path.exists(img):
                st.image(img, use_container_width=True)

        for title, filename in [
            ("Numerical Feature Distributions",  "plot_numerical.png"),
            ("Categorical Features vs Condition", "plot_categorical.png"),
            ("Engineered Features vs Condition",  "plot_engineered_features.png"),
            ("SMOTE — Class Balance Effect",      "plot_smote.png"),
        ]:
            st.subheader(title)
            img = os.path.join(STATIC_DIR, filename)
            if os.path.exists(img):
                st.image(img, use_container_width=True)
    else:
        st.warning("Dataset not found. Make sure heart_cleveland_upload.csv is in the Data/ folder.")