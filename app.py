from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Customer churn | ML + DL", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1180px} h1,h2,h3{color:#251f21}.stMetric{background:#f4efec;padding:14px;border-radius:10px}
[data-testid="stSidebar"]{background:#f4efec}.tag{display:inline-block;background:#dcebe7;color:#274e46;padding:5px 10px;border-radius:12px;margin-right:6px}
</style>""", unsafe_allow_html=True)
st.title("Customer churn prediction")
st.caption("A portfolio project comparing classical machine learning and deep learning on one fair test split.")
st.markdown('<span class="tag">Logistic Regression</span><span class="tag">Random Forest</span><span class="tag">Keras ANN</span>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Project overview", "Model comparison", "Try a prediction"])
with tab1:
    a,b,c=st.columns(3); a.metric("Dataset", "7,043 customers"); b.metric("Target", "Customer churn"); c.metric("Evaluation", "Held-out 20%")
    st.subheader("What this project demonstrates")
    st.write("Data cleaning, reusable preprocessing, class-aware training, fair evaluation, explainability, and a simple decision-support dashboard.")
    if (ROOT/"reports/eda_overview.png").exists(): st.image(str(ROOT/"reports/eda_overview.png"), caption="Exploratory analysis generated from the dataset")
    st.info("Dataset source and exact setup commands are in README.md. Model results appear only after you run training.")
with tab2:
    metrics_path=ROOT/"reports/metrics.csv"
    if metrics_path.exists():
        metrics=pd.read_csv(metrics_path); st.dataframe(metrics.style.format({c:"{:.3f}" for c in ["accuracy","precision","recall","f1","roc_auc"]}), width="stretch")
        st.caption("All models use the same stratified test set and a 0.50 decision threshold.")
        for img in ["roc_curves.png","confusion_matrices.png"]:
            if (ROOT/"reports"/img).exists(): st.image(str(ROOT/"reports"/img))
        imp=ROOT/"reports/permutation_importance.csv"
        if imp.exists(): st.subheader("What matters most"); st.bar_chart(pd.read_csv(imp).head(12).set_index("feature")["importance_mean"])
    else:
        st.warning("No trained results yet. Run `python src/train.py` after installing requirements.")
with tab3:
    st.write("Enter a customer profile. This is a learning demo, not a real retention decision system.")
    with st.form("prediction"):
        c1,c2,c3=st.columns(3)
        gender=c1.selectbox("Gender",["Female","Male"]); senior=c1.selectbox("Senior citizen",[0,1]); partner=c1.selectbox("Partner",["No","Yes"])
        dependents=c2.selectbox("Dependents",["No","Yes"]); tenure=c2.slider("Tenure (months)",0,72,12); contract=c2.selectbox("Contract",["Month-to-month","One year","Two year"])
        internet=c3.selectbox("Internet service",["DSL","Fiber optic","No"]); monthly=c3.number_input("Monthly charges",0.0,200.0,70.0); total=c3.number_input("Total charges",0.0,10000.0,float(monthly*tenure))
        submitted=st.form_submit_button("Estimate churn risk")
    if submitted:
        path=ROOT/"models/random_forest.joblib"
        if not path.exists(): st.error("Train the models first. See README.md.")
        else:
            row=pd.DataFrame([{"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,"PhoneService":"Yes","MultipleLines":"No","InternetService":internet,"OnlineSecurity":"No internet service" if internet=="No" else "No","OnlineBackup":"No internet service" if internet=="No" else "No","DeviceProtection":"No internet service" if internet=="No" else "No","TechSupport":"No internet service" if internet=="No" else "No","StreamingTV":"No internet service" if internet=="No" else "No","StreamingMovies":"No internet service" if internet=="No" else "No","Contract":contract,"PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":monthly,"TotalCharges":total}])
            risk=float(joblib.load(path).predict_proba(row)[0,1]); st.metric("Estimated churn probability",f"{risk:.1%}"); st.progress(risk)
            st.caption("Use as a model demonstration only. Probability is not a guarantee or an instruction to act.")
