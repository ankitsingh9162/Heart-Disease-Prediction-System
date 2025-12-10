import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")

# ================= LOAD MODELS =====================
scaler = joblib.load("scaler.pkl")
log_model = joblib.load("logistic_model.pkl")
knn_model = joblib.load("knn_model.pkl")
rf_model = joblib.load("rf_model.pkl")

feature_names = joblib.load("feature_names.pkl") if False else None

# ================= UI HEADER =====================
st.title("🫀 Heart Disease Prediction System")
st.markdown("### Machine Learning Based Medical Decision Support System")

# ================= SIDEBAR =====================
st.sidebar.header("🔍 Prediction Mode")
mode = st.sidebar.radio("Choose Mode", ["Single Patient", "Bulk CSV Upload"])

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "KNN", "Random Forest"]
)

model_map = {
    "Logistic Regression": log_model,
    "KNN": knn_model,
    "Random Forest": rf_model
}
model = model_map[model_choice]

# ================= SINGLE PATIENT =====================
if mode == "Single Patient":
    st.subheader("🧑 Single Patient Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 20, 100, 45)
        sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1])
        cp = st.number_input("Chest Pain Type", 0, 3, 1)
        trestbps = st.number_input("Resting BP", 90, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
        restecg = st.number_input("Rest ECG", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0,1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

    with col3:
        slope = st.number_input("Slope", 0, 2, 1)
        ca = st.number_input("CA", 0, 4, 0)
        thal = st.number_input("Thal", 0, 3, 2)

    if st.button("🔍 Predict"):
        user_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                                thalach,exang,oldpeak,slope,ca,thal]])
        user_scaled = scaler.transform(user_data)

        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error(f"🛑 High Risk of Heart Disease ({probability*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({(1-probability)*100:.2f}%)")

        # Probability Bar Chart
        prob_df = pd.DataFrame({
            "Status": ["No Disease", "Disease"],
            "Probability": [1-probability, probability]
        })

        fig, ax = plt.subplots()
        sns.barplot(x="Status", y="Probability", data=prob_df, ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)

# ================= BULK CSV UPLOAD =====================
if mode == "Bulk CSV Upload":
    st.subheader("📂 Bulk Patient Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview")
        st.dataframe(df.head())

        if st.button("⚡ Predict for All Patients"):
            X = df.values
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:,1]

            df["Prediction"] = preds
            df["Disease_Probability"] = probs

            st.success("Prediction Completed")
            st.dataframe(df.head())

            st.download_button(
                "⬇ Download Prediction Results",
                df.to_csv(index=False),
                file_name="heart_predictions.csv",
                mime="text/csv"
            )

# ================= MODEL COMPARISON DASHBOARD =====================
st.markdown("---")
st.subheader("📈 Model Accuracy Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("Logistic Accuracy", "✅ High")
col2.metric("KNN Accuracy", "✅ Medium")
col3.metric("Random Forest Accuracy", "🏆 Highest")

st.markdown("✅ Random Forest is the **recommended best model** for final deployment.")

# ================= FOOTER =====================
st.markdown("---")
st.caption("Developed by ML Engineer | UCI Heart Disease Dataset | Streamlit Deployment")
