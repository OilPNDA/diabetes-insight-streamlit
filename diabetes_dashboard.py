import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import shap
import numpy as np

# -------------------------------
# SECTION 1: LOAD & CLEAN DATA
# -------------------------------
st.title("Diabetes Prediction Dashboard")

st.header("1. Load Data")
uploaded_file = st.file_uploader("Upload your diabetes CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of dataset", df.head())

    # Replace 0 with median in key features
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_to_fix:
        median = df[df[col] != 0][col].median()
        df[col] = df[col].replace(0, median)

    # -------------------------------
    # SECTION 2: EDA
    # -------------------------------
    st.header("2. Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Outcome Distribution")
        st.bar_chart(df["Outcome"].value_counts())

    with col2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("BMI vs Outcome")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x="Outcome", y="BMI", ax=ax2)
    st.pyplot(fig2)

    # -------------------------------
    # SECTION 3: Model Training
    # -------------------------------
    st.header("3. Model Training & Prediction")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Confusion Matrix")
    st.write(
        pd.DataFrame(
            cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]
        )
    )

    st.subheader("Classification Report")
    st.write(pd.DataFrame(cr).transpose())

    # -------------------------------
    # SECTION 4: SHAP Explainability
    # -------------------------------
    st.header("4. SHAP Feature Importance")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("### SHAP Summary Plot")
    fig, ax = plt.subplots()
    fig_shap = shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
    st.pyplot(bbox_inches="tight", dpi=300)
    st.pyplot(fig)

    st.caption(
        "Blue = low value, Red = high value. More to right = more likely to predict diabetes."
    )

else:
    st.info("Please upload a CSV file to begin.")
