import streamlit as st
import numpy as np
import pickle

# Load scaler + model from pickle
scaler, model = pickle.load(open("model.pkl", "rb"))

st.title("🍷 Liver Disease Classification App")
st.write("Enter patient details to predict liver disease risk")

# Example input fields (you should adjust according to your dataset columns)
Age = st.number_input("Age", min_value=1, max_value=100, value=45)
Gender = st.selectbox("Gender", ["Male", "Female"])
Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0)
Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3)
Alkphos_Alkaline_Phosphotase = st.number_input("Alkphos Alkaline Phosphotase", min_value=0, value=200)
Sgpt_Alamine_Aminotransferase = st.number_input("Sgpt Alamine Aminotransferase", min_value=0, value=30)
Sgot_Aspartate_Aminotransferase = st.number_input("Sgot Aspartate Aminotransferase", min_value=0, value=40)
Total_Protiens = st.number_input("Total Proteins", min_value=0.0, value=6.5)
Albumin = st.number_input("Albumin", min_value=0.0, value=3.0)
AG_Ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, value=1.0)

# Convert gender to numeric (Male=1, Female=0)
Gender_num = 1 if Gender == "Male" else 0

# Collect all features into array
features = np.array([[Age, Gender_num, Total_Bilirubin, Direct_Bilirubin,
                      Alkphos_Alkaline_Phosphotase, Sgpt_Alamine_Aminotransferase,
                      Sgot_Aspartate_Aminotransferase, Total_Protiens,
                      Albumin, AG_Ratio]])

# Scale features
features_scaled = scaler.transform(features)

if st.button("🔍 Predict"):
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    # Handle case when classes are [1,2]
    if list(model.classes_) == [1,2]:
        prediction = 0 if prediction == 1 else 1   # remap 1→0, 2→1
        proba = proba[1]  # take probability of class "2" (disease)
    else:
        proba = proba[1]  # normal 0/1 case

    if prediction == 1:
        st.error(f"⚠️ High Risk of Liver Disease (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Low Risk of Liver Disease (Probability: {proba:.2f})")
