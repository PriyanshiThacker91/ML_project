import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/trained_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Cardiovascular Disease Prediction")

age = st.number_input("Age (years)")
gender = st.selectbox("Gender (1=Male,2=Female)", [1,2])
height = st.number_input("Height (cm)")
weight = st.number_input("Weight (kg)")
ap_hi = st.number_input("Systolic BP")
ap_lo = st.number_input("Diastolic BP")
chol = st.selectbox("Cholesterol", [1,2,3])
gluc = st.selectbox("Glucose", [1,2,3])
smoke = st.selectbox("Smoking", [0,1])
alco = st.selectbox("Alcohol", [0,1])
active = st.selectbox("Physical Activity", [0,1])

if st.button("Predict"):
    bmi = weight / ((height/100)**2)

    input_data = np.array([[gender,height,weight,ap_hi,ap_lo,chol,gluc,smoke,alco,active,age,bmi]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0]==1:
        st.error("High Risk of Cardiovascular Disease")
    else:
        st.success("Low Risk of Cardiovascular Disease")
