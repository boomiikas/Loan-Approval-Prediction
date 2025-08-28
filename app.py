import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('trained_model.sav', 'rb'))

st.title("Loan Approval Prediction App")

st.write("Enter applicant details below to predict loan approval:")

income = st.number_input("Income:", min_value=0, step=1000, value=55000)
age = st.number_input("Age:", min_value=18, max_value=100, value=35)
credit_score = st.number_input("Credit Score:", min_value=300, max_value=850, value=720)
loan_amount = st.number_input("Loan Amount:", min_value=1000, step=1000, value=20000)

features = np.array([[income, age, credit_score, loan_amount]])

if st.button("Predict Loan Status"):
    probability = model.predict_proba(features)

    st.subheader("Prediction Result")
    st.write(f"Probability of Approval: **{probability[0][1]*100:.2f}%**")
    st.write(f"Probability of Rejection: **{probability[0][0]*100:.2f}%**")

    if probability[0][1] > 0.5:
        st.success("Loan is **likely Approved**")
    else:
        st.error("Loan is **likely Rejected**")
