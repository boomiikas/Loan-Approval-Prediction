import streamlit as st
import pickle
import pandas as pd

# Load trained model and scaler
with open('rf_loan_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

st.title("Loan Approval Prediction App")
st.write("Enter applicant details below:")

# Streamlit inputs
income = st.number_input("Applicant Income:", min_value=0, step=1000, value=55000)
age = st.number_input("Age:", min_value=18, max_value=100, value=35)
credit_score = st.number_input("Credit Score:", min_value=300, max_value=850, value=720)
loan_amount = st.number_input("Loan Amount:", min_value=1000, step=1000, value=20000)
emp_exp = st.number_input("Years of Experience:", min_value=0, max_value=50, value=5)
loan_int_rate = st.number_input("Loan Interest Rate (%):", min_value=0.0, max_value=50.0, value=10.0)
cred_hist_length = st.number_input("Credit History Length (years):", min_value=0, max_value=50, value=10)

gender = st.selectbox("Gender:", ["Male", "Female"])
education = st.selectbox("Education:", ["Graduate", "Not Graduate"])
home_ownership = st.selectbox("Home Ownership:", ["Own", "Rent", "Mortgage"])
loan_intent = st.selectbox("Loan Intent:", ["Personal", "Debt Consolidation", "Education", "Home Improvement"])
previous_defaults = st.selectbox("Previous Loan Defaults:", ["No", "Yes"])

# Mapping categorical inputs
gender_map = {"Male": 0, "Female": 1}
education_map = {"Graduate": 0, "Not Graduate": 1}
home_map = {"Own": 0, "Rent": 1, "Mortgage": 2}
loan_intent_map = {"Personal": 0, "Debt Consolidation": 1, "Education": 2, "Home Improvement": 3}
defaults_map = {"No": 0, "Yes": 1}

# Create DataFrame for prediction
features = pd.DataFrame([{
    'person_income': income,
    'person_age': age,
    'credit_score': credit_score,
    'loan_amnt': loan_amount,
    'person_emp_exp': emp_exp,
    'loan_int_rate': loan_int_rate,
    'cb_person_cred_hist_length': cred_hist_length,
    'person_gender': gender_map[gender],
    'person_education': education_map[education],
    'person_home_ownership': home_map[home_ownership],
    'loan_intent': loan_intent_map[loan_intent],
    'previous_loan_defaults_on_file': defaults_map[previous_defaults],
    'loan_percent_income': (loan_amount/income)*100
}])

# Apply scaler only to numeric features used during training
numeric_cols = ['person_income','person_age','credit_score','loan_amnt','person_emp_exp',
                'loan_int_rate','cb_person_cred_hist_length','loan_percent_income']
features_scaled = features.copy()
features_scaled[numeric_cols] = loaded_scaler.transform(features[numeric_cols])

# Predict when button is clicked
if st.button("Predict Loan Status"):
    probability = loaded_model.predict_proba(features_scaled)
    st.subheader("Prediction Result")
    st.write(f"Probability of Approval: **{probability[0][1]*100:.2f}%**")
    st.write(f"Probability of Rejection: **{probability[0][0]*100:.2f}%**")

    if probability[0][1] > 0.5:
        st.success("Loan is **likely Approved**")
    else:
        st.error("Loan is **likely Rejected**")
