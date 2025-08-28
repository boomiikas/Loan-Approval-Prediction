# 🏦 Loan Approval Prediction  

This project predicts whether a bank should **approve or reject** a loan application using applicant data such as **income, age, credit score, and loan amount**.  
A **Random Forest Classifier** is trained on historical loan data to make predictions.  

🔗 **Live Demo**: [Loan Approval Prediction App](https://loan-approval-prediction-b5.streamlit.app/)  

---

## 📌 Features Used
- **income**: Applicant’s annual income  
- **age**: Applicant’s age  
- **credit_score**: Applicant’s credit score  
- **loan_amount**: Requested loan amount  

**Target Variable** → `loan_status` (Approved / Rejected)  

---

## ⚙️ Workflow
1. **Data Loading & Exploration**
   - Load dataset (`loan_approval.csv`)
   - Perform exploratory data analysis (EDA):
     - Summary statistics
     - Missing/duplicate value check
     - KDE plots, Boxplots, Histogram, Pie chart
     - Correlation heatmap  

2. **Preprocessing**
   - Encode target variable (`Approved=1`, `Rejected=0`)  
   - Train-test split (70% train, 30% test)  

3. **Model Training**
   - Random Forest Classifier with 100 estimators  
   - Trained on applicant features  

4. **Prediction**
   - Predict probability of loan approval for new applicants  
   - Example:  

     ```python
     new_applicant = [[55000, 35, 720, 20000]]
     model.predict_proba(new_applicant)
     ```

---

## 📊 Visualizations  

- **KDE plots for numerical features**  
  ![kde1](https://github.com/user-attachments/assets/7ef6d990-6e8c-4aea-9fc7-6b20074d37ca)  
  ![kde2](https://github.com/user-attachments/assets/9d62fdef-87c8-414d-a091-6fd3ee955368)  
  ![kde3](https://github.com/user-attachments/assets/549438f3-4bb2-4bbf-b15e-9ca98ac4df5b)  
  ![kde4](https://github.com/user-attachments/assets/dc0d7516-930e-451b-8c05-97586eb4ffab)  

- **Boxplots to detect outliers**  
  ![box1](https://github.com/user-attachments/assets/5786efaa-89ef-441b-8257-7ef011979522)  
  ![box2](https://github.com/user-attachments/assets/cd1ed421-acad-4d2e-8b30-a53f14860704)  
  ![box3](https://github.com/user-attachments/assets/b3c4b7b2-963c-49c8-b042-d3f5abcf0c35)  
  ![box4](https://github.com/user-attachments/assets/a7df0147-320b-4a43-abf3-384b185eb1fa)  

- **Histogram of credit scores**  
  ![hist](https://github.com/user-attachments/assets/4e081de6-b66f-43e6-bf12-1b0fd37be4ec)  

- **Pie chart of loan status distribution**  
  ![pie](https://github.com/user-attachments/assets/a8500548-8537-4aa0-aad8-a6e3e1e4ccc4)  

- **Correlation heatmap**  
  ![heatmap](https://github.com/user-attachments/assets/6d25697c-87ee-4e80-bc57-4111513d7894)  

---

## 🚀 Example Result
For an applicant with:  
- **Income** = 55,000  
- **Age** = 35  
- **Credit Score** = 720  
- **Loan Amount** = 20,000  

**Model Output:**  
- Probability of Rejection → **0.87**  
- Probability of Approval → **0.13**  

✅ The applicant has a **13% chance of loan approval**.  

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas, NumPy**  
- **Matplotlib, Seaborn**  
- **Scikit-learn**  

---

## 📌 Next Steps
- Hyperparameter tuning of Random Forest  
- Try other ML models (Logistic Regression, XGBoost)  
---

## 📸 Screenshots  
<img width="989" height="756" alt="image" src="https://github.com/user-attachments/assets/73bbb323-90eb-43ea-b1c1-e02ee8963aa4" />

---

💡 *This project demonstrates the use of machine learning in financial decision-making, ensuring data-driven and fair loan approvals.*  
