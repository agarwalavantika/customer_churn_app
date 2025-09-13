# Customer Churn Prediction App

An interactive Streamlit app that predicts whether a telecom customer will churn.  
Includes SHAP explainability to show feature impact for each prediction.

## Features
- Predict churn probability for a single customer
- Visual explanations (SHAP plots)
- User-friendly interface

### Input Form
<img width="1876" height="823" alt="image" src="https://github.com/user-attachments/assets/82c63198-d616-4e67-bd0f-f257614eb8b1" />


### Prediction Result
<img width="1818" height="758" alt="image" src="https://github.com/user-attachments/assets/9c847999-288b-4766-9dcb-094e6e32dc97" />


### Feature Impact (SHAP)
<img width="1460" height="996" alt="image" src="https://github.com/user-attachments/assets/5e22d79e-5b70-4758-9527-d379382b1eb7" />

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
