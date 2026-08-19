# Telecom Customer Churn Prediction

An end-to-end machine learning project that analyzes telecom customer churn, identifies high-risk customer segments, and predicts customers likely to leave — helping a business move from reactive churn analysis to proactive retention.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 features)

## Key Business Insights (EDA)

- Overall churn rate: **26.5%**
- Month-to-month contracts churn at **42%** vs just **3%** for two-year contracts
- Electronic check users churn at **45%** vs **15%** for auto-pay users
- First 6 months are the highest-risk window — churn drops sharply after year 1
- Fiber Optic users without Tech Support/Online Security churn the most

## Machine Learning

Three models were trained and compared — Logistic Regression, Random Forest, and XGBoost — with **Recall** prioritized over Accuracy, since missing an actual churner is costlier to the business than a false alarm.

The initial XGBoost model underperformed the simpler baseline. After hyperparameter tuning (RandomizedSearchCV, optimized for Recall), tuned XGBoost became the best-performing model:

- **Recall:** 81%
- **ROC-AUC:** 0.84

The final model powers a live risk-scoring system that segments customers into **Low / Medium / High / Critical** churn risk.

## Live App

Enter a customer's account and service details and get their churn probability + risk category in real time.

🔗 **[[https://telecom-customer-churn-prediction-9appegf8g3cweb2g8ejupo7.streamlit.app/]]**

## Files

| File | Description |
|---|---|
| `Churn_analysis_EDA.ipynb` | Exploratory data analysis and business insights |
| `Telecom_Churn_ML.ipynb` | Feature engineering, model training, tuning, evaluation |
| `Churn_app.py` | Streamlit app for live churn prediction |
| `Customer_Churn.csv` | Dataset used |

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit

## Author

**Mahesh Thakare**
[LinkedIn](https://www.linkedin.com/in/mahesh-thakare-75817b2a7/) · [GitHub](https://github.com/mahesh735-ai)
