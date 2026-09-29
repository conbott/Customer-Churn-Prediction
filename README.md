# Bank Customer Churn Prediction System

An end-to-end machine learning system designed to predict customer churn in retail banking, assess attrition probabilities, and recommend data-driven retention actions.

---

## 1. Project Overview

Customer attrition poses a significant financial risk to commercial and retail banking institutions. Identifying at-risk customers prior to account closure allows retention teams to intervene proactively.

This project implements a complete classification pipeline that:
- Ingests and processes customer demographic and transactional data (>160,000 records).
- Applies rigorous data preprocessing, scaling, and categorical encoding.
- Trains, evaluates, and compares multiple classification models using stratified validation.
- Deploys the optimal model through an interactive Streamlit decision-support application.

---

## 2. Dataset Information

The model was developed using a large-scale retail banking dataset consisting of 164,911 unique customer records after deduplication.

### Key Features:
- **Demographics:** `Geography` (France, Spain, Germany), `Gender`, `Age`.
- **Financial Profile:** `CreditScore`, `Balance`, `EstimatedSalary`.
- **Engagement Metrics:** `Tenure` (years with bank), `NumOfProducts`, `HasCrCard`, `IsActiveMember`.
- **Target Variable:** `Exited` (0 = Retained, 1 = Churned).
- **Target Distribution:** Imbalanced (~21.16% Churn Rate).

---

## 3. Exploratory Data Analysis Insights

1. **Geographic Concentration:** German customers exhibited a churn rate exceeding 32%, compared to approximately 16% in France and Spain.
2. **Age Correlation:** The median age of churned customers was 44–46 years, compared to 36–37 years for retained customers.
3. **Product Holding Dynamics:** Customers holding 2 products demonstrated the lowest attrition rate (<8%), while customers holding 3 or 4 products showed an extreme churn rate exceeding 80%.

---

## 4. Modeling & Performance Evaluation

Models were evaluated using Stratified 80/20 train/test splits to maintain class balance across sets. Evaluated metrics included Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Performance Summary:

| Metric | Logistic Regression | Random Forest Classifier |
| :--- | :--- | :--- |
| **Train Accuracy** | 83.48% | 87.42% |
| **Test Accuracy** | 83.46% | 86.51% |
| **Precision** | 71.82% | 74.21% |
| **Recall** | 39.11% | 55.84% |
| **F1-Score** | 50.64% | 63.73% |
| **ROC-AUC** | 0.8241 | 0.8876 |

### Model Selection:
The **Random Forest Classifier** was selected as the final production model due to its superior discriminative performance (ROC-AUC 0.8876 vs. 0.8241) and substantially higher Recall, ensuring a lower rate of missed churn cases.

---

## 5. Repository Structure

```text
CustomerChurn_Project/
├── saved_models/
│   └── customer_churn_pipeline.pkl   # Serialized end-to-end pipeline
├── Customer_churn_pred.ipynb         # Full analysis, EDA, training, and evaluation
├── churn_app.py                      # Interactive Streamlit application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
