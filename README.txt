# 🚀 SaaS User Behavior Funnel Analysis & Churn Prediction Dashboard

A professional **Data Analytics + Machine Learning project** focused on analyzing **user behavior in a SaaS product**, identifying **drop-off points in the conversion funnel**, performing **cohort and segmentation analysis**, and building a **churn prediction model** using Machine Learning.

This project was developed as part of a **Data Analytics Internship Task** and includes an interactive **Streamlit dashboard** for real-time business insights.

---

## 📌 Project Objective

The goal of this project is to analyze user journey data in a SaaS product and identify:

- Where users drop off in the funnel
- Which acquisition channels perform better
- How different user cohorts behave
- What factors contribute to churn
- How to predict churn using Machine Learning

---

## 🎯 Business Problem

A SaaS company is facing **high churn after the free trial phase** and wants to understand:

- Why users leave
- Which stages create friction
- Which users are more likely to upgrade
- How acquisition channels affect retention

This project solves these business questions using **Data Analytics, Visualization, and Predictive Modeling**.

---

## 📊 Features

### ✅ Data Cleaning & Preparation
- Handle missing values
- Remove duplicate records
- Convert timestamps
- Prepare event-level SaaS data

### ✅ Funnel Analysis
5-Step Conversion Funnel:

1. Landing
2. Sign-up
3. Onboarding
4. First Feature Use
5. Upgrade

Includes:
- Conversion Rate Analysis
- Drop-Off Rate Analysis
- Funnel Visualization

### ✅ Cohort Analysis
- User grouping by acquisition month
- Upgrade behavior comparison
- Monthly retention analysis

### ✅ Segmentation Analysis
Compare performance across:

- Organic Users
- Paid Users
- Referral Users

Additional segmentation:
- Country-wise analysis
- Device-wise analysis

### ✅ Churn Prediction (Machine Learning)
Predict whether a user is likely to churn using:

- Random Forest Classifier
- Feature Importance Analysis
- Accuracy Evaluation
- Live Prediction Interface

### ✅ Interactive Dashboard
Professional **Streamlit dashboard** with:

- KPI Cards
- Interactive Filters
- Funnel Analytics
- Cohort Heatmaps
- Segmentation Charts
- Churn Prediction
- Business Insights

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- Streamlit

---

## 📂 Project Structure

```text
SaaS-User-Funnel-Analysis/
│
├── data/
│   ├── saas_user_behavior_dataset.csv
│   └── cleaned_saas_user_behavior_dataset.csv
│
├── notebook/
│   └── analysis.ipynb
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   ├── funnel_analysis.csv
│   ├── cohort_analysis.csv
│   ├── segmentation_analysis.csv
│   ├── kpi_metrics.csv
│   └── churn_prediction_results.csv
│
├── screenshots/
│   ├── dashboard.png
│   ├── funnel_analysis.png
│   └── churn_prediction.png
│
├── requirements.txt
│
└── README.md