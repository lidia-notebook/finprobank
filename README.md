#  Bank Customer Classification — 2025 Marketing Focus  
**Created by:** Winona Timothea and Lidia Priskila  
**Reference:** [2025 Bank Marketing Trends — ABA Banking Journal](https://bankingjournal.aba.com/2025/01/2025-bank-marketing-trends/)

---

##  Executive Summary  
This notebook presents an end-to-end **machine learning classification project** developed to support the **2025 banking marketing objective of deposit growth**.  
The project leverages predictive analytics to identify customers most likely to subscribe to a **term deposit**, enhancing marketing precision, optimizing ROI, and strengthening long-term customer relationships.  

The workflow covers the complete lifecycle of a data-driven marketing project — from **data understanding and feature engineering** to **model evaluation and interpretability using SHAP**, followed by **artifact preparation for Streamlit deployment**.

---

##  Project Overview  
In 2025, the **banking industry’s top marketing focus remains deposit growth**, with a strong emphasis on building lasting customer relationships and improving retention.  
This project applies **machine learning** to classify customers based on their likelihood to subscribe to a term deposit, adapting insights from the **Bank Marketing Portugal dataset** to fit modern financial contexts.

---

##  Problem Statement  
Despite heavy investment in outbound marketing campaigns, conversion rates for term deposits remain low.  

**Key Challenges:**
1. **Inefficient Resource Allocation** — High outreach costs to customers unlikely to subscribe.  
2. **Customer Fatigue** — Over-contacting uninterested clients reduces trust and engagement.  

To align with the 2025 goal of **sustainable deposit growth**, this model enables **data-driven targeting**, helping marketing teams reduce campaign waste and focus on high-potential prospects.

---

##  Project Goals  

| **Objective Type** | **Goal Description** |
|:--------------------|:---------------------|
| **Machine Learning Objective** | Build a predictive model to classify customers’ likelihood of subscribing to a term deposit. |
| **Business Objective** | Improve campaign efficiency by minimizing false positives (uninterested customers) and maximizing true positives (real subscribers). |
| **Operational Objective** | Deploy a Streamlit app for marketers to simulate predictions and guide campaign strategies. |

---

##  Methodology and Tools  

**Tools Used**
- **Python** — Core analysis and modeling language  
- **Pandas, NumPy, Scikit-learn** — Data preprocessing, feature engineering, and modeling  
- **Matplotlib, Seaborn, Plotly** — Data visualization and EDA  
- **Joblib, JSON** — Artifact management and reproducibility  
- **Streamlit** — Interactive deployment for stakeholders  

**Model Used:** Logistic Regression (LogReg)  
Chosen for its **interpretability**, **transparency**, and suitability for **business-critical classification tasks**.

---

##  Evaluation Metrics  

| **Metric** | **Definition / Focus** | **Business Relevance** |
|:------------|:------------------------|:------------------------|
| **Recall** | Measures how many actual subscribers were correctly identified. | Ensures potential depositors are not missed. |
| **Precision** | Measures how many predicted “Yes” cases were truly subscribers. | Prevents wasted resources on uninterested contacts. |

> A balanced focus on **recall** and **precision** ensures optimal **ROI** — capturing genuine leads while reducing unnecessary campaign costs.

---

##  Phase Summaries  

| **Phase** | **Title** | **Key Activities** | **Output / Deliverable** |
|:-----------|:-----------|:------------------|:--------------------------|
| **Phase 1 — Data Understanding & Preparation** | Explored dataset structure, handled missing values, and performed targeted EDA. | Cleaned dataset (`bank_marketing_clean.csv`) ready for modeling. |
| **Phase 2 — Feature Engineering & Baseline Modeling** | Created new variables (age bins, job groupings, education levels), handled imbalance, benchmarked models. | Baseline comparison; **Logistic Regression** selected as final model. |
| **Phase 3 — Model Evaluation, Explainability & Insights** | Evaluated model using precision and recall, conducted SHAP analysis, and visualized feature influence. | Explainable model interpretation — key drivers: `emp.var.rate`, `cons.price.idx`, and `contact method`. |

---

##  SHAP Explainability Summary  
**SHAP (SHapley Additive exPlanations)** was applied to interpret how each feature contributed to the model’s predictions.  

- **Positive SHAP values** → Features that **increase** the likelihood of subscription.  
- **Negative SHAP values** → Features that **decrease** the likelihood of subscription.  

This interpretability bridges **data science insights** with **marketing strategy**, allowing teams to understand *why* certain segments respond better, and how to tailor outreach more effectively.



