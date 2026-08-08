# Customer Churn Prediction API

A machine learning project that predicts whether a telecom customer is likely to cancel their subscription (churn), and serves that prediction through a live REST API.

## Problem

Telecom companies lose revenue when customers cancel their service. If you can predict which customers are at risk *before* they leave, the business can proactively offer retention deals to the right people instead of guessing.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — ~7,000 customers, with account details, services subscribed, and whether they churned.

## Approach

1. **Data cleaning** — fixed a data quality issue where `TotalCharges` was stored as text due to blank values for new customers; converted to numeric and filled missing values.
2. **Exploratory analysis** — found that month-to-month contracts, low tenure, and high monthly charges were the strongest churn signals.
3. **Preprocessing pipeline** — built with scikit-learn: numeric features scaled, categorical features one-hot encoded.
4. **Modeling** — trained and compared two models:
   - Logistic Regression (baseline)
   - Random Forest (final model — selected for better recall on churners)
5. **API** — wrapped the trained model in a FastAPI application with a `/predict` endpoint that returns a churn prediction and probability score for any customer.

## Results

| Model | Recall (Churn) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.61 | 0.840 |
| Random Forest | 0.79 | — |

Recall was prioritized over plain accuracy, since missing an at-risk customer (false negative) is more costly to the business than a false alarm — and the dataset is imbalanced (~27% churn).

## Project structure
