import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from Models.regression import train_regression
from Models.classification import train_classifier
from Models.timeseries import forecast_scores
from Utils.metrics import regression_rmse

st.set_page_config(page_title="Student ML Analytics", layout="wide")

st.title("📊 Student Performance Analytics & Forecasting System")

data = pd.read_csv("data/student_performance.csv")

st.sidebar.header("🎯 Student Input")
hours = st.sidebar.slider("Hours Studied", 0, 12, 6)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 80)
internal = st.sidebar.slider("Internal Marks", 10, 30, 18)

reg_model = train_regression()
clf_model = train_classifier()

score_pred = reg_model.predict([[hours, attendance, internal]])[0]
risk_pred = clf_model.predict([[hours, attendance, internal]])[0]

col1, col2 = st.columns(2) 

with col1:
    st.subheader("📂 Dataset Preview")
    st.dataframe(data)

with col2:
    st.subheader("🎯 Prediction Results")
    st.success(f"📈 Predicted Final Score: {score_pred:.2f}")
    st.success("✅ SAFE" if risk_pred == 1 else "⚠️ AT RISK")

st.subheader("📊 Performance Analysis")

fig, ax = plt.subplots()
ax.scatter(data['hours_studied'], data['final_score'])
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Final Score")
st.pyplot(fig)

forecast = forecast_scores(data['final_score'].values)
st.subheader("⏳ Future Forecast")
st.write(forecast)

y_true = data['final_score']
y_pred = reg_model.predict(data[['hours_studied','attendance','internal_marks']])
rmse = regression_rmse(y_true, y_pred)
st.info(f"📉 Regression RMSE: {rmse:.2f}") 