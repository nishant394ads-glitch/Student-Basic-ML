import pandas as pd
from sklearn.linear_model import LinearRegression

def train_regression():
    data = pd.read_csv("data/student_performance.csv")

    X = data[['hours_studied', 'attendance', 'internal_marks']]
    y = data['final_score']

    model = LinearRegression()
    model.fit(X, y)

    return model
