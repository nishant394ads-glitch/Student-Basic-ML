import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train_classifier():
    data = pd.read_csv("data/student_performance.csv")

    # 1 = Safe, 0 = At Risk
    data['risk'] = data['final_score'].apply(lambda x: 1 if x >= 60 else 0)

    X = data[['hours_studied', 'attendance', 'internal_marks']]
    y = data['risk']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model
