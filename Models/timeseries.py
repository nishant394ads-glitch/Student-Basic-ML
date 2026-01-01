import numpy as np
from sklearn.linear_model import LinearRegression

def forecast_scores(scores, steps=3):
    time = np.arange(len(scores)).reshape(-1, 1)
    scores = np.array(scores).reshape(-1, 1)

    model = LinearRegression()
    model.fit(time, scores)

    future_time = np.arange(len(scores), len(scores) + steps).reshape(-1, 1)
    forecast = model.predict(future_time)

    return forecast.flatten()
