import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn import preprocessing
import math
import numpy as np

df = pd.read_csv('datasets/GOOGL.csv')

df = df[['Adj. Open', 'Adj. High', 'Adj. Close', 'Adj. Low', 'Adj. Volume']]

forecast_index = math.floor(len(df) * 0.01)
forecast_data = df['Adj. Close'].shift(-forecast_index)

dataframe = df
dataframe['Forecast'] = forecast_data
dataframe['HL_PCT'] = (dataframe['Adj. High'] - dataframe['Adj. Close']) / dataframe['Adj. Close'] * 100.0
dataframe['PCT_Change'] = (dataframe['Adj. Close'] - dataframe['Adj. Open']) / dataframe['Adj. Open'] * 100.0
dataframe = dataframe[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Forecast']]
dataframe.dropna(inplace=True)

X = np.array(dataframe.drop(['Forecast'], 1))
y = np.array(dataframe['Forecast'])
X = preprocessing.scale(X)
# X = X[:-forecast_index]
X_recent = X[-forecast_index:]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
confidence = classifier.score(X_test, y_test) * 100.0
print('\n\nConfidence:- {}'.format(confidence))

forecast_set = classifier.predict(X_recent)

indices = []

for i in range(len(X)):
	indices.append(i)

dataframe['Adj. Close'].plot()
plt.plot([len(dataframe['Adj. Close']) + x for x in range(forecast_index)], forecast_set, c="r")
plt.show()
