import numpy as np
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = [
	[4, 8, 15, 16, 23, 42, 1],
	[3, 7, 14, 15, 22, 41, 1],
	[5, 9, 16, 17, 25, 41, 1],
	[1, 2,  3,  4,  5,  6, 0],
	[1, 1,  2,  3,  4,  5, 0],
	[2, 3,  4,  5,  6,  7, 0]
]

data = np.array(data)

X = data[:, 0:6]
y = data[:, 6]

scaler = StandardScaler()

data = scaler.fit_transform(X)

dump(scaler, "../models/scaler.joblib")

rfc = RandomForestClassifier()

rfc = rfc.fit(X, y)

dump(rfc, "../models/rfc.joblib")

y_pred = rfc.predict(X)

print(classification_report(y_pred, y))
