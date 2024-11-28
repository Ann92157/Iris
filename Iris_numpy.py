import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score

def predict(X,  w):
    return np.dot(X, w) #w - вектор весов модели

def gradient_mse(X, y, w, alpha, iterations):
    m = len(X)
    for i in range(iterations):
        predictions = predict(X, w)
        gradient = (2 / m) * np.dot(X.T, (predictions - y))
        w = w - alpha * gradient
    return w

df = datasets.load_iris()
data = pd.DataFrame(data=df['data'], columns=df['feature_names'])
data['target'] = df['target']

data_for_binary = data.drop(index=data.index[data['target'] == 0])
data_for_binary['target'].replace({1: 0, 2: 1}, inplace=True)
X = data_for_binary[['petal length (cm)', 'petal width (cm)']] #отбираем признаки, наиболее скореллированные с видом по матрице корреляции
y = data_for_binary['target']

#разделение выборки на обучающую и тестовую
ratio = 0.8
total_rows = data_for_binary.shape[0]
train = int(ratio * total_rows)

X_train = X[0:train]
X_test = X[train:]
y_train = y[0:train]
y_test = y[train:]

X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

#обучение модели
alpha = 0.01
iterations = 1000
w = np.zeros(X_train.shape[1])

w = gradient_mse(X_train, y_train, w, alpha, iterations)

y_pred = predict(X_test, w)
y_pred_class = (y_pred >= 0.5).astype(int)

print(f"Коэффициенты модели: {w}") 

#вывод метрик модели
accuracy_lin = accuracy_score(y_test, y_pred_class)
precision_lin = precision_score(y_test, y_pred_class)
recall_lin = recall_score(y_test, y_pred_class)
print(f"Линейная регрессия - Accuracy: {accuracy_lin:.2f}")
print(f"Линейная регрессия - Precision: {precision_lin:.2f}")
print(f"Линейная регрессия - Recall: {recall_lin:.2f}")
