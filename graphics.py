import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

def plot_linear(X, y, linear_model):
    x_values = np.linspace(X['petal length (cm)'].min(), X['petal length (cm)'].max())
    y_values = -(-0.5 + linear_model.intercept_ + linear_model.coef_[0] * x_values) / linear_model.coef_[1]
    plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=y)
    plt.plot(x_values, y_values, color='green')
    plt.xlabel('PetalLengthCm')
    plt.ylabel('PetalWidthCm')
    plt.show()

def plot_ridge(X, y, ridge_model):
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = -(0.5-ridge_model.intercept_ + ridge_model.coef_[0] * x_values) / ridge_model.coef_[1]
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(x_values, y_values, color='blue', label='Разделяющая прямая')
    plt.xlabel('PetalLengthCm (scaled)')
    plt.ylabel('PetalWidthCm (scaled)')
    plt.title('Линейная регрессия с Ridge регуляризацией')
    plt.legend()
    plt.show()

def roc_auc_curve(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='green')
    plt.plot([0, 1], [0, 1], color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
