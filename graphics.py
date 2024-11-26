import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_data():
    snsdata = pd.read_csv("Iris.csv").drop(['Id'], axis=1)
    g = sns.pairplot(snsdata, hue='Species', markers='x')
    g = g.map_upper(plt.scatter)
    g = g.map_lower(sns.kdeplot)
    plt.savefig(f"data.png")
    plt.show()

def plot_linear(X, y, linear_model):
    x_values = np.linspace(X['PetalLengthCm'].min(), X['PetalLengthCm'].max())
    y_values = -(-0.5 + linear_model.intercept_ + linear_model.coef_[0] * x_values) / linear_model.coef_[1]
    plt.scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=y)
    plt.plot(x_values, y_values, color='green')
    plt.xlabel('PetalLengthCm')
    plt.ylabel('PetalWidthCm')
    plt.show()

def plot_ridge(X, y, ridge_model):
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = -(0.5-ridge_model.intercept_ + ridge_model.coef_[0] * x_values) / ridge_model.coef_[1]
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.plot(x_values, y_values, color='blue', label='Разделяющая прямая')
    plt.xlabel('PetalLengthCm (scaled)')
    plt.ylabel('PetalWidthCm (scaled)')
    plt.title('Линейная регрессия с Ridge регуляризацией')
    plt.legend()
    plt.show()
