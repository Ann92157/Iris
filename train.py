from data import load_data, split_data
from graphics import visualize_data
from graphics import plot_linear, plot_ridge, roc_auc_curve
from models import create_linear_model, create_logistic_model, create_linear_model_Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import yaml


#Загрузка конфига
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

#Обучение моделей
def train_models(config):

    #Визуализация исходных данных
    visualize_data()       
    df = pd.read_csv('Iris.csv')
    data_for_binary = df.drop(index=df.index[df['Species'] == 'Iris-setosa'])
    data_for_binary['Species'].replace({'Iris-versicolor':0, 'Iris-virginica':1}, inplace = True)
    X = data_for_binary[['PetalLengthCm', 'PetalWidthCm']] #отбираем признаки, наиболее скореллированные с видом по матрице корреляции
    y = data_for_binary['Species']
    X_train, X_test, y_train, y_test = split_data(
        X, y, config["test_size"], config["random_state"]
    )

    # Обучение линейной регрессии
    linear_model = create_linear_model()
    linear_model.fit(X_train, y_train)
    y_pred_lin = linear_model.predict(X_test)
    y_pred_class_lin = (y_pred_lin >= 0.5).astype(int) #порог для разделения классов
    accuracy_lin = accuracy_score(y_test, y_pred_class_lin)
    precision_lin = precision_score(y_test, y_pred_class_lin)
    recall_lin = recall_score(y_test, y_pred_class_lin)
    
    #Вывод метрик
    print(f"Линейная регрессия - Accuracy: {accuracy_lin:.2f}")
    print(f"Линейная регрессия - Precision: {precision_lin:.2f}")
    print(f"Линейная регрессия - Recall: {recall_lin:.2f}")

    plot_linear(X, y, linear_model)
    
    # Обучение логистической регрессии
    logistic_model = create_logistic_model(
        config["random_state"], config["C"]
    )

    logistic_model.fit(X_train, y_train)
    lr_y_pred = logistic_model.predict(X_test)
    accuracy_log = accuracy_score(y_test, lr_y_pred)
    precision_log = precision_score(y_test, lr_y_pred)
    recall_log = recall_score(y_test, lr_y_pred)

    #Вывод метрик
    print(f"Логистическая регрессия - Accuracy: {accuracy_log:.2f}")
    print(f"Логистическая регрессия - Precision: {precision_log:.2f}")
    print(f"Логистическая регрессия - Recall: {recall_log:.2f}")

    y_prob = logistic_model.predict_proba(X_test)[:, 1]
    roc_auc_curve(y_test, y_prob)

    # Обучение линейной регрессии c L2-регуляризацией
    Ridge_model = create_linear_model_Ridge(alpha = config["C"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, config["test_size"], config["random_state"]
    )
    Ridge_model.fit(X_train, y_train)
    y_pred = Ridge_model.predict(X_test)
    y_pred_class = (y_pred >= 0.5).astype(int)

    accuracy_ridge = accuracy_score(y_test, y_pred_class_lin)
    precision_ridge = precision_score(y_test, y_pred_class_lin)
    recall_ridge = recall_score(y_test, y_pred_class_lin)
    
    print(f"Линейная регрессия с регуляризацией- Accuracy: {accuracy_ridge:.2f}")
    print(f"Линейная регрессия с регуляризацией - Precision: {precision_ridge:.2f}")
    print(f"Линейная регрессия с регуляризацией - Recall: {recall_ridge:.2f}")

    plot_ridge(X_scaled, y, Ridge_model)

if __name__ == "__main__":
    config = load_config(r"config.yaml")
    train_models(config)
