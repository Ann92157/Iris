from data import load_data, split_data
from graphics import visualize_data
from models import create_linear_model, create_logistic_model
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml
import kagglehub
import os

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

if __name__ == "__main__":
    config = load_config(r"config.yaml")
    train_models(config)
