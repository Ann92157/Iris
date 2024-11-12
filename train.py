from data import load_data, split_data
from models import create_linear_model, create_logistic_model
import yaml

#Загрузка конфига
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

#Обучение моделей
def train_models(config):
    X, y = load_data("Iris.csv")
    X_train, X_test, y_train, y_test = split_data(
        X, y, config["test_size"], config["random_state"]
    )

    # Обучение линейной регрессии
    linear_model = create_linear_model()
    linear_model.fit(X_train, y_train)
    print("Uspekh")
    
    # Обучение логистической регрессии
    logistic_model = create_logistic_model(
        config["random_state"], config["C"], config["solver"]
    )
    logistic_model.fit(X_train, y_train)
    print("Uspekh")
