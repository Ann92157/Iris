import pandas as pd
from sklearn.model_selection import train_test_split

#Разделение на тестовую и обучающую выборки
def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
