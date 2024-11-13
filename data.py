import pandas as pd
from sklearn.model_selection import train_test_split

#загрузка данных
def load_data(data_path):
    data = pd.read_csv(data_path)
    #для бинарной классификации выбираем классы Versicolor и Virginica, т.к. они наименее различимы
    data = data.drop(index=data.index[data['Species'] == 'Iris-setosa'])
    X = data[['PetalLengthCm', 'PetalWidthCm']]
    data['Species'].replace({'Iris-versicolor':0, 'Iris-virginica':1}, inplace = True)
    y = data["Species"]
    return X, y

#Разделение на тестовую и обучающую выборки
def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
