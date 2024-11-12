from sklearn.linear_model import LinearRegression, LogisticRegression
# Модель линейной регрессии
def create_linear_model():
    return LinearRegression()

# Модель логистической регрессии
def create_logistic_model(random_state, C):
    return LogisticRegression(
        random_state=random_state,
        C=C
    )
