from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# Модель линейной регрессии
def create_linear_model():
    return LinearRegression()

# Модель логистической регрессии
def create_logistic_model(random_state, C):
    return LogisticRegression(
        random_state=random_state,
        penalty='l2',
        C=C
    )

# Модель линейной регрессии c L2 регуляризацией
def create_linear_model_Ridge(alpha):
    return Ridge(alpha = alpha)
