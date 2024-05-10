import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

data = pd.read_csv('Percept1.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

accuracy = perceptron.accuracy_score(y_test, y_pred)
print(accuracy)

