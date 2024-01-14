from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


def train():
    train_ds = pd.read_csv('train_data.csv')

    X = train_ds.iloc[:, 3:]
    y = train_ds['Survived']

    model = LogisticRegression()

    model.fit(X, y)

    test_ds = pd.read_csv('test_data.csv')
    X_test = test_ds.iloc[:, 3:]
    y_test = test_ds['Survived']
    score = model.score(X_test, y_test)

    print(score)


if __name__ == '__main__':
    train()
