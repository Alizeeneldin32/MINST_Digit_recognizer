from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# read the data into a pandas datafrome
def main():
    df_train = pd.read_csv('train.csv')
    print(df_train.shape)
    X = df_train.drop('label', axis=1)
    y = df_train['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=42)

    nb = MultinomialNB()

    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
main()