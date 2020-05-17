from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# read the data into a pandas datafrome
def main():
    df_train = pd.read_csv('train.csv')
    print(df_train.shape)
    X = df_train.drop('label', axis=1)
    y = df_train['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    ans_k = 0

    k_range = range(1, 8)
    scores = []

    for k in k_range:
        print("k = " + str(k) + " begin ")
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_vali)
        accuracy = accuracy_score(y_vali, y_pred)
        scores.append(accuracy)
        end = time.time()
        print(classification_report(y_vali, y_pred))
        print(confusion_matrix(y_vali, y_pred))

        print("Complete time: " + str(end - start) + " Secs.")
    print(scores)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K')
    plt.ylabel('Testing accuracy')
    plt.show()

    # knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=3, n_jobs=-1)
    # knn_clf.fit(X_train, y_train)
    # y_knn_pred = knn_clf.predict(X_test)
    #
    # accuracy_score(y_test, y_knn_pred)


main()
