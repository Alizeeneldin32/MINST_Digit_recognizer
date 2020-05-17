from sklearn.cluster import KMeans
import pandas as pd
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.metrics import confusion_matrix
#from sklearn.datasets.samples_generator import make_blobs

# read the data into a pandas dataframe
def main():
    df_train = pd.read_csv('train.csv')
    print(df_train.shape)
    X = df_train.drop('label', axis=1)
    y = df_train['label']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    #                                                     random_state=42)
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    # accuracy = accuracy_score(y_test, kpredict)
    # print(accuracy)
    cm = confusion_matrix(y, y_kmeans)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()


main()
