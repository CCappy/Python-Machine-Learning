import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer.data.txt')
    #Replaces unknown values with an outlier
    df.replace('?', -99999, inplace=True)
    #Drops the id column, it will break our k nearest neighbors
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'], 1)) #X for features (everything except class)
    y = np.array(df['class']) #y for class

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #Shuffles data and separates it into training and testing groups

    #Trains the program
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    #print(accuracy)

    #example_measures = np.array([[4, 2, 1, 1, 1 ,2 ,3, 2,1]])

    #prediction = clf.predict(example_measures)
    #print(prediction)
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))