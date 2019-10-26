from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k: 
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) #Faster cheat version of the distance formula
            distances.append([euclidean_distance, group])
    #i1 is the group up to k
    votes = [i[1] for i in sorted(distances) [:k]]
    #Most common comes as an array of a list, so you take 0 first and you get 
    #the list and then you take the 0th again so it gives the most common group and how many there were ("it basically just returns r or k")
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1) [0] [0]
    confidence = Counter(votes).most_common(1) [0] [1] / k
    #print(vote_result, confidence)
    return vote_result, confidence
accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    #Reason for this is because some come through as strings
    full_data = df.astype(float).values.tolist()
    #Shuffles our data
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int((test_size)*len(full_data))] #first 80% of the data
    test_data = full_data[-int((test_size)*len(full_data)):] #last 20% of the data

    #populating out dictionaries
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) #last value (benign or malignant) is not appended

    for i in test_data:
        test_set[i[-1]].append(i[:-1]) #last value (benign or malignant) is not appended

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set, data, k=5) #k is number of datasets to be trained by
            if group == vote:
                correct +=1
            #else:
                #print(confidence)
            total += 1

    #print('Accuracy: ', correct/total)
    accuracies.append(correct/total)
print(sum(accuracies) / len(accuracies))