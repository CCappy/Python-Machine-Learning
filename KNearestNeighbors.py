from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

#euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

#Two different catagories(labels), k and r
dataset = {'k' : [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

#This warns the user when they are trying to do something stupid.
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
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1) [0] [0]
    return vote_result

results = k_nearest_neighbors(dataset, new_features, k=3)
print(results)
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)
plt.scatter(new_features[0], new_features[1], color=results, s =100)
plt.show()