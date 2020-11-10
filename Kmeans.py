from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
path =  'data.txt'
data = pd.read_csv(path, header=None, names=['x1', 'x2'])
A = data
A = np.matrix(A.values)
def distance(a, b,ax=1):
    return np.linalg.norm(a - b, axis=ax)

def Kmeans(A,k):
    centers = A[:k,:]
    newCenters = np.zeros(centers.shape)
    clusters = np.zeros(len(A))

    while True:
        for i in range(len(A)):
            d = distance(A[i], centers)
            cluster = np.argmin(d)
            clusters[i] = cluster
        for i in range(k):
            points = [A[j] for j in range(len(A)) if clusters[j] == i]
            newCenters[i] = np.mean(points, axis = 0)
        if(centers == newCenters).all():
            break
        else:
            centers = newCenters
    return centers, clusters

centers,clusters = Kmeans(A,3)
print(centers)
print(clusters)
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots(figsize=(15,10))
for i in range(len(A)):
   ax.scatter(A[i].tolist()[0][0],A[i].tolist()[0][1], s=20, c=colors[int(clusters[i])])
ax.scatter(centers[:,0],centers[:,1],c=colors[3])
plt.show()