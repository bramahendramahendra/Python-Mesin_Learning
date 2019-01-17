import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.cluster import KMeans

#import dataset
dataset = pd.read_csv('TestsetTugas2.csv')

X = dataset.iloc[:,[1,2]]

#optimal number of cluster
wcss = []

for i in range(1,20):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plot.plot(range(1,20),wcss)
plot.title('Elbow Method')
plot.xlabel('Number of clusters')
plot.ylabel('WCSS')
plot.show()