import numpy as np
import matplotlib.pyplot as plt
import math, operator
import pandas as pd

def JarakEuclidean1301150031(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def LabelCluster1301150031(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def CentroidsBaru1301150031(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def RumusKMeans1301150031(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = JarakEuclidean1301150031(data_points[index_point], centroids[index_centroid])
            label = LabelCluster1301150031(distance, data_points[index_point], centroids)
            centroids[label[0]] = CentroidsBaru1301150031(label[1], centroids[label[0]])
            if iteration == (total_iteration - 1):
                cluster_label.append(label)
    return [cluster_label, centroids]


def PrintGraph1301150031(result,centroids):
    plt.rcParams['figure.figsize'] = (7,7)
    plt.style.use('ggplot')
    #data_train = pd.read_csv('TrainsetTugas2.csv')
    for output in result[0]:
        test = output[1];
        x1=test[0]
        x2=test[1]
        if output[0]==0:
            plt.scatter(x1,x2, c='r', s=10)
        elif output[0]==1:
            plt.scatter(x1,x2, c='b', s=10)
        else:
            plt.scatter(x1,x2, c='y', s=10)
    count=0
    while count<3:
        output1=centroids[count]
        y1=output1[0]
        y2=output1[1]
        plt.scatter(y1,y2, c='black', s=100)
        count +=1

def BuatCentroids1301150031():
    centroids = []
    centroids.append([15.0, 15.0])
    centroids.append([20.0, 25.0])
    centroids.append([25.0, 30.0])
    return np.array(centroids)


filename = os.path.dirname(__file__) + "\TrainsetTugas2.csv"
data = np.genfromtxt(filename, delimiter=",")
centroids = BuatCentroids1301150031()
total_iteration = 100

[cluster_label, new_centroids] = RumusKMeans1301150031(data, centroids, total_iteration)
PrintGraph1301150031([cluster_label, new_centroids],centroids)
  
print()
    
