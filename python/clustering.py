import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering as AGL


x = []
y = []
fix = []

def main():
    xy = []
    # with open('tackice.txt', 'r') as reader:
    #     line = reader.readline()
    #     while line != '':
    #         print(line)
    #         line = reader.readline()


    with open('tackice.txt', 'r') as reader:
        for line in reader:
            xy = line.split()
            fix.append([float(xy[0]), float(xy[1])])

    arr = np.array(fix)
    dbarr = arr.copy()
    print(arr.ndim) #the number of axes, or dimensions, of the array
    print(arr.shape) #you have a 2-D array with 2 rows and 3 columns, the shape of your array is (2, 3).
    print(arr.size) #will tell you the total number of elements of the array. This is the product of the elements of the array’s shape.

    #####################KMEANS##################################
    # kmeans = KMeans(n_clusters=16,
    #                 init="k-means++",
    #                 n_init=20,
    #                 max_iter=600,
    #                 random_state=None)
    # kmeans.fit(arr)
    # print("The lowest SSE value",kmeans.inertia_)
    # print("Final locations of the centroid",kmeans.cluster_centers_)
    # print("The number of iterations required to converge",kmeans.n_iter_)
    #
    # #print("Cluster assignments0",kmeans.labels_)
    #
    # identified_clusters = kmeans.fit_predict(arr)
    # print("Cluster assignments",identified_clusters)
    ##############################################################



    # df = pd.DataFrame(arr, columns=['x','y'])
    # print("df types: ",df.dtypes)
    # print("df ndim: ",df.ndim)
    # print("df shape: ",df.shape)
    # print("df size: ",df.size)
    #
    # df['clusters'] = identified_clusters

    ################DBSCAN############################
    # dbscan = DBSCAN(eps=11.4,min_samples=5).fit(dbarr)
    # labels = dbscan.labels_
    # print("labels_:", labels)
    #
    # df = pd.DataFrame(arr, columns=['x','y'])
    #
    # df['clusters'] = labels
    #####################################################
    ##################AgglomerativeClustering###########
    clustering = AGL(n_clusters=15,linkage="single",affinity="manhattan").fit(arr)
    labels = clustering.labels_
    df = pd.DataFrame(arr, columns=['x','y'])
    df['clusters'] = labels
    ####################################################

    plt.figure(1)
    plt.scatter(df['x'],df['y'],c=df['clusters'], cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    main()