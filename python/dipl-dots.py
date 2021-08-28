import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering as AGL
from sklearn.cluster import OPTICS
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection


x = []
y = []
fix = []
num_clusters = 16

def main():
    global num_clusters
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
    print(arr.ndim) #the number of axes, or dimensions, of the array
    print(arr.shape) #you have a 2-D array with 2 rows and 3 columns, the shape of your array is (2, 3).
    print(arr.size) #will tell you the total number of elements of the array. This is the product of the elements of the array’s shape.

    #####################KMEANS##################################

    # print("The lowest SSE value",kmeans.inertia_)
    # print("Final locations of the centroid\n",kmeans.cluster_centers_)
    # print("The number of iterations required to converge",kmeans.n_iter_)

    # clustered = False
    # while not clustered:
    #     kmeans = KMeans(n_clusters=num_clusters,
    #                     n_init=20,
    #                     max_iter=600,
    #                     algorithm='full',
    #                     random_state=None)
    #     identified_clusters = kmeans.fit_predict(arr)
    #     # clustering = AGL(n_clusters=num_clusters,linkage="single",affinity="manhattan").fit(arr)
    #     # identified_clusters = clustering.labels_
    #     print("Cluster assignments:\n",identified_clusters)
    #
    #     for i in range(len(identified_clusters)): #starting from 1 to make it from 0
    #         identified_clusters[i] = identified_clusters[i]-1
    #
    #     dots_per_cluster = np.zeros(num_clusters)
    #     for i in range(len(identified_clusters)):
    #         dots_per_cluster[identified_clusters[i]] = dots_per_cluster[identified_clusters[i]] + 1
    #
    #     new_num_clusters = num_clusters
    #     for i in range(len(dots_per_cluster)):
    #         if dots_per_cluster[i] > 20:
    #             new_num_clusters = new_num_clusters + 1
    #             break
    #     if new_num_clusters == num_clusters: #every cluster has <=15 dots
    #         clustered = True
    #     else:
    #         num_clusters = new_num_clusters
    #
    #
    # sum = 0
    # for i in range(len(dots_per_cluster)):
    #     sum = sum + dots_per_cluster[i]
    # print("Final cluster numbers: ", dots_per_cluster, "\nsum: ", sum)
    ##############################################################



    # df = pd.DataFrame(arr, columns=['x','y'])
    # df['clusters'] = identified_clusters
    # print("df types: ",df.dtypes)
    # print("df ndim: ",df.ndim)
    # print("df shape: ",df.shape)
    # print("df size: ",df.size)


    ################DBSCAN############################
    dbscan = DBSCAN(eps=11, min_samples=5, metric='euclidean').fit(arr)
    labels = dbscan.labels_
    print("labels_:", labels)

    df = pd.DataFrame(arr, columns=['x','y'])
    df['clusters'] = labels

    # #get only unique values
    # unique_cluster_numbers = []
    # for i in range(len(labels)):
    #     if labels[i] not in unique_cluster_numbers:
    #         unique_cluster_numbers.append(labels[i])
    #
    # print("Unique clusters:\n",unique_cluster_numbers)
    #
    # for i in range(len(unique_cluster_numbers)):
    #     fw = open(f'C:\\Users\\mn170387d\\Desktop\\clusters\\DBSCANcluster{unique_cluster_numbers[i]}.txt', "w+")
    #     num = 0
    #     for j in range(len(labels)):
    #         if(labels[j]==unique_cluster_numbers[i]):
    #             num = num + 1
    #     fw.write(f'{num}\n') #we got number of holes we will include in this file
    #     for j in range(len(labels)):
    #         if labels[j]==unique_cluster_numbers[i]:
    #             fw.write(f'{arr[j][0]} {arr[j][1]}\n')
    #     fw.close()


    #####################################################
    ##################AgglomerativeClustering###########
    # clustering = AGL(n_clusters=15,linkage="single",affinity="manhattan").fit(arr)
    # labels = clustering.labels_
    # df = pd.DataFrame(arr, columns=['x','y'])
    # df['clusters'] = labels
    ####################################################
    #################OPTICS#############################
    # clustering = OPTICS(min_samples=2, max_eps=11.5, metric='euclidean', cluster_method='dbscan', min_cluster_size=8).fit(arr)
    # labels = clustering.labels_
    # print("labels_:", labels)
    # df = pd.DataFrame(arr, columns=['x', 'y'])
    # df['clusters'] = labels
    #####################################################
    # kmeans = KMeans(n_clusters=27,
    #                 n_init=20,
    #                 max_iter=600,
    #                 algorithm='auto',
    #                 random_state=None)
    # identified_clusters = kmeans.fit_predict(arr)
    # df = pd.DataFrame(arr, columns=['x', 'y'])
    # df['clusters'] = identified_clusters

    plt.figure(1)
    plt.scatter(df['x'],df['y'],c=df['clusters'], cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    main()