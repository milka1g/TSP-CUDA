import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering as AGL
from sklearn.cluster import OPTICS


x = []
y = []
fix = []

def main():
    xy = []

    with open('cluster0.txt', 'r') as reader:
        for line in reader:
            xy = line.split()
            fix.append([float(xy[0]), float(xy[1])])

    arr = np.array(fix)
    print(arr.ndim) #the number of axes, or dimensions, of the array
    print(arr.shape) #you have a 2-D array with 2 rows and 3 columns, the shape of your array is (2, 3).
    print(arr.size) #will tell you the total number of elements of the array. This is the product of the elements of the array’s shape.

    dbscan = DBSCAN(eps=10, min_samples=5, metric='euclidean').fit(arr)
    labels = dbscan.labels_
    print("labels_:", labels)

    df = pd.DataFrame(arr, columns=['x', 'y'])
    df['clusters'] = labels


    xfrom = 0
    xto = 300
    yfrom = -100
    yto = 0

    plt.figure(1, figsize=(16,9), dpi=100)
    plt.scatter(df['x'], df['y'], c=df['clusters'], cmap='rainbow')
    #,c=df['clusters'], cmap='rainbow'
    plt.axvline(150)
    plt.axhline(-50)
    axes = plt.gca()
    axes.set_ylim(yfrom,yto)
    axes.set_xlim(xfrom,xto)
    # print("HAHAA",axes.get_ylim(), axes.get_xlim())
    plt.show()


if __name__ == '__main__':
    main()