import numpy as np
import os
import re


NUM_CLUSTERS = 0


def connectCentroidsAndSolution():
    global NUM_CLUSTERS
    csHashmap = {}
    fmeanssolotions = open(f'C:\\Users\\mn170387d\\Desktop\\elki16processed\\means_solutions.txt', "w+")
    with open(f'C:\\Users\\mn170387d\\Desktop\\elki16processed\\solution.txt', 'r') as solution, \
            open(f'C:\\Users\\mn170387d\\Desktop\\elki16processed\\cluster means.txt', 'r') as means:
        order = solution.readline()
        centroid = means.readline()
        while order != '' and centroid != '':
            NUM_CLUSTERS = NUM_CLUSTERS + 1
            orderInt = int(order.rstrip())
            coords = re.findall("\-?\d+\.\d+", centroid)
            centroidCoords = (float(coords[0]), float(coords[1]))
            csHashmap[orderInt] = centroidCoords
            fmeanssolotions.write(centroid.rstrip() + " " + order.rstrip() + '\n')
            order = solution.readline()
            centroid = means.readline()
    fmeanssolotions.close()
    return csHashmap

#\-?\d+\.\d+|\-?\d+ regex that reads floats and ints

def createCentroidToClusterHashmap():
    ccHashmap = {}
    directory = f'C:\\Users\\mn170387d\\Desktop\\elki16processed'
    for filename in os.listdir(directory):
        if 'cluster_' in filename:
            with open(os.path.join(directory, filename), 'r') as cluster:
                holes = []  # list of tuples, each tuple is one hole
                num = int(cluster.readline())  # number of holes, 17etc..
                for i in range(num):
                    line = cluster.readline()
                    coords = re.findall("\-?\d+\.\d+", line)
                    hole = (float(coords[0]), float(coords[1]))  # tuple
                    holes.append(hole)
                line = cluster.readline()
                centroidCoords = re.findall("\-?\d+\.\d+", line)  # key for our dict ccHashmap
                centroid = (float(centroidCoords[0]), float(centroidCoords[1]))
                ccHashmap[centroid] = holes

    return ccHashmap


def findClosestFromNeighborCluster(i, inext, coHmap, ccHmap,
                                   InOut):  # In hole will be first hole of the cluster, Out will be on last place, the parameter InOut tells that-> In==true
    bestNeighbor = 0
    bestDistance = float("inf")
    centroid = coHmap[i]
    centroidNeighbor = coHmap[inext]
    neighborCluster = ccHmap[centroidNeighbor]
    for d in range(
            len(neighborCluster)):  # for every hole in neighbor cluster we compute distance to i-th centroid and select the closest hole from that cluster
        distance = np.linalg.norm(np.array(centroid) - np.array(neighborCluster[d]))
        if distance < bestDistance:
            bestDistance = distance
            bestNeighbor = d
    # now we have to put that in the first/last place in our neighbor cluster, that will be fixed in CUDA kernel
    if InOut:
        temp = neighborCluster[0]
        ccHmap[centroidNeighbor][0] = ccHmap[centroidNeighbor][bestNeighbor]
        ccHmap[centroidNeighbor][bestNeighbor] = temp
    else:
        last = len(neighborCluster) - 1
        temp = neighborCluster[last]
        ccHmap[centroidNeighbor][last] = ccHmap[centroidNeighbor][bestNeighbor]
        ccHmap[centroidNeighbor][bestNeighbor] = temp

def exportToCppFolder(ccHmap):
    numcl = 0
    for key, values in ccHmap.items():
        fprocessed = open(f'C:\\Users\\mn170387d\\Desktop\\clusters\\cluster_{numcl}.txt', "w+")
        numcl = numcl + 1
        fprocessed.write(f'{len(values)}\n')
        for hole in values:
            fprocessed.write(f'{hole[0]} {hole[1]}\n')
        fprocessed.close()


def main():
    centroidOrderHashMap = connectCentroidsAndSolution()
    centroidClusterHashmap = createCentroidToClusterHashmap()
    #print("BEFORE InOut PROCESSING")
    # for key, val in centroidClusterHashmap.items():
    #     print(key, "===", val)
    # for key, val in centroidOrderHashMap.items():
    #     print(key, val)

    for i in range(NUM_CLUSTERS):
        findClosestFromNeighborCluster(i, (i+1) % NUM_CLUSTERS, centroidOrderHashMap, centroidClusterHashmap, True)
        findClosestFromNeighborCluster((i + 1) % NUM_CLUSTERS, i, centroidOrderHashMap, centroidClusterHashmap, False)

    # findClosestFromNeighborCluster(0, 1, centroidOrderHashMap, centroidClusterHashmap, True)
    # findClosestFromNeighborCluster(1, 0, centroidOrderHashMap, centroidClusterHashmap, False)


    # print("AFTER InOut PROCESSING")
    # for key, val in centroidClusterHashmap.items():
    #     print(key, "===", val)

    # writing to clusters folder that will go to CUDA
    exportToCppFolder(centroidClusterHashmap)

if __name__ == '__main__':
    main()
