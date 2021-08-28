import numpy as np
import os
import re
import matplotlib.pyplot as plt
import math

numholes = 0
holes = []
order = []

def euclidean(p1, p2):
    res = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    print("res1: ", res)
    res2 = np.linalg.norm(np.array(p1)-np.array(p2))
    print("res2: ", res2)

def readSolvedCluster(file):
    global numholes, holes
    dict = {}
    with open(f'C:\\Users\\mn170387d\\Desktop\\solvedcluster_3.txt', 'r') as reader:
        numh = reader.readline()
        numholes = int(numh)
        hole = reader.readline()
        while hole != '':
            coords = re.findall("\-?\d+\.\d+|\-?\d+", hole)
            dict[int(coords[2])] = (float(coords[0]), float(coords[1]))
            holes.append((float(coords[0]), float(coords[1])))
            order.append(int(coords[2]))
            print(coords)
            hole = reader.readline()
    return dict

def calculatePrice(holes, order):
    price = 0
    for i in range(numholes - 1):
        price = price + np.linalg.norm(np.array(holes[order[i]])-np.array(holes[order[i+1]]))
    return price


def main():
    solved = readSolvedCluster("heh")
    x = []
    y = []

    # for key, val in solved.items():
    #     print(key, ": ", val[0], " - ", val[1])
        # x.append(val[0])
        # y.append([val[1]])


    for i in range(len(holes)):
        val = holes[order[i]]
        x.append(val[0])
        y.append(val[1])

    price = calculatePrice(holes, order)
    print("COST:",price)


    xfrom = 0
    xto = 300
    yfrom = -100
    yto = 0
    plt.figure(1, figsize=(16, 9), dpi=100)
    plt.style.use('seaborn-whitegrid')
    axes = plt.gca()
    axes.set_ylim(yfrom, yto)
    axes.set_xlim(xfrom, xto)
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    main()
