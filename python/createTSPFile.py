import numpy as np

fix = []

def main():
    with open(f'C:\\Users\\mn170387d\\Desktop\\in263.txt', 'r') as reader:
        for line in reader:
            xy = line.split()
            print(xy)
            fix.append([float(xy[0]), float(xy[1])])

    print(reader.name)
    arr = np.array(fix)
    fw = open(f'C:\\Users\\mn170387d\\Desktop\\tspfiles\\{"clusterMeans"}.tsp',"w+")
    fw.write(f'NAME: Diplomski{reader.name}\n')
    fw.write(f'COMMENT: Simetricni TSP, klastering + CUDA\n')
    fw.write(f'TYPE: TSP\n')
    fw.write(f'DIMENSION: {len(arr)}\n')
    fw.write(f'EDGE_WEIGHT_TYPE : EUC_2D\n')
    fw.write(f'NODE_COORD_SECTION\n')

    for i in range(len(arr)):
        str = f'{i+1} {arr[i][0]} {arr[i][1]} \n'
        fw.write(str)

    fw.write(f'EOF\n')
    fw.close()

if __name__ == '__main__':
    main()