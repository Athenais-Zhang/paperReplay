import numpy as np
import random


def choosePivotPoints(len):
    if len == 1:
        return 0, 0
    index1 = random.randint(1, len)
    index2 = random.randint(1, len)
    while index1 == index2:
        index2 = random.randint(1, len)
    return index1, index2


def calcDis(x, y):
    distance = 0.0
    for i in range(x.shape[0] - 1):
        distance += (pow(x[i] - y[i], 2))
    distance = pow(distance, 0.5)
    return distance


def createTree(data, maxLeafSize=1):
    # make leaf
    if data.shape[0] == 0:
        return None
    if data.shape[0] <= maxLeafSize:
        T = {}
        T['val'] = np.mean(data[:, -1])
        T['leftChild'] = None
        T['rightChild'] = None
        return T

    T = {}
    if data.shape[0] > maxLeafSize:
        index1, index2 = choosePivotPoints(data.shape[0] - 1)
        set1 = []
        set2 = []
        T['leftX'] = data[index1]
        T['rightX'] = data[index2]
        for index in range(data.shape[0]):
            if index != index1 and index != index2:
                if calcDis(data[index], T['leftX']) <= calcDis(data[index], T['rightX']):
                    set1.append(data[index])
                else:
                    set2.append(data[index])
        T['leftChild'] = createTree(np.array(set1), maxLeafSize)
        T['rightChild'] = createTree(np.array(set2), maxLeafSize)
        if T['leftChild']==None:
            left={}
            left['val']=T['leftX'][-1]
            left['leftChild']=None
            left['rightChild']=None
            T['leftChild']=left
        if T['rightChild']==None:
            right={}
            right['val']=T['rightX'][-1]
            right['leftChild']=None
            right['rightChild']=None
            T['rightChild']=right
    return T
