import random
import numpy as np


def chooseRobjects(data, r, sampleObjWay):
    objects = []
    dataMaxIndex = len(data) - 1
    if dataMaxIndex <= 0 or len(set(np.array(data)[:, -1])) == 1:
        return None
    for i in range(r):
        obj = []
        if sampleObjWay == 1:
            index1 = random.randint(0, dataMaxIndex)
            index2 = random.randint(0, dataMaxIndex)
            while index1 == index2:
                index2 = random.randint(0, dataMaxIndex)
            obj = [data[index1], data[index2]]
        elif sampleObjWay == 2:
            index1 = random.randint(0, dataMaxIndex)
            index2 = random.randint(0, dataMaxIndex)
            while index1 == index2 or data[index1][-1] == data[index2][-1]:
                index2 = random.randint(0, dataMaxIndex)
            obj = [data[index1], data[index2]]

        objects.append(obj)
    return objects


def calcS(k, i):
    pass


def calcGini(dataset):
    if len(dataset) == 0:
        return 0
    resSet = set(np.array(dataset)[:, -1])
    totalDataNum = len(dataset)
    totalClassNum = len(resSet)
    resSetNum = {}
    for data in dataset:
        if resSetNum.__contains__(data[-1]):
            resSetNum[data[-1]] += 1
        else:
            resSetNum[data[-1]] = 1
    totalP = 0.0
    for classLabel in resSetNum:
        totalP += pow((resSetNum[classLabel] / totalDataNum), 2)
    return 1 - totalP


def chooseBestA(data, obj):
    # # objMetrix=obj[1]-obj[0]
    # allPxks = []
    # for ok in data:
    #     if ok == obj[0] or ok == obj[1]:
    #         continue
    #     else:
    #         pxk = sum(ok * obj[1] - ok * obj[0])
    #         pxkInfo = {'ok': ok, 'pxk': pxk}
    #         allPxks.append(pxkInfo)
    # set1 = []
    # set2 = []
    # for ok in data:
    #     for pxkInfo in allPxks:
    #         if pxkInfo['pxk']
    allPxks = []
    for okIndax in range(len(data)):
        if all(data[okIndax] == obj[0]) or all(data[okIndax] == obj[1]):
            allPxks.append(0)
            continue
        else:
            pxk = sum(data[okIndax] * obj[1] - data[okIndax] * obj[0])
            allPxks.append(pxk)
    gini = calcGini(data)
    bestGini = 1.0
    bestOk = obj[0]
    for okIndex in range(len(data)):
        set1 = []
        set2 = []
        if all(data[okIndex] == obj[0]) or all(data[okIndex] == obj[1]):
            continue
        pxk = allPxks[okIndex]
        for nIndex in range(len(data)):
            if nIndex == okIndex:
                continue
            else:
                if allPxks[nIndex] <= pxk:
                    set1.append(data[nIndex])
                else:
                    set2.append(data[nIndex])
        gq = (len(set1) * calcGini(set1) + len(set2) * calcGini(set2)) / (len(set1) + len(set2))
        if (gini - gq) < bestGini:
            bestGini = gini - gq
            bestOk = data[okIndex]
    return bestOk, bestGini


def chooseBestObjA(data, objects):
    BestObj = objects[0]
    BestA = objects[0][0]
    MinGini = 1.0
    for obj in objects:
        a, gini = chooseBestA(data, obj)
        if MinGini > gini:
            BestObj = obj
            BestA = a
            MinGini = gini
    return BestObj, BestA


def binDataSet(datas, obj, a):
    lSet = []
    rSet = []
    lSet.append(obj[0])
    rSet.append(obj[1])
    pxka = sum(a * obj[1] - a * obj[0])
    for data in datas:
        if all(data == obj[0]) or all(data == obj[1]):
            continue
        pxk = sum(data * obj[1] - data * obj[0])
        if pxk <= pxka:
            lSet.append(data)
        else:
            rSet.append(data)
    return lSet, rSet


def createTree(data, r=1, sampleObjWay=2):
    objects = chooseRobjects(data, r, sampleObjWay)
    if None == objects:
        if len(data)==0:
            # todo
            return None
        return {'res':data[0][-1]}
    obj, a = chooseBestObjA(data, objects)
    tree = {}
    tree['obj'] = obj
    tree['a'] = a
    lSet, rSet = binDataSet(data, obj, a)
    tree['left'] = createTree(lSet, r, sampleObjWay)
    tree['right'] = createTree(rSet, r, sampleObjWay)
    return tree
