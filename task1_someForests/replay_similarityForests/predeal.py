import pandas as pd
import numpy as np


def loadDataSet(fileName, splitChar):
    data = pd.read_csv(fileName)
    return data


def preDeal(data, concludeNa):
    if concludeNa:
        pass
    else:
        data = data.dropna()
    # print(data.info())
    return data.values


def getDataSet(fileName, splitChar, concludeNa):
    data = loadDataSet(fileName, splitChar)
    print("loading dataset.... wait a minute")
    dataset = preDeal(data, concludeNa)
    return dataset


def loadTrainAndTest(dataSet, k=10):
    testData = []
    trainData = []
    for index in range(len(dataSet)):
        if index % k == 0:
            testData.append(dataSet[index])
        else:
            trainData.append(dataSet[index])
    return np.array(trainData), np.array(testData)
