from numpy import mat

import predeal
import createTree as ct
import predict
# allData = predeal.getDataSet('../dataset/boston_housing_data.csv',',',False)
# trainData,testData = predeal.loadTrainAndTest(allData,10)
# print(str(trainData.shape)+"  "+str(testData.shape))

from sklearn import datasets
import numpy as np

# # boston = datasets.load_boston()
# # # print(type(boston))
# # print(boston['data'].shape)
# # print(boston['target'].shape)
# # print(boston['target'])
# #
# # # cancer = datasets.load_breast_cancer()
# # # print(cancer)
#
# iris=datasets.load_iris()
# print(iris['data'].shape)
# print(iris['target'].shape)
# print(iris['data'][0])
# print(set(iris['target']))
allDataInfo = datasets.load_diabetes()
# print(allDataInfo)
# allDataInfo = datasets.load_breast_cancer()
data = allDataInfo['data']
target = allDataInfo['target']
target = target.reshape(target.shape[0], 1)
# print(data.shape)
# print(target.shape)
entireData = np.hstack((data, target))
# entireData = entireData.dropna()
# print(entireData.shape)
#
trainData,testData=predeal.loadTrainAndTest(entireData)
# # print(trainData.shape)
# # print(testData.shape)
#
# tree=ct.createTree(trainData,1,1)
# print(tree)

import pandas as pd

# data = pd.read_csv('../dataset/breast_cancer.csv')
# data = data.dropna()
# trainData, testData = predeal.loadTrainAndTest(data.values)
tree = ct.createTree(trainData, 1, 1)
# print(tree)
mistakes = predict.calcAccuracy(tree,testData)
print('mistakes == '+str(mistakes))
print('len(testData) == '+str(len(testData)))
print('EER == '+str(mistakes/len(testData)))