import numpy as np

import predeal
import createTree as ct
import predict

allData = predeal.getDataSet('../dataset/boston_housing_data.csv',',',False)
trainData,testData = predeal.loadTrainAndTest(allData,10)
print(str(trainData.shape)+"  "+str(testData.shape))

# Tree = ct.createTree(trainData)
# print("tree is constructed!")
# print(Tree)

for i in range(10):
    Tree = ct.createTree(trainData,10)
    err=0.0
    for data in testData:
        res = predict.predict(Tree,data)
        # print(str(data[-1])+" 预测值："+str(res))
        err+=np.abs(res-data[-1])
    print("平均误差："+str(err/testData.shape[0]))

