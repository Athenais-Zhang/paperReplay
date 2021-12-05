from tkinter import Tk, Label, Entry, Button, IntVar, Checkbutton, END

import matplotlib

matplotlib.use('TKAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from numpy import mat, arange
import numpy as np

import modelTree
import createTree
import predict
from task1_someForests.recap_decisonTree.CART import predeal


def reDraw(rawData,testData,tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    reDraw.rawDat=rawData
    reDraw.testDat=testData
    if chkBtnVar.get():
        # 创建模型树、
        if tolN < 2:
            tolN = 2
        # 计算预测模型
        myTree = createTree.createTree(reDraw.rawDat, modelTree.modelLeaf,
                                     modelTree.modelErr, (tolS, tolN))
        # 根据预测模型和测试数据来预测结果，将结果存入yHat中
        yHat = predict.createForeCast(myTree, reDraw.testDat,
                                      predict.modelTreeEval)
    else:
        # 创建回归树
        myTree = createTree.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = predict.createForeCast(myTree, reDraw.testDat)
    print(myTree)
    print("========")
    # print(yHat)
    # for i in range(len(reDraw.testDat)):
    #     print("["+str(reDraw.testDat[i][-1])+"] "+str(yHat.tolist()[i][0]))
    # 绘制散点图，利用训练集当中的数据
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    # 绘制曲线图，利用测试数据和已有模型
    reDraw.a.plot(reDraw.testDat, yHat, lineWidth=2.0)
    reDraw.canvas.draw()

    #计算误差
    err=0.0
    totalNum=0
    for i in range(len(reDraw.testDat)):
        if np.isnan(yHat.tolist()[i][0]) or np.isnan(reDraw.testDat[i][-1]):
            continue
        err+= np.abs(yHat.tolist()[i][0]-reDraw.testDat[i][-1]) #np.power((yHat.tolist()[i][0]-reDraw.testDat[i][-1]), 2)
        totalNum+=1
    print("平均误差："+str(err/totalNum))
    return myTree


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()
    rawData,testData=loadnewData(chkBtnVarNa)
    return reDraw(rawData,testData,tolS, tolN)

def loadnewData(chkBtnVarNa):
    data = mat(predeal.getDataSet('../../dataset/boston_housing_data.csv', ' ', chkBtnVarNa))
    rawDat = []
    testDat = []
    lineIndex = 0
    for line in data:
        lineIndex += 1
        if lineIndex % 10 == 0:
            testDat.append(line.tolist()[0])
        else:
            rawDat.append(line.tolist()[0])
    # print(np.array(reDraw.rawDat).shape)
    # print(np.array(reDraw.testDat).shape)
    rawDat = np.array(rawDat)
    testDat = np.array(testDat)
    return rawDat,testDat

root = Tk()  # 创建Tk类型的根部件

# Label(root,text="Plot place holder").grid(row=0,columnspan=3)
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')

Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

chkBtnVarNa = IntVar()
chkBtnNa = Checkbutton(root, text="conclude na", variable=chkBtnVarNa)
chkBtnNa.grid(row=3, column=3, columnspan=2)

# 初始化一些与reDraw相关联的全局变量
# reDraw.rawDat = mat(createTree.loadDataSet('../00_dataset/boston_housing_data.csv',' '))
# reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)


# reDraw(1.0,10)
myTree = drawNewTree()

root.mainloop()