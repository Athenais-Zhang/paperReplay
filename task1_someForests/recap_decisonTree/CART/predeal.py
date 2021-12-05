import pandas as pd

def loadDataSet(fileName,splitChar):
    data=pd.read_csv(fileName)
    # print(data.info())
    return data


def preDeal(data,concludeNa):
    if concludeNa.get():
        pass
    else:
        data=data.dropna()
    print(data.info())
    return data.values


def getDataSet(fileName,splitChar,concludeNa):
    data=loadDataSet(fileName,splitChar)
    print("+++++++++++++++++++++++++++++++")
    dataset=preDeal(data,concludeNa)
    return dataset