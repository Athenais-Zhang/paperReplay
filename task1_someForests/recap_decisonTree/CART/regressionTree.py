from numpy.core import mean, var, shape


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    # var = mean(abs(x - x.mean()) ** 2)
    return var(dataSet[:, -1]) * shape(dataSet)[0]