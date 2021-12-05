from numpy import mat, linalg, power
from numpy.core import ones
from numpy.core.fromnumeric import shape


def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))