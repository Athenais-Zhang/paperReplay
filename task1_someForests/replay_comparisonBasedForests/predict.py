import createTree as ct

def predict(tree,data):
    if tree['leftChild']==None and tree['rightChild']==None:
        return tree['val']
    if tree['leftChild']==None:
        return predict(tree['rightChild'],data)
    if tree['rightChild']==None:
        return predict(tree['leftChild'],data)
    if ct.calcDis(tree['leftX'],data)<=ct.calcDis(tree['rightX'],data):
        return predict(tree['leftChild'],data)
    else:
        return predict(tree['rightChild'],data)