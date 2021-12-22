def calcAccuracy(tree, datas):
    mistakes=0
    for data in datas:
        res=predict(tree,data)
        # print('res=='+str(res))
        mistake=abs(res-data[-1])
        mistakes+=mistake
    return mistakes


def predict(tree, data):
    try:
        return tree['res']
    except:
        pxk = sum(data * tree['obj'][1] - data * tree['obj'][0])
        pxa = sum(tree['a'] * tree['obj'][1] - tree['a'] * tree['obj'][0])
        if pxk <= pxa:
            return predict(tree['left'], data)
        else:
            return predict(tree['right'], data)
