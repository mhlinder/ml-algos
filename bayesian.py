# http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
from pandas import read_csv
import numpy as np

train = read_csv('train.csv')

test = train.iloc[31500:]
train = train.iloc[:31500]

digits_p = {}
for i in range(10):
    d = train[train['label']==i].values[:,1:]
    digits_p[i] = np.mean(d, 0)

n = len(test)
test['guess'] = np.tile(np.nan, n)
for i in range(n):
    print i
    item = test.iloc[i].values[1:-1]
    m = {}
    for k in range(10):
        m[k] = 1
        for j in range(len(item)):
            m[k] = m[k] * digits_p[k][j]
