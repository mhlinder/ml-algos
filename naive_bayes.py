# http://www.kaggle.com/c/digit-recognizer
# http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
from pandas import read_csv
from numpy import zeros


Nj = 256

counts = {i: zeros([784,256]) for i in range(10)}

data = read_csv('kaggle/train.csv')
train = data

for i in range(len(data)):
    d = data.iloc[i].values
    label = d[0]
    d = d[1:]
    for j in range(784):
        counts[label][j][d[j]] = counts[label][j][d[j]] + 1
    if i% 1000 == 0:
        print i

thetas = {t: (counts[t] + 1) / float(len(counts[t]) + Nj) for t in counts.keys()}
