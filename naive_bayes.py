# http://www.kaggle.com/c/digit-recognizer
# http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf

from pandas import read_csv
from numpy import zeros, tile, nan
import sys

Ni = 784 # number of pixels in each picture---28x28
Nj = 256 # number values a pixel can take on
Nk = 10 # number of digits depicted by an image
l = 1 # laplace smoothing for l=1

# read training data; split into train, test subsets
data = read_csv('kaggle/train.csv')
train = data[len(data)/10:]
test = data[:len(data)/10]

# initialize dicts for counting occurences
counts_y = {i: 0 for i in range(Nk)} # counts for each y_k value
counts = {i: zeros([Ni, Nj]) for i in range(Nk)} # counts for each pixel in each X_i

# loop over training values, record each pixel's value
for i in range(len(train)):
    d = train.iloc[i].values
    # record label (digit depicted)
    label = d[0]
    counts_y[label] = counts_y[label] + 1

    # just pixels
    d = d[1:]
    for j in range(Ni):
        counts[label][j][d[j]] = counts[label][j][d[j]] + 1

    # user interface
    if i % 1000 == 0:
        sys.stdout.write("\rprocessed %i observations" % i)
        sys.stdout.flush()

# convert counts to proportions
pis = {i: counts_y[i] / float(len(train)) for i in range(Nk)} # proportion for each y_k value
thetas = {i: (counts[i] + l) / float(counts_y[i] + l*Nj) for i in range(nK)} # proportion for each X_i pixel

# classify each item in test set
test['guess'] = tile(nan, len(test))
for t in range(len(test)):
    # remove label, guess columns
    tt = test.iloc[t].values[1:-1]
    # 
    max_lh = [0, 0]
    # for each possible label/digit, calculate likelihood
    for k in range(Nk):
        m = 1.0
        for i in range(Ni):
            x_i = tt[i]
            m = m * thetas[k][i][x_i]
        m = m * pis[k]
        if m > max_lh[0]:
            max_lh[0] = m
            max_lh[1] = k
    test['guess'].iloc[t] = max_lh[1]
