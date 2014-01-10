# find average image for each digit, minimize dist from these averages
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas import read_csv
import numpy as np

data = read_csv('train.csv')
train = data[:len(data)/10.0]
test = data[len(data)/10.0:]
# train = read_csv('train.csv')
# test = read_csv('test.csv')

avgs = {i: None for i in range(10)}
for i in range(10):
    avg = np.average(train[train['label']==i].values[:,1:], axis=0)
    avgs[i] = avg

test['guess'] = np.tile(np.nan, len(test))
# test['label'] = np.tile(np.nan, len(test))
for i in range(len(test)):
    digit = test.iloc[i][:-1]
    dist = [np.inf, 'nan']
    for j in range(10):
        d = np.linalg.norm(digit - avgs[j])
        if d < dist[0]:
            dist[0] = d
            dist[1] = j
    test['guess'].iloc[i] = dist[1]
    # test['label'].iloc[i] = dist[1]

p_wrong = sum(test['guess'] != test['label']) / float(len(test))
# In [1]: p_wrong
# Out[1]: 0.20140211640211642

# labels = test['label']
# labels.to_csv('test_labels.csv',float_format="%.0f",header=True,index_label="ImageId")
