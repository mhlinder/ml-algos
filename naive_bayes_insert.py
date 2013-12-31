# http://www.kaggle.com/c/digit-recognizer
# http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf

from pandas import read_csv
from pymongo import MongoClient

def get_db():
    conn = MongoClient()
    return conn['digits']

# Insert into database
f = open('kaggle/train.csv')
lines_in = f.readlines()
f.close()

db = get_db()

keys = lines_in[0].rstrip().split(',')

values = []
i = 0
n = len(lines_in)
for line in lines_in[1:]:
    line = line.rstrip().split(',')
    line = [int(l) for l in line]

    d = dict(zip(keys, line))
    if i < n/10:
        db.train.insert(d)
    else:
        db.test.insert(d)

    i = i+1
    if i % 100 == 0:
        print i
