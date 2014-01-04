# run principle components analysis
# http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
from pandas import read_csv
from numpy import mean, zeros

train = read_csv('kaggle/train.csv')
pixels = train.iloc[:,1:].values # rows are images; columns are pixels

# demean
pixels_adj = zeros(pixels.shape)
for i in range(pixels.shape[1]):
    m = mean(pixels[:,i])
    pixels_adj[:,i] = pixels[:,i] - m
