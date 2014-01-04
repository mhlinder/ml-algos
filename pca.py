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

# generate covariance matrix
n = pixels_adj.shape[0]
col_n = pixels_adj.shape[1]
cov = zeros([col_n, col_n])

for i in range(col_n):
    for j in range(i,col_n):
        # find element-wise product of i and j pixels
        prod = pixels_adj[:,i] * pixels_adj[:,j]
        s_cov = sum(prod) / (n-1)

        cov[i,j] = s_cov
        if i != j: # cov is symmetric
            cov[j,i] = s_cov

    print i
