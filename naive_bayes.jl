# http://www.kaggle.com/c/digit-recognizer
# http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf

Ni = 784 # number of pixels in each picture---28x28
Nj = 256 # number values a pixel can take on
Nk = 10 # number of digits depicted by an image
l = 1 # laplace smoothing for l=1

# read in data
data = readdlm("kaggle/train.csv", ',', Int64, has_header=true);

cut = size(data[1])[1] / 10
train = data[1][1:end-cut,:]
test = data[1][end-cut:end,:]

counts_y = [i=>0 for i=0:9] # counts for each y_k value
counts = [i=>zeros(Ni,Nj) for i=0:9] # counts for each pixel in each X_i

# loop over training values, record each pixel's value
n = size(train)[1]
for i = 1:n
    d = train[i,:]
    # record label (digit depicted)
    label = d[1]
    counts_y[label] = counts_y[label] + 1

    # just pixels
    d = d[1:]
    for j = 1:Ni
        counts[label][j, d[j]+1] = counts[label][j, d[j]+1] + 1
    end

    # user interface
    if i % 1000 == 0
        println(i)
    end
end

# convert counts to proportions
pis = [i=>counts_y[i] / size(train)[1] for i = 0:Nk-1] # proportion of each y_k value
thetas = [i=>(counts[i] + l) / (counts_y[i] + l*Nj) for i = 0:Nk-1] # proportion for each X_i pixel

println("finished training data")

# classify each item in test set
test_labels = test[:,1]
test_guess = nans(size(test_labels))
n = size(test)[1]

for t = 1:n
    tt = test[t, 2:end]
    max_lh = [-Inf,-Inf]

    # for each possible label/digit, calculate likelihood
    for k = 0:Nk-1
        m = 0
        # loop over each pixel
        for i = 1:Ni
            x_i = tt[i]
            m = m + log(thetas[k][i, x_i+1])
        end
        m = m + log(pis[k])

        # update, if necessary
        if m > max_lh[1]
            max_lh[1] = m
            max_lh[2] = k
        end
    end
    test_guess[t] = max_lh[2]

    if t % 1000 == 0
        println(t)
    end
end

test_guess = int(test_guess)
