using DataFrames

data = readtable("train.csv");
test = data[1:nrow(data)/10, :]
train = data[nrow(data)/10:, :]

digits_p = Dict()
for i = 1:10
    i = i - 1
    d = train[train["label"] .== i, 2:]
    digits_p[i] = Dict()
    for j = 1:ncol(d)
        for k 
    end
end

guess = []
n = nrow(test)
