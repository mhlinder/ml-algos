# read in data
data = readdlm("kaggle/train.csv", ',', Int64, has_header=true);
train = data[1];
header = data[2];

pixels = train[:,2:];

# demean
pixels_adj = zeros(Float64, size(pixels));
for i = 1:size(pixels)[2]
    m = mean(pixels[:,i]);
    pixels_adj[:,i] = pixels[:,i] - m;
end

# generate covariance matrix
n = size(pixels_adj)[1];
col_n = size(pixels_adj)[2];
cov = zeros(Float64, col_n, col_n);

for i = 1:col_n
    for j = i:col_n
        prod = pixels_adj[:,i] .* pixels[:,j];
        s_cov = sum(prod) / (n-1)

        cov[i,j] = s_cov
        if i != j
            cov[j,i] = s_cov
        end
    end
    println(i)
end
