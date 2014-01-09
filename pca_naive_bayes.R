data1 <- read.csv('kaggle/train.csv',header=T)
# data2 <- read.csv('kaggle/test.csv',header=T)
data <- data.frame(data1)

nK <- 10
nPC <- 40

# calculate principal components
n <- nrow(data) / 10
test = data[1:n,]
train <- data[n:nrow(data),]
pca <- princomp(train[,2:ncol(train)])
pcs <- t(pca$loadings[,1:nPC]) # rows principal components

# conditional distributions look approximately normal
# columns are digits; rows are PCs
means <- mat.or.vec(nPC, nK)
sds <- mat.or.vec(nPC, nK)
for (digit in 0:9) {
    d <- train[train[,1]==digit,2:ncol(train)]
    d <- t(d) # rows are explanatory variables

    pc <- pcs %*% d
    pc <- t(pc) # rows are obs

    if (FALSE) { # Plot histograms for all digit, PC tuples
        for (com in 1:40)  {
            ggplot(data.frame(Princ.Comp = pc[,com]), aes(x=Princ.Comp, fill=1)) + geom_density() + opts(legend.position='none')
            ggsave(file=sprintf('pngs/d%d_c%s.png', digit, com))
        }
    }

    means[,digit+1] <- apply(pc, 2, mean)
    sds[,digit+1] <- apply(pc, 2, sd)
}

# calculate log likelihood fo# r each test value, conditional on digit
# llh <- mat.or.vec(nrow(test), nK) # log likelihoods
# for (digit in 0:9) {
    # # transform test values to principal components
    # d <- t(test[,2:ncol(test)])

    # pc <- pcs %*% d
    # pc <- t(pc)

    # # likelihood values
    # probs <- matrix(0, nrow(pc), ncol(pc))

    # # for each column (each PC), calculate the likelihood for that PC, conditional on digit
    # for (i in 1:ncol(pc)) {
        # probs[,i] <- dlnorm(pc[,i], mean=means[i,digit+1], sd=sds[i,digit+1])
    # }
    # # as dlnorm --- log normal --- was used, we sum over likelihood values
    # llh[,digit] <- apply(probs, 1, sum)
# }

# # digit is index-1 of maximum likelihood
# digits <- apply(llh, 1, which.max) - 1
