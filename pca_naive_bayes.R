indata1 <- read.csv('kaggle/train.csv',header=T)
indata2 <- read.csv('kaggle/test.csv',header=T)
indata2$label <- rep(NA, nrow(indata2))
indata <- rbind(indata1, indata2)

# calculate principal components of data
pca <- princomp(indata[,2:ncol(indata)])
nPC <- 40 # subset of princ. comp.
pcs <- t(pca$loadings[,1:nPC]) # rows principal components

# extract labels
labels <- indata[,1]

# transform data into pc
indata <- t(indata[,2:ncol(indata)])
indata <- data.frame(t(pcs %*% indata))
indata$label <- labels

# calculate naive bayes conditional probabilities
library("e1071")
n <- nrow(indata1)
dtrain <- indata[1:n, 1:nPC]
ltrain <- indata[1:n, nPC+1]

dtest <-indata[(n+1):nrow(indata), 1:nPC]
ltest <- indata[(n+1):nrow(indata), nPC+1]

bayes <- naiveBayes(dtrain, ltrain)
classified <- predict(bayes, indata[(n+1):nrow(indata),1:nPC], type="raw")

indata[(n+1):nrow(indata),]$label <- apply(classified,1,which.max) - 1
