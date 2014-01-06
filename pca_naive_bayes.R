data1 <- read.csv('kaggle/train.csv',header=T)
# data2 <- read.csv('kaggle/test.csv',header=T)
data <- data.frame(data1)


# calculate principal components
n <- nrow(data) / 10
train <- data[n:nrow(data), 2:ncol(data)]
pca <- princomp(train)
pcs <- t(pca$loadings[,1:40])

# classify test set
test <- data[1:n,]

for (digit in 0:9) {
    d <- test[test[,1]==digit, 2:ncol(test)]
    d <- t(d)

    pc <- pcs %*% d
    pc <- t(pc)

    if (FALSE) { # Plot histograms for all digit, PC tuples
        for (com in 1:40)  {
            ggplot(data.frame(Princ.Comp = pc[,com]), aes(x=Princ.Comp, fill=1)) + geom_density() + opts(legend.position='none')
            ggsave(file=sprintf('pngs/d%d_c%s.png', digit, com))
        }
    }
}
