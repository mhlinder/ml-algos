Simple Machine Learning Algorithms
=================================
These are some basic machine learning algorithms, applied to the Kaggle toy competition [Digit Recognizer](http://www.kaggle.com/c/digit-recognizer). These algos are not intended to be sophisticated, nor are they necessarily appropriate for this context; I've focused here on the implementation of algorithms, rather than solving a problem. To that end, these are general machine learning approaches, supervised and unsupervised, that can be applied to a variety of problems. They're written in a handful of languages, depending on personal preference, speed, learning to program better and built-in features, sometimes with duplicated code.

I have attempted to write some of these algorithms from scratch, for educational purposes, and in later instantiations of the same algorithm (see `pca_naive_bayes.R`) I use optimized libraries. There are, as I see it, two steps to this type of work: understanding the underlying algorithm (writing the algorithm from scratch), and using the algorithm (using a more robust, third-party implementation). This repo features both. Given are the files that implement a given solution, as well as the correct classification rates according to Kaggle for each algorithm.

### Benchmark: simple averaging
`averages_benchmark.py`
<br />
*Public Score: 0.80614*
<br />
As a benchmark, for all training observations with a given label, I calculate the average of each pixel. I then minimize the Euclidean distance between each test observation and these average images, using the "closest" digit as the classification rule. Sort of like KNN, n=1, on average digits.

### Naive Bayes
`naive_bayes.py`, `naive_bayes.jl`
<br />
*Public Score: 0.82671*
<br />
Assume a multinomial distribution with k=256 classes for each of 784 pixels; calculate conditional population proportions to find pixel distributions, conditional on digit. Assume independence between all pixel values. This is clearly naive: mutually proximal pixels are likely to have high correlation, and this implementation of the naive Bayes filter completely disregards the information contained in these correlations. There's no structure to the data, as it were, aside from 784 different explanatory variables.

### PCA and naive Bayes
`pca.jl`, `pca_naive_bayes.R`
<br />
*Public Score: 0.86771*
<br />
Calculate principal components for the training data set, and drop those principal components corresponding to low variance. This is dimension reduction. Perform naive Bayes classification on training data re-expressed in terms fo the principal components; because the explanatory variables are in terms of the principal components, these variables are continuous rather than discrete. Conditionally Gaussian distributions are used for calculation of the likelihood; plotting reveals that this is not an absurd assumption to make.
