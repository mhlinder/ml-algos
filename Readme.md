Simple Machine Learning Algorithms
=================================
These are some basic machine learning algorithms, applied to the Kaggle toy competition [Digit Recognizer](http://www.kaggle.com/c/digit-recognizer). These algos are not intended to be sophisticated, nor are they necessarily appropriate for this context; I've focused here on the implementation of algorithms, rather than solving a problem. To that end, these are general machine learning approaches, supervised and unsupervised, that can be applied to a variety of problems. They're written in a handful of languages, depending on personal preference, speed, learning to program better and built-in features, sometimes with duplicated code.

### Benchmark: simple averaging
`averages_benchmark.py`
*Public Score: 0.80614*
As a benchmark, for all training observations with a given label, I calculate the average of each pixel. I then minimize the Euclidean distance between each test observation and these average images, using the "closest" digit as the classification rule. Sort of like KNN, n=1, on average digits.

### Naive Bayes
`naive_bayes.py`, `naive_bayes.jl`
*Public Score: 0.82671*
Assume a multinomial distribution with k=256 classes for each 

### PCA and naive Bayes
*Public Score: 0.86771*
