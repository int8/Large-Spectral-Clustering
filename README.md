# Large Scale Spectral Clustering with Landmark-Based Representation

The purpose of this code is mainly educational. It is the implementation of unsupervised learning technique called:  Large Scale Spectral Clustering with Landmark-Based Representation

Quick example of how to use the code:

```julia
# Pkg.add("MNIST")
using MNIST;

```

Include the code and load the datasets.

```julia
include("LSC.jl");
include("Evaluation.jl");

# reading dataset (60k objects)
data, labels = MNIST.traindata(); # using testdata() instead will return smaller (10k objects) dataset
# normalizing it
data  = (data .- mean(data,2)) / std(data .- mean(data,2));
```

Now, to perform clustering run the following:
```julia
LSCMnistResult = LSCClustering(data, 10, 350, :Kmeans, 5, 0.5);
nmiValue = normalizedMutualInformation(getPartitionSet(LSCMnistResult .assignments, 10) , getPartitionSet(labels + 1 ,10), 60000);
```


Please visit http://int8.io/large-scale-spectral-clustering-with-landmark-based-representation for details (+ to see some experiments)
