# Document Clustering
HAC (Hierarchical Agglomerative/bottom-up Clustering) and k-means document clustering algorithms
- Implemented two document clustering algorithms: Hierarchical Agglomerative/bottom-up Clustering HAC (in Python) and k-means (in Spark).
- Represented each document as a unit vector (formed by normalizing its tf-idf vector by its Euclidean length) and used cosine function to measure their similarity/distance.
- Calculated centroid of merged clusters by taking average over the unit vectors in the cluster.
- HAC algorithm produces a desired number of clusters while k-means proceeds until the distance between cluster centers is more than a given convergence threshold. 	

Core Technology: Apache Spark, Python, numpy, scipy, heapq, AWS (Amazon EC2).

# Data Sets
Used the Bag-of-words dataset https://archive.ics.uci.edu/ml/datasets/Bag+of+Words at UCI Machine Learning Repository.

The data set contains five collections of documents. Documents have been pre-processed and each collection consists of two files: vocabulary file and document-word file. 

We will only use the document word file.

# Programs

## HAC

### About
- Used heap-based priority queue to store pairwise distances of clusters.
- To provide efficient removal of nodes that involve the clusters which have been merged, the programs uses the following approach :
  1. Instead of removing old nodes from heap, just keep these nodes in the heap.
  2. Every time you remove an element from the heap, check if this element is valid (does not contain old clusters) or not. If it is invalid, continue to pop another one.

### Input
The program takes 2 arguments
```
> python hac.py docword.txt k
```
- *docword.txt* is the document word file
- *k* is the desired number of clusters.

### Output
For each cluster, the program outputs document IDs that belong to this cluster on a new line.
```
96,50
79,86,93
97
4,65,69,70
…
```
## k-means (Modified Spark's Example Implementation)

### About
Modified the example k-means implementation in **Spark (version 2.1.0)**, so that it takes document-word file as the input, and outputs the number of nonzero values for each cluster.

### Input
The program takes 4 arguments
```
> bin/spark-submit kmeans.py inputfile k convergeDist output.txt
```
- *inputFile* is the document word file.
- *k* is the number of initial centroids.
- *convergeDist* is the given threshold.
- *output.txt* is the path to the output file.

### Output
For each cluster, the program outputs the number of its nonzero values.
```
687
600
509
560
…
```
It means that first cluster center has 687 nonzero values.
