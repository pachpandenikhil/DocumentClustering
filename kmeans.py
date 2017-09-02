from __future__ import print_function

import sys, math, warnings
import numpy as np
from itertools import islice
from scipy import spatial
from scipy.spatial import distance
from scipy.sparse import csc_matrix, SparseEfficiencyWarning
from pyspark.sql import SparkSession


# Returns unit vectors for each document as a sparse matrix
# Unit vector of a document is obtained from tf*idf vector
# of the document, normalized (divided) by its Euclidean length
def get_document_vectors(input_file):
    f = open(input_file)
    system_metadata = list(islice(f, 3))

    # extracting the number of documents in the system
    num_documents = int(system_metadata[0])

    # extracting the vocabulary of the system
    vocabulary = int(system_metadata[1])

    document_word_tf = csc_matrix((num_documents, vocabulary), dtype=np.int)
    word_documents = {}

    # adding term frequency and document ids for each word
    for line in iter(f):
        data = line.rstrip().split(' ')
        document_id = int(data[0]) - 1
        word_id = int(data[1]) - 1
        term_frequency = int(data[2])

        # adding term frequency
        document_word_tf[document_id, word_id] += term_frequency

        # adding document id to word map
        if word_id in word_documents:
            set_word_documents = word_documents[word_id]
            set_word_documents.add(document_id)
        else:
            set_word_documents = set()
            set_word_documents.add(document_id)
            word_documents[word_id] = set_word_documents

    # computing IDF for each word
    word_idf = compute_IDF_values(word_documents, num_documents)

    # computing TF*IDF for each word in each document
    document_tf_idf_vectors = compute_TF_IDF_values(document_word_tf, word_idf, num_documents, vocabulary)

    # normalizing document vectors
    document_vectors = normalize_document_vectors(document_tf_idf_vectors)

    return document_vectors


# Divides TF*IDF vectors of each document by its Euclidean length
# Returns the normalized document vectors
def normalize_document_vectors(document_tf_idf_vectors):
    rows = document_tf_idf_vectors.shape[0]
    cols = document_tf_idf_vectors.shape[1]
    document_vectors = csc_matrix((rows, cols), dtype=np.float)
    euclidean_lengths = {}

    # normalizing each document vector by its Euclidean length
    for i in range(rows):
        document_vector = document_tf_idf_vectors[i]
        euclidean_len = get_euclidean_length(document_vector)
        euclidean_lengths[i] = euclidean_len

    non_zero_elements = get_nonzero_elements(document_tf_idf_vectors)
    for non_zero_element in non_zero_elements:
        row = non_zero_element[0][0]
        col = non_zero_element[0][1]
        tf_idf = non_zero_element[1]
        document_vectors[row, col] = tf_idf / euclidean_lengths[row]

    return document_vectors


# Returns Euclidean length of a sparse vector
def get_euclidean_length(vector):
    euclidean_len = 0
    vector_array = vector.toarray()[0]
    for element in vector_array:
        euclidean_len += element ** 2
    return math.sqrt(euclidean_len)


# Returns the list of non zero elements from the sparse vector
# list = [( (row_idx, col_idx), value ), .....]
def get_nonzero_elements(matrix):
    non_zero_elements = []
    rows, cols = matrix.nonzero()
    for row, col in zip(rows, cols):
        non_zero_elements.append(((row, col), matrix[row, col]))
    return non_zero_elements


# Computes TF.IDF values for each word in document from TF values vector and IDF values map
# Returns TF.IDF vectors as sparse matrix
def compute_TF_IDF_values(document_word_tf, word_idf, num_documents, vocabulary):
    document_tf_idf_vectors = csc_matrix((num_documents, vocabulary), dtype=np.float)
    non_zero_elements = get_nonzero_elements(document_word_tf)

    # multiplying TF with IDF for each term in each document
    for non_zero_element in non_zero_elements:
        row = non_zero_element[0][0]
        col = non_zero_element[0][1]
        tf = non_zero_element[1]
        document_tf_idf_vectors[row, col] = tf * word_idf[col]

    return document_tf_idf_vectors


# Computes IDF values for each word from the vocabulary
# Returns map of word id to its IDF value
def compute_IDF_values(word_documents, N):
    word_idf = {}
    for word_id, set_word_documents in word_documents.iteritems():
        df = len(set_word_documents)  # number of documents where the word appears
        word_idf[word_id] = math.log(float(N + 1) / (df + 1), 2)
    return word_idf


# Returns the cosine distance between two vectors
def get_cosine_distance(document_1, document_2):
    document_1_array = document_1.toarray()[0]
    document_2_array = document_2.toarray()[0]
    return spatial.distance.cosine(document_1_array, document_2_array)


# Returns the eculedian distance between two vectors
def get_eculedian_distance(document_1, document_2):
    document_1_array = document_1.toarray()[0]
    document_2_array = document_2.toarray()[0]
    return distance.euclidean(document_1_array, document_2_array)


# Returns the centroid document with the least cosine distance to document 'd'
def closest_document(d, centers):
    best_index = 0
    closest = float("+inf")
    for i in range(len(centers)):
        temp_dist = get_cosine_distance(d[1], centers[i][1])
        if temp_dist < closest:
            closest = temp_dist
            best_index = i
    return best_index


# Returns the sum and the count of all the documents in each cluster
def get_cluster_stats(document_1_tuple, document_2_tuple):
    document_1 = document_1_tuple[0]
    document_2 = document_2_tuple[0]
    document_1_vector = np.array(document_1[1].toarray()[0])
    document_2_vector = np.array(document_2[1].toarray()[0])
    document_1_count = document_1_tuple[1]
    document_2_count = document_2_tuple[1]
    document_sum = document_1_vector + document_2_vector
    tot_count = document_1_count + document_2_count
    return ((0, csc_matrix(document_sum)), tot_count)


# Returns the centroid cluster from the sum and count of documents in the cluster
def get_new_clusters(cluster_stat):
    cluster_idx = cluster_stat[0]
    cluster_sum = cluster_stat[1][0][1]
    cluster_documents_count = cluster_stat[1][1]
    cluster_centroid = np.array(cluster_sum.toarray()[0])
    cluster_centroid /= cluster_documents_count
    return (cluster_idx, csc_matrix(cluster_centroid))


# Writes the output to the file line by line.
def write_output(output, file_path):
    with open(file_path, 'w') as file:
        for line in output:
            file.write(line + '\n')


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: kmeans <file> <k> <converge_dist> <output_file>", file=sys.stderr)
        exit(-1)

    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
    sc = spark.sparkContext

    # reading input parameters
    input_file = sys.argv[1]
    k = int(sys.argv[2])
    converge_dist = float(sys.argv[3])
    output_file = sys.argv[4]

    # reading document-word file
    warnings.simplefilter('ignore', SparseEfficiencyWarning)
    document_vectors = get_document_vectors(input_file)

    document_vectors_list = []
    document_id = 0
    for document_vector in document_vectors:
        document_vectors_list.append((document_id, document_vector))
        document_id += 1

    document_vectors_rdd = sc.parallelize(document_vectors_list).cache()
    document_vectors_rdd.sortByKey()
    initial_centroids = document_vectors_rdd.repartition(1).takeSample(False, k, 1)

    temp_dist = 1.0

    while temp_dist > converge_dist:
        closest = document_vectors_rdd.map(lambda d: (closest_document(d, initial_centroids), (d, 1)))
        cluster_stats = closest.reduceByKey(get_cluster_stats)
        new_clusters = cluster_stats.map(get_new_clusters).collect()
        temp_dist = sum(get_eculedian_distance(initial_centroids[iK][1], d) for (iK, d) in new_clusters)
        for (iK, d) in new_clusters:
            initial_centroids[iK] = (0, d)

    # generating output
    output = []
    for (idx, doc_vect) in initial_centroids:
        output.append(str(doc_vect.getnnz(None)))

    write_output(output, output_file)
    spark.stop()
