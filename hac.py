import sys, math, warnings
import numpy as np
from itertools import islice
from scipy import spatial
from heapq import heappush, heappop
from scipy.sparse import csc_matrix, SparseEfficiencyWarning


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
        euclidean_len += element**2
    return math.sqrt(euclidean_len)


# Returns the list of non zero elements from the sparse vector
# list = [( (row_idx, col_idx), value ), .....]
def get_nonzero_elements(matrix):
    non_zero_elements = []
    rows, cols = matrix.nonzero()
    for row, col in zip(rows,cols):
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
        df = len(set_word_documents)        # number of documents where the word appears
        word_idf[word_id] = math.log(float(N + 1)/(df + 1), 2)
    return word_idf


# Returns the cosine distance between two vectors
def get_distance(document_1, document_2):
    document_1_array = document_1.toarray()[0]
    document_2_array = document_2.toarray()[0]
    return spatial.distance.cosine(document_1_array, document_2_array)


# Returns initialization data before executing HAC algorithm
# Returns :
#   1. Initial clusters map {'1':1, '2':1, '3':1,....}
#   2. Initial centroids map {'1':<sparse vector for doc_id 1>, '2':<sparse vector for doc_id 2>, ........}
#   3. Initial heap(min-heap) containing pair-wise document cosine distances
def get_initialization_data(document_vectors):
    num_documents = document_vectors.shape[0]

    # creating initial clusters
    clusters = {}
    for i in range(num_documents):
        clusters[str(i)] = 1

    # computing initial centroids
    centroids = {}
    for i in range(num_documents):
        centroids[str(i)] = document_vectors[i]

    # adding documents to heap pairwise
    heap = []
    for i in range(num_documents - 1):
        document_1 = document_vectors[i]
        for j in range(i+1, num_documents):
            document_2 = document_vectors[j]
            distance = get_distance(document_1, document_2)
            heappush(heap, (distance, (str(i), str(j))))

    return clusters, centroids, heap


# Returns list of new clusters formed from the existing clusters
# in the clusters map and the new centroid
def get_new_clusters(centroids, clusters, new_cluster, centroid_vector):
    new_clusters = []
    for existing_cluster in clusters:
        existing_cluster_vector = centroids[existing_cluster]
        distance = get_distance(existing_cluster_vector, centroid_vector)
        new_cluster_tuple = (distance, (new_cluster, existing_cluster))
        new_clusters.append(new_cluster_tuple)
    return new_clusters


# Returns the centroid formed by taking the average over all the unit vectors in the cluster.
def get_centroid_vector(document_vectors, cluster_left, cluster_right):
    str_cluster = cluster_left + "," + cluster_right
    str_cluster_vectors = str_cluster.split(",")
    len_cluster_vectors = len(str_cluster_vectors)

    centroid_vector = np.array(document_vectors[int(str_cluster_vectors[0])].toarray()[0])

    for i in range(1, len_cluster_vectors):
        document_vector = document_vectors[int(str_cluster_vectors[i])].toarray()[0]
        centroid_vector += np.array(document_vector)

    centroid_vector /= len(str_cluster_vectors)

    return csc_matrix(centroid_vector)


# Executes Hierarchical Agglomerative Clustering to form k clusters
def perform_HAC_clustering(document_vectors, clusters, centroids, heap, k):
    while len(clusters) > k:
        cluster = heappop(heap)
        cluster_left = cluster[1][0]
        cluster_right = cluster[1][1]

        if cluster_left in clusters and cluster_right in clusters:
            clusters.__delitem__(cluster_left)
            clusters.__delitem__(cluster_right)

            centroid_vector = get_centroid_vector(document_vectors, cluster_left, cluster_right)

            new_clusters = get_new_clusters(centroids, clusters, cluster_left + "," + cluster_right, centroid_vector)
            for new_cluster in new_clusters:
                heappush(heap, new_cluster)
            clusters[cluster_left + "," + cluster_right] = 1
            centroids[cluster_left + "," + cluster_right] = centroid_vector
            centroids.__delitem__(cluster_left)
            centroids.__delitem__(cluster_right)
    return clusters


def print_output(clusters):
    output = []
    for cluster in clusters:
        cluster_documents = cluster.split(',')
        output_line = []
        for document_id in cluster_documents:
            output_line.append(int(document_id)+1)
        output.append(output_line)

    # printing to console
    for line in output:
        print ','.join(map(str, line))


# main execution
if __name__ == '__main__':

    # reading document-word file
    input_file = sys.argv[1]
    k = int(sys.argv[2])

    warnings.simplefilter('ignore', SparseEfficiencyWarning)
    document_vectors = get_document_vectors(input_file)

    clusters, centroids, heap = get_initialization_data(document_vectors)

    clusters = perform_HAC_clustering(document_vectors, clusters, centroids, heap, k)

    print_output(clusters)