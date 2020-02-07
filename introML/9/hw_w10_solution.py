import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib import markers
import itertools


# finds indices of closest clusters to be merged on next iteration
# clusters_matrix - clusters matrix
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# returns cluster indices and distance between them
def find_clusters_to_merge(clusters_matrix, distance_col, cluster_col):
    c1_index = np.argmin(clusters_matrix[:, distance_col])
    c2_index = clusters_matrix[c1_index, cluster_col]
    dist = np.amin(clusters_matrix[:, distance_col])
    return c1_index, int(c2_index), dist


# performs merge of clusters with indices c1_index, c2_index
# updates single-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def single_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    # keep minimum of two clusters in a new cluster (which will stay at c1 position)
    clusters_matrix[c1_index, 0: distance_col] = np.minimum(clusters_matrix[c1_index, 0: distance_col],
                                                            clusters_matrix[c2_index, 0: distance_col])
    # update diagonal element
    clusters_matrix[c1_index, c1_index] = np.inf
    # update the new cluster's corresponding column as well
    clusters_matrix[:, c1_index] = clusters_matrix[c1_index, 0: distance_col]
    # infinity out row and column for c2
    clusters_matrix[c2_index, :] = np.inf
    clusters_matrix[:, c2_index] = np.inf
    # if for some clusters c2 was the closest one, replace it with c1
    clusters_matrix[:, cluster_col][clusters_matrix[:, cluster_col] == float(c2_index)] = float(c1_index)
    # find closest cluster to the new one
    num_points = X_matrix.shape[0]
    clusters_matrix[c1_index, cluster_col] = np.argmin(clusters_matrix[c1_index, :num_points])
    clusters_matrix[c1_index, distance_col] = np.amin(clusters_matrix[c1_index, :num_points])
    # update label assignments in X
    X_matrix[:, -1][X_matrix[:, -1] == c2_index] = c1_index


# performs merge of clusters with indices c1_index, c2_index
# updates complete-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def complete_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    # keep maximum of two clusters in a new cluster (which will stay at c1 position)
    clusters_matrix[c1_index, 0: distance_col] = np.maximum(clusters_matrix[c1_index, 0: distance_col],
                                                            clusters_matrix[c2_index, 0: distance_col])
    # update the new cluster's corresponding column as well
    clusters_matrix[:, c1_index] = clusters_matrix[c1_index, 0: distance_col]
    # infinity out row and column for c2
    clusters_matrix[c2_index, :] = np.inf
    clusters_matrix[:, c2_index] = np.inf
    # update closest clusters for rows, where c1 or c2 were the closest one
    # that's because after merge metrics may change and completely different cluster may become the closest
    # for simplicity here we just update all rows
    num_points = X_matrix.shape[0]
    clusters_matrix[:, cluster_col] = np.argmin(clusters_matrix[:, :num_points], axis=1)
    clusters_matrix[:, distance_col] = np.amin(clusters_matrix[:, :num_points], axis=1)
    # update label assignments in X
    X_matrix[:, -1][X_matrix[:, -1] == c2_index] = c1_index


# performs merge of clusters with indices c1_index, c2_index
# updates average-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, use it for this method
def average_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    # calculate new distances for a new cluster (which will stay at c1 position)
    # collect indices of all points which belong to c1 or c2
    c1_2_points = np.argwhere((X_matrix[:, -1] == c1_index) | (X_matrix[:, -1] == c2_index))

    cluster_ids = np.unique(X_matrix[:, -1])
    for c in cluster_ids:
        if c != c1_index and c != c1_index:
            c_points = np.argwhere(X_matrix[:, -1] == c)
            sum = 0
            all_pairs = itertools.product(c_points, c1_2_points)
            for i, j in all_pairs:
                sum += distances_matrix[i, j]
            clusters_matrix[c1_index, c] = sum / (len(c_points) * len(c1_2_points))

    # update the new cluster's corresponding column as well
    clusters_matrix[:, c1_index] = clusters_matrix[c1_index, 0: distance_col]
    # infinity out row and column for c2
    clusters_matrix[c2_index, :] = np.inf
    clusters_matrix[:, c2_index] = np.inf
    # update closest clusters for all rows
    # after merge metrics may change and completely different cluster may become the closest
    num_points = X_matrix.shape[0]
    clusters_matrix[:, cluster_col] = np.argmin(clusters_matrix[:, :num_points], axis=1)
    clusters_matrix[:, distance_col] = np.amin(clusters_matrix[:, :num_points], axis=1)
    # update label assignments in X
    X_matrix[:, -1][X_matrix[:, -1] == c2_index] = c1_index


# the function which performs bottom-up (agglomerative) clustering
# merge_func - one of the three merge functions above, each with different linkage function
# X_matrix - data itself
# threshold - maximum merge distance, we stop merging if we reached it. if None, merge until there only is one cluster
def bottom_up_clustering(merge_func, X_matrix, distances_matrix, threshold=None):
    num_points = X_matrix.shape[0]

    # take dataset, add and initialize column for cluster membership
    X_data = np.c_[X_matrix, np.arange(0, num_points, 1)]

    # create clusters matrix, initially consisting of all points and pairwise distances
    # with last columns being distance to closest cluster and id of that cluster
    clusters = np.c_[distances_matrix, np.zeros((num_points, 2))]

    # ids of added columns - column with minimal distances, column with closest cluster ids
    dist_col_id = num_points
    clust_col_id = num_points + 1

    # calculate closest clusters and corresponding distances for each cluster
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)

    # array for keeping distances between clusters that we are merging
    merge_distances = np.zeros(num_points - 1)
    # main loop. at each step we are identifying and merging two closest clusters (wrt linkage function)
    for i in range(0, num_points - 1):
        c1_id, c2_id, distance = find_clusters_to_merge(clusters, dist_col_id, clust_col_id)
        # if threshold is set, we don't merge any further if we reached the desired max distance for merging
        if threshold is not None and distance > threshold:
            break
        merge_distances[i] = distance
        merge_func(c1_id, c2_id, X_data, clusters, dist_col_id, clust_col_id, distances_matrix)
        # uncomment when testing
        # print("Merging clusters #", c1_id, c2_id)
        # if i%30 == 0:
        #     for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        #         plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker, label=k)
        #     plt.show()

    # todo use the plot below to find the optimal threshold to stop merging clusters
    plt.plot(np.arange(0, num_points - 1, 1), merge_distances[:num_points - 1])
    plt.title("Merge distances over iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Distance")
    plt.show()

    for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker)
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# creating and populating matrix for storing pairwise distances
# diagonal elements are filled with np.inf to ease further processing
distances = squareform(pdist(X, metric='euclidean'))
np.fill_diagonal(distances, np.inf)

# seting up colors and marker types to use for plotting
markers = markers.MarkerStyle.markers
colormap = plt.cm.Dark2.colors

# performing bottom-up clustering with three different linkage functions
# todo set your own thresholds for each method.
# todo find thresholds by looking at plot titled "Merge distances over iterations" when threshold is set to None
bottom_up_clustering(single_link_merge, X, distances, threshold=10)
bottom_up_clustering(complete_link_merge, X, distances, threshold=60)
bottom_up_clustering(average_link_merge, X, distances, threshold=30)
