
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib import markers
import itertools


# In[122]:


# finds indices of closest clusters to be merged on next iteration
# clusters_matrix - clusters matrix
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# returns cluster indices and distance between them
def find_clusters_to_merge(clusters_matrix, distance_col, cluster_col):
    min_index = np.argmin(clusters_matrix[:, distance_col])
    min_dist = clusters_matrix[min_index, distance_col]
    target = clusters_matrix[min_index, cluster_col]
    
#     print("DEBUG FROM FIND CLUSTERS")
#     print(min_index, target, min_dist)

    
    return min_index, target, min_dist

def get_members(sets, c1, c2):
    sets[c1] = list(sets[c1] + sets[c2])
    sets[c2] = sets[c1]
    return sets[c1]

def get_participants(X_data):
    d = {}
    uniq = np.unique(X_data[:, -1])
    for v in uniq:
        i = int(v)
        d[i] = [i]
    return d

def update_clusters(X_matrix, clusters_matrix, members, cluster_id, val=np.inf):
    for index in members:
        for subindex in members:
            clusters_matrix[index, subindex] = val
            clusters_matrix[subindex, index] = val

        X_matrix[index, -1] = cluster_id


# In[135]:


# performs merge of clusters with indices c1_index, c2_index
# updates single-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def single_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix, sets):
    c1 = int(c1_index)
    c2 = int(c2_index)
    
    members_c1 = get_members(sets, c1, c2)
    update_clusters(X_matrix, clusters_matrix, members_c1, c1, val=np.inf)

#     print("c1 = {} c2 = {}, merge distance = {}".format(c1, c2, clusters_matrix[c1, c2]))
#     print(members_c1)
    
    
    for cluster_id in range(X_matrix.shape[0]):
        if cluster_id not in members_c1:
            v1 = clusters_matrix[c1, cluster_id]
            v2 = clusters_matrix[c2, cluster_id]
            min_dist = min(v1, v2)
            clusters_matrix[cluster_id, c1] = min_dist
            clusters_matrix[c1, cluster_id] = min_dist
            clusters_matrix[cluster_id, c2] = np.inf
            clusters_matrix[c2, cluster_id] = np.inf
        
    for cluster_id in range(X_matrix.shape[0]):
        min_index = np.argmin(clusters_matrix[cluster_id, :distance_col])
        min_dist = clusters_matrix[cluster_id, min_index]
        clusters_matrix[cluster_id, distance_col] = min_dist
        clusters_matrix[cluster_id, cluster_col] = min_index


# In[137]:


# performs merge of clusters with indices c1_index, c2_index
# updates complete-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def complete_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix, sets):
    c1 = int(c1_index)
    c2 = int(c2_index)
    
#     print("c1 = {} c2 = {}, merge distance = {}".format(c1, c2, clusters_matrix[c1, c2]))
#     print("Merging l1 = {} and l2 = {}".format(sets[c1], sets[c2]))


    members_c1 = get_members(sets, c1, c2)
#     print("Result merge = {}".format(members_c1))

    update_clusters(X_matrix, clusters_matrix, members_c1, c1, val=-1)
    
    
    for cluster_id in range(X_matrix.shape[0]):
        if cluster_id not in members_c1:
            v1 = clusters_matrix[c1, cluster_id]
            v2 = clusters_matrix[c2, cluster_id]
            max_dist = max(v1, v2)
            clusters_matrix[cluster_id, c1] = max_dist
            clusters_matrix[c1, cluster_id] = max_dist
            clusters_matrix[cluster_id, c2] = np.inf
            clusters_matrix[c2, cluster_id] = np.inf
        
        
    update_clusters(X_matrix, clusters_matrix, members_c1, c1, val=np.inf)

    for cluster_id in range(X_matrix.shape[0]):
        min_index = np.argmin(clusters_matrix[cluster_id, :distance_col])
        min_dist = clusters_matrix[cluster_id, min_index]
        clusters_matrix[cluster_id, distance_col] = min_dist
        clusters_matrix[cluster_id, cluster_col] = min_index


# In[140]:


# performs merge of clusters with indices c1_index, c2_index
# updates average-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, use it for this method
def average_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix, sets):
    c1 = int(c1_index)
    c2 = int(c2_index)
    
#     print("c1 = {} c2 = {}, merge distance = {}".format(c1, c2, clusters_matrix[c1, c2]))
#     print("Merging l1 = {} and l2 = {}".format(sets[c1], sets[c2]))


    members_c1 = get_members(sets, c1, c2)
#     print("Result merge = {}".format(members_c1))

    update_clusters(X_matrix, clusters_matrix, members_c1, c1, val=np.inf)
    
    for cluster_id in range(X_matrix.shape[0]):
        if cluster_id not in members_c1:
            v1 = clusters_matrix[c1, cluster_id]
            v2 = clusters_matrix[c2, cluster_id]
            dist = (v1 + v2) / 2
            clusters_matrix[cluster_id, c1] = dist
            clusters_matrix[c1, cluster_id] = dist
            clusters_matrix[cluster_id, c2] = np.inf
            clusters_matrix[c2, cluster_id] = np.inf

    for cluster_id in range(X_matrix.shape[0]):
        min_index = np.argmin(clusters_matrix[cluster_id, :distance_col])
        min_dist = clusters_matrix[cluster_id, min_index]
        clusters_matrix[cluster_id, distance_col] = min_dist
        clusters_matrix[cluster_id, cluster_col] = min_index


# In[144]:


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
    
    # My variable
    sets = get_participants(X_data)
    # main loop. at each step we are identifying and merging two closest clusters (wrt linkage function)
    for i in range(0, num_points - 1):
        c1_id, c2_id, distance = find_clusters_to_merge(clusters, dist_col_id, clust_col_id)
        # if threshold is set, we don't merge any further if we reached the desired max distance for merging
        if threshold is not None and distance > threshold:
            break
        merge_distances[i] = distance
        merge_func(c1_id, c2_id, X_data, clusters, dist_col_id, clust_col_id, distances_matrix, sets)
#         break
        # uncomment when testing
#         print("Merging clusters #", c1_id, c2_id)
#         if i%30 == 0:
#             for k, (marker, color) in zip(range(num_points), itertools.product(markerss, colormap)):
#                 plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker, label=k)
#             plt.show()
    
    # todo use the plot below to find the optimal threshold to stop merging clusters
    plt.plot(np.arange(0, num_points - 1, 1), merge_distances[:num_points - 1])
    plt.title("Merge distances over iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Distance")
    plt.show()

    for k, (marker, color) in zip(range(num_points), itertools.product(markerss, colormap)):
        plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker)
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()




# In[5]:


# importing the dataset
dataset = pd.read_csv('datasets/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# creating and populating matrix for storing pairwise distances
# diagonal elements are filled with np.inf to ease further processing
distances = squareform(pdist(X, metric='euclidean'))
np.fill_diagonal(distances, np.inf)

# seting up colors and marker types to use for plotting
markerss = markers.MarkerStyle.markers
colormap = plt.cm.Dark2.colors


# In[149]:



# performing bottom-up clustering with three different linkage functions
# todo set your own thresholds for each method.
# todo find thresholds by looking at plot titled "Merge distances over iterations" when threshold is set to None
bottom_up_clustering(single_link_merge, X, distances, threshold=10)
bottom_up_clustering(complete_link_merge, X, distances, threshold=10)
bottom_up_clustering(average_link_merge, X, distances, threshold=10)

