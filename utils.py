import numpy as np
import time

def max_clique(W):
    '''
    The `max_clique` function takes an input matrix `W` and returns a list of the four vertices with the highest sum of connections to other vertices.

    The function does the following:

    - It flattens the `W` matrix into a 1D array and sets it to `flat_matrix`.
    - It calculates the mean of `flat_matrix` and sets it to `mean_flat_matrix`.
    - It creates an array `v` where each element is the sum of the connections for the corresponding vertex in `W`, considering only connections with a value greater than `mean_flat_matrix`.
    - It sorts the `v` array in descending order and sets the result to `sorted_v`.
    - It returns the first four elements of `sorted_v`.
    
    This function is used to find the maximum clique in the input matrix, which is the clique with the highest sum of connections among its vertices. The returned list will contain the indices of the four vertices with the highest sum of connections.
    '''

    flat_matrix = np.array(W).flatten()
    mean_flat_matrix = np.mean(flat_matrix)
    v = np.sum(np.multiply(W, (W > mean_flat_matrix)), axis=1)
    sorted_v = np.argsort(v)[::-1]
    return sorted_v[:4]


def get_best_gain(n, vertex_list, triangle, W, no_vertex_list):
    '''
    The `get_best_gain` function takes in the following arguments:

    - `n`: an integer representing the number of vertices in the graph.
    - `vertex_list`: a list of vertices that are currently being considered for addition to the TMFG.
    - `triangle`: a list of three vertices that are already part of the TMFG.
    - `W`: the adjacency matrix of the graph.
    - `no_vertex_list`: a list of vertices that are not currently being considered for addition to the TMFG.
    
    The function returns the index and value of the vertex in vertex_list that would result in the highest gain when added to the TMFG. The gain is calculated as the sum of the connections between the vertex and the vertices in the triangle.

    The function does the following:

        . If `no_vertex_list` is None, it sets it to the list of vertices in the graph that are not in `vertex_list`.
        . It initializes `gvec` to None.
        . It iterates through the vertices in triangle and, for each vertex, selects the corresponding column in `W` and sets the elements in the `no_vertex_list` to 0. If `gvec` is None, it sets `gvec` to this modified column. Otherwise, it adds the modified column to `gvec`.
        . It returns the index and value of the maximum element in `gvec`.
    
    This function is used to determine which vertex in `vertex_list` would result in the highest gain when added to the TMFG. The resulting index and value are used to update the `max_clique_gains` and `best_vertex` lists in the `__compute_TMFG` function.
    '''
    gvec = None

    if no_vertex_list is None:
        original_range = range(n)
        no_vertex_list = np.setdiff1d(original_range, vertex_list)

    for tr in triangle:
        column = W[:][tr]
        column[no_vertex_list] = 0

        if gvec is None:
            gvec = column
        else:
            gvec = gvec + column

    index_max = np.argmax(gvec)
    max_element = np.max(gvec)
    return index_max, max_element