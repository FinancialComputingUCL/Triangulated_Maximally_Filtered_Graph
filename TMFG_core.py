import heapq
import numpy as np
import pandas as pd

from enum import Enum
from typing import Union, Optional
from numpy.linalg import inv


class OutputMode(Enum):
    LOGO = "logo"
    UNWEIGHTED_SPARSE_W_MATRIX = "unweighted_sparse_W_matrix"
    WEIGHTED_SPARSE_W_MATRIX = "weighted_sparse_W_matrix"


class TMFG:
    """
    The TMFG (Triangulated Maximally Filtered Graph) class implements the
    Triangulated Maximally Filtered Graph construction from a given
    correlation matrix. The TMFG is a sparse network representation technique
    useful in complex network analysis, particularly for large correlation
    matrices.

    The TMFG algorithm starts by choosing an initial 4-clique and then
    iteratively adds vertices that maximize a certain gain criterion until no
    vertices remain. The final output is the set of cliques, separators,
    and an adjacency matrix representing the TMFG.

    Attributes
    ----------
    _W : np.ndarray
        The input correlation (or similarity) matrix.
    _cov : np.ndarray
        The covariance matrix.
    _output_mode : OutputMode
        The desired output mode.
    _N : int
        The size (number of vertices) of the input matrix.
    _gains_pq : list[tuple[float, int, int]]
        A priority queue (implemented via heapq) holding tuples of
        (-gain, vertex, triangle_index) to select the best vertex insertion at
        each step.
    _cliques : list[list[int]]
        A list of 4-cliques formed during the TMFG construction.
    _separators : list[list[int]]
        A list of 3-clique separators corresponding to the edges added during
        vertex insertions.
    _triangles : list[list[int]]
        A list of current triangles. Each triangle is a 3-clique to which a new
        vertex can be added.
    _remaining_vertices_mask : np.ndarray
        A boolean mask indicating which vertices remain to be inserted into the
        TMFG.
    _J : np.ndarray
        The adjacency matrix of the TMFG once constructed. Initially None until
        computed.
    """

    def __init__(self):
        pass

    def fit(
        self,
        weights: Union[np.ndarray, pd.DataFrame],
        output: str,
        cov: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> None:
        """
        The `fit` method is a member of the `TMFG` class. It is used to fit
        the model to the input matrix W.

        Parameters
        ----------
        weights : np.ndarray or pd.DataFrame
            The input correlation (or similarity) matrix. Must be square and
            symmetric.
        output : str
            The desired output:
            - sparse inverse covariance matrix (output = 'logo')
            - sparse unweighted weights matrix (output = 'unweighted_sparse_W_matrix')
            - sparse weighted weights matrix (output = 'weighted_sparse_W_matrix')
        cov : np.ndarray or pd.DataFrame, optional
            The covariance matrix. Default is None.
        """
        if isinstance(weights, pd.DataFrame):
            weights = weights.to_numpy()
        self._W = weights.copy()

        if cov is not None:
            if isinstance(cov, pd.DataFrame):
                cov = cov.to_numpy()
            self._cov = cov.copy()

        self._output_mode = OutputMode(output)

        self._initialize()
        self._compute_TMFG()

    def transform(self) -> tuple[list[list[int]], list[list[int]], np.ndarray]:
        """
        Return the TMFG components after fitting the model.

        Returns
        -------
        cliques : list[list[int]]
            The list of 4-cliques constructed by TMFG.
        separators : list[list[int]]
            The list of 3-clique separators.
        J : np.ndarray
            The adjacency matrix depending on the output mode.
        """
        return self._cliques, self._separators, self._J

    def fit_transform(
        self,
        weights: Union[np.ndarray, pd.DataFrame],
        output: str,
        cov: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> tuple[list, list, np.ndarray]:
        """
        Fit the model to the input matrix and return the TMFG components.

        Parameters
        ----------
        weights : np.ndarray or pd.DataFrame
            The input correlation (or similarity) matrix. Must be square and
            symmetric.
        output : str
            The desired output:
            - sparse inverse covariance matrix (output = 'logo')
            - sparse unweighted weights matrix (output = 'unweighted_sparse_W_matrix')
            - sparse weighted weights matrix (output = 'weighted_sparse_W_matrix')
        cov : np.ndarray or pd.DataFrame, optional
            The covariance matrix. Default is None.

        Returns
        -------
        cliques : list[list[int]]
            The list of 4-cliques constructed by TMFG.
        separators : list[list[int]]
            The list of 3-clique separators.
        J : np.ndarray
            The adjacency matrix depending on the output mode.
        """
        self.fit(weights=weights, output=output, cov=cov)
        return self._cliques, self._separators, self._J

    def _initialize(self):
        """
        Perform initialization steps for the TMFG construction:
        - Initialize instance variables.
        - Select the initial 4-clique.
        - Create the initial 4 triangles (from the 4-clique).
        - Prepare the priority queue of vertex gains for each triangle.
        - Mark the initial 4-clique vertices as inserted.
        - Set diagonal of W to zero.
        """
        self._N = self._W.shape[1]

        # Priority queue for gains: (-gain, best_vertex, triangle_index)
        self._gains_pq = []

        self._cliques = []
        self._separators = []
        self._triangles = []

        self._remaining_vertices_mask = np.ones(self._N, dtype=bool)
        self._J = None

        # Find the initial 4-clique and create the initial triangles
        c0 = self._max_clique()
        self._cliques.append(c0)

        vx, vy, vz, vw = c0
        self._triangles = [[vx, vy, vz], [vx, vy, vw], [vx, vz, vw], [vy, vz, vw]]

        self._remaining_vertices_mask[c0] = False
        np.fill_diagonal(self._W, 0)

        self._triangle_columns_sum_cache = {}
        for i, t in enumerate(self._triangles):
            best_v, best_gain = self._get_best_gain(t)
            heapq.heappush(self._gains_pq, (-best_gain, best_v, i))

    def _compute_TMFG(self) -> tuple[list[list[int]], list[list[int]], np.ndarray]:
        """
        Compute the Triangulated Maximally Filtered Graph (TMFG).

        This method iteratively extracts the best gain vertex insertion from the
        priority queue and updates the network structure until all vertices have
        been inserted.

        Returns
        -------
        cliques : list[list[int]]
            The list of 4-cliques constructed by TMFG.
        separators : list[list[int]]
            The list of 3-clique separators.
        JS : np.ndarray
            The adjacency matrix of the unweighted TMFG (1/0 entries).

        Notes
        -----
        After computation, self._J will hold the adjacency matrix.
        """
        while len(self._cliques) < self._N - 3:
            # Each step, we pick the best gain vertex and the triangle to insert into
            gain, v, triangle_idx = heapq.heappop(self._gains_pq)
            if not self._remaining_vertices_mask[v]:
                # If vertex is already inserted, just recalculate for that triangle
                self._update_gains([triangle_idx])
            else:
                # Insert the vertex into the TMFG and update gains
                new_triangles_idxes = self._insert_vertex(v, triangle_idx)
                self._update_gains(new_triangles_idxes)

        self._get_adjacency_matrix()
        return self._cliques, self._separators, self._J

    def _max_clique(self) -> list[int]:
        """
        Identify the initial 4-clique to start the TMFG construction.

        The heuristic used is:
        - Compute a "score" for each vertex as the sum of edges above the mean
        value.
        - Select the top-4 scoring vertices.

        Returns
        -------
        np.ndarray
            Array of 4 vertex indices forming the initial 4-clique.
        """
        mean_val = np.mean(self._W)
        v = np.sum(np.multiply(self._W, (self._W > mean_val)), axis=1)
        # [::-1] added to prevent regression, but it is not needed
        return list(np.argsort(v)[-4:][::-1])

    def _triangle_columns_sum(self, triangle: tuple[int]) -> np.ndarray:
        """
        Compute the sum of edges connected to the given triangle. The results
        are cached because we might update the best gain for a triangle if
        the original best vertex is used.

        Parameters
        ----------
        triangle : tuple[int]
            A tuple of 3 vertex indices forming a triangle. Note that the
            triangle is a tuple to allow caching.

        Returns
        -------
        np.ndarray
            A 1D array where each element is the sum of the corresponding row's
            edges to the triangle vertices.
        """
        if triangle not in self._triangle_columns_sum_cache:
            self._triangle_columns_sum_cache[triangle] =  (
                self._W[:, triangle[0]] + self._W[:, triangle[1]] + self._W[:, triangle[2]]
            )
        return self._triangle_columns_sum_cache[triangle]

    def _get_best_gain(self, triangle: list[int]) -> tuple[int, float]:
        """
        Find the best vertex to add to a given triangle based on gain.

        The gain is defined as the sum of correlation values from candidate
        vertices to the vertices of the triangle. Only vertices not yet
        inserted are considered.

        Parameters
        ----------
        triangle : list[int]
            List of three vertex indices forming the triangle.

        Returns
        -------
        best_v : int
            The index of the vertex that yields the highest gain.
        best_gain : float
            The maximum gain value.
        """
        gvec = self._triangle_columns_sum(tuple(triangle))
        # Only consider vertices not yet inserted
        gvec *= self._remaining_vertices_mask

        best_v = np.argmax(gvec)
        best_gain = gvec[best_v]
        return best_v, best_gain

    def _insert_vertex(self, vertex: int, triangle_idx: int):
        """
        Insert a new vertex into the TMFG by "expanding" a triangle into three
        new triangles.

        This operation:
        - Adds a new 4-clique formed by the triangle and the new vertex.
        - Stores the original triangle as a separator.
        - Updates the set of triangles to reflect the insertion of the new vertex.

        Parameters
        ----------
        vertex : int
            The vertex index to be inserted.
        triangle_idx : int
            The index of the triangle into which the vertex will be inserted.

        Returns
        -------
        new_triangles_idxes : list[int]
            The indices of the new triangles created.
        """
        triangle = self._triangles[triangle_idx]
        del self._triangle_columns_sum_cache[tuple(triangle)]
        self._cliques.append(triangle + [vertex])
        self._separators.append(triangle)

        self._remaining_vertices_mask[vertex] = False

        vx, vy, vz = triangle
        new_triangles = [[vx, vy, vertex], [vy, vz, vertex], [vx, vz, vertex]]
        self._triangles[triangle_idx] = new_triangles[0]
        self._triangles.extend(new_triangles[1:])
        new_triangles_idxes = [
            triangle_idx,
            len(self._triangles) - 2,
            len(self._triangles) - 1,
        ]
        return new_triangles_idxes

    def _update_gains(self, new_triangles_idxes: list[int]):
        """
        Update the priority queue with new gain values for newly created or
        modified triangles.

        Parameters
        ----------
        new_triangles_idxes : list[int]
            Indices of the triangles whose gains need to be recalculated and
            re-pushed onto the queue.
        """
        for i in new_triangles_idxes:
            triangle = self._triangles[i]
            best_v, best_gain = self._get_best_gain(triangle)
            heapq.heappush(self._gains_pq, (-best_gain, best_v, i))

    def _get_adjacency_matrix(self):
        """
        Construct the adjacency matrix (J) of the TMFG based on output_mode.
        """
        if self._output_mode == OutputMode.LOGO:
            self._logo()
        elif self._output_mode == OutputMode.UNWEIGHTED_SPARSE_W_MATRIX:
            self._unweighted_sparse_W_matrix()
        elif self._output_mode == OutputMode.WEIGHTED_SPARSE_W_MATRIX:
            self._weighted_sparse_W_matrix()

    def _unweighted_sparse_W_matrix(self):
        """
        Construct the unweighted adjacency matrix of the TMFG.

        JS[i, j] = 1 if there is an edge between i and j in the TMFG, and 0 otherwise.
        """
        self._J = np.zeros((self._N, self._N))
        for c in self._cliques:
            self._J[np.ix_(c, c)] = 1
        np.fill_diagonal(self._J, 0)

    def _weighted_sparse_W_matrix(self):
        """
        Construct a matrix representation of the Triangulated Maximal
        Filtered Graph (TMFG). The resulting `J` matrix will have a value
        -1 <= 0 <= 1 for each pair of vertices that are connected in the TMFG
        and a value of 0 for each pair that are disconnected.
        """
        self._J = np.zeros((self._N, self._N))
        for c in self._cliques:
            self._J[np.ix_(c, c)] = self._W[np.ix_(c, c)]
        np.fill_diagonal(self._J, 0)

    def _logo(self):
        """
        Construct the sparse inverse covariance matrix of the TMFG.
        """
        self._J = np.zeros((self._N, self._N))
        for c in self._cliques:
            self._J[np.ix_(c, c)] += inv(self._cov[np.ix_(c, c)])
        for s in self._separators:
            self._J[np.ix_(s, s)] -= inv(self._cov[np.ix_(s, s)])
