import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import os

os.chdir("..")
from legacy_code.TMFG_core import TMFG as LegacyTMFG
from TMFG_core import TMFG


class TestTMFG(unittest.TestCase):
    def setUp(self):
        self.number_of_test_iterations = 100
        self.fast_tmfg = LegacyTMFG()
        self.optimised_tmfg = TMFG()

    def generate_symmetric_matrix(self, size):
        random_matrix = np.random.rand(size, size)
        return (random_matrix + random_matrix.T) / 2

    def test_unweighted_sparse_matrix(self):
        print("Testing 'unweighted_sparse_matrix' setup...")
        for _ in tqdm(range(self.number_of_test_iterations)):
            W = self.generate_symmetric_matrix(np.random.randint(10, 200))

            (
                fast_cliques,
                fast_separator,
                fast_adj_matrix,
            ) = self.fast_tmfg.fit_transform(
                pd.DataFrame(W), output="unweighted_sparse_W_matrix"
            )
            (
                optimised_cliques,
                optimised_separator,
                optimised_adj_matrix,
            ) = self.optimised_tmfg.fit_transform(
                weights=W, output="unweighted_sparse_W_matrix"
            )

            net = nx.from_numpy_array(optimised_adj_matrix)
            self.assertTrue(nx.is_chordal(net))
            self.assertTrue(nx.is_planar(net))

            self.assertEqual(
                [sorted(clique) for clique in fast_cliques],
                [sorted(clique) for clique in optimised_cliques],
            )
            self.assertEqual(
                [sorted(separator) for separator in fast_separator],
                [sorted(separator) for separator in optimised_separator],
            )
            self.assertTrue(np.array_equal(fast_adj_matrix, optimised_adj_matrix))

    def test_weighted_sparse_matrix(self):
        print("Testing 'weighted_sparse_matrix' setup...")
        for _ in tqdm(range(self.number_of_test_iterations)):
            W = self.generate_symmetric_matrix(np.random.randint(10, 500))

            (
                fast_cliques,
                fast_separator,
                fast_adj_matrix,
            ) = self.fast_tmfg.fit_transform(
                pd.DataFrame(W), output="weighted_sparse_W_matrix"
            )
            (
                optimised_cliques,
                optimised_separator,
                optimised_adj_matrix,
            ) = self.optimised_tmfg.fit_transform(
                weights=W, output="weighted_sparse_W_matrix"
            )

            net = nx.from_numpy_array(optimised_adj_matrix)
            self.assertTrue(nx.is_chordal(net))
            self.assertTrue(nx.is_planar(net))

            self.assertEqual(
                [sorted(clique) for clique in fast_cliques],
                [sorted(clique) for clique in optimised_cliques],
            )
            self.assertEqual(
                [sorted(separator) for separator in fast_separator],
                [sorted(separator) for separator in optimised_separator],
            )
            self.assertTrue(np.array_equal(fast_adj_matrix, optimised_adj_matrix))

    def test_logo_output(self):
        print("Testing 'logo_output' setup...")
        for _ in tqdm(range(self.number_of_test_iterations)):
            W = self.generate_symmetric_matrix(np.random.randint(10, 500))
            cov = np.cov(W)

            (
                fast_cliques,
                fast_separator,
                fast_adj_matrix,
            ) = self.fast_tmfg.fit_transform(
                pd.DataFrame(W), output="logo", cov=pd.DataFrame(cov)
            )
            (
                optimised_cliques,
                optimised_separator,
                optimised_adj_matrix,
            ) = self.optimised_tmfg.fit_transform(weights=W, output="logo", cov=cov)

            self.assertEqual(
                [sorted(clique) for clique in fast_cliques],
                [sorted(clique) for clique in optimised_cliques],
            )
            self.assertEqual(
                [sorted(separator) for separator in fast_separator],
                [sorted(separator) for separator in optimised_separator],
            )
            # Since the inverse of the covariance matrix is used in the logo
            # output, the results may not be exactly the same. We add a
            # tolerance of 1e-5.
            self.assertTrue(
                np.allclose(fast_adj_matrix, optimised_adj_matrix, rtol=1e-5, atol=1e-8)
            )

    def test_edge_case_all_ones_matrix(self):
        W = np.ones((100, 100))

        fast_cliques, fast_separator, fast_adj_matrix = self.fast_tmfg.fit_transform(
            pd.DataFrame(W), output="unweighted_sparse_W_matrix"
        )
        (
            optimised_cliques,
            optimised_separator,
            standard_adj_matrix,
        ) = self.optimised_tmfg.fit_transform(
            weights=W, output="unweighted_sparse_W_matrix"
        )

        self.assertEqual(
            [sorted(clique) for clique in fast_cliques],
            [sorted(clique) for clique in optimised_cliques],
        )
        self.assertEqual(
            [sorted(separator) for separator in fast_separator],
            [sorted(separator) for separator in optimised_separator],
        )
        self.assertTrue(np.array_equal(fast_adj_matrix, standard_adj_matrix))

    def test_pd_dataframe_input(self):
        W = pd.DataFrame(self.generate_symmetric_matrix(100))
        fast_cliques, fast_separator, fast_adj_matrix = self.fast_tmfg.fit_transform(
            W, output="unweighted_sparse_W_matrix"
        )
        (
            optimised_cliques,
            optimised_separator,
            optimised_adj_matrix,
        ) = self.optimised_tmfg.fit_transform(
            weights=W, output="unweighted_sparse_W_matrix"
        )

        self.assertEqual(
            [sorted(clique) for clique in fast_cliques],
            [sorted(clique) for clique in optimised_cliques],
        )
        self.assertEqual(
            [sorted(separator) for separator in fast_separator],
            [sorted(separator) for separator in optimised_separator],
        )
        self.assertTrue(np.array_equal(fast_adj_matrix, optimised_adj_matrix))


if __name__ == "__main__":
    unittest.main()
