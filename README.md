Fast_TMFG is an ultra-fast implementation of the Triangulated Maximally Fileterd Graph (TMFG).

The architecture is fully scikit-learn compatible. Consequently, it has three main methods:

	- `fit(self, c_matrix)`: Fits the model to the input matrix `c_matrix`. This method computes the Triangulated Maximal Filtered Graph (TMFG) based on the input matrix.
	- `transform(self)`: Returns the computed cliques and separators set of the model. The method also returns the TMFG adjacency matrix.
	- `fit_transform(self, c_matrix)`: Fits the model to the input matrix `c_matrix` and returns the computed cliques and separators set and the TMFG adjacency matrix.

We provide a detailed explanation of each function/method. Such an explanation is entirely generated through [ChatGPT](https://chat.openai.com).

For a fully understanding of the TMFG and for future research directions, we refer the interested reader to the follwing papers:

	- [Parsimonious modeling with information filtering networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.94.062306)
	- [Network filtering for big data: Triangulated maximally filtered graph] (https://academic.oup.com/comnet/article/5/2/161/2555365)
	- [Dependency structures in cryptocurrency market from high to low frequency] (https://arxiv.org/pdf/2206.03386.pdf)

If you use Fast-TMFG in a scientific publication, we would appreciate citations to the following paper:

---
@article{briola2022dependency,
  title={Dependency structures in cryptocurrency market from high to low frequency},
  author={Briola, Antonio and Aste, Tomaso},
  journal={arXiv preprint arXiv:2206.03386},
  year={2022}
}
---
