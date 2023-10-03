# Fast-TMFG

Fast_TMFG is an ultra-fast implementation of the Triangulated Maximally Fileterd Graph (TMFG). It is based on the work by [G. P. Massara](https://github.com/gprevide/MFCF-Pyton/tree/main/src) and is fully implemented by [A. Briola](https://github.com/AntoBr96).

The interface is fully scikit-learn compatible. Consequently, it has three main methods:
- `fit(weights, output)`: Fits the model to the input matrix `weights` (e.g. a squared correlation matrix). This method computes the Triangulated Maximal Filtered Graph (TMFG) based on the input matrix. The `output` parameter specifies what is the nature of the desired output:
  - sparse inverse covariance matrix (`output = 'logo'`)
  - sparse unweighted weights matrix (`output = 'unweighted_sparse_W_matrix'`)
  - sparse weighted weights matrix (`output = 'weighted_sparse_W_matrix'`)
- `transform()`: Returns the computed cliques and separators set of the model. The method also returns the TMFG adjacency matrix.
- `fit_transform(weights, output)`: Fits the model to the input matrix `weights` (e.g. a squared correlation matrix) and returns the computed cliques and separators set and the TMFG adjacency matrix. The `output` parameter specifies what is the nature of the desired output:
  - sparse inverse covariance matrix (`output = 'logo'`)
  - sparse unweighted weights matrix (`output = 'unweighted_sparse_W_matrix'`)
  - sparse weighted weights matrix (`output = 'weighted_sparse_W_matrix'`)

We provide a detailed explanation of each function/method. Such an explanation is entirely generated through [ChatGPT](https://chat.openai.com).

For a fully understanding of the TMFG, we refer the interested reader to the follwing papers:
- [Parsimonious modeling with information filtering networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.94.062306)
- [Network filtering for big data: Triangulated maximally filtered graph](https://academic.oup.com/comnet/article/5/2/161/2555365)
- [Dependency structures in cryptocurrency market from high to low frequency](https://arxiv.org/pdf/2206.03386.pdf)

# Installation
Install the latest version of the package using [PyPI](https://pypi.org/project/fast-tmfg/):
```pip3 install fast-tmfg```

# Usage Example
```python
import numpy as np
import pandas as pd

from fast_tmfg import *

def generate_random_df(num_rows, num_columns):
  data = np.random.randint(0, 100, size=(num_rows, num_columns))
  df = pd.DataFrame(data, columns=['col_{}'.format(i) for i in range(num_columns)])
  return df

df = generate_random_df(100, 50)
corr = np.square(df.corr())
model = TMFG()
cliques, seps, adj_matrix = model.fit_transform(corr, output='unweighted_sparse_W_matrix')
```

# How to cite us

If you use TMFG in a scientific publication, we would appreciate citations to the following paper:

```
@article{briola2022dependency,
  title={Dependency structures in cryptocurrency market from high to low frequency},
  author={Briola, Antonio and Aste, Tomaso},
  journal={arXiv preprint arXiv:2206.03386},
  year={2022}
}
```
