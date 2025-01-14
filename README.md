# Fast-TMFG

Fast_TMFG is an ultra-fast implementation of the Triangulated Maximally Fileterd Graph (TMFG). It is based on the work by [Guido Previde Massara](https://github.com/gprevide/MFCF-Pyton/tree/main/src) and is fully implemented [Yan Huang](https://github.com/yanh11), replacing the [previous version](legacy_code/TMFG_core.py) proposed by [Antonio Briola](https://github.com/AntoBr96) and [Tong Zheng](https://github.com/tz1003). The details of the optimised update can be found [here](legacy_code/README.md).

The interface is fully scikit-learn compatible. Consequently, it has three main methods:
- `fit(weights, cov, output)`: Fits the model to the input matrix `weights` (e.g. a squared correlation matrix) and input matrix `cov` (e.g. covariance matrix). This method computes the Triangulated Maximal Filtered Graph (TMFG) based on the input weight matrix. The `output` parameter specifies what is the nature of the desired output:
  - sparse inverse covariance matrix (`output = 'logo'`)
  - sparse unweighted weights matrix (`output = 'unweighted_sparse_W_matrix'`)
  - sparse weighted weights matrix (`output = 'weighted_sparse_W_matrix'`)
- `transform()`: Returns the computed cliques and separators set of the model. The method also returns the TMFG adjacency matrix.
- `fit_transform(weights, cov, output)`: Fits the model to the input matrix `weights` (e.g. a squared correlation matrix) and input matrix `cov` (e.g. covariance matrix), and returns the computed cliques and separators set and the TMFG adjacency matrix over the covariance matrix input. The `output` parameter specifies what is the nature of the desired output:
  - sparse inverse covariance matrix (`output = 'logo'`)
  - sparse unweighted weights matrix (`output = 'unweighted_sparse_W_matrix'`)
  - sparse weighted weights matrix (`output = 'weighted_sparse_W_matrix'`)

We provide a detailed explanation of each function/method. Such an explanation is entirely generated through [ChatGPT](https://chat.openai.com).

For a full understanding of the TMFG, we refer the interested reader to the following papers:
- [Parsimonious modelling with information filtering networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.94.062306)
- [Network filtering for big data: Triangulated maximally filtered graph](https://academic.oup.com/comnet/article/5/2/161/2555365)
- [Dependency structures in cryptocurrency market from high to low frequency](https://arxiv.org/pdf/2206.03386.pdf)

For the use of the TMFG as a topological regularization tool for the covariance selection problem, we further refer the interested reader to the following paper:
- [Topological regularization with information filtering networks](https://www.sciencedirect.com/science/article/pii/S0020025522005904)

# Installation
Install the latest version of the package using [PyPI](https://pypi.org/project/fast-tmfg/):
```pip3 install fast-tmfg```

# Usage Example
## Numpy Input (Preferred)
```python
import numpy as np
from fast_tmfg import *

data = np.random.randint(0, 100, size=(100, 50))
corr = np.square(np.corrcoef(data, rowvar=False))
cov = np.cov(data, rowvar=False)
model = TMFG()
cliques, seps, adj_matrix = model.fit_transform(weights=corr, cov=cov, output='logo')
```

## Pandas DataFrame Input
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
cov = df.cov()
model = TMFG()
cliques, seps, adj_matrix = model.fit_transform(weights=corr, cov=cov, output='logo')
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
