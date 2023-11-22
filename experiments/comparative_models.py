"""
    Competitor dimensionality reduction algorithms.
    Credit to Moor, Michael, et al. "Topological autoencoders." International conference on machine learning. PMLR, 2020.

"""
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from umap import UMAP

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE

except ImportError:
    from sklearn.manifold import TSNE