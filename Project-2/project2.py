from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import cr.sparse as crs
import cr.nimble as crn
from cr.sparse.dict import (
    gaussian_mtx,
    fourier_basis,
    cosine_basis,
    random_orthonormal_rows
)
from cr.sparse.pursuit import (
    mp,
    omp,
    cosamp,
    sp
)
from cr.sparse.cvx.adm import yall1
import cr.sparse.data as crdata
from cr.nimble.dsp import (
    nonzero_indices,
    nonzero_values
)


keys = random.split(random.PRNGKey(420))


N = 1000
M = 300
K = 50

nDict = gaussian_mtx(keys[0], N, M)

crn.has_orthogonal_rows(nDict)


x = crdata.sparse_normal_representations(keys[1], N, K, 1)

x = jnp.squeeze(x)


