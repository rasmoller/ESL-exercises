from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import cr.sparse as crs
import cr.sparse.data as crdata
from cr.nimble.dsp import (
    nonzero_indices,
    nonzero_values
)


