from learned_dictionary import AudioDictionaryLearner

import cr.sparse.pursuit.omp as omp
import cr.sparse.pursuit.sp as sp
import cr.sparse.cvx.adm.yall1 as bp

import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

A_1 = AudioDictionaryLearner().learn_dictionary()
A_2 = jnp.array()

class SparseApproximation():
    def __init__(self, algorithm: callable, dictionary: jnp.array):
        self.algorithm = algorithm
        self.dictionary = dictionary

    def approximate(self, input):
        X = self.algorithm(self.dictionary, input)
        return X.x


if __name__ == "__main__":
    for dict in [A_1, A_2]:
        for alg in [omp.solve, bp.solve, sp.solve]:
            cls = SparseApproximation(alg, dict)
            res = cls.approximate()
