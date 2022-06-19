import abc
import random

import numpy as np


class GenotypeFactory(object):

    @abc.abstractmethod
    def __call__(self):
        pass

    @classmethod
    def create_factory(cls, name, **kwargs):
        if name == "uniform_float":
            return UniformFloatFactory(n=kwargs["n"], r=kwargs["range"])
        raise ValueError("Invalid genotype factory name: {}".format(name))


class UniformFloatFactory(GenotypeFactory):

    def __init__(self, n, r):
        self.n = n
        self.l, self.u = r

    def __call__(self):
        return np.array([random.random() * (self.u - self.l) - self.l for _ in range(self.n)])
