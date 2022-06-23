import abc
import random

import numpy as np


class GenotypeFactory(object):

    def __init__(self, genotype_filter):
        self.genotype_filter = genotype_filter

    def create_population(self, pop_size):
        pop = []
        while len(pop) < pop_size:
            new_born = self.create()
            if self.genotype_filter(new_born):
                pop.append(new_born)
        return pop

    @abc.abstractmethod
    def create(self):
        pass

    @classmethod
    def create_factory(cls, name, genotype_filter, **kwargs):
        if name == "uniform_float":
            return UniformFloatFactory(genotype_filter=genotype_filter, n=kwargs["n"], r=kwargs["range"])
        raise ValueError("Invalid genotype factory name: {}".format(name))


class UniformFloatFactory(GenotypeFactory):

    def __init__(self, genotype_filter, n, r):
        super().__init__(genotype_filter)
        self.n = n
        self.l, self.u = r

    def create(self):
        return np.array([random.random() * (self.u - self.l) - self.l for _ in range(self.n)])
