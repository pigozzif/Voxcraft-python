import abc
import copy
import random

import numpy as np


class GeneticOperator(object):

    def __init__(self, phenotype_filter):
        self.phenotype_filter = phenotype_filter

    def apply(self, *args):
        new_born = self.propose(args)
        while not self.phenotype_filter(new_born):
            new_born = self.propose(args)
        return new_born

    @abc.abstractmethod
    def propose(self, *args):
        pass

    @abc.abstractmethod
    def get_arity(self):
        pass

    @classmethod
    def create_genetic_operator(cls, name, phenotype_filter, **kwargs):
        if name == "gaussian_mut":
            return GaussianMutation(phenotype_filter=phenotype_filter, mu=kwargs["mu"], sigma=kwargs["sigma"])
        elif name == "geometric_cx":
            return GeometricCrossover(phenotype_filter=phenotype_filter, upper=kwargs["upper"], lower=kwargs["lower"])
        raise ValueError("Invalid genetic operator name: {}".format(name))


class GaussianMutation(GeneticOperator):

    def __init__(self, phenotype_filter, mu, sigma):
        super().__init__(phenotype_filter)
        self.mu = mu
        self.sigma = sigma

    def propose(self, *args):
        if len(args) > 1:
            raise ValueError("More than one parent for mutation")
        child = copy.deepcopy(args[0])
        child += np.random.normal(self.mu, self.sigma, len(child))
        return child

    def get_arity(self):
        return 1


class GeometricCrossover(GeneticOperator):

    def __init__(self, phenotype_filter, upper, lower):
        super().__init__(phenotype_filter)
        self.upper = upper
        self.lower = lower

    def propose(self, *args):
        if len(args) > 2:
            raise ValueError("More than two parents for crossover")
        parent1, parent2 = args
        return np.array([v1 + (v2 - v1) * (random.random() * (self.upper - self.lower) + self.lower)
                         for v1, v2 in zip(parent1, parent2)])

    def get_arity(self):
        return 2
