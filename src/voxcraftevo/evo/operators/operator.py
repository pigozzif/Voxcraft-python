import abc
import copy
import random

import numpy as np


class GeneticOperator(object):

    @abc.abstractmethod
    def apply(self, *args):
        pass

    @abc.abstractmethod
    def get_arity(self):
        pass

    @classmethod
    def create_genetic_operator(cls, name, **kwargs):
        if name == "gaussian_mut":
            return GaussianMutation(mu=kwargs["kwargs"]["mu"], sigma=kwargs["kwargs"]["sigma"])
        elif name == "geometric_cx":
            return GeometricCrossover(upper=kwargs["kwargs"]["upper"], lower=kwargs["kwargs"]["lower"])
        raise ValueError("Invalid genetic operator name: {}".format(name))


class GaussianMutation(GeneticOperator):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def apply(self, *args):
        if len(args) > 1:
            raise ValueError("More than one parent for mutation")
        child = copy.deepcopy(args[0])
        child += np.random.normal(self.mu, self.sigma, len(child))
        return child

    def get_arity(self):
        return 1


class GeometricCrossover(GeneticOperator):

    def __init__(self, upper, lower):
        self.upper = upper
        self.lower = lower

    def apply(self, *args):
        if len(args) > 2:
            raise ValueError("More than two parents for crossover")
        parent1, parent2 = args
        return np.array([v1 + (v2 - v1) * (random.random() * (self.upper - self.lower) + self.lower)
                         for v1, v2 in zip(parent1, parent2)])

    def get_arity(self):
        return 2
