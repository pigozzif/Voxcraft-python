import abc

import numpy as np


class Selector(object):

    def select(self, population, n):
        return [self.select_individual(population) for _ in range(n)]

    @abc.abstractmethod
    def select_individual(self, population):
        pass


class WorstSelector(Selector):

    def select_individual(self, population):
        population.sort()
        return population[len(population) - 1]


class TournamentSelector(Selector):

    def __init__(self, size):
        self.size = size

    def select_individual(self, population):
        contenders = np.random.choice(population.individuals, self.size)
        return sorted(contenders, key=lambda x: x.fitness, reverse=True)[0]