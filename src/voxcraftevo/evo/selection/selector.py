import abc
from typing import List

from voxcraftevo.representations.population import Population
from voxcraftevo.representations.population import Individual


class Selector(object):

    def select(self, population: Population, n: int) -> List[Individual]:
        return [self.select_individual(population=population) for _ in range(n)]

    @abc.abstractmethod
    def select_individual(self, population: Population) -> Individual:
        pass

    @classmethod
    def create_selector(cls, name: str, **kwargs):
        if name == "worst":
            return WorstSelector()
        elif name == "tournament":
            return TournamentSelector(kwargs["tournament_size"])
        raise ValueError("Invalid selector name: {}".format(name))


class WorstSelector(Selector):

    def select_individual(self, population):
        population.sort()
        return population[len(population) - 1]


class TournamentSelector(Selector):

    def __init__(self, size: int, awarder=lambda x: sorted(x, reverse=True)[0]):
        self.size = size
        self.awarder = awarder

    def select_individual(self, population):
        contenders = population.sample(n=self.size)
        return self.awarder(contenders)
