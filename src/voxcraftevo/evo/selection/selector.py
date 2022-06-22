import abc


class Selector(object):

    def select(self, population, n):
        return [self.select_individual(population=population) for _ in range(n)]

    @abc.abstractmethod
    def select_individual(self, population):
        pass

    @classmethod
    def create_selector(cls, name, **kwargs):
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

    def __init__(self, size):
        self.size = size

    def select_individual(self, population):
        contenders = population.sample(n=self.size)
        return sorted(contenders, reverse=True)[0]
