import dataclasses
from functools import total_ordering

import numpy as np

from voxcraftevo.evo.selection.comparator import Comparator


@dataclasses.dataclass
@total_ordering
class Individual(object):
    id: int
    genotype: object
    solution: object
    comparator: Comparator
    fitness: dict = None
    age: int = 0
    evaluated: bool = False

    def __str__(self):
        return "Individual[id={0},age={1},fitness={2}]".format(self.id, self.age, self.fitness)

    def __eq__(self, other):
        return self.comparator.compare(ind1=self, ind2=other) == 0

    def __lt__(self, other):
        return self.comparator.compare(ind1=self, ind2=other) == -1


class Population(object):

    def __init__(self, pop_size, genotype_factory, solution_mapper, objectives_dict, comparator):
        self.genotype_factory = genotype_factory
        self.solution_mapper = solution_mapper
        self.objectives_dict = objectives_dict
        self.comparator = Comparator.create_comparator(name=comparator, objective_dict=objectives_dict)
        self._individuals = []
        self._max_id = 0
        # init random population (generation 0)
        for g in self.genotype_factory.create_population(pop_size=pop_size):
            self.add_individual(g)
        self.gen = 0

    def __str__(self):
        return "Population[size={0},best={1}]".format(len(self), self.get_best())

    def __len__(self):
        return len(self._individuals)

    def __getitem__(self, item):
        return self._individuals[item]

    def __iter__(self):
        return iter(self._individuals)

    def add_individual(self, genotype):
        self._individuals.append(Individual(id=self._max_id,
                                            genotype=genotype,
                                            solution=self.solution_mapper(genotype),
                                            comparator=self.comparator))
        self._max_id += 1

    def remove_individual(self, ind):
        self._individuals.remove(ind)

    def clear(self):
        self._individuals = []

    def update_ages(self):
        for ind in self:
            ind.age += 1

    def sort(self):
        self._individuals.sort(reverse=True)

    def get_best(self):
        self.sort()
        return self[0]

    def sample(self, n):
        return np.random.choice(self, size=n)
