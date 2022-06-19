import dataclasses


@dataclasses.dataclass
class Individual(object):
    id: int
    genotype: object
    solution: object
    fitness: float = None
    age: int = 0
    evaluated: bool = False

    def __str__(self):
        return "Individual[id={0},age={1},fitness={2}]".format(self.id, self.age, self.fitness)

    def __eq__(self, other):
        return self.id == other.id


class Population(object):

    def __init__(self, pop_size, genotype_factory, solution_mapper):
        self.genotype_factory = genotype_factory
        self.solution_mapper = solution_mapper
        self._individuals = []
        self._max_id = 0
        # init random population (generation 0)
        for _ in range(pop_size):
            self.add_random_individual()
        self.gen = 0
        self._i = 0

    def __str__(self):
        return "Population[size={0},best={1}]".format(len(self), self.get_best())

    def __len__(self):
        return len(self._individuals)

    def __getitem__(self, item):
        return self._individuals[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration
        self._i += 1
        return self._individuals[self._i - 1]

    def add_random_individual(self):
        genotype = self.genotype_factory()
        self.add_individual(genotype)

    def add_individual(self, genotype):
        self._individuals.append(Individual(self._max_id, genotype, self.solution_mapper(genotype)))
        self._max_id += 1

    def remove_individual(self, ind):
        self._individuals.remove(ind)

    def clear(self):
        self._individuals = []

    def update_ages(self):
        for ind in self:
            ind.age += 1

    def sort(self):
        self._individuals.sort(key=lambda x: x.fitness, reverse=True)

    def get_best(self):
        self.sort()
        return self[0]
