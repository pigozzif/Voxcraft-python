from src.voxcraftevo.evo.algorithms import EvolutionarySolver
from src.voxcraftevo.evo.selection.selector import Selector
from src.voxcraftevo.utils.utilities import weighted_random_by_dct


class NSGAII(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, fitness_func, offspring_size: int, remap,
                 genetic_operators, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, listener, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=remap,
                         genetic_operators=genetic_operators, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         listener=listener, comparator="pareto", **kwargs)
        self.parent_selector = Selector.create_selector(name="tournament", awarder=lambda x: self.crowded_comparison(x),
                                                        **kwargs)
        self.offspring_size = offspring_size
        self.fronts = {}
        self.dominates = {}
        self.dominated_by = {}
        self.crowding_distances = {}
        self.fast_non_dominated_sort()

    def crowded_comparison(self, individuals):
        reverse_fronts = {}
        for individual in individuals:
            for idx, front in self.fronts.items():
                if individual in front:
                    reverse_fronts[individual.id] = idx
                    break
        best = None
        for individual in individuals:
            if best is None:
                best = individual
            elif reverse_fronts[best.id] > reverse_fronts[individual.id]:
                best = individual
            elif self.crowding_distances[best.id] < self.crowding_distances[individual.id]:
                best = individual
        return best

    def fast_non_dominated_sort(self):
        self.fronts.clear()
        self.dominates.clear()
        self.dominated_by.clear()
        for p in self.pop:
            self.dominated_by[p] = 0
            for q in self.pop:
                if p > q:
                    if not self.dominates[p.id]:
                        self.dominates[p.id] = [q]
                    else:
                        self.dominates[p.id].append(q)
                elif p < q:
                    self.dominated_by[p.id] += 1
            if self.dominated_by[p.id] == 0:
                if not self.fronts[0]:
                    self.fronts[0] = [p]
                else:
                    self.fronts[0].append(p)
        i = 0
        while len(self.fronts[i]):
            self.fronts[i + 1] = []
            for p in self.fronts[i]:
                for q in self.dominates[p.id]:
                    self.dominated_by[q.id] -= 1
                    if self.dominated_by[q.id] == 0:
                        if not self.fronts[i + 1]:
                            self.fronts[i + 1] = [q]
                        else:
                            self.fronts[i + 1].append(q)
            i += 1

    def crowding_distance_assignment(self, individuals):
        for individual in individuals:
            self.crowding_distances[individual.id] = 0
        for rank, goal in self.pop.objectives_dict.items():
            individuals.sort(key=lambda x: x.fitness[rank], reverse=goal["maximize"])
            self.crowding_distances[individuals[0].id] = float("inf")
            self.crowding_distances[individuals[len(individuals) - 1].id] = float("inf")
            for i in range(1, len(individuals) - 1):
                self.crowding_distances[individuals[i].id] += individuals[i + 1].fitness[rank] - \
                                                              individuals[i - 1].fitness[rank]

    def build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def trim_population(self):
        self.fast_non_dominated_sort()
        i = 0
        n = 0
        while n + len(self.fronts[i]) <= self.pop_size - self.offspring_size:
            self.crowding_distance_assignment(self.fronts[i])
            n += len(self.fronts[i])
            i += 1
        self.fronts[i].sort(lambda x: self.crowding_distances[x.id], reverse=True)
        for j in range(self.pop_size - self.offspring_size - n):
            self.pop.remove_individual(ind=self.fronts[i][len(self.fronts[i]) - j])
        while len(self.fronts[i]):
            for ind in self.fronts[i]:
                self.pop.remove_individual(ind=ind)
            i += 1

    def evolve(self):
        if self.pop.gen == 0:
            self.fast_non_dominated_sort()
            for front in self.fronts.values():
                self.crowding_distance_assignment(front)
        for child_genotype in self.build_offspring():
            self.pop.add_individual(genotype=child_genotype)
        self.evaluate_individuals()
        # apply selection
        self.trim_population()
