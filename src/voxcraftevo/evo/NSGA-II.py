from src.voxcraftevo.evo.algorithms import EvolutionarySolver
from src.voxcraftevo.evo.selection.selector import Selector


class NSGAII(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, survival_selector: str, parent_selector: str,
                 fitness_func, offspring_size: int, remap, genetic_operators, data_dir, hist_dir,
                 pickle_dir, output_dir, executables_dir, listener, **kwargs):
        super().__init__(seed, pop_size, genotype_factory, solution_mapper, fitness_func, remap, genetic_operators,
                         data_dir, hist_dir, pickle_dir, output_dir, executables_dir, listener, **kwargs)
        self.survival_selector = Selector.create_selector(name=survival_selector, **kwargs)
        self.parent_selector = Selector.create_selector(name=parent_selector, **kwargs)
        self.offspring_size = offspring_size
        self.fronts = {}
        self.dominates = {}
        self.dominated_by = {}
        self.crowding_distances = {}

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
