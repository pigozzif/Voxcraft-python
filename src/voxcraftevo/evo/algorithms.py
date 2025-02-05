import abc
import os
import random
import time
import pickle
from typing import Dict

import numpy as np
import subprocess as sub

from .operators.operator import GeneticOperator
from .selection.filters import Filter
from .selection.selector import Selector
from ..fitness.evaluation import FitnessFunction
from ..listeners.listener import Listener
from ..representations.factory import GenotypeFactory
from ..representations.mapper import SolutionMapper
from ..representations.population import Population, Individual
from ..utils.utilities import weighted_random_by_dct


class Solver(object):

    def __init__(self, seed: int, fitness_func: FitnessFunction, data_dir: str, hist_dir: str, pickle_dir: str,
                 output_dir: str, executables_dir: str, logs_dir: str):
        self.seed = seed
        self.fitness_func = fitness_func
        self.start_time = None
        self.best_so_far = None
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            sub.call("mkdir {}".format(data_dir), shell=True)
        self.hist_dir = hist_dir
        if not os.path.isdir(self.hist_dir):
            sub.call("mkdir {}".format(hist_dir), shell=True)
        self.pickle_dir = pickle_dir
        if not os.path.isdir(self.pickle_dir):
            sub.call("mkdir {}".format(pickle_dir), shell=True)
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            sub.call("mkdir {}".format(output_dir), shell=True)
        self.executables_dir = executables_dir
        if not os.path.isdir(executables_dir):
            sub.call("mkdir {}".format(executables_dir), shell=True)
        for file in os.listdir(os.path.join("..", logs_dir)):
            if int(file.split(".")[1].split("_")[1]) == self.seed and "out" in file:
                self.log_file = os.path.join("/".join(os.getcwd().split("/")[:-1]), logs_dir, file)
                break
        else:
            raise IndexError

    def elapsed_time(self, units: str = "s") -> float:
        if self.start_time is None:
            self.start_time = time.time()
        s = time.time() - self.start_time
        if units == "s":
            return s
        elif units == "m":
            return s / 60.0
        elif units == "h":
            return s / 3600.0

    def save_checkpoint(self, pop: Population) -> None:
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        data = [self, random_state, numpy_random_state]

        with open(os.path.join(self.pickle_dir, "gen_{}.pickle".format(pop.gen)), "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_best(self, best: Individual) -> None:
        self.fitness_func.save_histories(individual=best, input_directory=self.data_dir, output_directory=self.hist_dir,
                                         executables_directory=self.executables_dir)
        sub.call("rm {}/*.vxd".format(self.data_dir), shell=True)

    def reload(self):
        pickled_pops = os.listdir(self.pickle_dir)
        last_gen = sorted(pickled_pops, key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)[0]
        with open(os.path.join(self.pickle_dir, last_gen), "rb") as handle:
            [optimizer, random_state, numpy_random_state] = pickle.load(handle)
        best = optimizer.pop.get_best()
        optimizer.save_best(best=best)

    @abc.abstractmethod
    def solve(self, max_gens: int, checkpoint_every: int):
        pass

    @classmethod
    def create_solver(cls, name: str, **kwargs):
        if name == "ga":
            return GeneticAlgorithm(**kwargs)
        elif name == "nsgaii":
            return NSGAII(**kwargs)
        raise ValueError("Invalid solver name: {}".format(name))


class EvolutionarySolver(Solver):

    def __init__(self, seed, pop_size: int, genotype_factory: str, solution_mapper: str, fitness_func, remap: bool,
                 genetic_operators: Dict[str, float], data_dir, hist_dir, pickle_dir, output_dir, executables_dir,
                 logs_dir, listener: Listener, comparator: str, genotype_filter: str = None, **kwargs):
        super().__init__(seed, fitness_func, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, logs_dir)
        self.pop_size = pop_size
        self.remap = remap
        self.continued_from_checkpoint = False
        self.pop = Population(pop_size=pop_size,
                              genotype_factory=GenotypeFactory.create_factory(name=genotype_factory,
                                                                              genotype_filter=Filter.create_filter(
                                                                                  genotype_filter), **kwargs),
                              solution_mapper=SolutionMapper.create_mapper(name=solution_mapper, **kwargs),
                              objectives_dict=self.fitness_func.create_objectives_dict(),
                              comparator=comparator)
        self.genetic_operators = {GeneticOperator.create_genetic_operator(name=k,
                                                                          genotype_filter=Filter.create_filter(
                                                                              genotype_filter), **kwargs):
                                      v for k, v in genetic_operators.items()}
        self.listener = listener

    def evaluate_individuals(self) -> None:
        num_evaluated = 0
        for ind in self.pop:
            if not ind.evaluated:
                self.fitness_func.create_vxd(ind=ind, directory=self.data_dir, record_history=False)
                num_evaluated += 1
        sub.call("echo " + "GENERATION {}".format(self.pop.gen), shell=True)
        sub.call("echo Launching {0} voxelyze individuals to-be-evaluated, out of {1} individuals".
                 format(num_evaluated, len(self.pop)), shell=True)
        output_file = os.path.join(self.output_dir, "output{0}_{1}.xml".format(self.seed, self.pop.gen))
        if num_evaluated > 0:
            while True:
                try:
                    sub.call("cd {0}; ./voxcraft-sim -i {1} -o {2}".format(self.executables_dir,
                                                                           os.path.join("..", self.data_dir),
                                                                           os.path.join("..", output_file)), shell=True)
                    # sub.call waits for the process to return
                    # after it does, we collect the results output by the simulator
                    break
                except IOError:
                    sub.call("echo Dang it! There was an IOError. I'll re-simulate this batch again...", shell=True)
                    pass
                except IndexError:
                    sub.call("echo Shoot! There was an IndexError. I'll re-simulate this batch again...", shell=True)
                    pass
            time.sleep(7)
        to_evaluate = list(filter(lambda x: not x.evaluated, self.pop))
        fitness = self.fitness_func.get_fitness(individuals=to_evaluate, output_file=self.output_dir + "/output{}_{}.xml".format(self.seed, self.pop.gen),
                                                log_file=self.log_file, gen=self.pop.gen)  # {"locomotion_score": min(ind.genotype[0] ** 2, 1.0), "sensing_score": min((ind.genotype[1] - 2) ** 2, 1.0)}
        for ind in to_evaluate:
            ind.fitness = fitness[ind.id]
            ind.evaluated = not self.remap

    def solve(self, max_gens, checkpoint_every) -> None:
        self.start_time = time.time()
        self.fitness_func.create_vxa(directory=self.data_dir)

        if not self.continued_from_checkpoint:  # generation zero
            self.evaluate_individuals()
        self.best_so_far = self.get_best()
        # iterate until stop conditions met
        while self.pop.gen < max_gens:
            sub.call("rm -rf {}/*".format(self.data_dir), shell=True)
            self.fitness_func.create_vxa(directory=self.data_dir)

            # checkpoint population
            if self.pop.gen % checkpoint_every == 0:  # and self.pop.gen > 0:
                sub.call("echo Saving checkpoint at generation {0}".format(self.pop.gen + 1), shell=True)
                self.save_checkpoint(pop=self.pop)

            # update population stats
            self.pop.gen += 1
            self.pop.update_ages()
            self.best_so_far = self.get_best()
            # update evolution
            self.listener.listen(solver=self)
            self.evolve()
        self.save_checkpoint(pop=self.pop)
        self.listener.listen(solver=self)
        sub.call("echo Saving history of run champ at generation {0}".format(self.pop.gen + 1), shell=True)
        self.save_best(best=self.pop.get_best())

    @abc.abstractmethod
    def evolve(self):
        pass

    def get_best(self) -> Individual:
        return self.pop.get_best()


class GeneticAlgorithm(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, survival_selector: str, parent_selector: str,
                 fitness_func, offspring_size: int, overlapping: bool, remap, genetic_operators, data_dir, hist_dir,
                 pickle_dir, output_dir, executables_dir, logs_dir, listener, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=remap,
                         genetic_operators=genetic_operators, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         logs_dir=logs_dir, listener=listener, comparator="lexicase", **kwargs)
        self.survival_selector = Selector.create_selector(name=survival_selector, **kwargs)
        self.parent_selector = Selector.create_selector(name=parent_selector, **kwargs)
        self.offspring_size = offspring_size
        self.overlapping = overlapping

    def build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def trim_population(self) -> None:
        while len(self.pop) > self.pop_size:
            self.pop.remove_individual(self.survival_selector.select(population=self.pop, n=1)[0])

    def evolve(self) -> None:
        # apply genetic operators
        if not self.overlapping:
            self.pop.clear()
        for child_genotype in self.build_offspring():
            self.pop.add_individual(genotype=child_genotype)
        # evaluate individuals
        self.evaluate_individuals()
        # apply selection
        self.trim_population()


class NSGAII(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, fitness_func, offspring_size: int, remap,
                 genetic_operators, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, logs_dir,
                 listener, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=remap,
                         genetic_operators=genetic_operators, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         logs_dir=logs_dir, listener=listener, comparator="pareto", **kwargs)
        self.offspring_size = offspring_size
        self.fronts = {}
        self.dominates = {}
        self.dominated_by = {}
        self.crowding_distances = {}
        self.parent_selector = Selector.create_selector(name="tournament_crowded",
                                                        crowding_distances=self.crowding_distances, fronts=self.fronts,
                                                        **kwargs)
        self.best_sensing = None
        self.best_locomotion = None

    def fast_non_dominated_sort(self) -> None:
        self.fronts.clear()
        self.dominates.clear()
        self.dominated_by.clear()
        for p in self.pop:
            self.dominated_by[p.id] = 0
            for q in self.pop:
                if p.id == q.id:
                    continue
                elif p > q:
                    if p.id not in self.dominates:
                        self.dominates[p.id] = [q]
                    else:
                        self.dominates[p.id].append(q)
                elif p < q:
                    self.dominated_by[p.id] += 1
            if self.dominated_by[p.id] == 0:
                if 0 not in self.fronts:
                    self.fronts[0] = [p]
                else:
                    self.fronts[0].append(p)
        if not self.fronts:
            self.fronts[0] = [ind for ind in self.pop]
            return
        i = 0
        while len(self.fronts[i]):
            self.fronts[i + 1] = []
            for p in self.fronts[i]:
                for q in self.dominates.get(p.id, []):
                    self.dominated_by[q.id] -= 1
                    if self.dominated_by[q.id] == 0:
                        self.fronts[i + 1].append(q)
            i += 1
        self.fronts.pop(i)
        self.crowding_distances.clear()
        for front in self.fronts.values():
            self.crowding_distance_assignment(individuals=front)

    def crowding_distance_assignment(self, individuals: list) -> None:
        for individual in individuals:
            self.crowding_distances[individual.id] = 0.0
        for rank, goal in self.pop.objectives_dict.items():
            individuals.sort(key=lambda x: x.fitness[goal["name"]], reverse=goal["maximize"])
            self.crowding_distances[individuals[0].id] = float("inf")
            self.crowding_distances[individuals[len(individuals) - 1].id] = float("inf")
            for i in range(1, len(individuals) - 1):
                self.crowding_distances[individuals[i].id] += (individuals[i + 1].fitness[goal["name"]] -
                                                               individuals[i - 1].fitness[goal["name"]]) / \
                                                              (abs(goal["best_value"] - goal["worst_value"]))

    def build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def trim_population(self) -> None:
        self.fast_non_dominated_sort()
        i = 0
        n = 0
        while n + len(self.fronts[i]) <= self.pop_size:
            n += len(self.fronts[i])
            i += 1
        self.fronts[i].sort(key=lambda x: self.crowding_distances[x.id])
        for j in range(len(self.fronts[i]) - self.pop_size + n):
            self.pop.remove_individual(ind=self.fronts[i][j])
        i += 1
        while i in self.fronts:
            for ind in self.fronts[i]:
                self.pop.remove_individual(ind=ind)
            i += 1

    def evolve(self) -> None:
        if self.pop.gen == 1:
            self.fast_non_dominated_sort()
        for child_genotype in self.build_offspring():
            self.pop.add_individual(genotype=child_genotype)
        self.evaluate_individuals()
        self.trim_population()
        temp_best_sensing = max(self.pop, key=lambda x: x.fitness["sensing_score"])
        self.best_sensing = temp_best_sensing if self.best_sensing is None else \
            max([temp_best_sensing, self.best_sensing], key=lambda x: x.fitness["sensing_score"])
        temp_best_locomotion = min(self.pop, key=lambda x: x.fitness["locomotion_score"])
        self.best_locomotion = temp_best_locomotion if self.best_locomotion is None else \
            min([temp_best_locomotion, self.best_locomotion], key=lambda x: x.fitness["locomotion_score"])

    def get_best(self) -> Individual:
        if not self.fronts:
            self.fast_non_dominated_sort()
        return min(self.fronts[0], key=lambda x: self.get_distance_from_diagonal(individual=x,
                                                                                 objectives_dict=self.pop.objectives_dict))

    def save_best(self, best: Individual) -> None:
        self.fitness_func.save_histories(individual=self.best_locomotion, input_directory=self.data_dir,
                                         output_directory=self.hist_dir,
                                         executables_directory=self.executables_dir)
        self.fitness_func.save_histories(individual=self.best_sensing, input_directory=self.data_dir,
                                         output_directory=self.hist_dir,
                                         executables_directory=self.executables_dir)

    @staticmethod
    def get_distance_from_diagonal(individual: Individual, objectives_dict: dict) -> float:
        s = 0.0
        for goal in objectives_dict.values():
            obj = individual.fitness[goal["name"]] / abs(goal["best_value"] - goal["worst_value"])
            if s == 0.0:
                s = obj
            else:
                s -= obj
        return abs(s)
