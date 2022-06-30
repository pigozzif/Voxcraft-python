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
                 output_dir: str, executables_dir: str):
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
        #if self.best_so_far is not None and self.best_so_far.id == best.id:
        #    return
        sub.call("rm {}/*".format(self.hist_dir), shell=True)
        self.fitness_func.save_histories(best=best, input_directory=self.data_dir, output_directory=self.hist_dir)
        sub.call("rm {}/*.vxd".format(self.data_dir), shell=True)

    def reload(self):
        pickled_pops = os.listdir(self.pickle_dir)
        last_gen = sorted(pickled_pops, reverse=True)[0]
        with open(os.path.join(self.pickle_dir, last_gen), "rb") as handle:
            [optimizer, random_state, numpy_random_state] = pickle.load(handle)
        best = optimizer.pop.get_best()
        optimizer.save_best(best=best)

    @abc.abstractmethod
    def solve(self, max_hours_runtime: int, max_gens: int, checkpoint_every: int, save_hist_every: int):
        pass


class EvolutionarySolver(Solver):

    def __init__(self, seed, pop_size: int, genotype_factory: str, solution_mapper: str, fitness_func, remap: bool,
                 genetic_operators: Dict[str, float], data_dir, hist_dir, pickle_dir, output_dir, executables_dir,
                 listener: Listener, comparator: str = "lexicase", genotype_filter: str = None, **kwargs):
        super().__init__(seed, fitness_func, data_dir, hist_dir, pickle_dir, output_dir, executables_dir)
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
        for ind in self.pop:
            if not ind.evaluated:
                ind.fitness = self.fitness_func.get_fitness(ind=ind, output_file=output_file)
                if not self.remap:
                    ind.evaluated = True

    def solve(self, max_hours_runtime, max_gens, checkpoint_every, save_hist_every) -> None:
        self.start_time = time.time()
        self.fitness_func.create_vxa(directory=self.data_dir)

        if not self.continued_from_checkpoint:  # generation zero
            self.evaluate_individuals()

        # iterate until stop conditions met
        while self.pop.gen < max_gens and self.elapsed_time(units="h") <= max_hours_runtime:
            sub.call("rm -rf {}/*".format(self.data_dir), shell=True)
            self.fitness_func.create_vxa(directory=self.data_dir)

            # checkpoint population
            if self.pop.gen % checkpoint_every == 0:  # and self.pop.gen > 0:
                sub.call("echo Saving checkpoint at generation {0}".format(self.pop.gen + 1), shell=True)
                self.save_checkpoint(pop=self.pop)

            # save history of best individual so far
            # if self.pop.gen % save_hist_every == 0:
            #     sub.call("echo Saving history of run champ at generation {0}".format(self.pop.gen + 1), shell=True)
            #     self.save_best(self.pop.get_best())
            # update population stats
            self.pop.gen += 1
            self.pop.update_ages()
            self.best_so_far = self.pop.get_best()
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


class GeneticAlgorithm(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, survival_selector: str, parent_selector: str,
                 fitness_func, offspring_size: int, overlapping: bool, remap, genetic_operators, data_dir, hist_dir,
                 pickle_dir, output_dir, executables_dir, listener, **kwargs):
        super().__init__(seed, pop_size, genotype_factory, solution_mapper, fitness_func, remap, genetic_operators,
                         data_dir, hist_dir, pickle_dir, output_dir, executables_dir, listener, **kwargs)
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
