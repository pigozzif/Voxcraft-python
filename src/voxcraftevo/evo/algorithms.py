import abc
import os
import random
import time
import pickle
import numpy as np
import subprocess as sub

from .operators.operator import GeneticOperator
from .selection.selector import Selector
from ..representations.factory import GenotypeFactory
from ..representations.mapper import SolutionMapper
from ..representations.population import Population
from ..utils.utilities import weighted_random_by_dct


class Solver(object):

    def __init__(self, seed, fitness_func):
        self.seed = seed
        self.fitness_func = fitness_func
        self.start_time = None
        self.best_so_far = None

    def elapsed_time(self, units="s"):
        if self.start_time is None:
            self.start_time = time.time()
        s = time.time() - self.start_time
        if units == "s":
            return s
        elif units == "m":
            return s / 60.0
        elif units == "h":
            return s / 3600.0

    def save_checkpoint(self, directory, pop):
        sub.call("mkdir {0}/pickledPops{1}".format(directory, self.seed), shell=True)

        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        data = [self, random_state, numpy_random_state]

        with open('{0}/pickledPops{1}/Gen_{2}.pickle'.format(directory, self.seed, pop.gen), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_best(self, best):
        if self.best_so_far is not None and self.best_so_far == best:
            return
        sub.call("rm histories/*", shell=True)
        sub.call("rm data{}/*.vxd".format(self.seed), shell=True)
        self.fitness_func.save_histories(best, "data{}".format(self.seed), "histories")

    @abc.abstractmethod
    def solve(self, max_hours_runtime, max_gens, checkpoint_every, save_hist_every, directory="."):
        pass


class EvolutionarySolver(Solver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, fitness_func, remap, **kwargs):
        super().__init__(seed, fitness_func)
        self.pop_size = pop_size
        # self.stop_condition = stop_condition # TODO
        self.remap = remap
        self.continued_from_checkpoint = False
        self.pop = Population(pop_size, GenotypeFactory.create_factory(genotype_factory, **kwargs),
                              SolutionMapper.create_mapper(solution_mapper, **kwargs))

    def evaluate_individuals(self):
        num_evaluated = 0
        for ind in self.pop.individuals:
            if not ind.evaluated:
                self.fitness_func.create_vxd(ind, "data{}".format(self.seed), False)
                num_evaluated += 1
        sub.call("echo " + "GENERATION {}".format(self.pop.gen), shell=True)
        sub.call("echo Launching {0} voxelyze individuals, out of {1} individuals".format(num_evaluated, len(self.pop)),
                 shell=True)
        output_file = "output/output{0}_{1}.xml".format(self.seed, self.pop.gen)
        while True:
            try:
                sub.call("cd executables; ./voxcraft-sim -i ../data{0} -o {1} -f".format(self.seed, os.path.join("..", output_file)), shell=True)
                sub.call("echo WE ARE HERE!", shell=True)
                # sub.call waits for the process to return
                # after it does, we collect the results output by the simulator
                break

            except IOError:
                sub.call("echo Dang it! There was an IOError. I'll re-simulate this batch again...", shell=True)
                pass

            except IndexError:
                sub.call("echo Shoot! There was an IndexError. I'll re-simulate this batch again...", shell=True)
                pass
        for ind in self.pop.individuals:
            if not ind.evaluated:
                self.fitness_func.get_fitness(ind, output_file)
                if not self.remap:
                    ind.evaluated = True

    def solve(self, max_hours_runtime, max_gens, checkpoint_every, save_hist_every, directory="."):
        self.start_time = time.time()
        self.fitness_func.create_vxa("data{}".format(str(self.seed)))

        if not self.continued_from_checkpoint:  # generation zero
            self.evaluate_individuals()

        # iterate until stop conditions met
        while self.pop.gen < max_gens and self.elapsed_time(units="h") <= max_hours_runtime:
            sub.call("rm data{}/*.vxd".format(self.seed), shell=True)

            # checkpoint population
            if self.pop.gen % checkpoint_every == 0:  # and self.pop.gen > 0:
                sub.call("echo Saving checkpoint at generation {0}".format(self.pop.gen + 1), shell=True)
                self.save_checkpoint(directory, self.pop)

            # save history of best individual so far
            if self.pop.gen % save_hist_every == 0:
                sub.call("echo Saving history of run champ at generation {0}".format(self.pop.gen + 1), shell=True)
                self.save_best(self.pop.get_best())
            # update population stats
            self.pop.gen += 1
            self.pop.update_ages()
            self.best_so_far = self.pop.get_best()
            # update evolution
            self.evolve()
        self.save_checkpoint(directory, self.pop)

    @abc.abstractmethod
    def evolve(self):
        pass


class GeneticAlgorithm(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, survival_selector, parent_selector,
                 fitness_func, remap, genetic_operators, offspring_size, overlapping, **kwargs):
        super().__init__(seed, pop_size, genotype_factory, solution_mapper, fitness_func, remap, **kwargs)
        self.survival_selector = Selector.create_selector(survival_selector, **kwargs)
        self.parent_selector = Selector.create_selector(parent_selector, **kwargs)
        self.genetic_operators = {GeneticOperator.create_genetic_operator(k, **kwargs): v for k, v in genetic_operators.items()}
        self.offspring_size = offspring_size
        self.overlapping = overlapping

    def build_offspring(self):
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(self.pop.individuals,
                                                                                 operator.get_arity())]
            children_genotypes.append(operator.apply(parents))
        return children_genotypes

    def trim_population(self):
        while len(self.pop) > self.pop_size:
            self.pop.individuals.remove(self.survival_selector.select(self.pop.individuals))

    def evolve(self):
        # apply genetic operators
        if not self.overlapping:
            self.pop.clear()
        for child_genotype in self.build_offspring():
            self.pop.add_individual(child_genotype)
        # evaluate individuals
        self.evaluate_individuals()
        # apply selection
        self.trim_population()
