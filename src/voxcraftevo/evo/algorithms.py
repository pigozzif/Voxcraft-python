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
            if "out" in file and int(file.split(".")[1].split("_")[1]) == self.seed:
                self.log_file = os.path.join("/".join(os.getcwd().split("/")[:-1]), logs_dir, file)
                break
        else:
            raise IndexError
        self.reload_log = os.path.join("/".join(os.getcwd().split("/")[:-1]), logs_dir, "reload_log.txt")
        self.future_best = 3533
        self.future_genotype = np.array([-0.7547392297507856,-1.7153990011485245,0.22602551423020073,0.36339037735787083,0.5733503667268705,0.5781164343072722,-0.9120319506978755,-0.830564155441279,-0.688265266094752,-1.6264913906231708,-0.4283617972032887,0.45212505186597657,3.5529953350808468,-0.2791635769776172,-0.18627663458437732,-0.24466889277071774,2.4064025113283853,-1.550649672093415,0.8920089830060864,-0.8295329012119962,-2.010175254404843,0.7205216889360125,1.76357593537527,-0.9627738552667654,0.2830324396989122,-0.26461301781810764,-1.5946004846089512,-0.046341196164248144,-2.2396424658554315,0.8270555799529369,-0.9718678924708153,1.1666030456317589,0.58293015356486,1.148152621354675,0.1447200030193421,1.6218133241604022,1.0622503544747575,-1.7656428463201002,-0.2816812625988552,1.463252851820343,0.2220224234402174,0.3393906421152279,-1.696895848684647,-0.5072162195329323,-1.948981049210405,-1.7988939100464112,1.5756151111781984,0.6893693327220908,0.6117120096523743,0.41897050066373936,0.6305256694819141,-1.4915900205297872,0.3587829554773202,0.27862112972611036,-1.157363859865529,-0.28512315413963657,1.9212279728946786,-0.10003898802079936,0.4497170790004579,0.5236712765448462,1.776877899686999,-1.303110098125532,-1.0819370054449213,0.024199703971659048,0.11197767340830156,-2.127017049942949,-1.4507824961807618,0.24505644911360286,-0.617973032431006,0.795209093263557,-1.2841452652381071,-0.21027144403484405,-0.7127233170435066,-0.6505656524934936,1.3972311749261457,-0.6934761111002425,-0.010419855203531767,1.0936345185530176,1.5936116632681114,0.0020405870996877684,-1.56911410017186,1.6038639295420454,-0.05690212298021627,0.5810680214029402,1.2104973570860587,-0.6568371336472197,-1.7925071461682154,-0.9303993742496659,-0.5921661886305372,0.18337124678849104,-0.7220970359068838,-2.1204939261768883,1.2171595679599672,2.9555057208727846,-0.4352923797786072,-0.5851904794323642,0.2206933755297687,-0.6272700632044484,-0.9860687267612387,1.0731501007831505,-0.21997243141581135,-0.05552607651346629,-1.3747184845434137,0.7221605978728396,-0.3416860215089328,1.3321954684732815,0.40717765569520453,1.6449657831381992,1.1316052593995296,-2.4188213025799854,0.40262473257868037,-0.28272856639513916,-1.2759305324477825,-0.5210353353418141,-0.44772274122323175,-1.1232165081066643,1.1822404924162082,-2.2891260709316086,-1.8084125126924278,0.43421812233028956,0.8094457052476324,-1.1930561283358256,-1.076477171985555,1.294665973229301,-0.4960372267054299,1.6567359141680595,0.27979253886532096,0.7482548100281076,1.0106585448077487,-1.6864821517566393,2.088404979556005,-1.2881608864682286,-0.4815912122619703,-1.3514612993235522,0.7867075054876347,-0.4552996610785143,-0.7338704642706242,-0.10615355195836307,0.6609905370348396,-0.4474987655583551,-0.3252783256266223,1.2044009149828032,0.588433701105195,1.116270009704031,-2.5191451436194456,-0.893701978879835,-0.43234232574060016,1.131556844277102,0.25771106695217144,-0.5287625233841442,0.6753113658069545,1.057605018638692,0.2593971287190877,1.2770854698118455,-1.3699991134583676,1.0339961995278624,0.42296893655664236,-2.433396314352212])

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
            if not ind.evaluated and ind.id == self.future_best:
                ind.genotype = self.future_genotype
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
            time.sleep(1)
        to_evaluate = list(filter(lambda x: not x.evaluated and ind.id != self.future_best, self.pop))
        fitness = self.fitness_func.get_fitness(individuals=to_evaluate, output_file=self.reload_log,
                                                gen=self.pop.gen)  # {"locomotion_score": min(ind.genotype[0] ** 2, 1.0), "sensing_score": min((ind.genotype[1] - 2) ** 2, 1.0)}
        for ind in to_evaluate:
            ind.fitness = fitness[ind.id]
            ind.evaluated = not self.remap

        for ind in self.pop:
            if ind.id == self.future_best:
                ind.fitness = self.fitness_func.get_fitness(individuals=[ind], output_file=self.log_file,
                                                            gen=self.pop.gen)
                ind.evaluated = not self.remap
                break

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
