import abc
import os
import random
import time
import pickle
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
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

PLOT = True


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
        self.reload_log_file = os.path.join("/".join(os.getcwd().split("/")[:-1]), logs_dir,
                                            "reload_{}.txt".format(seed))
        self.future_best = 2980
        self.future_genotype = np.array([-0.4050750107684763,2.7911664331466346,0.9354434450724496,0.4160558473752503,-0.2977111459530407,-1.3425811051415621,-1.1911451877488217,0.9960969260000668,-0.9950518748401087,0.33769990513550907,0.05413728110801852,-0.3034455051662579,1.512056268507844,-0.9616257097714672,-0.10450086145102072,-0.030298983220805195,-0.6788886831725595,1.373972806649061,-0.12516603798815962,-0.03185804813355225,-0.37972833365752706,0.754453913456356,0.1363275083091981,1.5437406618523868,-2.198827452066934,1.827641635819727,0.1040345863100886,0.2662066494916494,0.5622194647583687,0.21620698985558026,-0.47450546664476745,1.860949012982194,-1.1746851383526773,-0.7576958276586339,-0.5459870375414005,0.4852055068938355,1.6527428950943865,1.924619228237873,1.8763015812137454,-0.1859740986728331,-1.515646340879907,-0.6166824643816069,-1.1113273579694598,-0.8621555003714968,-0.8197276868139904,1.1723254998931585,-0.8281961586330427,1.0046128874565632,-0.8127701194043427,-2.0698013732380898,0.10226216316813114,0.6893347817830071,-1.051413673713884,-0.34978365942653245,0.125865869423199,-1.4187316140637436,-0.9232656795334734,0.750188008309971,0.5658981758016082,-0.06982794485122323,0.8055727199389529,0.6139979185150323,0.39115291960572507,-2.294820485376949,-0.014376134278690555,1.2335240639650076,-0.8680508409867266,0.6248039658127152,-0.23298727890098778,1.3519222361350582,-1.0920301865207451,-2.104174047686615,-1.9975533028253256,-1.5746148058520444,-1.0214087200846527,0.5639045345616865,0.06497825428969624,-0.1870599382066402,1.5324158803007821,-0.538970826947307,-1.526947320770669,-1.1445185387241463,-1.7259122043700175,2.0218341998101925,-0.6682653897741533,0.2936962472354637,2.412606128803483,-0.5604803692176692,1.6195081616559803,-1.4270846625690912,-0.16466640247772563,-0.6209345605533616,0.09301220288661356,-0.871444971223176,0.9151890063182571,1.1727429216723035,-0.7434454141711677,0.8394203187006519,-1.329015334186042,-0.817376582110644,-1.3920741194772808,0.43709262583854047,-0.648260250383655,-2.1139522591856057,0.4411371091195904,-0.29658225369822,-0.8942065872384625,-2.5697416466944434,-1.466478755941286,1.8222861291423138,-1.0900637856394098,-2.528271832900473,-1.2485385232202075,-1.777884452400738,-1.553092905423515,0.9277052091677823,1.5569178031820932,-1.4329720413166755,-0.8060800317428871,0.7390458162579188,0.6794904415984324,-2.603202904037663,0.4803896295685831,1.5731270434386564,-0.2184084458355768,1.2936080413045632,-0.2638045231476539,-2.2411821938357877,0.4575507981112133,1.011353549460033,-0.02160235482764203,-0.4553639281941281,0.29607168289603425,1.1108317268481036,0.6698834153891255,0.33027844690060715,0.3167449341197625,0.22527332799244032,-1.4169727299273527,-0.5695442431606763,1.4433379156254986,-0.23859266146615637,0.16763071075800293,2.2460425732299307,-1.3829214632310491,1.1817920338402175,1.2782780919962886,-0.3658387451267436,-0.5447701361472768,0.4085912217978442,-1.618624314616066,-0.18878072581508865,-0.8455986962054352,0.3039411491342802,-0.5850067208283344,0.5048216670636345,0.01626682711048688,-0.5747018853005019])

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
    def solve(self, max_hours_runtime: int, max_gens: int, checkpoint_every: int, save_hist_every: int):
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
        to_evaluate = list(filter(lambda x: not x.evaluated and x.id != self.future_best, self.pop))
        fitness = self.fitness_func.get_fitness(individuals=to_evaluate, output_file=self.reload_log_file,
                                                gen=self.pop.gen)  # {"locomotion_score": min(ind.genotype[0] ** 2, 1.0), "sensing_score": min((ind.genotype[1] - 2) ** 2, 1.0)}
        for ind in to_evaluate:
            ind.fitness = fitness[ind.id]
            ind.evaluated = not self.remap

        for ind in self.pop:
            if ind.id == self.future_best:
                ind.fitness = self.fitness_func.get_fitness(individuals=[ind], output_file=self.log_file,
                                                            gen=self.pop.gen)[ind.id]
                ind.evaluated = not self.remap
                break

    def solve(self, max_hours_runtime, max_gens, checkpoint_every, save_hist_every) -> None:
        self.start_time = time.time()
        self.fitness_func.create_vxa(directory=self.data_dir)

        if not self.continued_from_checkpoint:  # generation zero
            self.evaluate_individuals()
        self.best_so_far = self.get_best()
        # iterate until stop conditions met
        while self.pop.gen < max_gens and self.elapsed_time(units="h") <= max_hours_runtime:
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
        self._fronts_to_plot = {}
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
        if not PLOT:
            return
        if self.pop.gen == 1:
            self._fronts_to_plot[self.pop.gen] = self.fronts[0]
        elif self.pop.gen == 40:
            self._fronts_to_plot[self.pop.gen] = self.fronts[0]
        elif self.pop.gen == 80:
            self._fronts_to_plot[self.pop.gen] = self.fronts[0]
            for color, (gen, front) in zip(["orange", "blue", "red"], self._fronts_to_plot.items()):
                loc = [float(ind.fitness["locomotion_score"]) for ind in front],
                sens = [float(ind.fitness["sensing_score"]) for ind in front]
                plt.scatter(loc, sens, color=color, alpha=0.5, label=str(gen))
            plt.scatter([0.3, self.pop.objectives_dict[1]["best_value"]],
                        [self.pop.objectives_dict[0]["best_value"], self.pop.objectives_dict[1]["worst_value"]],
                        alpha=0.0)
            plt.xlabel("locomotion through the aperture (m)")
            plt.ylabel("affordance detection (% of timesteps correct)")
            plt.legend()
            plt.savefig("pareto_front_{}.png".format(self.seed))
            plt.clf()

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
