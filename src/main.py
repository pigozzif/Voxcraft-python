import os
import random
from time import time
import subprocess as sub
import argparse
import math

import numpy as np

from voxcraftevo.evo.algorithms import Solver
from voxcraftevo.listeners.listener import Listener
from voxcraftevo.utils.utilities import set_seed
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.configs.VXA import VXA
from voxcraftevo.configs.VXD import VXD
from voxcraftevo.fitness.evaluation import FitnessFunction


# actuation +/- 50%
# (1.5**(1/3)-1)/0.01 = 14.4714


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--solver", default="ga", type=str, help="solver for the optimization")
    parser.add_argument("--gens", default=40, type=int, help="generations for the ea")
    parser.add_argument("--popsize", default=100, type=int, help="population size for the ea")
    parser.add_argument("--history", default=100, type=int, help="how many generations for saving history")
    parser.add_argument("--checkpoint", default=1, type=int, help="how many generations for checkpointing")
    parser.add_argument("--time", default=48, type=int, help="maximum hours for the ea")
    parser.add_argument("--reload", default=0, type=int, help="restart from last pickled population")
    parser.add_argument("--execs", default="executables", type=str,
                        help="relative path to the dir containing Voxcraft executables")
    parser.add_argument("--logs", default="logs", type=str, help="relative path to the logs dir")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="relative path to output dir")
    parser.add_argument("--data_dir", default="data", type=str, help="relative path to data dir")
    parser.add_argument("--pickle_dir", default="pickledPops", type=str, help="relative path to pickled dir")
    parser.add_argument("--fitness", default="fitness_score", type=str, help="fitness tag")
    return parser.parse_args()


class MyListener(Listener):

    def listen(self, solver):
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([str(solver.seed), str(solver.pop.gen), str(solver.elapsed_time()),
                                             str(solver.best_so_far.fitness["locomotion_score"] +
                                                 solver.best_so_far.fitness["sensing_score"]),
                                             str(solver.best_so_far.id),
                                             str(np.median([ind.fitness["locomotion_score"] +
                                                            ind.fitness["sensing_score"] for ind in solver.pop])),
                                             str(min([ind.fitness["locomotion_score"] + ind.fitness["sensing_score"]
                                                      for ind in solver.pop])),
                                             str(solver.best_so_far.fitness["locomotion_score"]),
                                             str(np.median([ind.fitness["locomotion_score"] for ind in solver.pop])),
                                             str(min([ind.fitness["locomotion_score"] for ind in solver.pop])),
                                             str(solver.best_so_far.fitness["sensing_score"]),
                                             str(np.median([ind.fitness["sensing_score"] for ind in solver.pop])),
                                             str(min([ind.fitness["sensing_score"] for ind in solver.pop]))
                                             ]) + "\n")


class NSGAIIListener(Listener):

    def listen(self, solver):
        solver.fast_non_dominated_sort()
        pareto_front = solver.fronts[0]
        best_locomotion = min(pareto_front, key=lambda x: x.fitness["locomotion_score"])
        best_sensing = max(pareto_front, key=lambda x: x.fitness["sensing_score"])
        knee = solver.best_so_far
        stats = self._delimiter.join([str(solver.seed), str(solver.pop.gen), str(solver.elapsed_time()),
                                      str(best_sensing.id), str(best_locomotion.id),
                                      str(knee.fitness["locomotion_score"]), str(knee.fitness["sensing_score"])])
        locomotions = "/".join([str(ind.fitness["locomotion_score"]) for ind in solver.pop])
        sensing = "/".join([str(ind.fitness["sensing_score"]) for ind in solver.pop])
        pareto_locomotions = "/".join([str(ind.fitness["locomotion_score"]) for ind in pareto_front])
        pareto_sensing = "/".join([str(ind.fitness["sensing_score"]) for ind in pareto_front])
        gen_locomotion = ",".join([str(g) for g in best_locomotion.genotype])
        gen_sensing = ",".join([str(g) for g in best_sensing.genotype])
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([stats, locomotions, sensing, pareto_locomotions, pareto_sensing,
                                             gen_sensing, gen_locomotion]) + "\n")


class MyFitness(FitnessFunction):

    def __init__(self, fitness, solver):
        self.fitness = fitness
        self.immovable_left = None
        self.immovable_right = None
        self.soft = None
        self.special_passable = None
        self.special_impassable = None
        self.wall_left = None
        self.wall_right = None
        self.terrains = ["passable", "impassable"]
        self.solver = solver
        self.objective_dict = ObjectiveDict()

    @staticmethod
    def get_file_name(*args):
        return "-".join(list(args))

    @staticmethod
    def get_body_length(r_label):
        if r_label == "a":
            return 3
        elif r_label == "b":
            return 9
        elif r_label == "c":
            return 27
        raise ValueError("Unknown body size: {}".format(r_label))

    def create_objectives_dict(self):
        if self.solver != "nsgaii":
            self.objective_dict.add_objective(name="fitness_score", maximize=True, tag="<{}>".format(self.fitness),
                                              best_value=2.0, worst_value=-1.0)
        self.objective_dict.add_objective(name="locomotion_score", maximize=False,
                                          tag="<{}>".format("locomotion_score"),
                                          best_value=0.0, worst_value=5.0)
        self.objective_dict.add_objective(name="sensing_score", maximize=True, tag="<{}>".format("sensing_score"),
                                          best_value=1.0, worst_value=0.0)
        return self.objective_dict

    def create_vxa(self, directory):
        vxa = VXA(TempAmplitude=14.4714, TempPeriod=0.2, TempBase=0, EnableCollision=1)
        self.immovable_left = vxa.add_material(material_id=1, RGBA=(50, 50, 50, 255), E=10000, RHO=10, P=0.5,
                                               uDynamic=0.5, isFixed=1, isMeasured=0)
        self.immovable_right = vxa.add_material(material_id=2, RGBA=(0, 50, 50, 255), E=10000, RHO=10, P=0.5,
                                                uDynamic=0.5, isFixed=1, isMeasured=0)
        self.special_passable = vxa.add_material(material_id=3, RGBA=(255, 255, 255, 255), E=10000, RHO=10, P=0.5,
                                                 uDynamic=0.5, isFixed=1, isMeasured=0)
        self.soft = vxa.add_material(material_id=4, RGBA=(255, 0, 0, 255), E=10000, RHO=10, P=0.5, uDynamic=0.5,
                                     CTE=0.01, isMeasured=1)
        self.special_impassable = vxa.add_material(material_id=5, RGBA=(255, 255, 255, 255), E=10000, RHO=10, P=0.5,
                                                   uDynamic=0.5, isFixed=1, isMeasured=0)
        self.wall_left = vxa.add_material(material_id=6, RGBA=(50, 50, 50, 255), E=10000, RHO=10, P=0.5,
                                          uDynamic=0.5, isFixed=1, isMeasured=0)
        self.wall_right = vxa.add_material(material_id=7, RGBA=(0, 50, 50, 255), E=10000, RHO=10, P=0.5,
                                           uDynamic=0.5, isFixed=1, isMeasured=0)
        vxa.write(filename=os.path.join(directory, "base.vxa"))

    def create_vxd(self, ind, directory, record_history):
        for _, r_label in enumerate(["b"]):
            for _, p_label in enumerate(self.terrains):
                base_name = os.path.join(directory, self.get_file_name("bot_{:04d}".format(ind.id), r_label,
                                                                       p_label))
                body_length = self.get_body_length(r_label)
                world = self._create_world(body_length=body_length, p_label=p_label)

                vxd = VXD(NeuralWeights=ind.genotype, isPassable=p_label != "impassable")
                vxd.set_data(data=world)
                vxd.set_tags(RecordVoxel=record_history, RecordFixedVoxels=record_history,
                             RecordStepSize=100 if record_history else 0)
                vxd.write(filename=base_name + ".vxd")

    def _create_world(self, body_length, p_label):
        world = np.zeros((body_length * 3, body_length * 5, int(body_length / 3) + 1))

        start = math.floor(body_length * 1.5)
        left_edge = body_length
        right_edge = body_length * 2
        wall_position = body_length * 2
        lower_edge = body_length - 4
        world[start, lower_edge: wall_position - 4, 0] = self.soft
        world[left_edge: right_edge, start - 4, 0] = self.soft

        center = random.choice([start + i for i in range(-body_length // 2 + 1, body_length // 2 - 1)])
        aperture_size = random.choice([0, 1, 3]) if p_label == "impassable" else random.choice(
            [body_length + 2, body_length - 2, body_length])

        world[:center, wall_position, :2] = self.immovable_left
        world[center:, wall_position, :2] = self.immovable_right
        world[center - aperture_size // 2: center + aperture_size // 2, body_length * 2: body_length * 3 + 1, :] = 0

        left_wall = min(left_edge, center - aperture_size // 2) - random.choice(
            [i for i in range(1, body_length // 3 + 1)])
        right_wall = max(right_edge, center + aperture_size // 2) + random.choice(
            [i for i in range(1, body_length // 3 + 1)])
        left_wall_type = random.choice(["none", "straight", "knee"])
        right_wall_type = random.choice(["none", "straight", "knee"])
        if left_wall_type == "straight":
            world[left_wall, lower_edge: wall_position, :2] = self.wall_left
        elif left_wall_type == "knee":
            world[left_wall - 1, lower_edge: start, :2] = self.wall_left
            world[left_wall - 1: left_wall + body_length // 4 - 1, start, :2] = self.wall_left
            world[left_wall + body_length // 4 - 1, start: wall_position, :2] = self.wall_left
        if right_wall_type == "straight":
            world[right_wall, lower_edge: wall_position, :2] = self.wall_right
        elif right_wall_type == "knee":
            world[right_wall, lower_edge: start, :2] = self.wall_right
            world[right_wall - body_length // 4:right_wall + 1, start, :2] = self.wall_right
            world[right_wall - body_length // 4, start: wall_position, :2] = self.wall_right

        if p_label != "impassable":
            world[center, body_length * 5 - 1, 0] = self.special_passable
        else:
            world[center, 0, 0] = self.special_impassable

        return world

    def get_fitness(self, individuals, output_file):
        fitness = {}
        for ind in individuals:
            values = {obj: [] for obj in self.objective_dict}
            for _, r_label in enumerate(["b"]):
                for _, p_label in enumerate(self.terrains):
                    if p_label == "impassable":
                        terrain_id = 0
                    else:
                        terrain_id = 2
                    for obj in values:
                        name = self.objective_dict[obj]["name"]
                        values[obj].append(self.parse_fitness_from_history(output_file,
                                                                           fitness_tag="-".join(
                                                                               [str(ind.id), str(terrain_id), name]),
                                                                           worst_value=self.objective_dict[obj][
                                                                               "worst_value"]))
            fitness[ind.id] = {self.objective_dict[k]["name"]: min(v) if self.objective_dict[k]["maximize"] else max(v)
                               for k, v in values.items()}
        return fitness

    def save_histories(self, individual, input_directory, output_directory, executables_directory):
        sub.call("rm {}/*vxd".format(input_directory), shell=True)
        self.create_vxd(ind=individual, directory=input_directory, record_history=True)
        temp_dir = input_directory.replace("data", "temp")
        sub.call("mkdir {}".format(temp_dir), shell=True)
        self.create_vxa(directory=temp_dir)
        for file in os.listdir(input_directory):
            if file.endswith("vxd"):
                sub.call("cp {0} {1}/".format(os.path.join(input_directory, file), temp_dir), shell=True)
                sub.call("cd {0}; ./voxcraft-sim -i {1} -o output.xml > {2}".format(
                    executables_directory,
                    os.path.join("..", temp_dir),
                    os.path.join("..", output_directory, file.replace("vxd", "history"))), shell=True)
                sub.call("cd {}; rm output.xml".format(executables_directory), shell=True)
                sub.call("rm {}/*.vxd".format(temp_dir), shell=True)
        sub.call("rm -rf {}".format(temp_dir), shell=True)


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)

    pickle_dir = "{0}{1}".format(arguments.pickle_dir, arguments.seed)
    data_dir = "{0}{1}".format(arguments.data_dir, arguments.seed)
    sub.call("rm -rf {0}".format(pickle_dir), shell=True)
    sub.call("rm -rf {0}".format(data_dir), shell=True)

    seed = arguments.seed
    number_of_params = (9 * 9) + 9 + (9 * 8) + 8
    remap = True
    if arguments.solver == "ga":
        evolver = Solver.create_solver(name="ga", seed=seed, pop_size=arguments.popsize,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct", survival_selector="worst",
                                       parent_selector="tournament",
                                       fitness_func=MyFitness(arguments.fitness, arguments.solver), remap=remap,
                                       genetic_operators={"gaussian_mut": 1.0},
                                       offspring_size=arguments.popsize // 2, overlapping=True,
                                       data_dir=data_dir, hist_dir="history{}".format(seed),
                                       pickle_dir=pickle_dir, output_dir=arguments.output_dir,
                                       executables_dir=arguments.execs,
                                       listener=MyListener(file_path="{0}_{1}.csv".format(
                                           arguments.fitness, seed),
                                           header=["seed", "gen", "elapsed.time", "best.fitness_score", "best.id",
                                                   "median.fitness_score", "min.fitness_score", "best.locomotion_score",
                                                   "median.locomotion_score", "min.locomotion_score", "best"
                                                                                                      ".sensing_score",
                                                   "median.sensing_score", "min.sensing_score"]),
                                       tournament_size=5, mu=0.0, sigma=0.35, n=number_of_params,
                                       range=(-1, 1), upper=2.0, lower=-1.0)
    elif arguments.solver == "nsgaii":
        evolver = Solver.create_solver(name="nsgaii", seed=seed, pop_size=arguments.popsize,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=MyFitness(arguments.fitness, arguments.solver), remap=remap,
                                       genetic_operators={"gaussian_mut": 1.0},
                                       offspring_size=arguments.popsize // 2,
                                       data_dir=data_dir, hist_dir="history{}".format(seed),
                                       pickle_dir=pickle_dir, output_dir=arguments.output_dir,
                                       executables_dir=arguments.execs,
                                       logs_dir=arguments.logs,
                                       listener=NSGAIIListener(file_path="{0}_{1}.csv".format(
                                           arguments.fitness, seed),
                                           header=["seed", "gen", "elapsed.time", "best.sensing", "best.locomotion",
                                                   "knee.locomotion", "knee.sensing", "locomotions", "sensings",
                                                   "pareto.locomotions", "pareto.sensings", "best.sensing.g",
                                                   "best.locomotion.g"]),
                                       tournament_size=2, mu=0.0, sigma=0.35, n=number_of_params,
                                       range=(-1, 1), upper=2.0, lower=-1.0)
    else:
        raise ValueError("Invalid solver name: {}".format(arguments.solver))

    if arguments.reload:
        evolver.reload()
    else:
        evolver.solve(max_hours_runtime=arguments.time, max_gens=arguments.gens, checkpoint_every=arguments.checkpoint,
                      save_hist_every=arguments.history)
    start_time = time()
    sub.call("echo That took a total of {} minutes".format((time() - start_time) / 60.), shell=True)
