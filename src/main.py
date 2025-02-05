import os
import random
from time import time
import subprocess as sub
import argparse
import math
from lxml import etree

import numpy as np

from voxcraftevo.evo.algorithms import Solver
from voxcraftevo.listeners.listener import Listener
from voxcraftevo.utils.utilities import set_seed
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.configs.VXA import VXA
from voxcraftevo.configs.VXD import VXD
from voxcraftevo.fitness.evaluation import FitnessFunction


# (1.5**(1/3)-1)/0.01 = 14.4714


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--solver", default="nsgaii", type=str, help="solver for the optimization")
    parser.add_argument("--gens", default=199, type=int, help="generations for the ea")
    parser.add_argument("--popsize", default=100, type=int, help="population size for the ea")
    parser.add_argument("--checkpoint", default=1, type=int, help="how many generations for checkpointing")
    parser.add_argument("--reload", default=0, type=int, help="restart from last pickled population")
    parser.add_argument("--execs", default="executables", type=str,
                        help="relative path to the dir containing Voxcraft executables")
    parser.add_argument("--logs", default="logs", type=str, help="relative path to the logs dir")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="relative path to output dir")
    parser.add_argument("--data_dir", default="data", type=str, help="relative path to data dir")
    parser.add_argument("--pickle_dir", default="pickledPops", type=str, help="relative path to pickled dir")
    parser.add_argument("--fitness", default="fitness_score", type=str, help="fitness tag")
    parser.add_argument("--shape", default="gecko", type=str, help="shape to employ")
    parser.add_argument("--terrain", default="fixed", type=str, help="terrain for simulations")
    parser.add_argument("--remap", default=0, type=int, help="recompute fitness of parents")
    parser.add_argument("--rnn", default=1, type=int, help="use recurrent policy")
    return parser.parse_args()


class MyListener(Listener):

    def listen(self, solver):
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([str(solver.seed), str(solver.pop.gen), str(solver.elapsed_time()),
                                             str(solver.best_so_far.fitness["locomotion_score"]),
                                             str(np.median([ind.fitness["locomotion_score"] for ind in solver.pop])),
                                             str(max([ind.fitness["locomotion_score"] for ind in solver.pop]))
                                             ]) + "\n")


class NSGAIIListener(Listener):

    def listen(self, solver):
        solver.fast_non_dominated_sort()
        pareto_front = solver.fronts[0]
        best_locomotion = solver.best_locomotion
        best_sensing = solver.best_sensing
        if best_sensing is None:
            best_sensing = max(solver.pop, key=lambda x: x.fitness["sensing_score"])
        if best_locomotion is None:
            best_locomotion = min(solver.pop, key=lambda x: x.fitness["locomotion_score"])
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


class TestListener(Listener):

    def __init__(self, file_path, delimiter=";"):
        super().__init__(file_path, [], delimiter)

    def listen(self, solver):
        with open(self._file, "a") as file:
            header = ["seed", "elapsed.time", "shape", "k", "sensing.best.locomotion_score", "locomotion.best.locomotion_score",
                      "sensing.best.sensing_score", "locomotion.best.sensing_score"]
            file.write(self._delimiter.join(header) + "\n")
            #for ind in sorted(solver.pop, key=lambda x: x.id):
            #    if ind.id == 0:
            #        best_sensing = ind.fitness["_".join(["sensing_score", solver.fitness_func.shape, str(0)])]
            #    elif ind.id == 1:
            #        best_locomotion = ind.fitness["_".join(["locomotion_score", solver.fitness_func.shape, str(0)])]
            #best_locomotion = min([float(ind) for ind in solver.fitness_func.last_line.split(";")[9].split("/")])
            #best_sensing = max([float(ind) for ind in solver.fitness_func.last_line.split(";")[10].split("/")])
            for shape in solver.fitness_func.__SHAPES__:
                if shape == solver.fitness_func.shape:
                    continue
                for i in range(solver.fitness_func.k):
                    values = [str(solver.seed), str(solver.elapsed_time()), shape + "<-" + solver.fitness_func.shape,
                              str(i)]
                    for obj in ["locomotion_score", "sensing_score"]:
                        for ind in sorted(solver.pop, key=lambda x: x.id):
                            fit = ind.fitness["_".join([obj, shape, str(i)])]
                            if ind.id == 0:
                                values.append(str(fit))
                            elif ind.id == 1:
                                values.append(str(fit))
                    file.write(self._delimiter.join(values) + "\n")


class MyFitness(FitnessFunction):

    def __init__(self, solver, shape, terrain, is_recurrent):
        self.immovable_left = None
        self.immovable_right = None
        self.soft = None
        self.special_passable = None
        self.special_impassable = None
        self.wall_left = None
        self.wall_right = None
        self.world = terrain
        self.terrains = ["impassable", "passable_left", "passable_right"] if terrain == "fixed" \
            else ["passable", "impassable"] * int(terrain.split("-")[1])
        self.solver = solver
        self.shape = shape
        self.objective_dict = ObjectiveDict()
        self.is_recurrent = is_recurrent

    @staticmethod
    def get_file_name(*args):
        return "-".join(list(args))

    def get_body_length(self):
        if self.shape == "flatworm":
            return 4
        elif self.shape == "starfish":
            return 9
        elif self.shape == "gecko":
            return 6
        raise ValueError("Unknown shape: {}".format(self.shape))

    def create_objectives_dict(self):
        if self.solver == "ga":
            self.objective_dict.add_objective(name="locomotion_score", maximize=False,
                                              tag="<{}>".format("locomotion_score"),
                                              best_value=0.0, worst_value=5.0)
            return self.objective_dict
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
        self.wall_left = vxa.add_material(material_id=7, RGBA=(50, 50, 50, 255), E=10000, RHO=10, P=0.5,
                                          uDynamic=0.5, isFixed=1, isMeasured=0)
        self.wall_right = vxa.add_material(material_id=8, RGBA=(0, 50, 50, 255), E=10000, RHO=10, P=0.5,
                                           uDynamic=0.5, isFixed=1, isMeasured=0)
        vxa.write(filename=os.path.join(directory, "base.vxa"))

    def create_vxd(self, ind, directory, record_history, world_name=None):
        for _, r_label in enumerate([self.shape]):
            for terrain_id, p_label in enumerate(self.terrains):
                base_name = os.path.join(directory, self.get_file_name("bot_{:04d}".format(ind.id), str(terrain_id),
                                                                       r_label, p_label))
                body_length = self.get_body_length()
                world = self._create_world(body_length=body_length, p_label=p_label, world_name=self.world
                if world_name is None else world_name)

                if not self.is_recurrent:
                    vxd = VXD(NeuralWeightsX=ind.genotype, isPassable=p_label != "impassable", terrainID=terrain_id,
                              age=ind.age)
                else:
                    vxd = VXD(NeuralWeightsX=ind.genotype[:17 * 6],
                              NeuralWeightsH=ind.genotype[17 * 6: 17 * 6 + 6 * 6 + 6],
                              NeuralWeightsY=ind.genotype[17 * 6 + 6 * 6 + 6:], isPassable=p_label != "impassable",
                              terrainID=terrain_id, age=ind.age)
                vxd.set_data(data=world)
                vxd.set_tags(RecordVoxel=record_history, RecordFixedVoxels=record_history,
                             RecordStepSize=100 if record_history else 0)
                vxd.write(filename=base_name + ".vxd")

    def _create_world(self, body_length, p_label, world_name):
        if world_name == "fixed":
            return self._create_fixed_world(body_length=body_length, p_label=p_label)
        else:
            return self._create_random_world(body_length=body_length, p_label=p_label)

    def _create_fixed_world(self, body_length, p_label):
        world = np.zeros((body_length * 3, body_length * 5, int(body_length / 3) + 1))

        start = math.floor(body_length * 1.5)
        distance_from_wall = math.floor(body_length * 0.5)
        left_edge = body_length
        right_edge = body_length * 2
        wall_position = distance_from_wall * 2 + body_length * 2
        if self.shape == "starfish":
            world[start, distance_from_wall + body_length: wall_position - distance_from_wall, 0] = self.soft
            world[left_edge: right_edge, distance_from_wall + start, 0] = self.soft
        elif self.shape == "flatworm":
            world[start - body_length // 2: start + body_length // 2 + 1,
            distance_from_wall + start - body_length // 2: distance_from_wall + start + body_length // 2 + 1,
            0] = self.soft
        else:
            world[start + 1, distance_from_wall + start - 3: distance_from_wall + start + 3, 0] = self.soft
            world[start - 2, distance_from_wall + start - 3: distance_from_wall + start + 3, 0] = self.soft
            world[start - 3: start + 3, distance_from_wall + start - 2, 0] = self.soft
            world[start - 3: start + 3, distance_from_wall + start + 1, 0] = self.soft

        aperture_size = 1 if p_label == "impassable" else body_length - 1
        half = math.floor(body_length * 1.5)

        left_bank = half - int(aperture_size / 2) - 1
        right_bank = half + int(aperture_size / 2) + 1
        if p_label == "passable_left":
            left_bank -= math.ceil(aperture_size / 2)
            right_bank -= math.ceil(aperture_size / 2)
        elif p_label == "passable_right":
            left_bank += math.ceil(aperture_size / 2)
            right_bank += math.ceil(aperture_size / 2)
        world[:half, wall_position, :2] = self.immovable_left
        world[half:, wall_position, :2] = self.immovable_right
        world[left_bank + 1: right_bank, wall_position, :] = 0

        if p_label != "impassable":
            world[half, wall_position + body_length, 0] = self.special_passable
        else:
            world[half, wall_position + 1, 0] = self.special_impassable

        return world

    def _create_random_world(self, body_length, p_label):
        world = np.zeros((body_length * 3, body_length * 5, int(body_length / 3) + 1))

        start = math.floor(body_length * 1.5)
        distance_from_wall = math.floor(body_length * 0.5)
        left_edge = body_length
        right_edge = body_length * 2
        wall_position = distance_from_wall * 2 + body_length * 2
        lower_edge = body_length - 4
        if self.shape == "starfish":
            world[start, distance_from_wall + body_length: wall_position - distance_from_wall, 0] = self.soft
            world[left_edge: right_edge, distance_from_wall + start, 0] = self.soft
        elif self.shape == "flatworm":
            world[start - body_length // 2 + 1: start + body_length // 2,
            distance_from_wall + start - body_length // 2: distance_from_wall + start + body_length // 2 + 1,
            0] = self.soft
        else:
            world[start + 1, distance_from_wall + start - 3: distance_from_wall + start + 3, 0] = self.soft
            world[start - 2, distance_from_wall + start - 3: distance_from_wall + start + 3, 0] = self.soft
            world[start - 3: start + 3, distance_from_wall + start - 2, 0] = self.soft
            world[start - 3: start + 3, distance_from_wall + start + 1, 0] = self.soft

        center = random.choice([start + i - 2 for i in range(body_length // 2 + 1)])
        aperture_size = random.choice(np.arange(body_length // 2 + 1)) if p_label == "impassable" else random.choice(
            [body_length + 1, body_length + 2, body_length + 3])

        world[:center, wall_position, :2] = self.immovable_left
        world[center:, wall_position, :2] = self.immovable_right
        world[center - aperture_size // 2: center + aperture_size // 2, wall_position, :] = 0

        left_wall = min(left_edge, center - aperture_size // 2) - random.choice(
            [i for i in range(1, body_length // 3 + 1)])
        right_wall = max(right_edge, center + aperture_size // 2) + random.choice(
            [i for i in range(1, body_length // 3 + 1)])
        left_wall_type = random.choice(["none", "straight", "knee"])
        right_wall_type = random.choice(["none", "straight", "knee"])
        if left_wall_type == "straight":
            world[left_wall, lower_edge: wall_position, :2] = self.immovable_left
        elif left_wall_type == "knee":
            world[left_wall - 1, lower_edge: body_length // 2 + distance_from_wall + start, :2] = self.immovable_left
            world[left_wall - 1: left_wall + body_length // 4 - 1, body_length // 2 + distance_from_wall + start, :2] = \
                self.immovable_left
            world[left_wall + body_length // 4 - 1, body_length // 2 + distance_from_wall + start: wall_position, :2] = \
                self.immovable_left
        if right_wall_type == "straight":
            world[right_wall, lower_edge: wall_position, :2] = self.immovable_right
        elif right_wall_type == "knee":
            world[right_wall, lower_edge: body_length // 2 + distance_from_wall + start, :2] = self.immovable_right
            world[right_wall - body_length // 4:right_wall + 1, body_length // 2 + distance_from_wall + start, :2] = \
                self.immovable_right
            world[right_wall - body_length // 4, body_length // 2 + distance_from_wall + start: wall_position, :2] = \
                self.immovable_right

        if p_label != "impassable":
            world[center, wall_position + body_length, 0] = self.special_passable
        else:
            world[center, wall_position + 1, 0] = self.special_impassable

        return world

    def get_fitness(self, individuals, output_file, log_file, gen):
        fitness = {}
        root = etree.parse(output_file).getroot()
        for ind in individuals:
            values = {obj: [] for obj in self.objective_dict}
            for _, r_label in enumerate(["b"]):
                for terrain_id, p_label in enumerate(self.terrains):
                    for obj in values:
                        name = self.objective_dict[obj]["name"]
                        file_name = self.get_file_name("bot_{:04d}".format(ind.id), str(terrain_id), self.shape,
                                                       p_label)
                        test1 = self.parse_fitness_from_xml(root, bot_id=file_name, fitness_tag=name,
                                                            worst_value=self.objective_dict[obj][
                                                                "worst_value"])
                        test2 = self.parse_fitness_from_history(log_file,
                                                                fitness_tag="-".join(
                                                                    [str(ind.id), str(terrain_id),
                                                                     str(ind.age), name]),
                                                                worst_value=self.objective_dict[obj][
                                                                    "worst_value"])
                        values[obj].append(min(test1, test2) if self.objective_dict[obj]["maximize"]
                                           else max(test1, test2))
            fitness[ind.id] = {self.objective_dict[k]["name"]: min(v) if self.objective_dict[k]["maximize"] else max(v)
                               for k, v in values.items()}
        return fitness

    def save_histories(self, individual, input_directory, output_directory, executables_directory):
        sub.call("rm {}/*vxd".format(input_directory), shell=True)
        self.create_vxd(ind=individual, directory=input_directory, record_history=True, world_name="fixed")
        temp_dir = input_directory.replace("data", "temp")
        sub.call("mkdir {}".format(temp_dir), shell=True)
        self.create_vxa(directory=temp_dir)
        for file in os.listdir(input_directory):
            if file.endswith("vxd"):
                sub.call("cp {0} {1}/".format(os.path.join(input_directory, file), temp_dir), shell=True)
                sub.call("cd {0}; ./voxcraft-sim -i {1} -o output.xml > {2}".format(
                    executables_directory,
                    os.path.join("..", temp_dir),
                    os.path.join("..", output_directory, file.replace("vxd", "history"))),
                    shell=True)
                sub.call("cd {}; rm output.xml".format(executables_directory), shell=True)
                sub.call("rm {}/*.vxd".format(temp_dir), shell=True)
        sub.call("rm -rf {}".format(temp_dir), shell=True)

    @classmethod
    def create_fitness(cls, name, **kwargs):
        if name == "test":
            return TestFitness(**kwargs)
        return MyFitness(**kwargs)


class TestFitness(MyFitness):
    __SHAPES__ = ["flatworm", "starfish", "gecko"]

    def __init__(self, solver, shape, terrain, is_recurrent, file_name, k=1):
        super().__init__(solver, shape, terrain, is_recurrent)
        self.last_line = open(file_name).readlines()[-1]
        self.sensing_genotype = np.array([float(t) for t in self.last_line.split(";")[-2].split(",")])
        self.locomotion_genotype = np.array([float(t) for t in self.last_line.split(";")[-1].split(",")])
        self.k = k

    def get_body_length(self, shape):
        if shape == "flatworm":
            return 4
        elif shape == "starfish":
            return 9
        elif shape == "gecko":
            return 6
        raise ValueError("Unknown shape: {}".format(shape))

    def create_objectives_dict(self):
        for shape in self.__SHAPES__:
            if shape == self.shape:
                continue
            for i in range(self.k):
                self.objective_dict.add_objective(name="locomotion_score_{}_{}".format(shape, i), maximize=False,
                                                  tag="<{}>".format("locomotion_score"),
                                                  best_value=0.0, worst_value=5.0)
                self.objective_dict.add_objective(name="sensing_score_{}_{}".format(shape, i), maximize=True,
                                                  tag="<{}>".format("sensing_score"),
                                                  best_value=1.0, worst_value=0.0)
        return self.objective_dict

    def create_vxd(self, ind, directory, record_history, world_name=None):
        if ind.id == 0:
            ind.genotype = self.sensing_genotype
        elif ind.id == 1:
            ind.genotype = self.locomotion_genotype
        else:
            return
        for _, r_label in enumerate(self.__SHAPES__):
            if r_label == self.shape:
                continue
            for terrain_id, p_label in enumerate(self.terrains):
                for i in range(self.k):
                    base_name = os.path.join(directory, self.get_file_name("bot_{:04d}".format(ind.id), str(terrain_id),
                                                                           str(i), r_label, p_label))
                    body_length = self.get_body_length(r_label)
                    world = self._create_world(body_length=body_length, p_label=p_label, world_name=self.world)

                    if not self.is_recurrent:
                        vxd = VXD(NeuralWeightsX=ind.genotype, isPassable=p_label != "impassable", terrainID=terrain_id,
                                  age=ind.age)
                    else:
                        vxd = VXD(NeuralWeightsX=ind.genotype[:17 * 6],
                                  NeuralWeightsH=ind.genotype[17 * 6: 17 * 6 + 6 * 6 + 6],
                                  NeuralWeightsY=ind.genotype[17 * 6 + 6 * 6 + 6:], isPassable=p_label != "impassable",
                                  terrainID=terrain_id, age=ind.age)
                    vxd.set_data(data=world)
                    vxd.set_tags(RecordVoxel=record_history, RecordFixedVoxels=record_history,
                                 RecordStepSize=100 if record_history else 0)
                    vxd.write(filename=base_name + ".vxd")

    def get_fitness(self, individuals, output_file, log_file, gen):
        fitness = {}
        root = etree.parse(output_file).getroot()
        for ind in individuals:
            values = {obj: [] for obj in self.objective_dict}
            for terrain_id, p_label in enumerate(self.terrains):
                for obj in values:
                    name = self.objective_dict[obj]["name"]
                    r_label = name.split("_")[2]
                    i = name.split("_")[3]
                    file_name = self.get_file_name("bot_{:04d}".format(ind.id), str(terrain_id), str(i), r_label,
                                                   p_label)
                    test1 = self.parse_fitness_from_xml(root, bot_id=file_name,
                                                        fitness_tag="_".join(name.split("_")[:2]),
                                                        worst_value=self.objective_dict[obj]["worst_value"])
                    test2 = test1
                    values[obj].append(min(test1, test2) if self.objective_dict[obj]["maximize"]
                                       else max(test1, test2))
            fitness[ind.id] = {self.objective_dict[k]["name"]: min(v) if self.objective_dict[k]["maximize"] else max(v)
                               for k, v in values.items()}
        return fitness

    def save_histories(self, individual, input_directory, output_directory, executables_directory):
        return


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)

    pickle_dir = "{0}{1}".format(arguments.pickle_dir, arguments.seed)
    data_dir = "{0}{1}".format(arguments.data_dir, arguments.seed)
    sub.call("rm -rf {0}".format(data_dir), shell=True)

    seed = arguments.seed
    number_of_params = ((17 * 2) + 2) if not arguments.rnn else (17 * 6 + 6 * 6 + 6 + 6 * 2 + 2)
    if arguments.remap is None:
        arguments.remap = arguments.terrain.startswith("random")
    else:
        arguments.remap = bool(arguments.remap)
    if arguments.fitness == "test":
        arguments.gens = 0
        fitness = TestFitness(arguments.solver, arguments.shape, arguments.terrain, arguments.rnn == 1,
                              file_name="{0}_{1}.csv".format(arguments.shape, seed))
        listener = TestListener(file_path="test_{0}_{1}.csv".format(arguments.shape, seed))
    elif arguments.solver == "ga":
        fitness = MyFitness(arguments.solver, arguments.shape, arguments.terrain, arguments.rnn == 1)
        listener = NSGAIIListener(file_path="{0}_{1}.csv".format(arguments.shape, seed),
                                  header=["seed", "gen", "elapsed.time", "best.sensing", "best.locomotion",
                                          "knee.locomotion", "knee.sensing", "locomotions", "sensings",
                                          "pareto.locomotions", "pareto.sensings", "best.sensing.g",
                                          "best.locomotion.g"])
    else:
        fitness = MyFitness(arguments.solver, arguments.shape, arguments.terrain, arguments.rnn == 1)
        listener = MyListener(file_path="{0}_{1}.csv".format(arguments.shape, seed),
                              header=["seed", "gen", "elapsed.time", "best.locomotion_score",
                                      "median.locomotion_score", "min.locomotion_score"])
    if arguments.solver == "ga":
        evolver = Solver.create_solver(name="ga", seed=seed,
                                       pop_size=arguments.popsize,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       survival_selector="worst",
                                       parent_selector="tournament",
                                       fitness_func=fitness,
                                       remap=arguments.remap,
                                       genetic_operators={"gaussian_mut": 1.0},
                                       offspring_size=arguments.popsize // 2,
                                       overlapping=True,
                                       data_dir=data_dir,
                                       hist_dir="history{}".format(seed),
                                       pickle_dir=pickle_dir,
                                       output_dir=arguments.output_dir,
                                       executables_dir=arguments.execs,
                                       logs_dir=arguments.logs,
                                       listener=listener,
                                       tournament_size=5,
                                       mu=0.0,
                                       sigma=0.35,
                                       n=number_of_params,
                                       range=(-1, 1),
                                       upper=2.0,
                                       lower=-1.0)
    elif arguments.solver == "nsgaii":
        evolver = Solver.create_solver(name="nsgaii",
                                       seed=seed,
                                       pop_size=arguments.popsize,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=fitness,
                                       remap=arguments.remap,
                                       genetic_operators={"gaussian_mut": 1.0},
                                       offspring_size=arguments.popsize // 2,
                                       data_dir=data_dir,
                                       hist_dir="history{}".format(seed),
                                       pickle_dir=pickle_dir,
                                       output_dir=arguments.output_dir,
                                       executables_dir=arguments.execs,
                                       logs_dir=arguments.logs,
                                       listener=listener,
                                       tournament_size=2,
                                       mu=0.0,
                                       sigma=0.35,
                                       n=number_of_params,
                                       range=(-1, 1),
                                       upper=2.0,
                                       lower=-1.0)
    else:
        raise ValueError("Invalid solver name: {}".format(arguments.solver))

    if arguments.reload:
        evolver.reload()
    else:
        evolver.solve(max_gens=arguments.gens, checkpoint_every=arguments.checkpoint)
    start_time = time()
    sub.call("echo That took a total of {} minutes".format((time() - start_time) / 60.), shell=True)
