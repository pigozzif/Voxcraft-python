import os
from time import time
import subprocess as sub
import argparse
import math

import numpy as np
from lxml import etree

from voxcraftevo.listeners.listener import Listener
from voxcraftevo.utils.utilities import set_seed
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.configs.VXA import VXA
from voxcraftevo.configs.VXD import VXD
from voxcraftevo.evo.algorithms import GeneticAlgorithm
from voxcraftevo.fitness.evaluation import FitnessFunction


# actuation +/- 50%
# (1.5**(1/3)-1)/0.01 = 14.4714


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--gens", default=501, type=int, help="generations for the ea")
    parser.add_argument("--popsize", default=4, type=int, help="population size for the ea")
    parser.add_argument("--history", default=100, type=int, help="how many generations for saving history")
    parser.add_argument("--checkpoint", default=1, type=int, help="how many generations for checkpointing")
    parser.add_argument("--time", default=48, type=int, help="maximumm hours for the ea")
    parser.add_argument("--reload", default=0, type=int, help="restart from last pickled population")
    parser.add_argument("--execs", default="executables", type=str,
                        help="relative path to the dir containing Voxcraft executables")
    parser.add_argument("--output_dir", default="output", type=str, help="relative path to output dir")
    parser.add_argument("--data_dir", default="data", type=str, help="relative path to data dir")
    parser.add_argument("--pickle_dir", default="pickledPops", type=str, help="relative path to pickled dir")
    parser.add_argument("--fitness", default="fitness_score", type=str, help="fitness tag")
    return parser.parse_args()


class MyListener(Listener):

    def listen(self, solver):
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([str(solver.seed), str(solver.pop.gen), str(solver.elapsed_time()),
                                             str(solver.best_so_far.fitness["fitness"]), str(solver.best_so_far.id),
                                             str(np.median([ind.fitness["fitness"] for ind in solver.pop])),
                                             str(min([ind.fitness["fitness"] for ind in solver.pop]))]) + "\n")


class MyFitness(FitnessFunction):

    def __init__(self, fitness):
        self.fitness = fitness
        self.immovable_left = None
        self.immovable_right = None
        self.soft = None
        self.special = None
        self.terrains = ["passable_left"] if "locomotion" in fitness else ["passable_left", "passable_right",
                                                                           "impassable"]

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
        objective_dict = ObjectiveDict()
        objective_dict.add_objective(name="fitness", maximize=True, tag="<{}>".format(self.fitness))
        return objective_dict

    def create_vxa(self, directory):
        vxa = VXA(TempAmplitude=14.4714, TempPeriod=0.2, TempBase=0)
        self.immovable_left = vxa.add_material(material_id=1, RGBA=(50, 50, 50, 255), E=5e10, RHO=1e8, isFixed=1,
                                               isMeasured=0)
        self.immovable_right = vxa.add_material(material_id=2, RGBA=(0, 50, 50, 255), E=5e10, RHO=1e8, isFixed=1,
                                                isMeasured=0)
        self.special = vxa.add_material(material_id=3, RGBA=(255, 255, 255, 255), E=5e10, RHO=1e8, isFixed=1,
                                        isMeasured=0)
        self.soft = vxa.add_material(material_id=4, RGBA=(255, 0, 0, 255), E=10000, RHO=10, P=0.5, uDynamic=0.5,
                                     CTE=0.01, isMeasured=1)
        vxa.write(filename=os.path.join(directory, "base.vxa"))

    def create_vxd(self, ind, directory, record_history):
        for _, r_label in enumerate(["b"]):
            for _, p_label in enumerate(self.terrains):
                base_name = os.path.join(directory, self.get_file_name("bot_{:04d}".format(ind.id), r_label,
                                                                       p_label))
                body_length = self.get_body_length(r_label)
                world = np.zeros((body_length * 3, body_length * 5, int(body_length / 3) + 1))

                start = math.floor(body_length * 1.5)
                half_thickness = math.floor(body_length / 6)
                world[start - half_thickness: start + half_thickness + 1, body_length - 1: body_length * 2 - 1,
                :half_thickness + 1] = self.soft
                world[body_length: body_length * 2, start - half_thickness - 1: start + half_thickness,
                :half_thickness + 1] = self.soft

                aperture_size = round(body_length * (0.25 if p_label == "impassable" else 0.75))
                half = math.floor(body_length * 1.5)
                if "locomotion" not in self.fitness:
                    world[:half, body_length * 2, :] = self.immovable_left
                    world[half:, body_length * 2, :] = self.immovable_right

                left_bank = half - int(aperture_size / 2) - 1
                right_bank = half + int(aperture_size / 2) + 1
                if p_label == "passable_left":
                    left_bank -= math.ceil(aperture_size / 2)
                    right_bank -= math.ceil(aperture_size / 2)
                elif p_label == "passable_right":
                    left_bank += math.ceil(aperture_size / 2)
                    right_bank += math.ceil(aperture_size / 2)
                if "locomotion" not in self.fitness:
                    world[left_bank, body_length * 2: body_length * 3 + 1, :] = self.immovable_left
                    world[right_bank, body_length * 2: body_length * 3 + 1, :] = self.immovable_right
                    world[left_bank + 1: right_bank, body_length * 2: body_length * 3 + 1, :] = 0

                if p_label != "impassable":
                    world[math.floor(body_length * 1.5), body_length * 5 - 1, 0] = self.special

                vxd = VXD(NeuralWeights=ind.genotype, isPassable=p_label != "impassable")
                vxd.set_data(data=world)
                vxd.set_tags(RecordVoxel=record_history, RecordFixedVoxels=record_history,
                             RecordStepSize=100 if record_history else 0)
                vxd.write(filename=base_name + ".vxd")

    def get_fitness(self, ind, output_file):
        root = etree.parse(output_file).getroot()
        values = []
        for _, r_label in enumerate(["b"]):
            for _, p_label in enumerate(self.terrains):
                values.append(float(
                    self.parse_fitness(root, self.get_file_name("bot_{:04d}".format(ind.id), r_label,
                                                                p_label), self.fitness).text))

        return {"fitness": min(values)}

    def save_histories(self, best, input_directory, output_directory):
        local_dir = "{0}/vxd_{1}".format(input_directory, best.id)
        self.create_vxd(ind=best, directory=input_directory, record_history=True)
        sub.call("mkdir temp", shell=True)
        self.create_vxa(directory="temp")
        for file in os.listdir(local_dir):
            if file.endswith("vxd"):
                print(file)
                sub.call("cp {} temp/".format(os.path.join(local_dir, file)), shell=True)
                sub.call("cd executables; ./voxcraft-sim -i {0} -o output.xml > {1}".format(
                    os.path.join("..", "temp"),
                    os.path.join("..", output_directory, file.replace("vxd", "history"))), shell=True)
                # sub.call("rm temp/{}".format(os.path.join(local_dir, file)), shell=True)
        # sub.call("rm -rf temp", shell=True)


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)

    pickle_dir = "{0}{1}".format(arguments.pickle_dir, arguments.seed)
    data_dir = "{0}{1}".format(arguments.data_dir, arguments.seed)
    sub.call("rm -rf {0}{1}".format(pickle_dir, arguments.seed), shell=True)
    sub.call("rm -rf {0}{1}".format(data_dir, arguments.seed), shell=True)

    seed = arguments.seed
    evolver = GeneticAlgorithm(seed=seed, pop_size=arguments.popsize, genotype_factory="uniform_float",
                               solution_mapper="direct", survival_selector="worst", parent_selector="tournament",
                               fitness_func=MyFitness(arguments.fitness), remap=False,
                               genetic_operators={"gaussian_mut": 1.0},
                               offspring_size=arguments.popsize // 2, overlapping=True,
                               data_dir=data_dir, hist_dir="history{}".format(seed),
                               pickle_dir=pickle_dir, output_dir=arguments.output_dir,
                               executables_dir=arguments.execs, listener=MyListener(file_path="{}.csv".format(seed),
                                                                                    header=["seed", "gen", "elapsed"
                                                                                                           ".time",
                                                                                            "best.fitness", "best.id",
                                                                                            "median.fitness",
                                                                                            "min.fitness"]),
                               tournament_size=5, mu=0.0, sigma=0.35, n=(12 * 8) + 8,
                               range=(-1, 1), upper=2.0, lower=-1.0)

    if arguments.reload:
        evolver.reload()
    else:
        evolver.solve(max_hours_runtime=arguments.time, max_gens=arguments.gens, checkpoint_every=arguments.checkpoint,
                      save_hist_every=arguments.history)
    start_time = time()
    sub.call("echo That took a total of {} minutes".format((time() - start_time) / 60.), shell=True)
