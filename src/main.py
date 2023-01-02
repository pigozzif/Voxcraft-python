import os
from time import time
import subprocess as sub
import argparse
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import imageio

from voxcraftevo.evo.algorithms import Solver
from voxcraftevo.listeners.listener import Listener
from voxcraftevo.utils.utilities import set_seed
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.fitness.evaluation import FitnessFunction


# (1.5**(1/3)-1)/0.01 = 14.4714


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--solver", default="kmeans", type=str, help="solver for the optimization")
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
    parser.add_argument("--num_dims", default=10, type=int, help="number of problem dimensions")
    return parser.parse_args()


class MyListener(Listener):

    def listen(self, solver):
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([str(solver.pop.gen), str(solver.elapsed_time()),
                                             str(solver.get_best_fitness()), str(np.nan), str(np.nan)]
                                            ) + "\n")


class VizListener(Listener):

    def __init__(self, file_path: str, header: Iterable[str]):
        super().__init__(file_path, header)
        self._inner_listener = MyListener(file_path=file_path, header=header)
        self.images = []
        os.system("rm -rf frames")
        os.makedirs("frames")

    def listen(self, solver) -> None:
        self._inner_listener.listen(solver=solver)
        r_min, r_max = -1.0, 1.1
        x_axis = np.arange(r_min, r_max, 0.05)
        y_axis = np.arange(r_min, r_max, 0.05)
        x, y = np.meshgrid(x_axis, y_axis)
        results = np.array([[MyFitness.point_aiming([x[i, j], y[i, j]]) for i in range(len(x_axis))]
                            for j in range(len(y_axis))])
        plt.pcolormesh(x_axis, y_axis, results, cmap="plasma")
        plt.scatter(0.0, 0.0, marker="o", color="white")
        plt.scatter([ind.genotype[0] for ind in solver.pop], [ind.genotype[1] for ind in solver.pop], marker="x",
                    color="red")
        image = "frames/{}.png".format(solver.pop.gen)
        plt.savefig(image)
        self.images.append(image)
        plt.clf()

    def save_gif(self):
        with imageio.get_writer("test.gif", mode="I") as writer:
            for filename in self.images:
                image = imageio.imread(filename)
                writer.append_data(image)
        os.system("rm -rf frames")


class MyFitness(FitnessFunction):

    def __init__(self):
        self.objective_dict = ObjectiveDict()

    def create_objectives_dict(self):
        self.objective_dict.add_objective(name="fitness_score", maximize=False,
                                          tag="<{}>".format("fitness_score"),
                                          best_value=0.0, worst_value=float("inf"))
        return self.objective_dict

    def create_vxa(self, directory):
        pass

    def create_vxd(self, ind, directory, record_history):
        pass

    def get_fitness(self, individuals):
        fitness = {}
        for ind in individuals:
            fitness[ind.id] = {"fitness_score": self.point_aiming(ind.genotype)}
        return fitness

    def save_histories(self, individual, input_directory, output_directory, executables_directory):
        pass

    @staticmethod
    def point_aiming(x, target=0.0):
        return np.sum([(i - target) ** 2 for i in x])


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)

    pickle_dir = "{0}{1}".format(arguments.pickle_dir, arguments.seed)
    data_dir = "{0}{1}".format(arguments.data_dir, arguments.seed)

    seed = arguments.seed
    number_of_params = arguments.num_dims
    listener = VizListener(file_path="my.{}.txt".format(seed), header=["iteration", "elapsed.time", "best.fitness",
                                                                       "avg.test", "std.test"])
    if arguments.solver == "es":
        evolver = Solver.create_solver(name="es", seed=seed, pop_size=arguments.popsize, num_dims=number_of_params,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=MyFitness(),
                                       data_dir=data_dir, hist_dir="history{}".format(seed),
                                       pickle_dir=pickle_dir, output_dir=arguments.output_dir,
                                       executables_dir=arguments.execs,
                                       logs_dir=None,
                                       listener=listener,
                                       sigma=0.1, sigma_decay=0.999, sigma_limit=0.01, l_rate_init=0.02,
                                       l_rate_decay=0.999, l_rate_limit=0.001, n=number_of_params, range=(-1, 1),
                                       upper=2.0, lower=-1.0)
    elif arguments.solver == "kmeans":
        evolver = Solver.create_solver(name="kmeans", seed=seed, pop_size=arguments.popsize, num_dims=number_of_params,
                                       num_modes=2, genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=MyFitness(),
                                       data_dir=data_dir, hist_dir="history{}".format(seed),
                                       pickle_dir=pickle_dir, output_dir=arguments.output_dir,
                                       executables_dir=arguments.execs,
                                       logs_dir=None,
                                       listener=listener,
                                       sigma=0.1, sigma_decay=0.999, sigma_limit=0.01, l_rate_init=0.02,
                                       l_rate_decay=0.999, l_rate_limit=0.001, n=number_of_params, range=(-1, 1),
                                       upper=2.0, lower=-1.0)
    else:
        raise ValueError("Invalid solver name: {}".format(arguments.solver))
    start_time = time()
    evolver.solve(max_hours_runtime=arguments.time, max_gens=arguments.gens, checkpoint_every=arguments.checkpoint,
                  save_hist_every=arguments.history)
    if isinstance(listener, VizListener):
        listener.save_gif()
    sub.call("echo That took a total of {} seconds".format(time() - start_time), shell=True)
