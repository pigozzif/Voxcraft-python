import random
import numpy as np
from time import time
import cPickle
import subprocess as sub
from glob import glob
import argparse

from evo.softbot import Genotype, Phenotype, Population
from evo.algorithms import Optimizer
from evo.utilities import natural_sort
from evo.objectives import ObjectiveDict
from evo.evaluation import evaluate_population
from evo.mutation import create_new_children_through_mutation
from evo.selection import pareto_selection


# actuation +/- 50%
# (1.5**(1/3)-1)/0.01 = 14.4714


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--debug", default=0, type=int, help="debug")
    parser.add_argument("--reload", default=0, type=int, help="reload experiment")
    parser.add_argument("--gens", default=1001, type=int, help="generations for the ea")
    parser.add_argument("--popsize", default=8 * 2 - 1, type=int, help="population size for the ea")
    parser.add_argument("--history", default=100, type=int, help="how many generations for saving history")
    parser.add_argument("--checkpoint", default=1, type=int, help="how many generations for checkpointing")
    parser.add_argument("--time", default=47, type=int, help="maximumm hours for the ea")
    return parser.parse_args()


class MyGenotype(Genotype):

    def __init__(self, num_weights):
        Genotype.__init__(self)
        self.weights = [random.random() * 2.0 - 1.0 for _ in range(num_weights)]

    def express(self):
        pass


class MyPhenotype(Phenotype):

    def __init__(self, genotype):
        super().__init__(genotype)

    def is_valid(self):
        return True


def create_optimizer(args):
    sub.call("mkdir data{}".format(args.seed), shell=True)
    sub.call("cp base.vxa data{}/".format(args.seed), shell=True)
    if args.debug:
        print("DEBUG MODE")
        sub.call("rm a{}_id0_fit-1000000000.hist".format(args.seed), shell=True)
        sub.call("rm -r pickledPops{0} && rm -r data{0}".format(args.seed), shell=True)

    if args.reload:
        sub.call("rm voxcraft-sim && rm vx3_node_worker", shell=True)
        sub.call("cp /users/s/k/skriegma/sim/build/voxcraft-sim .", shell=True)
        sub.call("cp /users/s/k/skriegma/sim/build/vx3_node_worker .", shell=True)

    sub.call("mkdir pickledPops{}".format(args.seed), shell=True)
    sub.call("mkdir data{}".format(args.seed), shell=True)
    sub.call("cp base.vxa data{}/".format(args.seed), shell=True)
    # Now specify the objectives for the optimization.
    # Creating an objectives dictionary
    my_objective_dict = ObjectiveDict()

    # Adding an objective named "fitness", which we want to maximize.
    # This information is returned by Voxelyze in a fitness .xml file, with a tag named "distance"
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<fitness_score>")

    if args.debug:
        # quick test to make sure evaluation is working properly:
        my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=args.popsize)
        my_pop.seed = args.seed
        evaluate_population(my_pop, record_history=True)
        exit()

    if len(glob("pickledPops{}/Gen_*.pickle".format(args.seed))) == 0:
        # Initializing a population of SoftBots
        my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=args.popsize)
        my_pop.seed = args.seed

        # Setting up our optimization
        my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

    else:
        successful_restart = False
        pickle_idx = 0
        while not successful_restart:
            try:
                pickled_pops = glob("pickledPops{}/*".format(args.seed))
                last_gen = natural_sort(pickled_pops, reverse=True)[pickle_idx]
                with open(last_gen, 'rb') as handle:
                    [optimizer, random_state, numpy_random_state] = cPickle.load(handle)
                successful_restart = True

                my_pop = optimizer.pop
                my_optimization = optimizer
                my_optimization.continued_from_checkpoint = True
                my_optimization.start_time = time()

                random.setstate(random_state)
                np.random.set_state(numpy_random_state)

                print("Starting from pickled checkpoint: generation {}".format(my_pop.gen))

            except EOFError:
                # something went wrong writing the checkpoint : use previous checkpoint and redo last generation
                sub.call("touch IO_ERROR_$(date +%F_%R)", shell=True)
                pickle_idx += 1
                pass

    return my_optimization


if __name__ == "__main__":
    args = parse_args()
    optimizer = create_optimizer(args)
    start_time = time()
    optimizer.run(max_hours_runtime=args.time, max_gens=args.gens,
                  checkpoint_every=args.checkpoint, save_hist_every=args.history)
    print("That took a total of {} minutes".format((time() - start_time) / 60.))
    # finally, record the history of best robot at end of evolution so we can play it back in VoxCad
    optimizer.pop.individuals = [optimizer.pop.individuals[0]]
    evaluate_population(optimizer.pop, record_history=True)
