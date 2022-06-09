import hashlib
import math

from lxml import etree
import subprocess as sub
import numpy as np

from ..configs.VXA import VXA
from ..configs.VXD import VXD


def get_body_length(r_label):
    if r_label == "a":
        return 3
    elif r_label == "b":
        return 9
    elif r_label == "c":
        return 27
    else:
        raise ValueError("Unknown body size: {}".format(r_label))


def create_world(record_history, seed, ind, r_label, p_label):
    base_name = "data" + str(seed) + "/bot_{:04d}".format(ind.id) + r_label + p_label

    vxa = VXA(TempAmplitude=14.4714, TempPeriod=0.2, TempBase=0, NeuralWeights=ind.genotype.weights,
              isPassable=p_label == "passable")
    body_length = get_body_length(r_label)
    immovable = vxa.add_material(RGBA=(50, 50, 50, 255), E=5e10, RHO=1e8, isFixed=1)
    soft = vxa.add_material(RGBA=(255, 0, 0, 255), E=10000, RHO=10, P=0.5, uDynamic=0.5, CTE=0.01)
    special = vxa.add_material(RGBA=(255, 255, 255, 255), E=5e10, RHO=1e8, isFixed=1)
    vxa.write("data" + str(seed) + "/base.vxa")
    vxa.write(base_name + ".vxa")

    world = np.zeros((body_length * 3, body_length * 5, int(body_length / 3) + 1))

    start = math.floor(body_length * 1.5)
    half_thickness = math.floor(body_length / 6)
    world[start - half_thickness: start + half_thickness + 1, body_length: body_length * 2, :half_thickness + 1] = soft
    world[body_length: body_length * 2, start - half_thickness: start + half_thickness + 1, :half_thickness + 1] = soft

    aperture_size = round(body_length * (0.75 if p_label == "passable" else 0.25))
    world[:, body_length * 2, :] = immovable
    world[:, body_length * 3, :] = immovable
    world[math.floor(body_length * 1.5) - int(aperture_size / 2) - 1, body_length * 2: body_length * 3 + 1,
    :] = immovable
    world[math.floor(body_length * 1.5) + int(aperture_size / 2) + 1, body_length * 2: body_length * 3 + 1,
    :] = immovable
    world[
    math.floor(body_length * 1.5) - int(aperture_size / 2): math.floor(body_length * 1.5) + int(aperture_size / 2) + 1,
    body_length * 2: body_length * 3 + 1, :] = 0
    world[math.floor(body_length * 1.5), body_length * 5 - 1, 0] = special

    vxd = VXD()
    vxd.set_data(world)
    vxd.set_tags(record_history=record_history, RecordVoxel=1)
    vxd.write(base_name + ".vxd")


def evaluate_population(pop, record_history=False):
    seed = pop.seed

    N = len(pop)
    if record_history:
        N = 1  # only evaluate the best ind in the pop

    # clear old robot files from the data directory
    sub.call("rm data{}/*".format(seed), shell=True)

    # remove old sim output.xml if we are saving new stats
    if not record_history:
        sub.call("rm output{0}_{1}.xml".format(seed, pop.gen), shell=True)

    num_evaluated_this_gen = 0

    # hash all inds in the pop
    if not record_history:

        for n, ind in enumerate(pop):

            ind.teammate_ids = []
            ind.duplicate = False
            data_string = ""
            for name, details in ind.genotype.to_phenotype_mapping.items():
                data_string += details["state"].tostring()
                m = hashlib.md5()
                m.update(data_string)
                ind.md5 = m.hexdigest()

            if (ind.md5 in pop.already_evaluated) and len(
                    ind.fit_hist) == 0:  # line 141 mutations.py clears fit_hist for new designs
                # print "dupe: ", ind.id
                ind.duplicate = True

            # It's still possible to get duplicates in generation 0.
            # Then there's two inds with the same md5, age, and fitness (because one will overwrite the other).
            # We can adjust mutations so this is impossible
            # or just don't evaluate th new yet duplicate design.
    sub.call("mkdir data{}".format(str(seed)), shell=True)
    # evaluate new designs
    for r_num, r_label in enumerate(['a', 'b', 'c']):
        for p_num, p_label in enumerate(["passable", "impassable"]):
            for n, ind in enumerate(pop[:N]):

                # don't evaluate if invalid
                if not ind.phenotype.is_valid():
                    for rank, goal in pop.objective_dict.items():
                        if goal["name"] != "age":
                            setattr(ind, goal["name"], goal["worst_value"])

                    print("Skipping invalid individual")

                # if it's a new valid design, or if we are recording history, create a vxd
                # new designs are evaluated with teammates from the entire population (new and old).
                elif (ind.md5 not in pop.already_evaluated) or record_history:

                    num_evaluated_this_gen += 1
                    pop.total_evaluations += 1

                    create_world(record_history, seed, ind, r_label, p_label)

    # ok let's finally evaluate all the robots in the data directory

    if record_history:  # just save history, don't assign fitness
        print("Recording the history of the run champ")
        for r_num, r_label in enumerate(['a', 'b', 'c']):
            for p_num, p_label in enumerate(["passable", "impassable"]):
                sub.call("cp " + "data" + str(seed) + "/bot_{:04d}".format(ind.id) + r_label + p_label + ".vxa" +
                         " data{}".format(str(seed) + str(r_label)), shell=True)
                sub.call('cp data' + str(seed) + '/bot_{:04d}'.format(ind.id) + '{}.vxd'.format(
                    r_label + p_label) + ' data{}'.format(
                    str(seed) + str(r_label)), shell=True)
                sub.call(
                    "cd executables; ./voxcraft-sim -i ../data{0} > ../{0}_id{1}_fit{2}.hist".format(
                        str(seed) + str(r_label) + str(p_label),
                        pop[0].id,
                        int(100 * pop[0].fitness)), shell=True)
                sub.call("rm -r data{}".format(str(seed) + str(r_label) + str(p_label)), shell=True)

    else:  # normally, we will just want to update fitness and not save the trajectory of every voxel

        print("GENERATION {}".format(pop.gen))

        print("Launching {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop)))

        while True:
            try:
                sub.call("cd executables; ./voxcraft-sim -i ../data{0} -o ../output{1}_{2}.xml".format(seed, seed, pop.gen), shell=True)
                # sub.call waits for the process to return
                # after it does, we collect the results output by the simulator
                root = etree.parse("output{0}_{1}.xml".format(seed, pop.gen)).getroot()
                break

            except IOError:
                print("Dang it! There was an IOError. I'll re-simulate this batch again...")
                pass

            except IndexError:
                print("Shoot! There was an IndexError. I'll re-simulate this batch again...")
                pass

        for ind in pop:

            if ind.phenotype.is_valid() and ind.md5 not in pop.already_evaluated:

                for r_num, r_label in enumerate(['a', 'b', 'c']):
                    for p_num, p_label in enumerate(["passable", "impassable"]):
                        body_length = get_body_length(r_label)
                        print(root.findall("detail/bot_{:04d}".format(ind.id) + r_label + p_label + "/fitness_score"))
                        ind.fit_hist += [float(
                            root.findall("detail/bot_{:04d}".format(ind.id) + r_label + p_label + "/fitness_score")[
                                0].text) / body_length]

                ind.fitness = np.min(ind.fit_hist)
                print("Assigning ind {0} fitness {1}".format(ind.id, ind.fitness))

                pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                                  for rank, details in
                                                  pop.objective_dict.items()]
