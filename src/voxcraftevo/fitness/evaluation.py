import hashlib
import math

from lxml import etree
import subprocess as sub
import numpy as np

from ..configs.VXA import VXA
from ..configs.VXD import VXD


def get_file_name(*args):
    return "-".join(list(args))


def parse_fitness(root, bot_id):
    detail = root.findall("detail/")
    for d in detail:
        if d.tag == bot_id:
            return d.findall("fitness_score")[0]
    raise IndexError


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
    base_name = "data" + str(seed) + "/" + get_file_name("bot_{:04d}".format(ind.id), r_label, p_label)

    vxa = VXA(TempAmplitude=14.4714, TempPeriod=0.2, TempBase=0)
    body_length = get_body_length(r_label)
    immovable_left = vxa.add_material(RGBA=(50, 50, 50, 255), E=5e10, RHO=1e8, isFixed=1)
    immovable_right = vxa.add_material(RGBA=(0, 50, 50, 255), E=5e10, RHO=1e8, isFixed=1)
    # special = vxa.add_material(RGBA=(255, 255, 255, 255), E=5e10, RHO=1e8, isFixed=1)
    soft = vxa.add_material(RGBA=(255, 0, 0, 255), E=10000, RHO=10, P=0.5, uDynamic=0.5, CTE=0.01)
    vxa.write("data{}/base.vxa".format(str(seed)))

    world = np.zeros((body_length * 3, body_length * 5, int(body_length / 3) + 1))

    start = math.floor(body_length * 1.5)
    half_thickness = math.floor(body_length / 6)
    world[start - half_thickness: start + half_thickness + 1, body_length - 1: body_length * 2 - 1,
    :half_thickness + 1] = soft
    world[body_length: body_length * 2, start - half_thickness - 1: start + half_thickness, :half_thickness + 1] = soft

    aperture_size = round(body_length * (0.25 if p_label == "impassable" else 0.75))
    half = math.floor(body_length * 1.5)
    world[:half, body_length * 2, :] = immovable_left
    world[half:, body_length * 2, :] = immovable_right

    left_bank = half - int(aperture_size / 2) - 1
    right_bank = half + int(aperture_size / 2) + 1
    if p_label == "passable_left":
        left_bank -= math.ceil(aperture_size / 2)
        right_bank -= math.ceil(aperture_size / 2)
    elif p_label == "passable_right":
        left_bank += math.ceil(aperture_size / 2)
        right_bank += math.ceil(aperture_size / 2)
    world[left_bank, body_length * 2: body_length * 3 + 1, :] = immovable_left
    world[right_bank, body_length * 2: body_length * 3 + 1, :] = immovable_right
    world[left_bank + 1: right_bank, body_length * 2: body_length * 3 + 1, :] = 0

    # world[math.floor(body_length * 1.5), body_length * 5 - 1, 0] = special

    vxd = VXD(NeuralWeights=ind.genotype.weights, isPassable=p_label != "impassable")
    vxd.set_data(world)
    vxd.set_tags(RecordVoxel=record_history, RecordFixedVoxels=record_history, RecordStepSize=100 if record_history else 0)
    vxd.write(base_name + ".vxd")


def evaluate_population(pop, record_history=False):
    seed = pop.seed

    N = len(pop)
    if record_history:
        N = 1  # only evaluate the best ind in the pop

    # clear old robot files from the data directory
    sub.call("rm data{}/*.vxd".format(seed), shell=True)

    # remove old sim output.xml if we are saving new stats
    # if not record_history:
    #     sub.call("rm ../output/output{0}_{1}.xml".format(seed, pop.gen), shell=True)

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
    # sub.call("rm -rf executables/workspace", shell=True)
    # evaluate new designs
    for r_num, r_label in enumerate(['b']):
        for p_num, p_label in enumerate(["passable_left", "passable_right", "impassable"]):
            for n, ind in enumerate(pop[:N]):

                # don't evaluate if invalid
                if not ind.phenotype.is_valid():
                    for rank, goal in pop.objective_dict.items():
                        if goal["name"] != "age":
                            setattr(ind, goal["name"], goal["worst_value"])

                    sub.call("echo Skipping invalid individual", shell=True)

                # if it's a new valid design, or if we are recording history, create a vxd
                # new designs are evaluated with teammates from the entire population (new and old).
                elif (ind.md5 not in pop.already_evaluated) or record_history:

                    num_evaluated_this_gen += 1
                    pop.total_evaluations += 1

                    create_world(record_history, seed, ind, r_label, p_label)

    # ok let's finally evaluate all the robots in the data directory
    if record_history:  # just save history, don't assign fitness
        sub.call("echo Recording the history of the run champ", shell=True)
        sub.call("rm histories/*", shell=True)
        for r_num, r_label in enumerate(['b']):
            for p_num, p_label in enumerate(["passable_left", "passable_right", "impassable"]):
                sub.call("mkdir data{}".format(str(seed) + get_file_name(r_label, p_label)), shell=True)
                sub.call("cp data{0}/base.vxa data{1}/".format(str(seed), str(seed) + get_file_name(r_label, p_label)), shell=True)
                sub.call(
                    'cp data' + str(seed) + '/' + get_file_name("bot_{:04d}".format(ind.id), r_label, p_label) + ".vxd" + " data{}".format(
                        str(seed) + get_file_name(r_label, p_label)), shell=True)
                sub.call("cd executables; ./voxcraft-sim -i ../data{0} > ../histories/{0}_id{1}_fit{2}.hist -f".format(
                    str(seed) + get_file_name(r_label, p_label), pop[0].id,
                    int(100 * pop[0].fitness)),
                    shell=True)
                sub.call("rm -r data{}".format(str(seed) + get_file_name(r_label, p_label)), shell=True)

    else:  # normally, we will just want to update fitness and not save the trajectory of every voxel

        sub.call("echo " + "GENERATION {}".format(pop.gen), shell=True)

        sub.call("echo Launching {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop)),
                 shell=True)

        while True:
            try:
                sub.call(
                    "cd executables; ./voxcraft-sim -i ../data{0} -o ../output/output{0}_{1}.xml -f".format(seed, pop.gen),
                    shell=True)
                sub.call("echo WE ARE HERE!", shell=True)
                # sub.call waits for the process to return
                # after it does, we collect the results output by the simulator
                root = etree.parse("./output/output{0}_{1}.xml".format(seed, pop.gen)).getroot()
                break

            except IOError:
                sub.call("echo Dang it! There was an IOError. I'll re-simulate this batch again...", shell=True)
                pass

            except IndexError:
                sub.call("echo Shoot! There was an IndexError. I'll re-simulate this batch again...", shell=True)
                pass

        for ind in pop:

            if ind.phenotype.is_valid() and ind.md5 not in pop.already_evaluated:

                for r_num, r_label in enumerate(['b']):
                    for p_num, p_label in enumerate(["passable_left", "passable_right", "impassable"]):
                        sub.call("echo " + str(parse_fitness(root, get_file_name("bot_{:04d}".format(ind.id), r_label, p_label)).text), shell=True)
                        ind.fit_hist += [float(
                            parse_fitness(root, get_file_name("bot_{:04d}".format(ind.id), r_label, p_label)).text)]

                ind.fitness = np.min(ind.fit_hist)
                sub.call("echo Assigning ind {0} fitness {1}".format(ind.id, ind.fitness), shell=True)

                pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                                  for rank, details in
                                                  pop.objective_dict.items()]