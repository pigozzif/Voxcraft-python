from random import random


def weighted_random_by_dct(dct):
    rand_val = random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k
    raise RuntimeError("Could not sample from dictionary")
