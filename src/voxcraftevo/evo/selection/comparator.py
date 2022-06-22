import abc

from voxcraftevo.utils.utilities import dominates


class Comparator(object):

    def __init__(self, objective_dict):
        self.objective_dict = objective_dict

    @abc.abstractmethod
    def compare(self, ind1, ind2):
        pass

    @classmethod
    def create_comparator(cls, name, objective_dict):
        if name == "lexicase":
            return LexicaseComparator(objective_dict=objective_dict)
        raise ValueError("Invalid comparator name: {}".format(name))


class LexicaseComparator(Comparator):

    def compare(self, ind1, ind2):
        for rank in reversed(range(len(self.objective_dict))):
            goal = self.objective_dict[rank]
            d = dominates(ind1=ind1, ind2=ind2, attribute_name=goal["name"], maximize=goal["maximize"])
            if d == 1:
                return 1
            elif d == -1:
                return -1
        return 0
