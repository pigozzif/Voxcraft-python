import abc


class SolutionMapper(object):

    @abc.abstractmethod
    def __call__(self, genotype):
        pass

    @classmethod
    def create_mapper(cls, name, **kwargs):
        if name == "direct":
            return lambda x: x
        raise ValueError("Invalid mapper name: {}".format(name))
