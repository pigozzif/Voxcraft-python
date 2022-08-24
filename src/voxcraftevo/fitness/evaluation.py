import abc

from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.representations.population import Individual


class FitnessFunction(object):

    @staticmethod
    def parse_fitness(root, bot_id: str, fitness_tag: str):
        detail = root.findall("detail/")
        for d in detail:
            if d.tag == bot_id:
                return d.findall(fitness_tag)[0]
        raise IndexError

    @abc.abstractmethod
    def create_objectives_dict(self) -> ObjectiveDict:
        pass

    @abc.abstractmethod
    def create_vxa(self, directory: str) -> None:
        pass

    @abc.abstractmethod
    def create_vxd(self, ind: Individual, directory: str, record_history: bool) -> None:
        pass

    @abc.abstractmethod
    def get_fitness(self, ind: Individual, output_file: str) -> dict:
        pass

    @abc.abstractmethod
    def save_histories(self, best: Individual, input_directory: str, output_directory: str, executables_directory: str)\
            -> None:
        pass
