import abc


class FitnessFunction(object):

    @staticmethod
    def parse_fitness(root, bot_id):
        detail = root.findall("detail/")
        for d in detail:
            if d.tag == bot_id:
                return d.findall("fitness_score")[0]
        raise IndexError

    @abc.abstractmethod
    def create_objectives_dict(self):
        pass

    @abc.abstractmethod
    def create_vxa(self, directory):
        pass

    @abc.abstractmethod
    def create_vxd(self, ind, directory, record_history):
        pass

    @abc.abstractmethod
    def get_fitness(self, ind, output_file):
        pass

    @abc.abstractmethod
    def save_histories(self, best, input_directory, output_directory):
        pass
