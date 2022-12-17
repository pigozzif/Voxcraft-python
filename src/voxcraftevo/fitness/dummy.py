import torch
import numpy as np

NUM_SENSORS = 1
NUM_SIGNALS = 6


class DummyVoxel(object):

    def __init__(self, x, y, z, input_dim, output_dim):
        self.x = x
        self.y = y
        self.z = z
        self.inputs = np.zeros(input_dim)
        self.outputs = np.zeros(output_dim)
        self.curr_signals = np.zeros(6)
        self.last_signals = np.zeros(6)

    def get_last_signals(self, body):
        for d in range(NUM_SIGNALS):
            adjacent_voxel = None
            for voxel in body:
                off = self.get_offset(d=d)
                if voxel.x == self.x + off[0] and voxel.y == self.y + off[1] and voxel.z == self.z + off[2]:
                    adjacent_voxel = voxel
                    break
            self.inputs[d + NUM_SENSORS] = adjacent_voxel.last_signals[d] if adjacent_voxel else 0.0

    def update_signals(self):
        for d in range(NUM_SIGNALS):
            self.last_signals[d] = self.curr_signals[d]

    @classmethod
    def get_offset(cls, d):
        if d == 0:
            return [1, 0, 0]
        elif d == 1:
            return [-1, 0, 0]
        elif d == 2:
            return [0, 1, 0]
        elif d == 3:
            return [0, -1, 0]
        elif d == 4:
            return [0, 0, 1]
        else:
            return [0, 0, -1]

    @classmethod
    def create_brain(cls, genotype):
        brain = torch.nn.Sequential(
            torch.nn.Linear(in_features=NUM_SIGNALS + NUM_SENSORS, out_features=1 + NUM_SIGNALS),
            torch.nn.Tanh()
        )
        state_dict = brain.state_dict()
        start = 0
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(np.array(genotype[start:start + num]).reshape(state_dict[key].shape))
            start += num
        brain.load_state_dict(state_dict)
        return brain


def dummy_simulation(genotype, steps, idx, is_passable, terrain_id, age, log_file, record_file=None):
    brain = DummyVoxel.create_brain(genotype=genotype)
    body = [DummyVoxel(i, 5, 0, 7, 7) for i in range(9)] + [DummyVoxel(5, i, 0, 7, 7) for i in range(4)] + \
           [DummyVoxel(5, i, 0, 7, 7) for i in range(5, 9)]
    votes = []

    for i in range(steps):

        for voxel in body:
            if is_passable == 0:
                voxel.inputs[0] = 1.0 if voxel.x == 4 and voxel.y == 0 else -1.0
            else:
                voxel.inputs[0] = 1.0 if voxel.x > 5 else -1.0
            voxel.get_last_signals(body=body)
            voxel.outputs = brain(torch.from_numpy(voxel.inputs).float()).detach().numpy()
            for d in range(NUM_SIGNALS):
                voxel.curr_signals[d] = voxel.outputs[1 + d]

        for voxel in body:
            voxel.update_signals()

        votes.append(1 if len(list(filter(lambda x: x.outputs[0] >= 0.0, body))) >
                          len(list(filter(lambda x: x.outputs[0] < 0.0, body))) else 0)

        if record_file is not None:
            with open(record_file, "a") as file:
                file.write("{}: ".format(i))
                for voxel in body:
                    file.write("{1},{2},{3},{4}/".format(voxel.outputs[0], voxel.x, voxel.y, voxel.z, voxel.inputs[0]))
                file.write("\n")

    sensing = sum([v == is_passable for v in votes]) / steps
    if log_file is not None:
        with open(log_file, "a") as file:
            file.write("{0}-{1}-{2}-sensing_score: {3}\n".format(idx, terrain_id, age, sensing))
