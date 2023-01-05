import torch
import numpy as np

NUM_SENSORS = 1
NUM_SIGNALS = 6


class DummyVoxel(object):

    def __init__(self, x, y, z, input_dim, output_dim, genotype):
        self.x = x
        self.y = y
        self.z = z
        self.inputs = np.zeros(input_dim)
        self.outputs = np.zeros(output_dim)
        self.curr_signals = np.zeros(6)
        self.last_signals = np.zeros(6)
        self.h = torch.zeros(1, 1, 6)
        self.rnn = torch.nn.RNN(input_size=NUM_SIGNALS + NUM_SENSORS, hidden_size=6, num_layers=1)
        self.fc = torch.nn.Sequential(torch.nn.Linear(in_features=6, out_features=1 + NUM_SIGNALS),
                                      torch.nn.Tanh()
                                      )
        self._create_brain(genotype)

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

    def _create_brain(self, genotype):
        state_dict = self.rnn.state_dict()
        start = 0
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(np.array(genotype[start:start + num]).reshape(state_dict[key].shape))
            start += num
        self.rnn.load_state_dict(state_dict)
        state_dict = self.fc.state_dict()
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(np.array(genotype[start:start + num]).reshape(state_dict[key].shape))
            start += num
        self.fc.load_state_dict(state_dict)

    def think(self, inputs):
        out, hidden = self.rnn(inputs.reshape(1, 1, -1), self.h)
        out = out.contiguous().view(-1, 6)
        out = self.fc(out)
        self.h = hidden
        return out.squeeze()


def dummy_simulation(genotype, steps, idx, is_passable, terrain_id, age, log_file, record_file=None):
    body = [DummyVoxel(i, 4, 0, 7, 7, genotype) for i in range(9)] + \
           [DummyVoxel(4, i, 0, 7, 7, genotype) for i in range(4)] + \
           [DummyVoxel(4, i, 0, 7, 7, genotype) for i in range(5, 9)]
    votes = []

    for i in range(steps):

        for voxel in body:
            if i >= 300:
                voxel.inputs[0] = -1.0
            elif is_passable == 0:
                voxel.inputs[0] = 1.0 if voxel.x == 4 and voxel.y == 0 else -1.0
            else:
                voxel.inputs[0] = 1.0 if voxel.x > 5 else -1.0
            voxel.get_last_signals(body=body)
            outputs = voxel.think(inputs=torch.from_numpy(voxel.inputs).float())
            voxel.outputs = outputs.detach().numpy()
            for d in range(NUM_SIGNALS):
                voxel.curr_signals[d] = voxel.outputs[1 + d]

        for voxel in body:
            voxel.update_signals()
        if is_passable == 1:
            votes.append(len(list(filter(lambda x: x.outputs[0] >= 0.0, body))) / 17.0)
        else:
            votes.append(len(list(filter(lambda x: x.outputs[0] < 0.0, body))) / 17.0)

        if record_file is not None:
            with open(record_file, "a") as file:
                file.write("{}: ".format(i))
                for voxel in body:
                    file.write("{0},{1},{2},{3},{4}/".format(voxel.outputs[0], voxel.x, voxel.y, voxel.z,
                                                             1 if voxel.inputs[0] == 1.0 else 0))
                file.write("\n")

    sensing = votes[-1]  # sum(votes) / steps
    if log_file is not None:
        with open(log_file, "a") as file:
            file.write("{0}-{1}-{2}-sensing_score: {3}\n".format(idx, terrain_id, age, sensing))
    else:
        print("sensing: {}".format(sensing))
