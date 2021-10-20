from torch.utils.data import Dataset
import torch
import os
import pickle
import json
from problems.pcb_route.state_pcb_route import StatePcbRoute
import copt
import copy
import numpy as np
import random
from utils.beam_search import beam_search


class PcbRoute(object):
    NAME = 'PcbRoute'

    @staticmethod
    def get_costs(dataset, pi, penalty_per_node=1e4, normalize=False, normalize_per_node=1000):
        """
        # Currently, both invalid and valid tour can be evaluated. Later to switch to valid tour by backtracking.
        # Valid or invalid here is determined by seeing whether it can route all the data pairs, but not every node
        # is visited only once.
        """

        penalty = penalty_per_node * pi.size()[1]
        # dataset: (batch_size, graph_size, node_dim)
        # pi: (batch_size, graph_size)
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"
        # 'out' here is used to make the data format and the data type of the new tensor align with the old one.

        costs = []
        for problem, order in zip(dataset, pi):
            if not normalize:
                eval_dict = copt.evaluate(list(map(tuple, problem.type(torch.long).tolist())), order.type(torch.long).tolist())
            else:
                std = np.sqrt((650-350)**2/12)
                inv_norm = torch.cat([problem[:, 0][:, None] * std + 500, problem[:, 1][:, None] * std + 500,
                                      problem[:, 2][:, None] * std + 1500, problem[:, 3][:, None] * std + 500], dim=1)
                eval_dict = copt.evaluate(list(map(tuple, inv_norm.type(torch.long).tolist())),
                                          order.type(torch.long).tolist())
            # measure 1: overall cost
            # measure 2: average cost of valid routing, which can also be evaluated by this vector
            # measure 3: success rate, which can also be evaluated by this vector
            # print(eval_dict)
            if not eval_dict['success']:
                if normalize:
                   costs.append(penalty / (normalize_per_node * pi.size()[1]))
                else:
                    costs.append(penalty)
            else:
                if normalize:
                    costs.append(eval_dict['measure'] / (normalize_per_node * pi.size()[1]))
                else:
                    costs.append(eval_dict['measure'])
        return torch.tensor(costs), None
        # return (batch_size,), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PcbRouteDataset(*args, **kwargs)

    @staticmethod
    def make_dataset_BC(*args, **kwargs):
        ds = PcbRouteDatasetBC(*args, **kwargs)
        demos = copy.deepcopy(ds.demos)
        del ds.demos
        demos_reward = copy.deepcopy(ds.demos_rewards)
        del ds.demos_rewards

        return ds, demos, demos_reward

    @staticmethod
    def make_dataset_DAPG_demos(*args, **kwargs):
        ds = PcbRouteDatasetDAPGdemos(*args, **kwargs)
        demos = copy.deepcopy(ds.demos)
        del ds.demos
        demos_reward = copy.deepcopy(ds.demos_rewards)
        del ds.demos_rewards

        return ds, demos, demos_reward

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePcbRoute.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        raise NotImplementedError
        # assert model is not None, "Provide model"
        #
        # fixed = model.precompute_fixed(input)

        # def propose_expansions(beam):
        #     return model.propose_expansions(
        #         beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
        #     )
        #
        # state = PcbRoute.make_state(
        #     input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        # )
        #
        # return beam_search(state, beam_size, propose_expansions)


class PcbRouteDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, normalize=False):
        # size: graph_size
        # num_samples: val_size

        super(PcbRouteDataset, self).__init__()
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.json'

            with open(filename, 'r') as f:
                data = json.load(f)
                self.data = [torch.tensor(list(map(tuple, row[0])), dtype=torch.float) for row in (data[offset:offset + num_samples])]
        else:
            # Generate data on the fly...(can not be parallelized)
            self.data = []
            for _ in range(num_samples):
                data2append = torch.tensor(copt.getProblem(size), dtype=torch.float)
                self.data.append(data2append)
            # self.data = [torch.tensor(copt.getProblem(size), dtype=torch.float) for _ in range(num_samples)]

        self.size = len(self.data)
        if normalize:
            for i in range(len(self.data)):
                d = self.data[i]
                std = np.sqrt((650-350)**2/12)
                self.data[i] = torch.cat([(d[:, 0][:, None] - 500)/std,
                                          ((d[:, 1][:, None] - 500)/std),
                                          ((d[:, 2][:, None] - 1500)/std),
                                          ((d[:, 3][:, None] - 500)/std)], dim=1)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class PcbRouteDatasetBC(Dataset):

    def __init__(self, filename, num_samples=1000000, offset=0, normalize=False, normalize_cost=1000):
        # size: graph_size
        # num_samples: val_size

        assert os.path.splitext(filename)[1] == '.json', 'Provided demo file is not json file.'

        super(PcbRouteDatasetBC, self).__init__()
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.json'

            with open(filename, 'r') as f:
                data = json.load(f)
                self.data = [torch.tensor(list(map(tuple, row[0])), dtype=torch.float) for row in
                             (data[offset:offset + num_samples])]
                graph_size = len(self.data[0])
                self.demos = [row[1]['order'] for row in
                              (data[offset:offset + num_samples])]
                if normalize:
                    self.demos_rewards = [row[1]['measure'] / (graph_size * normalize_cost) for row in
                                          (data[offset:offset + num_samples])]
                else:
                    self.demos_rewards = [row[1]['measure'] for row in
                                          (data[offset:offset + num_samples])]
        else:
            raise ValueError('Demo file not provided!')
        self.size = len(self.data)
        if normalize:
            for i in range(len(self.data)):
                d = self.data[i]
                std = np.sqrt((650-350)**2/12)
                self.data[i] = torch.cat([(d[:, 0][:, None] - 500) / std,
                                          ((d[:, 1][:, None] - 500) / std),
                                          ((d[:, 2][:, None] - 1500) / std),
                                          ((d[:, 3][:, None] - 500) / std)], dim=1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class PcbRouteDatasetDAPGdemos(Dataset):

    def __init__(self, filename, num_samples=1000000, normalize=False, normalize_cost=1000):
        # size: graph_size
        # num_samples: val_size

        assert os.path.splitext(filename)[1] == '.json', 'Provided demo file is not json file.'

        super(PcbRouteDatasetDAPGdemos, self).__init__()
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.json'

            with open(filename, 'r') as f:
                data = json.load(f)
                data_samples = random.choices(data, k=num_samples)
                self.data = [torch.tensor(list(map(tuple, row[0])), dtype=torch.float) for row in data_samples]
                graph_size = len(self.data[0])
                self.demos = [row[1]['order'] for row in data_samples]
                if normalize:
                    self.demos_rewards = [row[1]['measure'] / (graph_size * normalize_cost) for row in data_samples]
                else:
                    self.demos_rewards = [row[1]['measure'] for row in data_samples]

        else:
            raise ValueError('Demo file not provided!')
        self.size = len(self.data)
        if normalize:
            for i in range(len(self.data)):
                d = self.data[i]
                std = np.sqrt((650 - 350) ** 2 / 12)
                self.data[i] = torch.cat([(d[:, 0][:, None] - 500) / std,
                                          ((d[:, 1][:, None] - 500) / std),
                                          ((d[:, 2][:, None] - 1500) / std),
                                          ((d[:, 3][:, None] - 500) / std)], dim=1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]