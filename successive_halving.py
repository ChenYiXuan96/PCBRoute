from typing import Any, Iterator

import numpy as np
import math
import argparse
from argparse import Namespace
import random
import time
import torch
import os
import copy
import pickle
import pprint as pp
from run import run


def get_options():
    parser = argparse.ArgumentParser(
        description="Doing hyper-parameters tuning using successive halving.")
    parser.add_argument('--n_para_sets', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path', type=str, default='HPtuning')
    parser.add_argument('--run_name', type=str, default='PcbRoute5_checkpoint')
    parser.add_argument('--problem', type=str, default='PcbRoute')
    parser.add_argument('--graph_size', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='outputs')
    opts = parser. parse_args()
    return opts


class SuccessiveHalving(object):
    """
    Applies successhalving on a model for n configurations max r ressources.
    """
    def __init__(self, opts):
        self.n_para_sets = opts.n_para_sets
        self.seed = opts.seed
        random.seed(self.seed)
        self.run_name = opts.run_name
        self.problem = opts.problem
        self.graph_size = opts.graph_size
        self.output_dir = opts.output_dir
        self.para_sets = self._get_hyperparameter_configurations()
        self.evaluation = []

        path = '{}_{}'.format(opts.path, time.strftime("%Y%m%dT%H%M%S"))
        self.path = os.path.join(opts.path, path, 'tuning.p')
        os.makedirs(os.path.join(opts.path, path), exist_ok=True)

        assert np.log2(opts.n_para_sets).is_integer(), 'number of parameter sets must be exponential to 2'

    def _get_hyperparameter_configurations(self):
        def get_para_set(i):
            para_set = Namespace()
            para_set.problem = 'PcbRoute'
            para_set.graph_size = 5
            para_set.batch_size = random.choice([64, 128, 256, 512])
            para_set.epoch_size = 4096
            para_set.val_size = 64
            para_set.val_dataset = None
            para_set.model = 'attention'
            # para_set.embedding_dim = random.choice([64, 128])
            para_set.embedding_dim = 128
            # para_set.hidden_dim = random.choice([64, 128])
            para_set.hidden_dim = 128
            para_set.n_encode_layers = random.choice([2, 3, 4])
            para_set.tanh_clipping = 10
            para_set.normalization = 'batch'
            para_set.lr_model = random.choice([0.001, 0.0001])
            para_set.lr_critic = 0.0001
            para_set.lr_decay = np.exp(random.uniform(np.log(0.9), np.log(0.999)))
            para_set.eval_only = False
            para_set.n_epochs = 1
            para_set.seed = 1234
            para_set.max_grad_norm = 1
            para_set.no_cuda = False
            para_set.exp_beta = 0.8
            para_set.baseline = 'rollout'
            para_set.bl_alpha = 0.05
            para_set.bl_warmup_epochs = 1
            para_set.eval_batch_size = 1
            para_set.checkpoint_encoder = False
            para_set.shrink_size = None
            para_set.data_distribution = None
            para_set.log_step = 50
            para_set.log_dir = 'logs'
            para_set.run_name = "PcbRoute5_checkpoint"
            para_set.output_dir = 'outputs'
            para_set.epoch_start = 0  # TODO
            para_set.checkpoint_epochs = 5
            para_set.load_path = None
            para_set.resume = None  # TODO
            para_set.no_tensorboard = False
            para_set.no_progress_bar = False
            para_set.penalty_per_node = np.exp(random.uniform(np.log(5000), np.log(200000)))
            para_set.use_cuda = False
            para_set.save_dir = ''

            para_set.use_cuda = torch.cuda.is_available() and not para_set.no_cuda
            para_set.run_name = "{}_{}_{}".format(para_set.run_name, time.strftime("%Y%m%dT%H%M%S"), i)
            para_set.save_dir = os.path.join(
                para_set.output_dir,
                "{}_{}".format(para_set.problem, para_set.graph_size),
                para_set.run_name
            )
            if para_set.bl_warmup_epochs is None:
                para_set.bl_warmup_epochs = 1 if para_set.baseline == 'rollout' else 0
            assert (para_set.bl_warmup_epochs == 0) or (para_set.baseline == 'rollout')
            assert para_set.epoch_size % para_set.batch_size == 0, "Epoch size must be integer multiple of batch size!"

            return para_set

        para_sets = []
        for i in range(self.n_para_sets):
            this_para_set = get_para_set(i)
            para_sets.append(this_para_set)

        return para_sets

    def _evaluate_halve(self, iter, final=False):

        def iter2n(iter):
            if iter == 0:
                return 1
            else:
                return 2 ** (iter + 1) - 2 ** iter

        if not final:
            evaluated_para_sets = []
            for para_set in self.para_sets:
                avg_cost, avg_cost_over_valid_routes, success_rate, this_epoch, save_dir = run(copy.deepcopy(para_set))
                new_para_set = copy.deepcopy(para_set)
                new_para_set.resume = save_dir + '/epoch-{}.pt'.format(this_epoch)
                new_para_set.n_epochs = iter2n(iter)
                # opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))

                new_para_set.run_name = "{}_{}".format(self.run_name, time.strftime("%Y%m%dT%H%M%S"))
                new_para_set.save_dir = os.path.join(
                    self.output_dir,
                    "{}_{}".format(self.problem, self.graph_size),
                    new_para_set.run_name
                )

                evaluated_para_sets.append((success_rate, new_para_set))

            evaluated_para_sets.sort(key=lambda pair: pair[0], reverse=True)
            this_result = list(map(lambda pair: pair[1], evaluated_para_sets))
            self.para_sets = this_result[:math.ceil(len(evaluated_para_sets) / 2)]
            for para_set in evaluated_para_sets[math.ceil(len(evaluated_para_sets) / 2):]:
                self.evaluation.append((para_set[1], para_set[0]))

            return
        else:
            avg_cost, avg_cost_over_valid_routes, success_rate, this_epoch, save_dir = run(self.para_sets[0])
            self.evaluation.append((self.para_sets[0], success_rate))
            return self.para_sets[0], success_rate, save_dir

    def apply(self):

        n_iters = int(np.log2(self.n_para_sets)) + 1
        results = None
        for i in range(n_iters):
            print('=' * 20 + '\ncurrently processing iteration {}...\n'.format(i) + '=' * 20)
            final = True if i == n_iters - 1 else False
            results = self._evaluate_halve(i, final)  # results: (best_para_set, best_avg_cost, best_save_dir)

        print('=' * 20, '\nresults: (best_para_set, best_avg_cost, best_save_dir)\n', '=' * 20)
        # print(results)

        with open(self.path, 'wb') as file:
            pickle.dump((results, self.evaluation), file)

        return results, self.evaluation


def main():
    opts = get_options()
    sh = SuccessiveHalving(opts)
    results, evaluation_list = sh.apply()
    pp.pprint(vars(results[0]))
    pp.pprint(results[1])
    pp.pprint(results[2])
    # pp.pprint(evaluation_list)


if __name__ == '__main__':
    main()
