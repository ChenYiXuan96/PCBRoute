import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from copy import deepcopy
import numpy as np
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

import copt


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor  # (batch_size, graph_size, embedding_dim) node_embed
    context_node_projected: torch.Tensor  # (batch_size, 1, embed_dim) projected graph_embed
    glimpse_key: torch.Tensor  # (n_heads, batch_size, num_steps, graph_size, head_dim) key
    glimpse_val: torch.Tensor  # (n_heads, batch_size, num_steps, graph_size, head_dim) val
    logit_key: torch.Tensor  # (batch_size, 1, graph_size, embedding_dim) query (?)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
                # glimpse_key and logit_key are all keys in equation (6) in the paper,
                # but in different steps
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 penalty_per_node=1e-4):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_pcb_route = problem.NAME == 'PcbRoute'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        self.penalty_per_node = penalty_per_node

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)

            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        elif self.is_pcb_route:  # PcbRoute
            step_context_dim = 2 * embedding_dim
            node_dim = 4

            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)

        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, is_BC=False, normalize=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        # Encoding...
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        # Decoding...
        _log_p, pi = self._inner(input, embeddings, is_BC)
        # return (batch_size, graph_size, graph_size), (batch_size, graph_size)
        # pi is the solution route

        # evaluating...
        if self.is_pcb_route:
            cost, mask = self.problem.get_costs(input, pi, self.penalty_per_node, normalize=normalize)
        else:
            cost, mask = self.problem.get_costs(input, pi)
        # (batch_size,), None # So mask is useless here

        # Log likelihood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        # ll(log likelihood): (batch_size,)
        if is_BC:
            return _log_p
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def get_demo_logp(self, input, demo, on_DAPG=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param demo: (batch_size, graph_size) demo routing
        :return:
        """
        if not on_DAPG:
            # Policy Gradient should be be implemented on evaluating ideal distribution
            with torch.no_grad():
                # Encoding...
                embeddings, _ = self.embedder(self._init_embed(input))

                # Decoding...
                _log_p, pi = self._BC_decoder(input, embeddings, demo, on_DAPG=on_DAPG)
                # return (batch_size, graph_size, graph_size), (batch_size, graph_size)
                # pi is the solution route
                # pi should be the same as demo

                ll = self._calc_log_likelihood(_log_p, pi, None)

                assert (pi == demo).all(), 'Demo output wrong'
            return _log_p
        else:
            # Encoding...
            embeddings, _ = self.embedder(self._init_embed(input))

            # Decoding...
            _log_p, pi = self._BC_decoder(input, embeddings, demo, on_DAPG=on_DAPG)
            # return (batch_size, graph_size, graph_size), (batch_size, graph_size)
            # pi is the solution route
            # pi should be the same as demo

            ll = self._calc_log_likelihood(_log_p, pi, None)

            assert (pi == demo).all(), 'Demo output wrong'
            return _log_p, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        # _log_p: (batch_size, graph_size, graph_size)
        # a (pi): (batch_size, graph_size)
        # mask: None
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        # log_p: (batch_size, graph_size)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)  # sum cuz it's log_p rather than p # return: (batch_size,)

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def _inner(self, input, embeddings, on_BC=False):
        # (batch_size, graph_size, node_dim)
        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)
        # list of Namedtuples(fixed per batch), as shown in attention model fixed line 20

        batch_size = state.ids.size(0)  # batch_size

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            # shrink_size is always None, all_finished == True when step >= graph_size

            # if self.shrink_size is not None:
            #     unfinished = torch.nonzero(state.get_finished() == 0)
            #     if len(unfinished) == 0:
            #         break
            #     unfinished = unfinished[:, 0]
            #     # Check if we can shrink by at least shrink_size and if this leaves at least 16
            #     # (otherwise batch norm will not work well and it is inefficient anyway)
            #     if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
            #         # Filter states
            #         state = state[unfinished]
            #         fixed = fixed[unfinished]

            # Trying backtracking...
            # if self.is_pcb_route:
            #     log_p, mask = self._get_log_p_constrained(fixed, state, input, deepcopy(sequences))
            # else:
            #     log_p, mask = self._get_log_p(fixed, state)

            if not on_BC:
                log_p, mask = self._get_log_p(fixed, state, on_BC=on_BC)
            else:
                log_p, mask, log_p_not_masked = self._get_log_p(fixed, state, on_BC=on_BC)
            # log_p (batch_size, 1, graph_size)
            # mask (batch_size, 1, graph_size)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            # Trying backtracking...
            # if self.is_pcb_route:
            #     selected, is_success = self._pcb_backtracking(log_p.exp()[:, 0, :], mask[:, 0, :], input, deepcopy(sequences))
            # else:
            #     selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            # selected: (batch_size,)

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            # shrink_size is None at the first attempt, so this part can be ignored.
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            if not on_BC:
                outputs.append(log_p[:, 0, :])  # Just squeeze out the dimension of length 1
            else:
                outputs.append(log_p_not_masked[:, 0, :])  # Just squeeze out the dimension of length 1
            # (batch_size, graph_size)
            sequences.append(selected)
            # (batch_size,)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)
        # Pay attention to the difference between stack and concat!
        # return (batch_size, graph_size, graph_size), (batch_size, graph_size)

    def _BC_decoder(self, input, embeddings, demo, on_DAPG=False):
        # input:(batch_size, graph_size, node_dim)
        # embeddings: (batch_size, graph_size, node_dim)
        # demo: (batch_size, graph_size)
        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)
        # list of Namedtuples(fixed per batch), as shown in attention model fixed line 20

        batch_size = state.ids.size(0)  # batch_size

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            # shrink_size is always None, all_finished == True when step >= graph_size

            log_p, mask, log_p_not_masked = self._get_log_p(fixed, state, on_BC=True)

            # log_p (batch_size, 1, graph_size)
            # mask (batch_size, 1, graph_size)

            # Select the indices of the next nodes in the sequences, result (batch_size) long

            # selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            selected = demo[:, i]
            # selected: (batch_size,)

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'

            # Collect output of step
            if not on_DAPG:
                outputs.append(log_p_not_masked[:, 0, :])  # Just squeeze out the dimension of length 1
            else:
                outputs.append(log_p[:, 0, :])  # Just squeeze out the dimension of length 1
            # (batch_size, graph_size)
            sequences.append(selected)
            # (batch_size,)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)
        # Pay attention to the difference between stack and concat!
        # return (batch_size, graph_size, graph_size), (batch_size, graph_size)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        # prob: (batch_size, graph_size)
        # mask: (batch_size, graph_size)
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            # (values, indices) <- return
            _, selected = probs.max(1)
            # selected: (batch_size,)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            # selected (batch_size,)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                # if samples are masked...
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _pcb_backtracking(self, probs, mask, input, sequences):
        # prob: (batch_size, graph_size)
        # mask: (batch_size, graph_size)
        # input: (batch_size, graph_size, node_dim)
        # sequences: [(batch_size,), ..., (batch_size,)]  # (graph_size, batch_size)
        assert (probs == probs).all(), "Probs should not contain any nans"
        probs = deepcopy(probs)

        if self.decode_type == "greedy":
            # (values, indices) <- return
            _, selected = probs.max(1)
            # selected: (batch_size,)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            # selected (batch_size,)

            def is_valid_partial_route(history_node):
                history_node = list(map(tuple, history_node))
                indexes = list(range(len(history_node)))
                result = copt.evaluate(history_node, indexes)

                return result['success']

            def is_valid_partial_route_batch(input, sequences, selected):
                sequences.append(selected)
                sequences_list = []
                for tensor in sequences:
                    sequences_list.append(tensor.tolist())
                sequences = sequences_list
                print(sequences)
                history = torch.tensor(np.array(sequences).T)
                history_ = history[:, :, None].expand(*history.shape, 4)
                history_nodes = torch.gather(input, 1, torch.tensor(history_))
                return all(list(map(is_valid_partial_route, history_nodes.tolist())))

            while not is_valid_partial_route_batch(input, sequences.copy(), selected):
                print('Sampled bad values(invalid partial solution), resampling!')
                selected = probs.multinomial(1).squeeze(1)


            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                # if samples are masked...
                print('Sampled bad values(involving masked nodes), resampling!')
                selected = probs.multinomial(1).squeeze(1)

            # Check whether the current sampling can produce a valid partial routing...


        else:
            assert False, "Unknown decode type"
        return selected


    def _precompute(self, embeddings, num_steps=1):  # seldom see num_step != 1

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # Guess: embeddings: (batch_size, graph_size, embedding_dim)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]  # projected graph_embed

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
        # All three tensors: (batch_size, 1, graph_size, embedding_dim)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),  # (n_heads, batch_size, num_steps, graph_size, head_dim)
            self._make_heads(glimpse_val_fixed, num_steps),  # (n_heads, batch_size, num_steps, graph_size, head_dim)
            logit_key_fixed.contiguous()  # contiguous in memory, i.e. the shape remains unchanged and in the same order
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True, on_BC=False):

        # Compute query = context node embedding
        # _get_parallel_step_context returns (batch_size, 1 <- (num_step), context_dim <- (embed_dim * 2))
        # project_step_context returns (batch_size, 1, embed_dim) -> projected prev_a and first_a
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
        # query: (batch_size, 1, embed_dim)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)  # state is not used

        # Compute the mask
        # visited: (batch_size, 1, n_loc), where False indicates unvisited, True indicates visited
        mask = state.get_mask()  # This returns visited_ of state

        # Compute logits (unnormalized log_p)
        if not on_BC:
            log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, on_BC)
        else:
            log_p, glimpse, log_p_not_masked = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, on_BC)
        # log_p (batch_size, 1, graph_size)
        # glimpse (batch_size, num_steps, embed_dim) # Useless here cuz projected twice

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        if not on_BC:
            return log_p, mask
        else:
            return log_p, mask, torch.log_softmax(log_p_not_masked, dim=-1)

    def _get_log_p_constrained(self, fixed, state, input, sequences, normalize=True):

        # Compute query = context node embedding
        # _get_parallel_step_context returns (batch_size, 1 <- (num_step), context_dim <- (embed_dim * 2))
        # project_step_context returns (batch_size, 1, embed_dim) -> projected prev_a and first_a
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
        # query: (batch_size, 1, embed_dim)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)  # state is not used

        # Compute the mask
        # visited: (batch_size, 1, n_loc), where False indicates unvisited, True indicates visited
        mask = state.get_mask_constrained(input, sequences)  # This returns visited_ of state

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        # log_p (batch_size, 1, graph_size)
        # glimpse (batch_size, num_steps, embed_dim) # Useless here cuz projected twice

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_log_p_backtracking(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        # _get_parallel_step_context returns (batch_size, 1 <- (num_step), context_dim <- (embed_dim * 2))
        # project_step_context returns (batch_size, 1, embed_dim) -> projected prev_a and first_a
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
        # query: (batch_size, 1, embed_dim)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        # visited: (batch_size, 1, n_loc), where False indicates unvisited, True indicates visited
        mask = state.get_mask()  # This returns visited_ of state

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        # log_p (batch_size, 1, graph_size)
        # glimpse (batch_size, num_steps, embed_dim) # Useless here cuz projected twice

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()  # (batch_size, 1)
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    # state.first_a (batch_size, 1)
                    # current_node (batch_size, 1)
                    # codes below are kinda hard to understand. Here are some tips:
                    # torch.cat -> (batch_size, (first_a, current_node), duplicate over dim_embed)
                    # embeddings (batch_size, graph_size, dim_embed)
                    # index array has different shape with input, and the output corresponds with the index
                    # gather -> (batch_size, 2, dim_embed)
                    # view -> (batch_size, 1, 2 * dim_embed)
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )

            # W_placeholder (2 * embed_dim)
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, on_BC=False):
        """
        # glimpse_key: torch.Tensor  # (n_heads, batch_size, num_steps, graph_size, head_dim) key
        # glimpse_val: torch.Tensor  # (n_heads, batch_size, num_steps, graph_size, head_dim) val
        # logit_key: torch.Tensor  # (batch_size, 1, graph_size, embedding_dim) key for producing logits
        # query: (batch_size, 1, embed_dim)  # Later change to multi-heads

        # mask: (batch_size, 1, n_loc), where False indicates unvisited, True indicates visited
        """
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        # key_size = head_dim

        # Compute the glimpse, rearrange dimensions so the dimensions are
        # (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities
        # (n_heads, batch_size, num_steps, graph_size)
        # the size above is given officially, yet I guess it is
        # (n_heads, batch_size, num_steps, 1, graph_size) instead
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # mask_inner is defaultly true
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        # Again, (n_heads, batch_size, num_steps, 1, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        # After permute: (batch_size, num_steps, 1, n_heads, val_size)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
        # glimpse: (batch_size, num_steps, 1, embed_dim)

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        # final_Q: (batch_size, num_steps, 1, embed_dim)
        # logit_K.transpose: (batch_size, 1, embedding_dim, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        # logits (batch_size, 1, graph_size)

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping  # tanh_clipping is C in the paper appendix A

        logits_backup = logits.clone()
        if self.mask_logits:  # always True
            logits[mask] = -math.inf

        if not on_BC:
            return logits, glimpse.squeeze(-2)
        else:
            return logits, glimpse.squeeze(-2), logits_backup

    def _get_attention_node_data(self, fixed, state):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
