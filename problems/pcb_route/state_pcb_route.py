import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import copt


class StatePcbRoute(NamedTuple):
    # Fixed input
    # This is just type hints, not requirements!
    loc: torch.Tensor  # loc (location) is training batch: (batch_size, graph_size, node_dim)

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows  # vertical arange function (batch_size, 1)

    # State
    first_a: torch.Tensor  # (batch_size, 1)
    prev_a: torch.Tensor  # (batch_size, 1)
    visited_: torch.Tensor  # Keeps track of nodes that have been visited  # (batch_size, 1, graph_size)
    # lengths: torch.Tensor  # (batch_size, 1)
    # cur_coord: torch.Tensor  # None
    i: torch.Tensor  # Keeps track of step  # (1)

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key]
                # lengths=self.lengths[key],
                # cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StatePcbRoute, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        # loc (location) is training batch: (batch_size, graph_size, node_dim)
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StatePcbRoute(
            loc=loc,
            # dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),  # dist: distance[batch, first_node, second_node, distance]
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension # a vertical vector
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil # Just ignore it
            ),
            # lengths=torch.zeros(batch_size, 1, device=loc.device),  #lengths of the routes up to now
            # cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        # the method will never be called
        raise NotImplementedError

        # return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        # selected: (batch_size,)

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]  # Guess: it's the coordinates of the previous node (verified)
        # lengths = self.lengths
        # if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
        #     lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        # update visited, i.e. put the visited node to 1...
        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            # visited_: (batch_size, 1, graph_size)
            # prev_a: (batch_size, 1)
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             # lengths=lengths, cur_coord=cur_coord,
                             i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)  # step >= graph_size

    def get_current_node(self):
        # (batch_size, 1)
        return self.prev_a

    def get_mask(self):
        # visited: (batch_size, 1, n_loc), where 0 indicates unvisited, 1 indicates visited
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_mask_constrained(self):
        # visited: (batch_size, 1, n_loc), where 0 indicates unvisited, 1 indicates visited

        # TODO: need to update masking here...
        # # update mask nodes that we can not visit here
        #
        # def is_valid_partial_route(history_node):
        #     history_node = list(map(tuple, history_node))
        #     indexes = list(range(len(history_node)))
        #     result = copt.evaluate(history_node, indexes)
        #
        #     return result['success']
        #
        # def is_valid_partial_route_batch(input, sequences, selected):
        #     sequences_list = []
        #     for tensor in sequences:
        #         sequences_list.append(tensor.tolist())
        #     sequences = sequences_list
        #     history = torch.tensor(np.array(sequences).T)
        #     history_ = history[:, :, None].expand(*history.shape, 4)
        #     history_nodes = torch.gather(input, 1, torch.tensor(history_))
        #     return all(list(map(is_valid_partial_route, history_nodes.tolist())))

        # while not is_valid_partial_route_batch(input, sequences.copy(), selected):

        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
