from torch.utils.data import Sampler
import numpy as np

class SuperResSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size, n_sq_sum_threshold=None, drop_last=False, shuffle=True):
        """
        Args:
            n_nodes_array: array of the number of nodes (tracks + cells)
            batch_size: batch size
            n_sq_sum_threshold: string that we can eval(). a batch will alsways have n_sq_sum < n_sq_sum_threshold
            drop_last: drop the last batch if it is smaller than batch_size
        """
        super().__init__() # n_nodes_array.size)
        self.dataset_size = n_nodes_array.size
        self.n_nodes_array = n_nodes_array
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_sq_sum_threshold = eval(n_sq_sum_threshold)
        self.shuffle = shuffle

        self.index_to_batch = {}
        running_idx = -1

        n_nodes_args_sorted = np.argsort(n_nodes_array)

        tmp_batch = []; tmp_batch_n_max = 0
        for n_idx in n_nodes_args_sorted:
            tmp_batch_n_max = max(tmp_batch_n_max, n_nodes_array[n_idx])
            n_sq_sum = tmp_batch_n_max**2 * (len(tmp_batch) + 1)

            # make a batch if we are above the threshold or len(tmp_batch) == batch_size
            if (n_sq_sum >= self.n_sq_sum_threshold) or (len(tmp_batch) == batch_size):
                assert len(tmp_batch) > 0, "SSLSampler: computed batch size=0 encountered"
                running_idx += 1
                self.index_to_batch[running_idx] = tmp_batch

                tmp_batch = []
                tmp_batch_n_max = n_nodes_array[n_idx]

            tmp_batch.append(n_idx)

        # add the last batch
        if len(tmp_batch) > 0 and not self.drop_last:
            running_idx += 1
            self.index_to_batch[running_idx] = tmp_batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            batch_order = np.random.permutation(np.arange(self.n_batches))
        else:
            batch_order = np.arange(self.n_batches)
        for i in batch_order:
            yield self.index_to_batch[i]
