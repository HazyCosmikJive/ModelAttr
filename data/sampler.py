import numpy as np
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler


class BalanceSampler(WeightedRandomSampler):
    '''
    Balance sampler to sovle imbalanced data problem
    '''
    def __init__(self, dataset, class_weight=None):
        self.data_size = len(dataset)
        self.class_weight = class_weight
        sample_weights = self.generate_sample_weights(dataset.infolist)
        
        super().__init__(sample_weights, self.data_size)


    def count(self, infolist):
        # count class distribution
        label_cnt = defaultdict(int)
        for path, label in infolist:
            label_cnt[label] += 1
        
        return label_cnt

    def generate_sample_weights(self, infolist):
        weights = list()
        indices = np.arange(self.data_size)
        label_cnt = self.count(infolist)

        for idx in indices:
            _, label = infolist[idx]
            if self.class_weight is None:
                w = 1.0 / label_cnt[label]
            else:
                w = self.class_weight[label]
            weights.append(w)

        weights = torch.DoubleTensor(weights)

        return weights