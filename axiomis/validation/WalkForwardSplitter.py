from dataclasses import dataclass

import pandas as pd
import numpy as np

from exceptions.WalkForwardError import *

@dataclass
class WalkForwardSplitter:

    nb_train_buckets: int
    nb_validation_buckets: int
    nb_test_buckets: int
    num_buckets: int

    if (nb_train_buckets + nb_validation_buckets + nb_test_buckets > num_buckets):
        raise WalkForwardError("Impossible split. Train/Validate/Test buckets exceed num_buckets")
    
    def bucketize(self,data_array: np.ndarray) -> list[np.ndarray]:
        bucket_list = list()

        data_size = data_array.shape[0]

        bucket_size = data_size // self.num_buckets
        remainder = data_size % self.num_buckets
        
        bucket_start = 0

        for bucket_idx in range(self.num_buckets):
            bucket_end = bucket_start + bucket_size

            if remainder > 0:
                bucket_end += 1
                remainder -= 1
            
            bucket_list.append(data_array[bucket_start:bucket_end])
            bucket_start = bucket_end

        return bucket_list

    def train_validation_test(self,data_array: np.array) -> list[list]:
        return
