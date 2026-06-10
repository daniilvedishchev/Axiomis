from dataclasses import dataclass

import pandas as pd
import numpy as np

from exceptions.WalkForwardError import *

@dataclass
class WalkForwardSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    split_id: int

@dataclass
class WalkForwardSplitter:

    nb_train_buckets: int
    nb_validation_buckets: int
    nb_test_buckets: int
    num_buckets: int
    bucket_step: int

    if (nb_train_buckets + nb_validation_buckets + nb_test_buckets > num_buckets):
        raise WalkForwardError("Impossible split. Train/Validate/Test buckets exceed num_buckets")
    
    def bucketize(self,data_array: np.ndarray) -> list[np.ndarray]:
        bucket_list: list = list()

        data_size: int = data_array.shape[0]

        bucket_size: int = data_size // self.num_buckets
        remainder: int = data_size % self.num_buckets
        
        bucket_start: int = 0

        for bucket_idx in range(self.num_buckets):
            bucket_end: int = bucket_start + bucket_size

            if remainder > 0:
                bucket_end += 1
                remainder -= 1
            
            bucket_list.append(data_array[bucket_start:bucket_end])
            bucket_start = bucket_end

        return bucket_list
    
    def moving_buckets(self):
        return self.nb_train_buckets + self.nb_validation_buckets + self.nb_test_buckets

    def train_validation_test(self,data_array: np.ndarray) -> list[WalkForwardSplit]:

        buckets: list[np.ndarray] = self.bucketize(data_array)
        splits: list[WalkForwardSplit] = list()
        max_displacement: int = (self.num_buckets - self.moving_buckets) // self.bucket_step

        for displacement in range(max_displacement + 1):
            split: WalkForwardSplit

            start_train = displacement * self.bucket_step
            end_train = start_train + self.nb_train_buckets
            end_validation = end_train + self.nb_validation_buckets
            end_test = end_validation + self.nb_test_buckets

            split = WalkForwardSplit(
                train=buckets[start_train:end_train],
                validation=buckets[end_train:end_validation],
                test=buckets[end_validation:end_test],
                split_id=displacement
            )
            splits.append(split)
        return splits
